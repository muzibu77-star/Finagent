"""LangGraph orchestration that turns a draft review into a workpaper.

Five nodes run in order.  A model extracts the numbers of the draft and the line
item each one names; the deterministic checker judges every extracted claim;
code finds numbers of one metric, year and unit written with different values
inside the draft; a model proposes a one-sentence revision for each wrong
number, which code accepts only if the sentence is otherwise unchanged; and code
assembles the workpaper.  Only the two model nodes call a model, every verdict
comes from the checker, and the revision node never changes one.
"""

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import sqlite3
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.llm import ApiChatModel
from src.evaluation.draft_pilot import STATED
from src.model.api_client import ApiClient
from src.verify.checker import Checker
from src.verify.extract import PROMPT, line_items, locate, parse
from src.verify.pipeline import judge

# Amounts and percentages the checker can look up; a difference in percentage
# points is derived from two ratios and is left to a person.
UNITS = ('亿元', '万元', '百万元', '%')
BREAK = '。；'

REVISE = '''下面是一篇年报点评里的一句话，其中"{number}"写错了，正确的值是{expected}。请只把这个数字改成正确的值，句子的其余部分（其他数字、单位和文字）一个字都不要改，输出改写后的完整句子。

只输出 JSON：{{"sentence": "改写后的句子"}}

原句：
{sentence}'''

SUMMARY_LABELS = {'consistent': '一致', 'inconsistent': '不一致', 'unsupported': '无依据',
                  'ambiguous': '存疑', 'missed': '抽取遗漏需人工'}


class ReviewState(TypedDict):
    """The data carried between the five nodes of the review graph."""

    text: str
    code: str
    company: str
    fiscal_year: int
    as_of: str
    items: list[dict]
    verdicts: dict[int, dict]
    consistency: list[dict]
    suggestions: dict[int, dict]
    workpaper: dict


def plain(text: str) -> str:
    """The text without digit grouping, so ``1,234.56`` and ``1234.56`` compare equal."""
    return text.replace(',', '').replace('，', '')


def written_value(number: str):
    """The value a written number denotes, or the plain text when it is unparsable."""
    try:
        return Decimal(plain(number))
    except InvalidOperation:
        return plain(number)


def numbers_in(text: str) -> list[dict]:
    """Every checkable amount or percentage of the draft, in reading order."""
    found = []
    for match in STATED.finditer(text):
        if match.group(2) not in UNITS:
            continue
        try:
            Decimal(plain(match.group(1)))
        except InvalidOperation:
            continue
        found.append({'start': match.start(1), 'text': match.group(), 'unit': match.group(2)})
    return found


def sentence_at(text: str, start: int) -> str:
    """The sentence of the draft that holds the number starting at ``start``.

    The draft is split on Chinese full stops and semicolons; the break itself is
    not part of the sentence, so a revision only has to reproduce the words.
    """
    begin = max((text.rfind(ch, 0, start) for ch in BREAK), default=-1) + 1
    ends = [position for ch in BREAK if (position := text.find(ch, start)) != -1]
    end = min(ends) if ends else len(text)
    return text[begin:end].strip()  # a paragraph break is not part of the sentence


def intra_document(items: list[dict], text: str) -> list[dict]:
    """Numbers of one metric, year and unit written with different values.

    Claims are grouped by type, line item, fiscal year and unit; a group whose
    members write different values names the same figure twice, so every member
    is reported with the positions on both sides.
    """
    groups: dict[tuple, dict[int, str]] = {}
    for item in items:
        label, number = item.get('label'), str(item['number'])
        start = locate(text, number, item.get('occurrence'))
        if start is None or not isinstance(label, str) or not label:
            continue
        groups.setdefault((item.get('type'), label, item.get('year'), item.get('unit')),
                          {})[start] = number
    findings = []
    for (kind, label, year, unit), seen in groups.items():
        if len(seen) < 2 or len({written_value(number) for number in seen.values()}) < 2:
            continue
        findings.append({'type': kind, 'label': label, 'year': year, 'unit': unit,
                         'positions': list(seen), 'numbers': list(seen.values())})
    return sorted(findings, key=lambda finding: finding['positions'][0])


def rewritten_sentence(content: str) -> str | None:
    """The sentence in a revision answer; a malformed answer yields nothing."""
    found = re.search(r'\{.*\}', content, re.S)
    if not found:
        return None
    try:
        answer = json.loads(found.group())
    except json.JSONDecodeError:
        return None
    sentence = answer.get('sentence')
    return sentence.strip() if isinstance(sentence, str) and sentence.strip() else None


def evidence_facts(db: sqlite3.Connection, ids: list[int]) -> dict[int, dict]:
    """The file, page, character range, row label, column header and unit of each fact."""
    if not ids:
        return {}
    marks = ', '.join('?' * len(ids))
    rows = db.execute(
        'SELECT f.id, f.page, f.start, f.end, f.row_label, f.column_header, f.unit,'
        ' d.id AS document_id, d.title AS document, d.published'
        ' FROM facts f JOIN documents d ON d.id = f.document_id'
        f' WHERE f.id IN ({marks})', ids).fetchall()
    return {row['id']: {'id': row['id'], 'document': row['document'],
                        'document_id': row['document_id'], 'published': row['published'],
                        'page': row['page'], 'start': row['start'], 'end': row['end'],
                        'row_label': row['row_label'], 'column_header': row['column_header'],
                        'unit': row['unit']} for row in rows}


def workpaper(checker: Checker, state: ReviewState) -> dict:
    """One row per number of the draft: verdict, evidence, revision, consistency."""
    verdicts, suggestions = state['verdicts'], state['suggestions']
    findings = {position: finding for finding in state['consistency']
                for position in finding['positions']}
    numbers = numbers_in(state['text'])
    known = evidence_facts(checker.db, [fact for verdict in verdicts.values()
                                        for fact in verdict.get('evidence', [])])
    rows, summary = [], {key: 0 for key in SUMMARY_LABELS}
    for number in numbers:
        verdict = verdicts.get(number['start'], {'verdict': 'missed'})
        summary[verdict['verdict']] += 1
        finding = findings.get(number['start'])
        rows.append({
            'start': number['start'], 'text': number['text'], 'verdict': verdict['verdict'],
            'error': verdict.get('error'), 'expected': verdict.get('expected'),
            'label': verdict.get('label'), 'year': verdict.get('year'),
            'type': verdict.get('type'), 'reason': verdict.get('reason', ''),
            'evidence': [known.get(fact, {'id': fact}) for fact in verdict.get('evidence', [])],
            'suggestion': suggestions.get(number['start'], {}),
            'consistency': {'intra_document': True, 'positions': finding['positions'],
                            'numbers': finding['numbers']} if finding else None})
    return {'review': {key: state[key] for key in ('code', 'company', 'fiscal_year', 'as_of')},
            'text': state['text'], 'numbers': rows,
            'summary': {'numbers': len(rows), **summary}}


def workpaper_markdown(workpaper: dict) -> str:
    """The workpaper as a table a person can read."""
    review, summary = workpaper['review'], workpaper['summary']
    lines = [f"# 工作底稿：{review['company']}{review['fiscal_year']}年年度报告点评", '',
             f"发行人 {review['code']}，写稿截止日 {review['as_of']}，"
             f"稿件数字 {summary['numbers']} 个。", '', '| 结论 | 数量 |', '|---|---|']
    for key, label in SUMMARY_LABELS.items():
        lines.append(f'| {label} | {summary.get(key, 0)} |')
    lines += ['', '| 位置 | 原文 | 结论 | 错因 | 应有值 | 证据 | 修改建议 | 稿内一致性 |',
              '|---|---|---|---|---|---|---|---|']
    for row in workpaper['numbers']:
        evidence = '；'.join(
            f"{item.get('document')} 第{item.get('page')}页 [{item.get('start')},{item.get('end')}) "
            f"{item.get('row_label')}／{item.get('column_header')}／{item.get('unit')}"
            for item in row['evidence'])
        suggestion = row['suggestion'].get('sentence') or row['suggestion'].get('note', '')
        finding = row['consistency']
        consistency = ('稿内不一致：同一指标另写作 ' + '、'.join(finding['numbers'])
                       if finding else '')
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            row['start'], row['text'], SUMMARY_LABELS.get(row['verdict'], row['verdict']),
            row['error'] or row['reason'], row['expected'] or '', evidence, suggestion, consistency))
    return '\n'.join(lines) + '\n'


def build_graph(checker: Checker, llm) -> CompiledStateGraph:
    """The five-node review graph over ``ReviewState``."""
    db = checker.db

    def extract(state: ReviewState) -> dict:
        labels = line_items(db, state['code'], state['fiscal_year'])
        prompt = PROMPT.format(company=state['company'], year=state['fiscal_year'],
                               labels='\n'.join(labels) or '（无）', review=state['text'])
        return {'items': parse(llm.invoke([HumanMessage(content=prompt)]).content)}

    def check(state: ReviewState) -> dict:
        return {'verdicts': judge(checker, state['text'], state['items'], state['code'],
                                  state['as_of'])}

    def consistency(state: ReviewState) -> dict:
        return {'consistency': intra_document(state['items'], state['text'])}

    def revise(state: ReviewState) -> dict:
        suggestions = {}
        for item in state['items']:
            number = str(item['number'])
            start = locate(state['text'], number, item.get('occurrence'))
            verdict = state['verdicts'].get(start)
            if not verdict or verdict['verdict'] != 'inconsistent' or not verdict.get('expected'):
                continue
            sentence = sentence_at(state['text'], start)
            answer = llm.invoke([HumanMessage(content=REVISE.format(
                number=number, expected=verdict['expected'], sentence=sentence))])
            rewritten = rewritten_sentence(answer.content)
            wanted = sentence.replace(number, verdict['expected'])
            if rewritten is not None and plain(rewritten) == plain(wanted):
                suggestions[start] = {'accepted': True, 'sentence': rewritten}
            else:
                suggestions[start] = {'accepted': False, 'sentence': None, 'note': '需人工改写'}
        return {'suggestions': suggestions}

    def report(state: ReviewState) -> dict:
        return {'workpaper': workpaper(checker, state)}

    graph = StateGraph(ReviewState)
    for name, node in (('extract', extract), ('check', check), ('consistency', consistency),
                       ('revise', revise), ('report', report)):
        graph.add_node(name, node)
    for before, after in ((START, 'extract'), ('extract', 'check'), ('check', 'consistency'),
                          ('consistency', 'revise'), ('revise', 'report'), ('report', END)):
        graph.add_edge(before, after)
    return graph.compile()


def review_draft(draft: dict, *, library: Path, model: str, url: str, budget_usd: float,
                 cache: Path) -> dict:
    """Run the review graph on one draft (``text``, ``code``, ``company``, ``fiscal_year``,
    ``as_of``) and return its workpaper."""
    checker = Checker(sqlite3.connect(f'file:{library}?mode=ro', uri=True))
    llm = ApiChatModel(client=ApiClient(model, budget_usd, Path(cache), url=url))
    state = build_graph(checker, llm).invoke(
        {key: draft[key] for key in ('text', 'code', 'company', 'fiscal_year', 'as_of')})
    return state['workpaper']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True,
                        help='the frozen review set the draft is taken from')
    parser.add_argument('--review-id', required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='the workpaper JSON; the markdown goes beside it as .md and the '
                             'API cache as api_cache.jsonl in the same directory')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--url', default='https://api.deepseek.com/chat/completions')
    parser.add_argument('--budget-usd', type=float, required=True)
    args = parser.parse_args()
    db = sqlite3.connect(f'file:{args.library}?mode=ro', uri=True)
    review = None
    for line in args.reviews.read_text().splitlines():
        row = json.loads(line)
        if row['id'] == args.review_id:
            review = row
    if review is None:
        raise SystemExit(f'review {args.review_id} is not in {args.reviews}')
    published = db.execute('SELECT published FROM documents WHERE id = ?',
                           (review['document_id'],)).fetchone()[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paper = review_draft({'text': review['text'], 'code': review['code'],
                          'company': review['company'], 'fiscal_year': review['fiscal_year'],
                          'as_of': published},
                         library=args.library, model=args.model, url=args.url,
                         budget_usd=args.budget_usd,
                         cache=args.output.parent / 'api_cache.jsonl')
    args.output.write_text(json.dumps(paper, ensure_ascii=False, indent=1) + '\n')
    args.output.with_suffix('.md').write_text(workpaper_markdown(paper))
    print(json.dumps(paper['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
