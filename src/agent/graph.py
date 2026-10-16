"""LangGraph orchestration that turns a draft review into a workpaper.

Four nodes run in order.  A model extracts the numbers of the draft and the line
item each one names; the deterministic checker judges every extracted claim; a
model proposes a one-sentence revision for each wrong number, which code accepts
only if the sentence is otherwise unchanged; and code assembles the workpaper.
Only the model nodes call a model and only the checker settles a verdict, except
that the optional investigate node - between the checker and the revision node -
lets the tool agent re-open the numbers the checker handed to a person; such a
number can only become ``inconsistent``, never ``consistent``, and the revision
node rewrites none of them.

A model node that fails - the call raises, or its answer cannot be parsed -
never aborts the run and never settles a verdict: the numbers it did not reach
stay in the workpaper as numbers a person has to judge, and the reason is
written to the ``errors`` list beside the nodes that failed.
"""

import argparse
from decimal import Decimal, InvalidOperation
from functools import partial
import json
from pathlib import Path
import re
import sqlite3
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.llm import ApiChatModel
from src.agent.single import MEANING, investigate_draft
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
    """The data carried between the nodes of the review graph."""

    text: str
    code: str
    company: str
    fiscal_year: int
    as_of: str
    items: list[dict]
    verdicts: dict[int, dict]
    agents: dict[int, dict]
    suggestions: dict[int, dict]
    errors: list[str]
    workpaper: dict


def plain(text: str) -> str:
    """The text without digit grouping, so ``1,234.56`` and ``1234.56`` compare equal."""
    return text.replace(',', '').replace('，', '')


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


def written_number(number: dict) -> str:
    """The digits of a ``numbers_in`` entry, without its unit."""
    return number['text'][:len(number['text']) - len(number['unit'])].strip()


def occurrence_at(text: str, number: str, start: int) -> int | None:
    """Which occurrence of ``number`` ``locate`` counts at ``start``."""
    for occurrence in range(1, text.count(number) + 2):
        if locate(text, number, occurrence) == start:
            return occurrence
    return None


def note(state: ReviewState, message: str) -> list[str]:
    """The run's errors with one more entry; a failed node keeps the earlier ones."""
    return (state.get('errors') or []) + [message]


def handover(verdicts: dict[int, dict], text: str) -> list[dict]:
    """The numbers the checker left to a person: missed, unsupported or ambiguous.

    Each entry carries the number without its unit, its occurrence in the draft
    and the written text, so an agent answer can be located back into the draft.
    """
    handed = []
    for entry in numbers_in(text):
        verdict = verdicts.get(entry['start'], {}).get('verdict', 'missed')
        if verdict not in ('missed', 'unsupported', 'ambiguous'):
            continue
        number = written_number(entry)
        occurrence = occurrence_at(text, number, entry['start'])
        if occurrence is not None:
            handed.append({'number': number, 'occurrence': occurrence,
                           'start': entry['start'], 'text': entry['text']})
    return handed


def agent_note(agent: dict) -> str:
    """The workpaper note for a number the tool agent judged."""
    if agent['verdict'] == 'consistent':
        return 'Agent 认为一致，仍需人工确认'
    expected = agent.get('expected') or '未给应有值'
    basis = f"（依据：{agent['basis']}）" if agent.get('basis') else ''
    return f'Agent 认为有误：应为 {expected}{basis}，需人工确认'


def sentence_span(text: str, start: int) -> tuple[int, int]:
    """The bounds of the sentence that holds the number starting at ``start``."""
    begin = max((text.rfind(ch, 0, start) for ch in BREAK), default=-1) + 1
    ends = [position for ch in BREAK if (position := text.find(ch, start)) != -1]
    end = min(ends) if ends else len(text)
    return begin, end


def sentence_at(text: str, start: int) -> str:
    """The sentence of the draft that holds the number starting at ``start``.

    The draft is split on Chinese full stops and semicolons; the break itself is
    not part of the sentence, so a revision only has to reproduce the words.
    """
    begin, end = sentence_span(text, start)
    return text[begin:end].strip()  # a paragraph break is not part of the sentence


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
    """One row per number of the draft: verdict, evidence and revision."""
    verdicts, suggestions = state['verdicts'], state['suggestions']
    agents = state.get('agents') or {}
    numbers = numbers_in(state['text'])
    known = evidence_facts(checker.db, [fact for verdict in verdicts.values()
                                        for fact in verdict.get('evidence', [])])
    rows = []
    summary = {key: 0 for key in SUMMARY_LABELS}
    summary.update(agent_flagged=0, agent_hint_consistent=0)
    for number in numbers:
        verdict = verdicts.get(number['start'], {'verdict': 'missed'})
        summary[verdict['verdict']] += 1
        agent = agents.get(number['start'])
        if verdict.get('source') == 'agent':
            source = 'agent'
            summary['agent_flagged'] += 1
        elif number['start'] in verdicts:
            source = 'checker'
        else:
            source = ''
        if agent and agent['verdict'] == 'consistent':
            summary['agent_hint_consistent'] += 1
        rows.append({
            'start': number['start'], 'text': number['text'], 'verdict': verdict['verdict'],
            'source': source, 'agent': agent,
            'error': verdict.get('error'), 'expected': verdict.get('expected'),
            'label': verdict.get('label'), 'year': verdict.get('year'),
            'type': verdict.get('type'), 'reason': verdict.get('reason', ''),
            'evidence': [known.get(fact, {'id': fact}) for fact in verdict.get('evidence', [])],
            'suggestion': suggestions.get(number['start'], {})})
    return {'review': {key: state[key] for key in ('code', 'company', 'fiscal_year', 'as_of')},
            'text': state['text'], 'numbers': rows, 'errors': state.get('errors') or [],
            'summary': {'numbers': len(rows), **summary}}


def workpaper_markdown(workpaper: dict) -> str:
    """The workpaper as a table a person can read."""
    review, summary = workpaper['review'], workpaper['summary']
    lines = [f"# 工作底稿：{review['company']}{review['fiscal_year']}年年度报告点评", '',
             f"发行人 {review['code']}，写稿截止日 {review['as_of']}，"
             f"稿件数字 {summary['numbers']} 个。", '', '| 结论 | 数量 |', '|---|---|']
    for key, label in SUMMARY_LABELS.items():
        lines.append(f'| {label} | {summary.get(key, 0)} |')
    lines += ['', '| 位置 | 原文 | 结论 | 错因 | 应有值 | 证据 | 修改建议 |',
              '|---|---|---|---|---|---|---|']
    for row in workpaper['numbers']:
        evidence = '；'.join(
            f"{item.get('document')} 第{item.get('page')}页 [{item.get('start')},{item.get('end')}) "
            f"{item.get('row_label')}／{item.get('column_header')}／{item.get('unit')}"
            for item in row['evidence'])
        suggestion = row['suggestion'].get('sentence') or row['suggestion'].get('note', '')
        if row.get('agent'):
            suggestion = agent_note(row['agent'])
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            row['start'], row['text'], SUMMARY_LABELS.get(row['verdict'], row['verdict']),
            row['error'] or row['reason'], row['expected'] or '', evidence, suggestion))
    return '\n'.join(lines) + '\n'


def build_graph(checker: Checker, llm, *, agent_llm=None, investigator=None) -> CompiledStateGraph:
    """The review graph over ``ReviewState``.

    ``investigator`` is called as ``investigator(draft, handed)`` for the numbers
    the checker left to a person; ``None`` leaves its node out, so the graph is
    the four-node flow without the tool agent.
    ``agent_llm`` answers the revision prompt; ``None`` uses ``llm``, so a run
    that names no agent model sends every request with the same body as before.
    """
    db = checker.db

    def extract(state: ReviewState) -> dict:
        labels = line_items(db, state['code'], state['fiscal_year'])
        prompt = PROMPT.format(company=state['company'], year=state['fiscal_year'],
                               labels='\n'.join(labels) or '（无）', review=state['text'])
        try:
            answer = llm.invoke([HumanMessage(content=prompt)]).content
        except Exception as error:  # a failed extraction leaves every number to a person
            return {'items': [],
                    'errors': note(state, f'extract failed: {type(error).__name__}: {error}')}
        items = parse(answer)
        if not items:
            return {'items': [],
                    'errors': note(state, f'unparsable extract answer: {answer[:200]!r}')}
        return {'items': items}

    def check(state: ReviewState) -> dict:
        return {'verdicts': judge(checker, state['text'], state['items'], state['code'],
                                  state['as_of'])}

    def investigate(state: ReviewState) -> dict:
        handed = handover(state['verdicts'], state['text'])
        if not handed:
            return {}
        draft = {key: state[key] for key in ('text', 'code', 'company', 'fiscal_year', 'as_of')}
        try:
            run = investigator(draft, handed)
        except Exception as error:  # every checker verdict survives an agent failure
            return {'errors': note(state, f'investigate failed: {type(error).__name__}: {error}')}
        if not run['items']:
            return {'errors': note(state,
                                   f'unparsable investigate answer: {run["answer"][:200]!r}')}
        positions = {entry['start'] for entry in handed}
        verdicts, agents = dict(state['verdicts']), {}
        for item in run['items']:
            start = locate(state['text'], str(item.get('number')), item.get('occurrence'))
            if start is None or start not in positions or start in agents:
                continue
            agent_verdict = MEANING.get(item.get('verdict'))
            if agent_verdict not in ('consistent', 'inconsistent'):
                continue
            agents[start] = {'verdict': agent_verdict, 'expected': item.get('expected') or None,
                             'basis': item.get('basis') or None}
            if agent_verdict == 'inconsistent':
                previous = verdicts.get(start, {})
                verdicts[start] = {'verdict': 'inconsistent', 'source': 'agent',
                                   'expected': agents[start]['expected'],
                                   'label': previous.get('label'), 'year': previous.get('year'),
                                   'type': previous.get('type'), 'error': None, 'reason': '',
                                   'evidence': previous.get('evidence', [])}
        return {'verdicts': verdicts, 'agents': agents}

    def revise(state: ReviewState) -> dict:
        suggestions, errors = {}, state.get('errors') or []
        model = agent_llm if agent_llm is not None else llm
        for item in state['items']:
            number = str(item['number'])
            start = locate(state['text'], number, item.get('occurrence'))
            verdict = state['verdicts'].get(start)
            if (not verdict or verdict['verdict'] != 'inconsistent'
                    or verdict.get('source') == 'agent' or not verdict.get('expected')):
                continue
            sentence = sentence_at(state['text'], start)
            try:
                answer = model.invoke([HumanMessage(content=REVISE.format(
                    number=number, expected=verdict['expected'], sentence=sentence))])
            except Exception as error:  # one failed rewrite leaves that number to a person
                suggestions[start] = {'accepted': False, 'sentence': None, 'note': '需人工改写'}
                errors = errors + [f'revise failed: {type(error).__name__}: {error}']
                continue
            rewritten = rewritten_sentence(answer.content)
            wanted = sentence.replace(number, verdict['expected'])
            if rewritten is not None and plain(rewritten) == plain(wanted):
                suggestions[start] = {'accepted': True, 'sentence': rewritten}
            else:
                suggestions[start] = {'accepted': False, 'sentence': None, 'note': '需人工改写'}
        return {'suggestions': suggestions, 'errors': errors}

    def report(state: ReviewState) -> dict:
        return {'workpaper': workpaper(checker, state)}

    ordered = [('extract', extract), ('check', check)]
    if investigator is not None:
        ordered.append(('investigate', investigate))
    ordered += [('revise', revise), ('report', report)]
    graph = StateGraph(ReviewState)
    for name, node in ordered:
        graph.add_node(name, node)
    graph.add_edge(START, ordered[0][0])
    for (before, _), (after, _) in zip(ordered, ordered[1:]):
        graph.add_edge(before, after)
    graph.add_edge(ordered[-1][0], END)
    return graph.compile()


def review_draft(draft: dict, *, library: Path, model: str, url: str, budget_usd: float,
                 cache: Path, investigate: bool = True, agent_model: str | None = None,
                 agent_url: str | None = None, agent_cache: Path | None = None) -> dict:
    """Run the review graph on one draft (``text``, ``code``, ``company``, ``fiscal_year``,
    ``as_of``) and return its workpaper.

    With ``investigate`` the tool agent re-opens the numbers the checker handed
    to a person; without it the graph is the four-node flow without the agent.
    ``model`` extracts the numbers; the investigate and revision nodes use
    ``agent_model`` on ``agent_url`` through ``agent_cache``, all of which
    default to the extraction ones, so a run that names no agent model is
    served, billed and cached exactly as before.
    """
    checker = Checker(sqlite3.connect(f'file:{library}?mode=ro', uri=True))
    llm = ApiChatModel(client=ApiClient(model, budget_usd, Path(cache), url=url))
    agent_model = agent_model or model
    agent_url = agent_url or url
    agent_cache = Path(agent_cache) if agent_cache else Path(cache)
    if agent_model == model and agent_url == url and agent_cache == Path(cache):
        agent_llm = llm
    else:
        agent_llm = ApiChatModel(
            client=ApiClient(agent_model, budget_usd, agent_cache, url=agent_url))
    investigator = None
    if investigate:
        investigator = partial(investigate_draft, library=Path(library), model=agent_model,
                               url=agent_url, budget_usd=budget_usd, cache=agent_cache)
    state = build_graph(checker, llm, agent_llm=agent_llm, investigator=investigator).invoke(
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
