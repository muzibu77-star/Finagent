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

``review_events`` runs the same graph and yields one progress event per node, for
the streaming review page to show what each stage did; the workpaper it stores
carries those events back as its ``trace``, so the page can replay a finished run
without calling a model.

When the page is watching a run it wants more than one event per node, so
``review_draft`` hands the graph a ``progress`` callback and every node reports
what it does as it does it: the extraction answer as it is written and each
claim the moment it can be parsed, every conclusion of the checker, every tool
call of the review agent and each revision as it is decided.  Those live events
are only streamed - the entries they carry are also folded into the ``detail`` of
the node event, which is what the workpaper's ``trace`` keeps, so a replay shows
the same steps without any raw model text.  Nothing about a run changes with the
callback: the same nodes, the same requests, the same conclusions.
"""

import argparse
from collections.abc import Callable, Iterator
from decimal import Decimal, InvalidOperation
from functools import partial
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.llm import ApiChatModel
from src.agent.recheck import backfill
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

NODE_LABELS = {'extract': '抽取 Agent', 'check': '核查器', 'investigate': '复查 Agent',
               'revise': '修订 Agent', 'report': '底稿'}
TOOL_RESULT_CHARS = 160


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
    tools: list[dict]
    errors: list[str]
    workpaper: dict
    check_rows: list[dict]
    agent_rows: list[dict]
    revise_rows: list[dict]


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


def summarise(content) -> str:
    """One tool result, cut to ``TOOL_RESULT_CHARS`` so a trace stays small."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    text = text.strip()
    return text[:TOOL_RESULT_CHARS] + '…' if len(text) > TOOL_RESULT_CHARS else text


def tool_event(progress: Callable[[dict], None], name: str, args, result) -> None:
    """One tool call of the review agent, as the page sees it while it happens."""
    progress({'event': 'tool', 'name': name, 'args': args, 'result': summarise(result)})


def complete_objects(text: str) -> list[str]:
    """Every complete top-level ``{...}`` of a half-written answer, in order.

    The extraction answer is a JSON array that names one claim per object; while
    it is still being written, only the objects that have closed can be read, and
    those are what the page marks in the draft.  Braces inside a string do not
    count, so a claim that quotes one cannot cut an object short.
    """
    found, depth, begin, quoted, escaped = [], 0, None, False, False
    for at, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == '{':
            if depth == 0:
                begin = at
            depth += 1
        elif char == '}':
            if depth:
                depth -= 1
                if depth == 0:
                    found.append(text[begin:at + 1])
    return found


def extraction_entry(text: str, item: dict) -> dict:
    """One extracted claim as a progress row: where it is and what it says."""
    return {'start': locate(text, str(item.get('number')), item.get('occurrence')),
            'number': str(item.get('number')), 'type': item.get('type'),
            'label': item.get('label'), 'year': item.get('year')}


class ExtractWatcher:
    """The extraction answer as it arrives: raw text out, parsed claims in order.

    Each text segment the model writes is reported as it comes and every object
    that has closed is read as one claim, so the page can mark the numbers of the
    draft while the extractor is still writing.  The claim list of the node's own
    event is the parsed answer, which is what the workpaper's trace keeps.
    """

    def __init__(self, progress: Callable[[dict], None], text: str) -> None:
        self.progress, self.text, self.buffer, self.read = progress, text, '', 0

    def feed(self, piece: str) -> None:
        self.buffer += piece
        self.progress({'event': 'delta', 'text': piece})
        whole = complete_objects(self.buffer)
        while self.read < len(whole):
            content = whole[self.read]
            self.read += 1
            try:
                item = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get('number'):
                self.progress({'event': 'item', **extraction_entry(self.text, item)})


def candidate_values(db: sqlite3.Connection, ids: list[int]) -> dict[int, dict]:
    """The value, unit and page of each evidence fact, for the progress rows."""
    if not ids:
        return {}
    marks = ', '.join('?' * len(ids))
    rows = db.execute(f'SELECT id, page, text, unit FROM facts WHERE id IN ({marks})',
                      ids).fetchall()
    return {row['id']: {'id': row['id'], 'page': row['page'], 'text': row['text'],
                        'unit': row['unit']} for row in rows}


def check_entries(checker: Checker, text: str, verdicts: dict[int, dict]) -> list[dict]:
    """One progress row per number of the draft: its conclusion and its candidates.

    The candidates are the figures of the source library the checker weighed, or
    would have weighed, for that number; naming them is a report, not a verdict.
    """
    values = candidate_values(checker.db, [fact for verdict in verdicts.values()
                                           for fact in verdict.get('evidence', [])])
    entries = []
    for number in numbers_in(text):
        verdict = verdicts.get(number['start'], {'verdict': 'missed'})
        entries.append({'start': number['start'], 'text': number['text'],
                        'verdict': verdict['verdict'], 'label': verdict.get('label'),
                        'year': verdict.get('year'), 'reason': verdict.get('reason', ''),
                        'error': verdict.get('error'), 'expected': verdict.get('expected'),
                        'candidates': [values.get(fact, {'id': fact})
                                       for fact in verdict.get('evidence', [])]})
    return entries


def tool_calls(messages: list[dict] | None) -> list[dict]:
    """What a tool-agent run called, in order: the tool, its arguments, what it read.

    ``messages`` is the run's message trace as ``investigate_draft`` returns it.
    Each ``tool`` message belongs to the call that precedes it, which is the
    order the agent made them in; the result is shortened, since the trace is
    written into the workpaper for the page's replay.
    """
    calls: list[dict] = []
    for message in messages or []:
        for call in message.get('tool_calls') or []:
            calls.append({'name': call.get('name'), 'args': call.get('args') or {},
                          'result': None})
        if message.get('type') == 'tool':
            for call in calls:
                if call['result'] is None:
                    call['result'] = summarise(message.get('content'))
                    break
    return calls


def agent_note(agent: dict) -> str:
    """The workpaper note for a number the tool agent judged.

    Figures the agent's own lookups turned up beside the drafted number (or
    beside the value it expects) are named after it.  They are evidence for a
    person to open, not a verdict, and they never change the note's meaning.
    """
    if agent['verdict'] == 'consistent':
        note = 'Agent 认为一致，仍需人工确认'
    else:
        expected = agent.get('expected') or '未给应有值'
        basis = f"（依据：{agent['basis']}）" if agent.get('basis') else ''
        note = f'Agent 认为有误：应为 {expected}{basis}，需人工确认'
    cited = agent.get('evidence') or []
    return note + f"（{cited[0]['mark']} {len(cited)} 条）" if cited else note


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


def conclusion_counts(verdicts: dict[int, dict], text: str) -> dict[str, int]:
    """How many numbers of the draft each conclusion holds, in the workpaper's wording."""
    counts = {label: 0 for label in SUMMARY_LABELS.values()}
    for number in numbers_in(text):
        verdict = verdicts.get(number['start'], {}).get('verdict', 'missed')
        counts[SUMMARY_LABELS[verdict]] += 1
    return counts


def node_detail(node: str, state: ReviewState, earlier: dict) -> dict:
    """What one node of the run did, in the numbers its card shows.

    ``state`` holds every output up to and including this node; ``earlier`` the
    outputs before it, which the investigate card needs: the numbers it was
    handed are the ones the checker left to a person, not what is left of them
    after the agent has answered.

    Beside the card's counts, each event carries the entries the page replays:
    the claims of the extraction, the conclusion and candidates of every number
    the checker judged, the lookups and conclusions of the agent, and the
    revisions.  The extraction keeps no raw model text, only its claims.
    """
    if node == 'extract':
        return {'extracted': len(state.get('items') or []),
                'items': [extraction_entry(state['text'], item)
                          for item in state.get('items') or []]}
    if node == 'check':
        return {'conclusions': conclusion_counts(state.get('verdicts') or {}, state['text']),
                'numbers': state.get('check_rows') or []}
    if node == 'investigate':
        agents = state.get('agents') or {}
        return {'handed': len(handover(earlier.get('verdicts') or {}, state['text'])),
                'judged': len(agents),
                'flagged': sum(1 for entry in agents.values()
                               if entry['verdict'] == 'inconsistent'),
                'tools': state.get('tools') or [],
                'numbers': state.get('agent_rows') or []}
    if node == 'revise':
        suggestions = state.get('suggestions') or {}
        return {'advised': len(suggestions),
                'accepted': sum(1 for entry in suggestions.values() if entry.get('accepted')),
                'numbers': state.get('revise_rows') or []}
    return {'numbers': state['workpaper']['summary']['numbers']}


def review_events(graph, state: ReviewState) -> Iterator[dict]:
    """One progress event per node of ``graph``, in the order the nodes run.

    The events are what the review page streams: the node's name and label, the
    time it started, how long it took and the highlight of what it did.  The
    event of the ``report`` node also carries the workpaper that node built, under
    ``workpaper``, for the caller to keep; nothing else about the run changes.
    """
    started, at = time.monotonic(), time.strftime('%H:%M:%S')
    seen: dict = dict(state)  # the draft, then every node's output as it lands
    for chunk in graph.stream(state, stream_mode='updates'):
        for node, update in chunk.items():
            now = time.monotonic()
            earlier, seen = seen, {**seen, **update}
            event = {'node': node, 'label': NODE_LABELS.get(node, node), 'at': at,
                     'seconds': round(now - started, 2),
                     'detail': node_detail(node, seen, earlier)}
            if 'workpaper' in update:
                event['workpaper'] = update['workpaper']
            yield event
            started, at = now, time.strftime('%H:%M:%S')


def build_graph(checker: Checker, llm, *, agent_llm=None, investigator=None,
                progress=None) -> CompiledStateGraph:
    """The review graph over ``ReviewState``.

    ``investigator`` is called as ``investigator(draft, handed)`` for the numbers
    the checker left to a person; ``None`` leaves its node out, so the graph is
    the four-node flow without the tool agent.
    ``agent_llm`` answers the revision prompt; ``None`` uses ``llm``, so a run
    that names no agent model sends every request with the same body as before.

    ``progress``, when given, is called with one event as each node does its
    work: the extraction text as it is written, each claim parsed from it, every
    conclusion of the checker, every tool call of the agent and every revision.
    Without it the nodes report nothing and the extraction is asked for exactly
    as it was before, in one non-streaming request.
    """
    db = checker.db

    def extract(state: ReviewState) -> dict:
        labels = line_items(db, state['code'], state['fiscal_year'])
        prompt = PROMPT.format(company=state['company'], year=state['fiscal_year'],
                               labels='\n'.join(labels) or '（无）', review=state['text'])
        watcher = ExtractWatcher(progress, state['text']) if progress is not None else None
        try:
            answer = llm.invoke([HumanMessage(content=prompt)],
                                **({'on_delta': watcher.feed} if watcher else {})).content
        except Exception as error:  # a failed extraction leaves every number to a person
            return {'items': [],
                    'errors': note(state, f'extract failed: {type(error).__name__}: {error}')}
        items = parse(answer)
        if not items:
            return {'items': [],
                    'errors': note(state, f'unparsable extract answer: {answer[:200]!r}')}
        return {'items': items}

    def check(state: ReviewState) -> dict:
        verdicts = judge(checker, state['text'], state['items'], state['code'], state['as_of'])
        rows = check_entries(checker, state['text'], verdicts)
        if progress is not None:
            for row in rows:
                progress({'event': 'check', **row})
        return {'verdicts': verdicts, 'check_rows': rows}

    def investigate(state: ReviewState) -> dict:
        handed = handover(state['verdicts'], state['text'])
        if not handed:
            return {'tools': [], 'agent_rows': []}
        draft = {key: state[key] for key in ('text', 'code', 'company', 'fiscal_year', 'as_of')}
        try:
            run = investigator(draft, handed)
        except Exception as error:  # every checker verdict survives an agent failure
            return {'errors': note(state, f'investigate failed: {type(error).__name__}: {error}')}
        if not run['items']:
            return {'errors': note(state,
                                   f'unparsable investigate answer: {run["answer"][:200]!r}')}
        positions = {entry['start']: entry for entry in handed}
        facts = run.get('facts') or []
        verdicts, agents, rows = dict(state['verdicts']), {}, []
        for item in run['items']:
            start = locate(state['text'], str(item.get('number')), item.get('occurrence'))
            if start is None or start not in positions or start in agents:
                continue
            agent_verdict = MEANING.get(item.get('verdict'))
            if agent_verdict not in ('consistent', 'inconsistent'):
                continue
            answer = positions[start]
            entry = {'verdict': agent_verdict, 'expected': item.get('expected') or None,
                     'basis': item.get('basis') or None}
            # The agent names no fact id in its answer; the facts its own
            # ``find_number`` calls returned are attached here instead, as
            # evidence for the person.  They add nothing to the verdict below.
            cited = backfill(facts, verdict=agent_verdict, stated=answer['number'],
                             unit=answer['text'][len(answer['number']):].strip(),
                             expected=entry['expected'])
            if cited:
                entry['evidence'] = cited
            agents[start] = entry
            if agent_verdict == 'inconsistent':
                previous = verdicts.get(start, {})
                verdicts[start] = {'verdict': 'inconsistent', 'source': 'agent',
                                   'expected': entry['expected'],
                                   'label': previous.get('label'), 'year': previous.get('year'),
                                   'type': previous.get('type'), 'error': None, 'reason': '',
                                   'evidence': previous.get('evidence', [])}
            rows.append({'start': start, 'text': answer['text'], 'verdict': agent_verdict,
                         'expected': entry['expected'], 'basis': entry['basis'],
                         'note': agent_note(entry), 'evidence': entry.get('evidence') or []})
        if progress is not None:  # the conclusions the agent reached, once it reached them
            for row in rows:
                progress({'event': 'agent', **row})
        # The run's tool calls travel with the verdicts for the progress page;
        # they settle nothing and the workpaper's rows do not read them.
        return {'verdicts': verdicts, 'agents': agents, 'agent_rows': rows,
                'tools': tool_calls(run.get('messages'))}

    def revise(state: ReviewState) -> dict:
        suggestions, errors, rows = {}, state.get('errors') or [], []
        model = agent_llm if agent_llm is not None else llm
        for item in state['items']:
            number = str(item['number'])
            start = locate(state['text'], number, item.get('occurrence'))
            verdict = state['verdicts'].get(start)
            if (not verdict or verdict['verdict'] != 'inconsistent'
                    or verdict.get('source') == 'agent' or not verdict.get('expected')):
                continue
            sentence = sentence_at(state['text'], start)
            row = {'start': start, 'number': number, 'sentence': sentence,
                   'expected': verdict['expected'], 'rewritten': None, 'accepted': False,
                   'note': None}
            try:
                answer = model.invoke([HumanMessage(content=REVISE.format(
                    number=number, expected=verdict['expected'], sentence=sentence))])
            except Exception as error:  # one failed rewrite leaves that number to a person
                suggestions[start] = {'accepted': False, 'sentence': None, 'note': '需人工改写'}
                errors = errors + [f'revise failed: {type(error).__name__}: {error}']
            else:
                rewritten = rewritten_sentence(answer.content)
                wanted = sentence.replace(number, verdict['expected'])
                if rewritten is not None and plain(rewritten) == plain(wanted):
                    suggestions[start] = {'accepted': True, 'sentence': rewritten}
                    row.update(rewritten=rewritten, accepted=True)
                else:
                    suggestions[start] = {'accepted': False, 'sentence': None,
                                          'note': '需人工改写'}
            if not suggestions[start]['accepted']:
                row['note'] = '需人工改写'
            rows.append(row)
            if progress is not None:
                progress({'event': 'revise', **row})
        return {'suggestions': suggestions, 'errors': errors, 'revise_rows': rows}

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
                 agent_url: str | None = None, agent_cache: Path | None = None,
                 on_event: Callable[[dict], None] | None = None) -> dict:
    """Run the review graph on one draft (``text``, ``code``, ``company``, ``fiscal_year``,
    ``as_of``) and return its workpaper, whose ``trace`` holds the progress events.

    With ``investigate`` the tool agent re-opens the numbers the checker handed
    to a person; without it the graph is the four-node flow without the agent.
    ``model`` extracts the numbers; the investigate and revision nodes use
    ``agent_model`` on ``agent_url`` through ``agent_cache``, all of which
    default to the extraction ones, so a run that names no agent model is
    served, billed and cached exactly as before.  ``on_event`` receives every
    progress event as the node finishes, which is how the streaming page follows
    a run; a caller that passes none still gets them in the workpaper's trace.

    A caller that passes ``on_event`` is watching the run, so the nodes also
    stream: the extraction answer arrives as it is written, and the checker's
    conclusions, the agent's tool calls and the revisions follow one by one.
    Those live events are not kept - the workpaper's ``trace`` holds the entries
    of each node under its ``detail``, which is what a replay reads - and a run
    without ``on_event`` asks the models for exactly what it asked for before.
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
        if on_event is not None:
            investigator = partial(investigator, on_call=partial(tool_event, on_event))
    graph = build_graph(checker, llm, agent_llm=agent_llm, investigator=investigator,
                        progress=on_event)
    events: list[dict] = []
    paper: dict = {}
    for event in review_events(graph, {key: draft[key]
                                       for key in ('text', 'code', 'company',
                                                   'fiscal_year', 'as_of')}):
        paper = event.pop('workpaper', paper)
        events.append(event)
        if on_event is not None:
            on_event(event)
    paper['trace'] = events
    # the page labels the flow with the models that actually ran, not the form's choice
    paper['models'] = {'extract': model, 'agent': agent_model}
    return paper


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
