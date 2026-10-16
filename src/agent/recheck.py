"""Bind one agent answer back to the facts the agent's own run read.

The investigate agent answers in prose: a verdict, the value it expects when it
says the draft is wrong, and a sentence of basis.  A person then has to find the
figure in the annual report by hand.  The run's ``find_number`` calls are
recorded in code (``src/agent/single.py``), so once the answer is parsed this
module looks through those facts for the value the answer rests on - the number
the draft writes for a "consistent" answer, the value the answer expects for an
"inconsistent" one - and returns the matching facts with their page and
character range, which the evidence view already knows how to render.

The facts accompany the note and settle nothing: a number can never become
``consistent`` through them.  Only money is searched, because a growth rate or a
ratio of the draft is not a figure of any table.
"""

import re
from decimal import Decimal

from src.verify.checker import SCALES, close, decimal

SAME_VALUE = '原文中有同值，请核对科目'  # a same value under another line item is possible
EXPECTED_VALUE = '原文应有值'
SELF_CONTRADICTED = 'Agent 自相矛盾：应有值与稿件相同'
LIMIT = 3  # facts attached to one answer, in the order the agent read them

NUMBER = re.compile(r'([-+]?[\d,]+(?:\.\d+)?)\s*(万亿元|百万元|万元|亿元|元|%)?')


def in_unit(fact, unit: str) -> Decimal:
    """The fact's value in the unit of the draft's number.

    A money unit scales the fact as the checker does; anything else is compared
    as written, so a percentage stays in percentage points.
    """
    value = decimal(fact['text']) * Decimal(fact['scale'] or 1)
    return value / SCALES[unit] if unit in SCALES else value


def written(text: str, unit: str) -> tuple[str, str] | None:
    """The digits ``text`` writes and the unit they are in, ``unit`` when unstated.

    ``None`` when ``text`` holds no number at all, so an answer that only says
    why it judged the draft matches no fact.
    """
    found = NUMBER.search(str(text or ''))
    if found is None:
        return None
    return found.group(1), (found.group(2) or unit)


def evidence(fact) -> dict:
    """One found fact in the shape the evidence view reads: where, and what it says."""
    return {'id': fact['id'], 'document_id': fact['document_id'], 'page': fact['page'],
            'start': fact['start'], 'end': fact['end'], 'text': fact['text'],
            'unit': fact['unit'], 'row_label': fact['row_label'],
            'column_header': fact['column_header'], 'title': fact['title']}


def backfill(facts: list[dict], *, verdict: str, stated: str, unit: str,
             expected: str | None = None) -> list[dict]:
    """Up to ``LIMIT`` of a run's facts that carry the value one answer rests on.

    ``facts`` are what the run's ``find_number`` calls returned, in call order;
    ``verdict`` is the agent's own verdict, ``stated`` the number the draft
    writes, ``unit`` its written unit and ``expected`` the value the agent says
    the draft should carry.  A fact matches when its value, converted to the
    unit the answer is written in, is equal to that written value at the written
    precision - the checker's own ``close()``.  An empty list means the run saw
    no such figure, which is not a conclusion about the number.
    """
    if unit not in SCALES:
        return []
    if verdict == 'consistent':
        mark, told = SAME_VALUE, written(stated, unit)
    elif verdict == 'inconsistent' and expected:
        mark, told = EXPECTED_VALUE, written(expected, unit)
    else:
        return []
    if told is None:
        return []
    digits, wanted = told
    drafted = written(stated, unit)
    if mark == EXPECTED_VALUE and drafted and close(drafted[0], decimal(digits)
                                                    * SCALES[wanted] / SCALES[drafted[1]]):
        mark = SELF_CONTRADICTED  # the "wrong" answer expects the very number drafted
    found: list[dict] = []
    seen = set()
    for fact in facts:
        if fact['id'] in seen:
            continue
        seen.add(fact['id'])
        if close(digits, in_unit(fact, wanted)):
            found.append({**evidence(fact), 'mark': mark})
            if len(found) == LIMIT:
                break
    return found
