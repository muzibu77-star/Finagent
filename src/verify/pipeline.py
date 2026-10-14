"""Turn extracted claims into per-number verdicts with the deterministic checker.

``judge`` is the middle of the end-to-end review check: given a review's text,
the claims a model extracted from it, the issuer code and the date the draft
was written, it returns a verdict for every number keyed by the number's start
offset in the text.  It is factored out so the same step can be reused by the
LangGraph orchestration, the service page and the MCP tool.
"""

import re

from src.verify.checker import SCALES, Checker, Claim
from src.verify.extract import checkable, locate, matches


def judge(checker: Checker, text: str, items: list[dict], code: str,
          published: str) -> dict[int, dict]:
    """Verdicts for ``items`` keyed by their start offset in ``text``.

    ``items`` are the claims a model extracted from ``text`` (``parse``'s
    output), ``code`` is the issuer and ``published`` the date the draft was
    written, which bounds the evidence and the corrections that apply.  A claim
    that cannot be checked gets an ``unsupported`` verdict carrying the reason;
    the shape of every verdict is unchanged from the end-to-end runner.
    """
    verdicts: dict[int, dict] = {}
    for item in items:
        start = locate(text, str(item['number']), item.get('occurrence'))
        if start is None or start in verdicts:
            continue
        kind, unit, label = item.get('type'), item.get('unit'), item.get('label')
        year = item.get('year')
        if kind not in ('amount', 'growth', 'change', 'ratio') or not isinstance(year, int):
            verdicts[start] = {'verdict': 'unsupported', 'reason': 'not a checkable claim'}
            continue
        if not checkable(item.get('phrase')):
            verdicts[start] = {'verdict': 'unsupported', 'reason': 'quarter, segment or own-method ratio'}
            continue
        if item.get('scope') == '母公司':
            verdicts[start] = {'verdict': 'unsupported', 'reason': 'parent-company claim'}
            continue
        if kind == 'ratio':
            top, bottom = item.get('numerator'), item.get('denominator')
            if unit != '%' or not isinstance(top, list) or not isinstance(bottom, list):
                verdicts[start] = {'verdict': 'unsupported', 'reason': 'ratio parts not named'}
                continue
            claim = Claim(code=code, metric='', fiscal_year=year,
                          stated=str(item['number']), unit=unit, as_of=published,
                          kind='ratio', numerator=top, denominator=bottom)
        else:
            expected_unit = unit == '%' if kind == 'growth' else unit in SCALES
            if not label or not expected_unit:
                verdicts[start] = {'verdict': 'unsupported', 'reason': 'no line item or unit'}
                continue
            if not matches(item.get('phrase'), label):
                verdicts[start] = {'verdict': 'unsupported', 'label': label,
                                   'reason': 'line item does not fit the wording'}
                continue
            claim = Claim(code=code, metric=label, fiscal_year=year,
                          stated=str(item['number']), unit=unit, as_of=published,
                          kind={'amount': 'direct'}.get(kind, kind),
                          direction={'up': 1, 'down': -1}.get(item.get('direction')))
        if not re.fullmatch(r'-?[\d,，]*\.?\d+', claim.stated.strip()):
            verdicts[start] = {'verdict': 'unsupported', 'reason': 'number not written plainly'}
            continue
        verdict = checker.check(claim)
        verdicts[start] = {'verdict': verdict.verdict, 'error': verdict.error,
                           'expected': verdict.expected, 'label': label, 'year': year,
                           'type': kind, 'reason': verdict.reason,
                           'evidence': verdict.evidence}
    return verdicts
