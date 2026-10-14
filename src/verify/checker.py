"""Deterministic checking of one structured claim against the source library.

A claim names an issuer, a metric, a fiscal year, a stated number and a unit,
plus the date the draft is written (``as_of``).  Only documents disclosed on or
before that date are evidence, and a later correction replaces the figure it
corrects.  The verdict is one of ``consistent``, ``inconsistent``,
``unsupported`` (nothing in scope to compare with) or ``ambiguous`` (several
candidate figures disagree).  For an inconsistent claim the most likely kind of
mistake is named by testing what the stated number does equal.
"""

from dataclasses import dataclass, field
from decimal import Decimal
import re
import sqlite3

SCALES = {'元': Decimal(1), '万元': Decimal(10) ** 4, '百万元': Decimal(10) ** 6,
          '亿元': Decimal(10) ** 8, '万亿元': Decimal(10) ** 12}
CURRENT = re.compile(r'本期|本年|期末|年末|本报告期')
PRIOR = re.compile(r'上期|上年|期初|年初')
RESTATED_AWAY = re.compile(r'重述前|调整前')
# A bare "资产负债表" counts too: its columns ("本集团"／"本公司") decide the scope,
# and every lookup below also requires the consolidated scope.
STATEMENT = re.compile(r'(?:合并(?:及(?:母)?公司)?)?(?:利润表|资产负债表|现金流量表)|主要会计数据')
PARENT = re.compile(r'(?:母公司|合并及(?:母)?公司)(?:利润表|资产负债表|现金流量表)')


@dataclass
class Claim:
    code: str
    metric: str
    fiscal_year: int
    stated: str
    unit: str
    as_of: str
    kind: str = 'direct'            # direct | growth | change | ratio | date
    cited_year: int | None = None    # a "数据来源：……Y年年度报告" note
    agenda: str = ''                 # for a dated policy statement
    numerator: list = field(default_factory=list)    # ratio: line items, "-" to subtract
    denominator: list = field(default_factory=list)
    direction: int | None = None     # growth or change: +1 up, -1 down, None unstated


@dataclass
class Verdict:
    verdict: str
    error: str | None = None
    expected: str | None = None
    evidence: list = field(default_factory=list)
    reason: str = ''


def norm(label: str | None) -> str:
    label = re.sub(r'\s', '', label or '')
    label = re.sub(r'[（(][^（()）]*[)）]', '', label)
    label = re.sub(r'^(?:[一二三四五六七八九十]+、|\d+[.、．]|其中[:：]|加[:：]|减[:：])', '', label)
    return re.sub(r'[:：、，,]', '', label)


def decimal(text: str) -> Decimal:
    text = text.replace(',', '').replace('，', '').strip()
    negative = text.startswith('(') and text.endswith(')')
    value = Decimal(text.strip('()%'))
    return -value if negative else value


def places(stated: str) -> int:
    return len(stated.split('.')[1]) if '.' in stated else 0


def close(stated: str, value: Decimal, slack: Decimal = Decimal('0.5')) -> bool:
    """Equal at the written precision; ``slack`` is in units of the last digit."""
    return abs(abs(decimal(stated)) - abs(value)) <= Decimal(1).scaleb(-places(stated)) * slack


OUTFLOW = re.compile(r'使用|流出|支付|净减少|亏损|损失')


def wrong_way(claim: 'Claim', change: Decimal, values: tuple[Decimal, Decimal]) -> bool:
    """A stated rise that was a fall, or the reverse.

    Outflows and losses are left alone: tables show them as negative numbers, and
    "少流出" is a fall in words but a rise in the signed figure.
    """
    if claim.direction is None or change == 0 or OUTFLOW.search(claim.metric):
        return False
    if any(value < 0 for value in values):
        return False
    return (change > 0) != (claim.direction > 0)


def role(header: str | None, fiscal_year: int) -> str | None:
    header = header or ''
    if RESTATED_AWAY.search(header):
        return None  # the comparative as first reported, superseded by the restated one
    # A written year decides first: "2024年末" in a 2025 report is the comparative
    # even though "年末" alone would read as the current period.
    years = set(re.findall(r'(20\d{2})\s*年', header))
    if years:
        if years == {str(fiscal_year)}:
            return 'current'
        return 'prior' if years == {str(fiscal_year - 1)} else None
    if PRIOR.search(header):
        return 'prior'
    return 'current' if CURRENT.search(header) else None


class Checker:
    def __init__(self, library: sqlite3.Connection) -> None:
        self.db = library
        self.db.row_factory = sqlite3.Row
        self._facts: dict[str, list[sqlite3.Row]] = {}

    def documents(self, code: str, as_of: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM documents WHERE code = ? AND published <= ? "
            "AND kind IN ('annual', 'correction') ORDER BY published", (code, as_of)).fetchall()

    def facts(self, document_id: str) -> list[sqlite3.Row]:
        if document_id not in self._facts:
            self._facts[document_id] = self.db.execute(
                'SELECT * FROM facts WHERE document_id = ?', (document_id,)).fetchall()
        return self._facts[document_id]

    def figures(self, claim: Claim, year: int, scope: str = '合并') -> list[tuple]:
        """``(yuan, fact, document)`` for the metric and fiscal year, newest disclosure last."""
        wanted, found = norm(claim.metric), []
        for document in self.documents(claim.code, claim.as_of):
            report_year = document['fiscal_year']
            if report_year not in (year, year + 1):
                continue
            side = 'current' if report_year == year else 'prior'
            for fact in self.facts(document['id']):
                if fact['scale'] is None or norm(fact['row_label']) != wanted:
                    continue
                title = fact['title'] or ''
                pattern = PARENT if scope == '母公司' else STATEMENT
                if not pattern.search(title) and document['kind'] != 'correction':
                    continue
                if (fact['scope'] or ('合并' if '主要会计数据' in title else None)) != scope:
                    continue
                if document['kind'] == 'correction':
                    if report_year != year:
                        continue
                elif role(fact['column_header'], report_year) != side:
                    continue
                found.append((decimal(fact['text']) * Decimal(fact['scale']), fact, document))
        # A correction disclosed later replaces what it corrects.
        corrections = [item for item in found if item[2]['kind'] == 'correction']
        return corrections or [item for item in found if item[2]['fiscal_year'] == year] or found

    def check(self, claim: Claim) -> Verdict:
        if claim.kind == 'date':
            return self.check_date(claim)
        if claim.kind == 'ratio':
            return self.check_ratio(claim)
        if claim.kind == 'change':
            return self.check_change(claim)
        if claim.unit == '%':
            return self.check_growth(claim)
        if claim.unit not in SCALES:
            return Verdict('unsupported', reason=f'unit {claim.unit} is not an amount')
        current = self.figures(claim, claim.fiscal_year)
        if not current:
            return Verdict('unsupported', reason='no figure for this metric and year in scope')
        values = {round(value / SCALES[claim.unit], places(claim.stated)) for value, _, _ in current}
        if len(values) > 1:
            return Verdict('ambiguous', evidence=[f['id'] for _, f, _ in current],
                           reason='disclosed figures for the metric disagree')
        value, fact, document = current[-1]
        expected = value / SCALES[claim.unit]
        shown = str(round(expected, places(claim.stated)))
        if claim.cited_year is not None and claim.cited_year != claim.fiscal_year:
            cited = [d for d in self.documents(claim.code, claim.as_of)
                     if d['fiscal_year'] == claim.cited_year and d['kind'] == 'annual']
            if cited and not any(f['digits'] == fact['digits'] for f in self.facts(cited[0]['id'])):
                return Verdict('inconsistent', 'source', shown, [fact['id'], cited[0]['id']],
                               f'cited report {claim.cited_year} does not contain the figure')
        if close(claim.stated, expected):
            return Verdict('consistent', None, shown, [fact['id']])
        if close(claim.stated, expected, Decimal(1)):
            return Verdict('consistent', None, shown, [fact['id']], 'rounding difference')
        return Verdict('inconsistent', self.diagnose(claim, expected), shown, [fact['id']])

    def diagnose(self, claim: Claim, expected: Decimal) -> str:
        for value, _, _ in self.figures(claim, claim.fiscal_year - 1):
            if close(claim.stated, value / SCALES[claim.unit]):
                return 'period'
        for value, _, _ in self.figures(claim, claim.fiscal_year, scope='母公司'):
            if close(claim.stated, value / SCALES[claim.unit]):
                return 'scope'
        superseded = [value for value, fact, document in self._all_versions(claim)]
        if any(close(claim.stated, value / SCALES[claim.unit]) for value in superseded):
            return 'superseded'
        for unit, scale in SCALES.items():
            if unit != claim.unit and close(claim.stated, expected * SCALES[claim.unit] / scale):
                return 'unit'
        written, right = claim.stated.replace(',', ''), str(round(expected, places(claim.stated)))
        if len(written) == len(right) and sum(a != b for a, b in zip(written, right)) == 1:
            return 'digit'
        return 'value'

    def _all_versions(self, claim: Claim):
        """Original-report figures that a correction in scope has replaced."""
        found = []
        for document in self.documents(claim.code, claim.as_of):
            if document['kind'] != 'annual' or document['fiscal_year'] != claim.fiscal_year:
                continue
            for fact in self.facts(document['id']):
                if fact['scale'] is not None and norm(fact['row_label']) == norm(claim.metric):
                    found.append((decimal(fact['text']) * Decimal(fact['scale']), fact, document))
        return found if any(d['kind'] == 'correction' for d in self.documents(claim.code, claim.as_of)) else []

    def comparative(self, claim: Claim) -> tuple[list, list]:
        """Both years from the claim year's own report, as an analyst would compare.

        A report's comparative column may restate last year's figure, so the
        previous report is used only when the claim year's report has no pair.
        """
        found = {'current': [], 'prior': []}
        for document in self.documents(claim.code, claim.as_of):
            if document['kind'] != 'annual' or document['fiscal_year'] != claim.fiscal_year:
                continue
            for fact in self.facts(document['id']):
                title = fact['title'] or ''
                if (fact['scale'] is None or norm(fact['row_label']) != norm(claim.metric)
                        or not STATEMENT.search(title)
                        or (fact['scope'] or ('合并' if '主要会计数据' in title else None)) != '合并'):
                    continue
                side = role(fact['column_header'], claim.fiscal_year)
                if side:
                    found[side].append((decimal(fact['text']) * Decimal(fact['scale']), fact, document))
        if found['current'] and found['prior']:
            return found['current'], found['prior']
        return self.figures(claim, claim.fiscal_year), self.figures(claim, claim.fiscal_year - 1)

    def check_growth(self, claim: Claim) -> Verdict:
        current, prior = self.comparative(claim)
        if not current or not prior:
            return Verdict('unsupported', reason='both years are needed for a change')
        now, before = current[-1][0], prior[-1][0]
        if before <= 0:
            return Verdict('unsupported', reason='no positive base year')
        change = (now - before) / abs(before) * 100
        shown = str(round(abs(change), places(claim.stated)))
        evidence = [current[-1][1]['id'], prior[-1][1]['id']]
        if wrong_way(claim, change, (current[-1][0], prior[-1][0])):
            return Verdict('inconsistent', 'direction', shown, evidence, 'the figure moved the other way')
        if close(claim.stated, change):
            return Verdict('consistent', None, shown, evidence)
        if close(claim.stated, change, Decimal(1)):
            return Verdict('consistent', None, shown, evidence, 'rounding difference')
        mistaken = (now - before) / abs(now) * 100 if now else None
        error = 'calculation' if mistaken is not None and close(claim.stated, mistaken) else 'value'
        return Verdict('inconsistent', error, shown, evidence)

    def amount_of(self, claim: Claim, label: str) -> tuple | None:
        """One unambiguous current-year figure for a line item, else ``None``."""
        found = self.figures(Claim(claim.code, label, claim.fiscal_year, '0', '元', claim.as_of),
                             claim.fiscal_year)
        values = {value for value, _, _ in found}
        return found[-1] if len(values) == 1 else None

    def sum_of(self, claim: Claim, terms: list[str]) -> tuple[Decimal, list] | None:
        total, evidence = Decimal(0), []
        for term in terms:
            sign = -1 if term.startswith('-') else 1
            found = self.amount_of(claim, term.lstrip('-+'))
            if found is None:
                return None
            total += sign * found[0]
            evidence.append(found[1]['id'])
        return total, evidence

    def check_ratio(self, claim: Claim) -> Verdict:
        """A share or margin: numerator over denominator, as a percentage."""
        if claim.unit != '%' or not claim.numerator or not claim.denominator:
            return Verdict('unsupported', reason='ratio needs both parts')
        top, bottom = self.sum_of(claim, claim.numerator), self.sum_of(claim, claim.denominator)
        if top is None or bottom is None or bottom[0] == 0:
            return Verdict('unsupported', reason='a ratio part has no unambiguous figure')
        value = top[0] / bottom[0] * 100
        if abs(value) >= 10 ** 6:
            return Verdict('unsupported', reason='ratio parts give an implausible percentage')
        shown, evidence = str(round(value, places(claim.stated))), top[1] + bottom[1]
        if close(claim.stated, value):
            return Verdict('consistent', None, shown, evidence)
        if close(claim.stated, value, Decimal(1)):
            return Verdict('consistent', None, shown, evidence, 'rounding difference')
        return Verdict('inconsistent', 'calculation', shown, evidence)

    def check_change(self, claim: Claim) -> Verdict:
        """An increase or decrease in amount: this year minus the comparative year."""
        if claim.unit not in SCALES:
            return Verdict('unsupported', reason=f'unit {claim.unit} is not an amount')
        current, prior = self.comparative(claim)
        if not current or not prior:
            return Verdict('unsupported', reason='both years are needed for a change')
        if len({v for v, _, _ in current}) > 1 or len({v for v, _, _ in prior}) > 1:
            return Verdict('ambiguous', reason='disclosed figures for the metric disagree')
        change = (current[-1][0] - prior[-1][0]) / SCALES[claim.unit]
        shown = str(round(abs(change), places(claim.stated)))
        evidence = [current[-1][1]['id'], prior[-1][1]['id']]
        if wrong_way(claim, change, (current[-1][0], prior[-1][0])):
            return Verdict('inconsistent', 'direction', shown, evidence, 'the figure moved the other way')
        if close(claim.stated, change):
            return Verdict('consistent', None, shown, evidence)
        if close(claim.stated, change, Decimal(1)):
            return Verdict('consistent', None, shown, evidence, 'rounding difference')
        return Verdict('inconsistent', 'calculation', shown, evidence)

    def check_date(self, claim: Claim) -> Verdict:
        """A dated executive meeting: find the release by its agenda, compare the date."""
        rows = self.db.execute(
            "SELECT d.id, d.published, p.text FROM documents d JOIN pages p ON p.document_id = d.id "
            "WHERE d.kind = 'executive-meetings' AND d.published <= ?", (claim.as_of,)).fetchall()
        key = re.sub(r'\s', '', claim.agenda)[:24]
        matches = [row for row in rows if key and key in re.sub(r'\s', '', row['text'])]
        if not matches:
            return Verdict('unsupported', reason='no release with this agenda in scope')
        if len(matches) > 1:
            return Verdict('ambiguous', evidence=[row['id'] for row in matches])
        row = matches[0]
        found = re.search(r'国务院总理李强(\d{1,2})月(\d{1,2})日', row['text'])
        if not found:
            return Verdict('unsupported', evidence=[row['id']], reason='meeting date not found')
        actual = f"{row['published'][:4]}年{int(found.group(1))}月{int(found.group(2))}日"
        if claim.stated == actual:
            return Verdict('consistent', None, actual, [row['id']])
        return Verdict('inconsistent', 'date', actual, [row['id']])

