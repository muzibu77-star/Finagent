"""Run the deterministic checker on frozen seed claims with their own structure.

Each seed claim already carries its issuer, metric, year, number and unit, so
this measures checking alone: retrieval of the right figure, comparison and
naming the mistake.  Extraction of claims from free text is measured separately.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from src.verify.checker import Checker, Claim

NAMES = {'E1': 'digit', 'E2': 'unit', 'E3': 'period', 'E4': 'scope', 'E5': 'calculation',
         'E6': 'superseded', 'E7': 'date', 'E8': 'source', 'E9': 'digit'}
TODAY = '2026-10-14'


def structured(record: dict) -> Claim:
    """The claim exactly as the seed generator wrote it."""
    if record['kind'] == 'date':
        agenda = record['text'].split('国务院常务会议', 1)[1].rstrip('。')
        return Claim(code='', metric='国务院常务会议', fiscal_year=0, stated=record['stated'],
                     unit='', as_of=TODAY, kind='date', agenda=agenda)
    cited = record.get('cited_report')
    return Claim(code=record['code'], metric=record['metric'], fiscal_year=record['fiscal_year'],
                 stated=record['stated'], unit=record['unit'], as_of=record['as_of'],
                 kind='growth' if record['kind'] == 'derived' else 'direct',
                 cited_year=cited[1] if cited else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--claims', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checker = Checker(sqlite3.connect(f'file:{args.library}?mode=ro', uri=True))
    rows, table = [], {}
    for line in args.claims.read_text().splitlines():
        record = json.loads(line)
        verdict = checker.check(structured(record))
        truth = record['error'] or 'consistent'
        rows.append({'id': record['id'], 'truth': truth, 'verdict': verdict.verdict,
                     'error': verdict.error, 'expected': verdict.expected,
                     'evidence': verdict.evidence, 'reason': verdict.reason})
        cell = table.setdefault(truth, Counter())
        cell[verdict.verdict] += 1
        if truth != 'consistent' and verdict.verdict == 'inconsistent':
            cell['named correctly'] += verdict.error == NAMES[truth]
    (args.output_dir / 'records.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    summary = {}
    for truth, cell in sorted(table.items()):
        total = sum(cell[v] for v in ('consistent', 'inconsistent', 'unsupported', 'ambiguous'))
        summary[truth] = {'claims': total, **{v: cell[v] for v in
                          ('consistent', 'inconsistent', 'unsupported', 'ambiguous')}}
        if truth != 'consistent':
            summary[truth]['named correctly'] = cell['named correctly']
    errors = [cell for truth, cell in table.items() if truth != 'consistent']
    flagged = sum(cell['inconsistent'] for cell in errors)
    total_errors = sum(sum(cell[v] for v in ('consistent', 'inconsistent', 'unsupported',
                                              'ambiguous')) for cell in errors)
    correct = table.get('consistent', Counter())
    summary['overall'] = {
        'error_claims': total_errors, 'detected': flagged,
        'detection_rate': round(flagged / total_errors, 4) if total_errors else None,
        'correct_claims': sum(correct.values()), 'false_alarms': correct['inconsistent'],
        'false_alarm_rate': round(correct['inconsistent'] / max(1, sum(correct.values())), 4),
        'handed_to_human': sum(cell['unsupported'] + cell['ambiguous'] for cell in table.values())}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary['overall']))


if __name__ == '__main__':
    main()
