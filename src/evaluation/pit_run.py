"""Replay verified disclosure dates on the original PDF snapshot, without model calls."""

import argparse
import json
from pathlib import Path

from src.data.report_corpus import ReportCorpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    corpus = ReportCorpus('data/staged/reports_v2/corpus.json', 'reports-v1')
    records = []
    for cutoff, expected in [('2023-03-22', 0), ('2023-03-23', 6),
                             ('2024-03-20', 6), ('2024-03-21', 11)]:
        payload = {'company': 'CHL', 'period': ['2022', '2023'],
                   'snapshot_id': 'reports-v1', 'as_of': cutoff}
        allowed = corpus.allowed(payload)
        hits = corpus.search('营运收入 2023 资本开支', allowed)
        assert len(allowed) == expected
        assert all(corpus.documents[key]['published_at'] <= cutoff for key in allowed)
        assert {hit['evidence_id'] for hit in hits} <= allowed
        rejected = set(corpus.documents) - allowed
        for key in rejected:
            try:
                corpus.read(key, allowed)
            except ValueError:
                continue
            raise AssertionError('source authorization bypass')
        records.append({'as_of': cutoff, 'eligible_pages': len(allowed),
                        'eligible_ids': sorted(allowed), 'retrieved': hits,
                        'rejected_reads': len(rejected)})
    result = {'passed': True, 'snapshot': corpus.snapshot_id, 'corpus_sha256': corpus.fingerprint,
              'records': records,
              'limits': 'Publication-date replay at inclusive calendar-day granularity. Source snapshot preserves actual later ingestion; no claim that these documents were locally ingested in 2024. Revised-version behavior is separately covered by synthetic tests; annual-report section publication dates remain unknown and are excluded.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'passed': True, 'counts': [r['eligible_pages'] for r in records]}))
    corpus.close()


if __name__ == '__main__':
    main()
