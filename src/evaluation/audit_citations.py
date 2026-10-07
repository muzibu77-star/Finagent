"""Re-score evidence aliases without modifying frozen generations or numeric scores."""

import argparse
import json
from pathlib import Path

from src.evidence.identity import evidence_identity


def audit(run: Path, root: Path) -> dict:
    def rows(name):
        return list(map(json.loads,(root/f'{name}.jsonl').read_text().splitlines()))
    manifest={r['evidence_id']:r for r in rows('manifest')}
    evidence={r['evidence_id']:r for r in rows('evidence')}
    tasks={r['task_id']:r for r in manifest.values()}
    identities={key:evidence_identity(doc,manifest[key]['source_report_id'])
                for key,doc in evidence.items()}
    records=list(map(json.loads,(run/'records.jsonl').read_text().splitlines()))
    audited=[]
    for r in records:
        key=r.get('selected_evidence')
        expected=tasks[r['task_id']]['evidence_id']
        same=key in identities and identities[key]==identities[expected]
        audited.append({'task_id':r['task_id'],'condition':r['condition'],
            'numeric_correct':r['correct'],'source_supported':same,
            'correct_with_source':r['correct'] and same,
            'alias_corrected':same and key!=expected})
    result={'protocol':'same source report AND identical structured visible evidence; evidence ID aliases permitted',
        'numeric_scores_unchanged':True,'results':{c:{
            'total':sum(r['condition']==c for r in audited),
            'correct_with_source':sum(r['correct_with_source'] for r in audited if r['condition']==c),
            'alias_corrected':sum(r['alias_corrected'] for r in audited if r['condition']==c),
        } for c in sorted({r['condition'] for r in audited})},'records':audited}
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.run,Path('data/staged/m0_frozen_v1'))
    target=args.run/'citation_audit.json'
    if target.exists(): raise ValueError('citation audit already exists; preserve original')
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['results']))


if __name__=='__main__':
    main()
