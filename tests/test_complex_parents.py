import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.training.complex_autonomous import validate_complex_parents


class ComplexParentTests(unittest.TestCase):
    def test_frozen_pairs_and_conversations_cannot_claim_heldout_sources(self):
        manifest = [dict(task_id=key, source_report_id=source, dataset='finqa',
                         original_split=split, official_eligible=True)
                    for key, source, split in [('a', 'fin:A/2010', 'train'),
                                               ('b', 'fin:A/2011', 'train'),
                                               ('x', 'fin:X/2010', 'test')]]
        provenance = {
            'pair': dict(kind='finqa_pair', parent_task_ids=['a', 'b'],
                         source_report_ids=['fin:A/2010', 'fin:A/2011']),
            'dialogue': dict(kind='convfinqa_train', conversation_id='conversation',
                             source_report_ids=['fin:A/2010'])}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'provenance.json').write_text(json.dumps(provenance))
            archive = Path('data/raw/convfinqa/data.zip')
            archive_bytes = b'fake archive; actual annotations are mocked'
            lock = dict(complex_training=True, split='train',
                        conversation_archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
                        sha256={'provenance': hashlib.sha256((root / 'provenance.json').read_bytes()).hexdigest()})
            (root / 'lock.json').write_text(json.dumps(lock))
            digest = hashlib.sha256((root / 'lock.json').read_bytes()).hexdigest()
            samples = [dict(task_id=f'{key}:autonomous:0', parent_task_id=key,
                            source_report_ids=meta['source_report_ids'])
                       for key, meta in provenance.items()]
            original = Path.read_bytes
            def read_bytes(path):
                return archive_bytes if path == archive else original(path)
            with patch.object(Path, 'read_bytes', read_bytes), patch(
                    'src.training.complex_autonomous.training_conversations',
                    return_value=[dict(id='conversation', filename='A/2010/page_1.pdf')]):
                validate_complex_parents(samples, manifest, root, digest)
                for invalid in [samples + samples[:1],
                                [{**samples[0], 'source_report_ids': ['fin:X/2010']}],
                                [{**samples[0], 'parent_task_id': 'x'}],
                                [{**samples[0], 'task_id': 'pair:autonomous:01'}]]:
                    with self.assertRaises(ValueError):
                        validate_complex_parents(invalid, manifest, root, digest)
                with self.assertRaises(ValueError):
                    validate_complex_parents(samples, manifest, root, 'changed')
