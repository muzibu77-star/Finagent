import unittest

from src.training.autonomous_sft import successful_trace, validate_training_parents


class AutonomousTraceTests(unittest.TestCase):
    def setUp(self):
        self.label = {'expected': 30, 'evidence_id': 'gold'}
        self.identities = {'gold': 'report-A-content', 'alias': 'report-A-content',
                           'wrong': 'report-B-content'}

    def row(self, value=30, evidence='alias', execution='succeeded'):
        return {'execution': execution, 'business': 'answered', 'report': {
            'calculations': [{'value': value, 'facts': {'f0': {'evidence_id': evidence}}}]}}

    def test_actual_source_alias_allowed(self):
        self.assertTrue(successful_trace(self.row(), self.label, self.identities))

    def test_executable_wrong_answer_or_source_rejected(self):
        for row in (self.row(value=31), self.row(evidence='wrong'),
                    self.row(execution='failed')):
            self.assertFalse(successful_trace(row, self.label, self.identities))

    def test_uncited_and_missing_report_rejected(self):
        row = self.row()
        row['report']['calculations'][0]['facts'] = {}
        self.assertFalse(successful_trace(row, self.label, self.identities))
        row['report'] = None
        self.assertFalse(successful_trace(row, self.label, self.identities))

    def test_derived_actions_remain_in_the_training_source(self):
        manifest = [{'task_id': key, 'source_report_id': 'report', 'dataset': 'finqa',
                     'original_split': split, 'official_eligible': True}
                    for key, split in [('train', 'train'), ('heldout', 'test')]]
        sample = {'task_id': 'train:autonomous:0', 'parent_task_id': 'train',
                  'source_report_id': 'report'}
        validate_training_parents([sample, {**sample, 'task_id': 'train:autonomous:1'}], manifest)
        bad = [[sample, sample], [{**sample, 'source_report_id': 'other'}],
               [{**sample, 'parent_task_id': 'heldout', 'task_id': 'heldout:autonomous:0'}],
               [{**sample, 'task_id': 'different:autonomous:0'}],
               [{**sample, 'task_id': 'train:autonomous:01'}]]
        for samples in bad:
            with self.assertRaises(ValueError):
                validate_training_parents(samples, manifest)
