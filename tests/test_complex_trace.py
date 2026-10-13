import unittest
from src.training.autonomous_sft import successful_trace


class ComplexTraceTests(unittest.TestCase):
    def setUp(self):
        self.identities = {'a': 'report-A', 'alias': 'report-A', 'b': 'report-B'}
        self.label = {'expected': [10, 20], 'evidence_ids': ['a', 'b']}
        self.row = {'execution': 'succeeded', 'business': 'answered', 'report': {
            'calculations': [{'value': 10, 'facts': {'f0': {'evidence_id': 'alias'}}},
                             {'value': 20, 'facts': {'f1': {'evidence_id': 'b'}}}]}}

    def test_two_results_require_exact_order_and_sources(self):
        self.assertTrue(successful_trace(self.row, self.label, self.identities))
        self.row['report']['calculations'].reverse()
        self.assertFalse(successful_trace(self.row, self.label, self.identities))

    def test_missing_calculation_is_not_a_successful_pair(self):
        self.row['report']['calculations'].pop()
        self.assertFalse(successful_trace(self.row, self.label, self.identities))

    def test_later_turn_requires_its_own_exact_result(self):
        label = {'expected': [20], 'evidence_ids': ['b']}
        self.row['report']['calculations'] = self.row['report']['calculations'][1:]
        self.assertTrue(successful_trace(self.row, label, self.identities))
        self.row['report']['calculations'][0]['value'] = 19
        self.assertFalse(successful_trace(self.row, label, self.identities))

    def test_arithmetic_success_with_wrong_source_is_rejected(self):
        self.row['report']['calculations'][1]['facts'] = {'f0': {'evidence_id': 'a'}}
        self.assertFalse(successful_trace(self.row, self.label, self.identities))
