import unittest
from src.tools.benchmark_tools import execute_action, execute_with_results


class CalculationPrecisionTests(unittest.TestCase):
    def test_display_zero_is_not_a_zero_carried_denominator(self):
        facts = {'f0': {'value_token': '1', 'evidence_id': 'e'},
                 'f1': {'value_token': '1000000', 'evidence_id': 'e'}}
        first = {'steps': [{'operation': 'divide', 'args': ['f0', 'f1']}]}
        saved = execute_with_results(first, facts, {})
        self.assertEqual(saved['value'], 0)
        result = execute_with_results(
            {'steps': [{'operation': 'divide', 'args': ['f0', 'c0']}]},
            facts, {'c0': saved})
        self.assertEqual(result['value'], 1000000)
        self.assertEqual(set(result['facts']), {'f0', 'f1'})
        self.assertEqual(execute_action(first, facts)['value'], 0)

    def test_genuinely_zero_carried_denominator_still_rejected(self):
        facts = {'f0': {'value_token': '1', 'evidence_id': 'e'}}
        saved = execute_with_results(
            {'steps': [{'operation': 'subtract', 'args': ['f0', 'f0']}]}, facts, {})
        with self.assertRaisesRegex(ValueError, 'zero denominator'):
            execute_with_results(
                {'steps': [{'operation': 'divide', 'args': ['f0', 'c0']}]},
                facts, {'c0': saved})
