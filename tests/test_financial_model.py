import unittest
from unittest.mock import patch

from src.model.financial import predict


class FinancialModelTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {'evidence_id': 'e1', 'table': [['Revenue', '120', '150']]}
        self.config = {'generation': {}, 'model': {'device': 'cpu'}}

    def run_output(self, text, stopped=True):
        with patch('src.model.financial.generate', return_value={
            'text': text, 'stopped_on_eos': stopped}):
            return predict(None, None, self.evidence, 'Change?', 'calculator', self.config)

    def test_native_nested_actions_execute_with_citations(self):
        output = ('<tool_call>\n<function=calculate>\n<parameter=steps>\n'
                  '[{"operation":"subtract","args":["f1","f0"]}]\n'
                  '</parameter>\n</function>\n</tool_call>')
        row = self.run_output(output)
        self.assertNotIn('error', row)
        self.assertEqual(row['prediction'], 30)
        self.assertEqual(row['calculation']['fact_ids'], ['f0', 'f1'])
        self.assertIn('error', self.run_output(output + output))
        self.assertIn('error', self.run_output(output, False))

    def test_answer_without_required_tool_is_failure(self):
        self.assertIn('error', self.run_output('30'))

    def test_model_cannot_supply_unreferenced_numeric_arguments(self):
        output = ('<tool_call>\n<function=calculate>\n<parameter=steps>\n'
                  '[{"operation":"subtract","args":["150","120"]}]\n'
                  '</parameter>\n</function>\n</tool_call>')
        self.assertIn('error', self.run_output(output))
