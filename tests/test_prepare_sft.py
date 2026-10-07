import unittest

from src.training.prepare_sft import convert_program


class PrepareSftTests(unittest.TestCase):
    def test_exact_program_conversion(self):
        action, result = convert_program('subtract(150, 120), divide(#0, 120)',
            {'evidence_id':'e','table':[['Revenue','120','150']]}, .25)
        self.assertEqual(action['steps'][0]['args'], ['f1','f0'])
        self.assertEqual(result['value'], .25)

    def test_ambiguous_values_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            convert_program('add(10, 20)',
                {'evidence_id':'e','table':[['10','20'],['10','30']]},30)

    def test_gold_disagreement_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            convert_program('add(10, 20)', {'evidence_id':'e','table':[['10','20']]},31)

    def test_unsupported_table_operation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            convert_program('table_sum(Revenue, none)',
                {'evidence_id':'e','table':[['Revenue','10','20']]},30)
