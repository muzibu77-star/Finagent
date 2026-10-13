import unittest

from src.training.prepare_sft import annotated_facts, convert_program


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

    def test_released_row_annotation_resolves_original_fact_id(self):
        evidence = {'evidence_id': 'e', 'table': [['A', '10'], ['B', '10', '20']]}
        refs = annotated_facts(evidence, {'table_1': 'B is 10 and 20'})
        action, result = convert_program('add(10, 20)', evidence, 30, refs)
        self.assertEqual(action['steps'][0]['args'], ['f1', 'f2'])
        self.assertEqual(result['facts']['f1']['location']['row'], 1)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            convert_program('add(10, 20)', evidence, 30, {'f0', 'f1', 'f2'})

    def test_table_reduction_remaps_prior_results(self):
        action, result = convert_program('table_average(Revenue, none), multiply(#0, const_2)',
            {'evidence_id': 'e', 'table': [['Revenue', '10', '20', '30']]}, 40,
            table_reductions=True)
        self.assertEqual(action['steps'][-1]['args'], ['#2', 'const_2'])
        self.assertEqual(result['value'], 40)
