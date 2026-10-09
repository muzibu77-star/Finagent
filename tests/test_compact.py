import unittest

from src.evidence.compact import evidence_view
from src.tools.benchmark_tools import extract_numeric_facts


class EvidenceViewTests(unittest.TestCase):
    def test_cited_rows_keep_header_coordinates_and_unit_context(self):
        evidence = {'evidence_id': 'e', 'table': [['Item', '2023'],
                    ['Revenue', '100'], ['Debt', '200']], 'pre_text': ['USD millions']}
        facts = extract_numeric_facts(evidence)
        selected = [key for key, fact in facts.items() if fact['location'].get('row') == 2]
        view = evidence_view(evidence, facts, selected)
        self.assertEqual([row['original_row'] for row in view['table_rows']], [0, 2])
        self.assertEqual(view['pre_text'], ['USD millions'])
        self.assertEqual(evidence['table'][1], ['Revenue', '100'])

    def test_foreign_fact_rejected(self):
        with self.assertRaisesRegex(ValueError, 'another'):
            evidence_view({'evidence_id': 'a'}, {'f': {'evidence_id': 'b'}}, ['f'])


if __name__ == '__main__':
    unittest.main()
