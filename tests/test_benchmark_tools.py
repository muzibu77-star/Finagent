import unittest

from src.tools.benchmark_tools import execute_action, extract_numeric_facts, parse_action
from src.retrieval.bm25 import EvidenceIndex


class BenchmarkToolTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {'evidence_id': 'e1', 'table': [['Revenue', '120', '150']]}
        self.facts = extract_numeric_facts(self.evidence)

    def test_growth_reference_chain(self):
        result = execute_action({'steps': [
            {'operation': 'subtract', 'args': ['f1', 'f0']},
            {'operation': 'divide', 'args': ['#0', 'f0']},
        ]}, self.facts)
        self.assertEqual(result['value'], .25)
        self.assertEqual(result['fact_ids'], ['f0', 'f1'])
        self.assertEqual(result['facts']['f0']['location']['column'], 1)

    def test_literals_and_forward_refs_rejected(self):
        for arg in ('120', '__import__("os")', '#0'):
            with self.assertRaises(ValueError):
                execute_action({'steps': [{'operation': 'add', 'args': [arg, 'f1']}]}, self.facts)

    def test_exponent_budget(self):
        with self.assertRaises(ValueError):
            execute_action({'steps': [{'operation': 'exp', 'args': ['f0', 'f1']}]}, self.facts)

    def test_percent_official_semantics(self):
        facts = extract_numeric_facts({'evidence_id': 'e1', 'table': [['25%', '1']]})
        self.assertEqual(execute_action({'steps': [
            {'operation': 'multiply', 'args': ['f0', 'f1']} ]}, facts)['value'], .25)

    def test_parenthesized_table_negative(self):
        facts = extract_numeric_facts({'evidence_id': 'e', 'table': [['(120)', '150']]})
        result = execute_action({'steps': [
            {'operation': 'subtract', 'args': ['f1', 'f0']}]}, facts)
        self.assertEqual(result['value'], 270)

    def test_json_only(self):
        self.assertEqual(parse_action('```json\n{"steps": []}\n```'), {'steps': []})
        with self.assertRaises(ValueError): parse_action('hello')

    def test_retrieval_and_gold_guard(self):
        index = EvidenceIndex()
        try:
            index.add(self.evidence)
            self.assertEqual(index.search('Revenue'), ['e1'])
            with self.assertRaises(ValueError): index.add({**self.evidence, 'answer': 'secret'})
            self.assertEqual(index.search('" OR --'), [])
        finally:
            index.close()
