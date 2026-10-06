import dataclasses
from decimal import Decimal
import unittest

from src.evidence.models import Evidence, Fact
from src.tools.calculator import calculate


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {
            'e1': Evidence('e1', 'doc', 'v1', '120', 'hash', row=1, column=1),
            'e2': Evidence('e2', 'doc', 'v1', '150', 'hash', row=1, column=2),
        }
        self.facts = {
            'a': Fact('a', 'ACME', 'revenue', Decimal(120), 'money', 'USD',
                      Decimal(1000000), '2022', 'consolidated', ('e1',), 'validated'),
            'b': Fact('b', 'ACME', 'revenue', Decimal(150), 'money', 'USD',
                      Decimal(1000000), '2023', 'consolidated', ('e2',), 'validated'),
        }

    def call(self, op='growth_rate', ids=None):
        return calculate('c1', op, ids or ['a', 'b'], self.facts, self.evidence)

    def change(self, key, **values):
        self.facts[key] = dataclasses.replace(self.facts[key], **values)

    def test_growth_and_references(self):
        result = self.call()
        self.assertEqual(Decimal(result.value), Decimal('.25'))
        self.assertEqual(result.fact_ids, ('a', 'b'))
        self.assertEqual(result.unit, 'ratio')

    def test_invalid_reference(self):
        with self.assertRaises(ValueError): self.call(ids=['a', 'missing'])

    def test_evidence_value_mismatch(self):
        self.change('a', value=Decimal(121))
        with self.assertRaises(ValueError): self.call()

    def test_unit_currency_scope_mismatch(self):
        for field, value in [('unit', 'number'), ('currency', 'EUR'), ('scope', 'parent')]:
            self.setUp()
            self.change('b', **{field: value})
            with self.assertRaises(ValueError): self.call()

    def test_period_mismatch(self):
        with self.assertRaises(ValueError): self.call('sum')

    def test_zero_base(self):
        self.change('a', value=Decimal(0))
        self.evidence['e1'] = dataclasses.replace(self.evidence['e1'], text='0')
        with self.assertRaises(ValueError): self.call()

    def test_negative_base_returns_difference(self):
        self.change('a', value=Decimal(-120))
        self.evidence['e1'] = dataclasses.replace(self.evidence['e1'], text='(120)')
        result = self.call()
        self.assertEqual(result.operation, 'difference')
        self.assertEqual(Decimal(result.value), Decimal(270000000))
        self.assertIsNotNone(result.note)

    def test_scale_conversion(self):
        self.change('b', period='2022', scale=Decimal(1))
        self.assertEqual(Decimal(self.call('sum').value), Decimal(120000150))

    def test_unreviewed_metadata_rejected(self):
        self.change('a', status='candidate')
        with self.assertRaises(ValueError): self.call()

    def test_percentage_points(self):
        for key in ('a', 'b'):
            self.change(key, unit='percent', currency=None, scale=Decimal(1), period='2022')
        self.assertEqual(Decimal(self.call('percentage_points').value), Decimal(30))

    def test_generated_code_rejected(self):
        with self.assertRaises(ValueError): self.call('__import__("os")')
