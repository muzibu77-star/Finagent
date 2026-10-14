from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import struct
import unittest
from unittest.mock import patch

from src.evaluation.review_labels import (
    adjudicate, agreement, blind_sample, calculate, compare, counts,
    split_summary, validate,
)


class ReviewLabelsTest(unittest.TestCase):
    def test_half_unit_inclusive_and_rounding_band(self):
        self.assertEqual(compare('1.00亿元', Decimal('1.005'))['label'], 'correct')
        self.assertEqual(compare('1.00亿元', Decimal('1.006'))['error_type'],
                         'rounding')
        self.assertEqual(compare('1.00亿元', Decimal('1.01'))['error_type'],
                         'rounding')
        self.assertIsNone(compare('1.00亿元', Decimal('1.011'))['error_type'])

    def test_decrease_does_not_erase_sign(self):
        self.assertEqual(compare('3.1%', Decimal('-3.07'), -1)['label'], 'correct')
        result = compare('3.1%', Decimal('3.07'), -1)
        self.assertEqual(result['label'], 'wrong')
        self.assertEqual(result['expected'], '-3.1%')

    def test_threshold_claim_is_not_an_equality(self):
        self.assertEqual(compare('12%', Decimal('5.25'),
                                 comparison_operator='<=')['label'], 'correct')
        self.assertEqual(compare('12%', Decimal('12.01'),
                                 comparison_operator='<=')['label'], 'wrong')
        self.assertEqual(compare('50%', Decimal('51'),
                                 comparison_operator='>=')['label'], 'correct')

    def test_reject_invalid_numbers(self):
        for text, value in [('1亿元x', '1'), ('1亿元', 'NaN')]:
            with self.assertRaises(ValueError):
                compare(text, Decimal(value))

    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.executescript(
            'CREATE TABLE facts (id, document_id, page, start, end, text);'
            'CREATE TABLE pages (document_id, page, text);'
            'CREATE TABLE documents (id, code, fiscal_year, kind);'
            "INSERT INTO documents VALUES ('doc','000002',2023,'annual');"
            "INSERT INTO pages VALUES ('doc',1,'营业收入100');"
            "INSERT INTO facts VALUES (1,'doc',1,4,7,'100');"
        )
        self.review = {
            'id': 'r', 'text': '收入1.00亿元', 'document_id': 'doc',
            'fiscal_year': 2023, 'code': '000002',
            'model': 'example', 'closed_book': True,
            'numbers': [{'start': 2, 'text': '1.00亿元', 'unit': '亿元',
                         'supported': False}],
        }
        self.label = {
            'review_id': 'r', 'start': 2, 'text': '1.00亿元', 'metric': '营业收入',
            'period': '2023年', 'scope': '合并', 'label': 'correct',
            'error_type': None, 'expected': '1.00亿元', 'note': '',
            'evidence': {'formula': 'a / 100',
                         'operands': [{'name': 'a', 'fact_id': 1, 'value': '100'}]},
        }

    def tearDown(self):
        self.db.close()

    def test_coverage_and_duplicates(self):
        self.assertEqual(validate([self.review], [self.label], self.db)['remaining'], 0)
        for labels in ([], [self.label, self.label]):
            with self.assertRaises(ValueError):
                validate([self.review], labels, self.db)
        self.assertEqual(validate([self.review], [], self.db,
                                  complete=False)['remaining'], 1)

    def test_invalid_fact_span_and_document(self):
        for statement in (
            'UPDATE facts SET start=0',
            "UPDATE facts SET document_id='other'",
            'DELETE FROM facts',
        ):
            self.db.execute('SAVEPOINT test')
            self.db.execute(statement)
            with self.assertRaises(ValueError):
                validate([self.review], [self.label], self.db)
            self.db.execute('ROLLBACK TO test')
            self.db.execute('RELEASE test')

    def test_missing_evidence_and_error_type(self):
        for changes in ({'evidence': {}}, {'label': 'wrong'}, {'text': '2.00亿元'}):
            label = deepcopy(self.label)
            label.update(changes)
            with self.assertRaises(ValueError):
                validate([self.review], [label], self.db)

    def test_blind_sample_omits_weak_labels_and_is_stable(self):
        reviews = []
        for index in range(21):
            review = deepcopy(self.review)
            review['id'] = str(index)
            reviews.append(review)
        sample = blind_sample(reviews)
        self.assertEqual(len(sample), 3)
        self.assertEqual(sample, blind_sample(list(reversed(reviews))))
        self.assertTrue(all('supported' not in item for item in sample))
        self.assertTrue(all('label' not in item for item in sample))

    def test_agreement_requires_the_exact_sample(self):
        result = agreement([self.review], [self.label], [self.label])
        self.assertEqual(result['label_agreement'], 1)
        self.assertTrue(result['passes_95_percent'])
        other = {**self.label, 'label': 'unverifiable'}
        self.assertFalse(agreement([self.review], [self.label], [other])[
            'passes_95_percent'])
        for rows in ([], [self.label, self.label]):
            with self.assertRaises(ValueError):
                agreement([self.review], [self.label], rows)

    def test_counts_distinguish_unfinished_and_weak_label_disagreements(self):
        summary = counts([self.review], [self.label])
        self.assertEqual(summary['remaining'], 0)
        self.assertEqual(summary['groups'][0]['supported_false_correct'], 1)
        summary = counts([self.review], [])
        self.assertEqual(summary['remaining'], 1)
        self.assertEqual(summary['groups'][0]['labeled'], 0)

    def test_calculation_uses_exact_decimal_and_preserves_losses(self):
        ops = [{'name': 'a', 'value': '-0.3'}, {'name': 'b', 'value': '-0.1'}]
        self.assertEqual(calculate('(abs(a) - abs(b)) / abs(b) * 100', ops),
                         Decimal('200'))
        self.assertEqual(calculate('a - b', ops), Decimal('-0.2'))
        for formula in ('a ** 2', 'float(a)', 'a.real', '__import__("os")', '0.1'):
            with self.assertRaises(ValueError):
                calculate(formula, ops)

    def test_adjudication_requires_explicit_selection(self):
        label = adjudicate(self.review, 2, '营业收入', '2023年度',
                           [{'name': 'a', 'fact_id': 1, 'value': '1.007'}], 'a')
        self.assertEqual(label['error_type'], 'rounding')
        self.assertEqual(label['expected'], '1.01亿元')
        with self.assertRaises(ValueError):
            adjudicate(self.review, 99, '营业收入', '2023年度', [], '1')

    def test_recomputed_evidence_must_match_label(self):
        for field, value in [('expected', '2.00亿元'), ('formula', 'a / 50')]:
            label = deepcopy(self.label)
            if field == 'formula':
                label['evidence'][field] = value
            else:
                label[field] = value
            with self.assertRaises(ValueError):
                validate([self.review], [label], self.db)

    def test_previous_quarter_requires_explicit_prior_period_reference(self):
        self.db.execute("INSERT INTO documents VALUES ('prior','000002',2022,'annual')")
        self.db.execute("INSERT INTO pages VALUES ('prior',1,'营业收入100')")
        self.db.execute("UPDATE facts SET document_id='prior'")
        label = deepcopy(self.label)
        label['period'] = '2023年第四季度较2022年第四季度'
        operand = label['evidence']['operands'][0]
        operand.update(period='2022年第四季度', previous_period_reference='第四季度同比')
        validate([self.review], [label], self.db)
        del operand['previous_period_reference']
        with self.assertRaises(ValueError):
            validate([self.review], [label], self.db)

    def test_full_claim_preserves_frozen_extraction_key(self):
        review = deepcopy(self.review)
        review['text'] = '收入1，001.00亿元'
        review['numbers'][0].update(start=4, text='001.00亿元')
        label = deepcopy(self.label)
        label.update(start=4, text='001.00亿元', label='wrong', error_type='value')
        label['evidence']['claim_value'] = {
            'value': '1001.00', 'unit': '亿元', 'start': 2, 'end': 12,
            'quote': '1，001.00亿元',
        }
        validate([review], [label], self.db)
        label['evidence']['claim_value']['value'] = '2001.00'
        with self.assertRaises(ValueError):
            validate([review], [label], self.db)

    def test_summary_only_completes_after_independent_sample(self):
        result = split_summary([self.review], [self.label], [])
        self.assertEqual(result['status'], 'first_pass_complete')
        self.assertIsNone(result['second_pass'])
        result = split_summary([self.review], [self.label], [self.label])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['second_pass']['supported_false_correct'], 1)
        with self.assertRaises(ValueError):
            split_summary([self.review], [], [self.label])

    def test_image_exception_requires_empty_page_and_valid_coordinates(self):
        self.db.execute("UPDATE pages SET text=''")
        label = deepcopy(self.label)
        operand = label['evidence']['operands'][0]
        operand.pop('fact_id')
        operand.update(
            source_type='pdf_image', document_id='doc', page=1,
            pdf_path='data/raw/source.pdf',
            image_path='data/staged/eval_v2_labels/evidence/source.png',
            quote='营业收入100',
            bbox={'x0': 0, 'y0': 0, 'x1': 50, 'y1': 20,
                  'image_width': 100, 'image_height': 80},
        )
        config = json.dumps({'sources': [
            {'id': 'doc', 'path': 'data/raw/source.pdf'},
        ]})
        header = b'\x89PNG\r\n\x1a\n' + b'\0' * 8 + struct.pack('>II', 100, 80)
        with (patch.object(Path, 'read_text', return_value=config),
              patch.object(Path, 'read_bytes', return_value=header),
              patch.object(Path, 'is_file', return_value=True)):
            validate([self.review], [label], self.db)
            operand['bbox']['x1'] = 101
            with self.assertRaisesRegex(ValueError, 'coordinates'):
                validate([self.review], [label], self.db)
            operand['bbox']['x1'] = 50
            self.db.execute("UPDATE pages SET text='已有文本'")
            with self.assertRaisesRegex(ValueError, 'empty text page'):
                validate([self.review], [label], self.db)

    def test_corrections_do_not_inflate_blind_agreement(self):
        baseline = deepcopy(self.label)
        baseline.update(label='wrong', error_type='value')
        result = split_summary([self.review], [self.label], [self.label],
                               first_before_blind=[baseline])
        self.assertEqual(result['labels']['correct'], 1)
        self.assertEqual(result['second_pass']['label_agreement'], 0)
        self.assertEqual(result['status'], 'requires_rework')


if __name__ == '__main__':
    unittest.main()
