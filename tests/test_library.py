import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.data.library import SCHEMA, Library, add_page, digits


def document(db, key, code, published, publisher='某公司'):
    db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               (key, 'annual', publisher, code, 2024, '年度报告', 'http://x', published, 'h', 1))


def fact(db, key, text):
    db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, row_label) '
               'VALUES (?, 1, 0, 1, ?, ?, ?, ?)', (key, json.dumps([0, 0, 1, 1]), text,
                                                   digits(text), '营业收入'))


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / 'library.sqlite'
        db = sqlite3.connect(path)
        db.executescript(SCHEMA)
        document(db, 'a-2024', '600000', '2025-03-30')
        document(db, 'a-2024-fix', '600000', '2025-06-10')
        document(db, 'b-2024', '000001', '2025-04-20', '另一公司')
        add_page(db, 'a-2024', 1, '本年营业收入增长，毛利率改善')
        add_page(db, 'a-2024-fix', 1, '更正后营业收入')
        add_page(db, 'b-2024', 1, '研发费用增加')
        fact(db, 'a-2024', '1,234.50')
        fact(db, 'a-2024-fix', '(1,234.50)')
        fact(db, 'b-2024', '1234.5')
        db.commit()
        db.close()
        self.library = Library(path)

    def tearDown(self):
        self.library.close()
        self.directory.cleanup()

    def test_digits_ignore_grouping_and_sign_marks_only(self):
        self.assertEqual(digits('(1,234.50)'), '1234.50')
        self.assertEqual(digits('-12.3%'), '12.3')
        self.assertNotEqual(digits('1,234.50'), digits('1,234.5'))

    def test_number_lookup_respects_issuer_and_cutoff(self):
        found = self.library.find_number('1234.50', code='600000')
        self.assertEqual([f['document_id'] for f in found], ['a-2024-fix', 'a-2024'])
        early = self.library.find_number('1,234.50', code='600000', as_of='2025-06-09')
        self.assertEqual([f['document_id'] for f in early], ['a-2024'])
        self.assertEqual(self.library.find_number('1,234.50', code='000001'), [])

    def test_cutoff_includes_the_disclosure_day(self):
        hits = self.library.search_pages('营业收入', as_of='2025-03-30')
        self.assertEqual([h['document_id'] for h in hits], ['a-2024'])

    def test_page_search_and_text(self):
        hits = self.library.search_pages('研发费用')
        self.assertEqual(hits[0]['document_id'], 'b-2024')
        self.assertEqual(self.library.page_text('b-2024', 1), '研发费用增加')
        with self.assertRaises(KeyError):
            self.library.page_text('b-2024', 9)
        self.assertEqual(self.library.search_pages('！？'), [])


if __name__ == '__main__':
    unittest.main()
