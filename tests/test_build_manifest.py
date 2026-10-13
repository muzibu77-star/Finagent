import unittest

from src.data.build_manifest import financial_source, source_owner, visible_evidence


class ManifestTests(unittest.TestCase):
    def test_report_group_ignores_page(self):
        self.assertEqual(financial_source('ADI/2019/page_1.pdf'),
                         financial_source('ADI/2019/page_99.pdf'))

    def test_invalid_report_fails(self):
        with self.assertRaises(ValueError):
            financial_source('unknown.pdf')

    def test_holdout_priority(self):
        self.assertEqual(source_owner({'train', 'test', 'dev'}), 'test')
        self.assertEqual(source_owner({'train', 'dev'}), 'dev')

    def test_gold_cannot_enter_evidence(self):
        row = {'table': [['Revenue', '12']], 'pre_text': ['Reported revenue'],
               'post_text': [], 'qa': {'answer': 'secret'},
               'model_input': 'secret', 'gold_inds': {'secret': 'secret'}}
        visible = visible_evidence('finqa', row)
        self.assertEqual(set(visible), {'table', 'pre_text', 'post_text'})
        self.assertNotIn('secret', str(visible))
