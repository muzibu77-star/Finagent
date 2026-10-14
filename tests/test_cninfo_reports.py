import unittest

from src.data.cninfo_reports import classify

YEARS = [2023, 2024, 2025]


class ClassifyTest(unittest.TestCase):
    def test_annual_report(self):
        self.assertEqual(classify('贵州茅台2023年年度报告', 'annual', YEARS), ('annual', 2023))

    def test_summaries_and_translations_are_not_sources(self):
        for title in ('贵州茅台2023年年度报告摘要', '贵州茅台2023年年度报告（英文版）',
                      '关于披露2023年年度报告的提示性公告'):
            self.assertIsNone(classify(title, 'annual', YEARS))

    def test_revised_report_is_a_correction_even_in_the_annual_category(self):
        self.assertEqual(classify('某公司2024年年度报告（更正后）', 'annual', YEARS),
                         ('correction', 2024))
        self.assertEqual(classify('关于<em>2024年年度报告</em>的更正公告', 'correction', YEARS),
                         ('correction', 2024))

    def test_other_years_and_report_types_are_skipped(self):
        self.assertIsNone(classify('某公司2021年年度报告', 'annual', YEARS))
        self.assertIsNone(classify('某公司2024年半年度报告', 'annual', YEARS))

    def test_bank_style_title_and_forwarded_subsidiary_report(self):
        self.assertEqual(classify('招商银行股份有限公司2023年度报告', 'annual', YEARS),
                         ('annual', 2023))
        self.assertIsNone(classify('中国平安：平安银行股份有限公司2023年年度报告', 'annual',
                                   YEARS, '中国平安'))
        self.assertEqual(classify('中国平安2023年年度报告', 'annual', YEARS, '中国平安'),
                         ('annual', 2023))
        self.assertEqual(classify('中国移动：2023年年度报告', 'annual', YEARS, '中国移动'),
                         ('annual', 2023))

    def test_misspelled_title_and_undated_correction_notice(self):
        self.assertEqual(classify('紫金矿业集团股份有限公司2024年年报报告', 'annual', YEARS),
                         ('annual', 2024))
        self.assertEqual(classify('关于定期报告财务报表附注部分相关文字错误的更正公告',
                                  'correction', YEARS), ('correction', None))
        self.assertIsNone(classify('关于回购股份的补充公告', 'correction', YEARS))


if __name__ == '__main__':
    unittest.main()
