import unittest

from src.verify.extract import checkable, locate, matches, parse


class ExtractTest(unittest.TestCase):
    def test_parse_takes_the_array_and_survives_bad_answers(self):
        self.assertEqual(parse('好的：\n[{"number": "1.5", "unit": "%"}] 完'), [{'number': '1.5', 'unit': '%'}])
        self.assertEqual(parse('[{"number": "1.5",'), [])
        self.assertEqual(parse('没有数字'), [])

    def test_locate_finds_the_nth_standalone_occurrence(self):
        text = '收入12.3亿元，增长12.3%，其中112.3亿元'
        self.assertEqual(locate(text, '12.3', 1), 2)
        self.assertEqual(locate(text, '12.3', 2), 11)
        self.assertIsNone(locate(text, '12.3', None))
        self.assertEqual(locate('仅一次 7.5%', '7.5', 9), 4)

    def test_wording_must_fit_the_line_item(self):
        self.assertTrue(matches('归母净利润', '归属于母公司股东的净利润'))
        self.assertTrue(matches('货币资金', '货币资金'))
        self.assertFalse(matches('归属于母公司股东权益', '归属于母公司股东的净利润'))
        self.assertFalse(matches('少数股东权益', '少数股东损益'))
        self.assertFalse(matches(None, '货币资金'))
        self.assertTrue(matches('同比增长', '营业收入'))
        self.assertTrue(matches('归属于母公司股东权益', '归属于上市公司股东的净资产'))

    def test_total_and_plain_line_items_are_not_each_other(self):
        # Both are reported with different values, so the "总" decides the item.
        self.assertFalse(matches('营业收入', '营业总收入'))
        self.assertFalse(matches('营业总收入', '营业收入'))
        self.assertFalse(matches('营业成本', '营业总成本'))
        self.assertFalse(matches('营业总成本', '营业成本'))
        self.assertTrue(matches('营业总收入', '营业总收入'))
        # "利润总额" keeps "总额" as one word: dropping the "总" leaves "利润额",
        # which is not "利润", so the pair stays with the existing noun rule.
        self.assertTrue(matches('利润总额', '利润'))
        self.assertTrue(matches('利润', '利润总额'))

    def test_quarters_segments_and_own_method_ratios_are_left_to_a_person(self):
        self.assertFalse(checkable('四季度收入'))
        self.assertFalse(checkable('加权平均净资产收益率'))
        self.assertTrue(checkable('营业收入'))
        self.assertTrue(checkable(None))


if __name__ == '__main__':
    unittest.main()
