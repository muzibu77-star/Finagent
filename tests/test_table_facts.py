import unittest

from src.data.table_facts import DECLARED, context, grid, header_rows, stated_unit
from src.data.table_map import table_cells


def text(kind, content):
    return {'type': kind, 'content': [{'type': 'text', 'content': content}]}


class TableFactsTest(unittest.TestCase):
    def test_context_reads_title_unit_and_date_before_the_table(self):
        found = context([text('paragraph_title', '合并资产负债表'), text('text', '2017 年 12 月 31 日'),
                         text('text', '编制单位: 某公司'), text('text', '单位:元 币种:人民币')])
        self.assertEqual((found['title'], found['unit'], found['scale'], found['currency'],
                          found['statement_date']),
                         ('合并资产负债表', '元', '1', '人民币', '2017年12月31日'))

    def test_a_new_title_discards_the_previous_unit(self):
        found = context([text('paragraph_title', '合并利润表'), text('text', '单位：人民币百万元'),
                         text('paragraph_title', '主要客户')])
        self.assertEqual(found['title'], '主要客户')
        self.assertIsNone(found['unit'])

    def test_amounts_in_prose_are_not_a_unit_line(self):
        found = context([text('text', '本年实现营业收入人民币100亿元。')])
        self.assertIsNone(found['unit'])

    def test_compound_units_keep_their_scale(self):
        found = context([text('text', '（除特别注明外，金额单位为人民币百万元）')])
        self.assertEqual((found['unit'], found['scale']), ('百万元', '1000000'))

    def test_header_stops_at_first_value_or_single_label_row(self):
        cells = table_cells('<table><tr><td rowspan="2">项目</td><td colspan="2">账面余额</td></tr>'
                            '<tr><td>期末</td><td>期初</td></tr>'
                            '<tr><td>流动资产:</td><td colspan="2"></td></tr>'
                            '<tr><td>货币资金</td><td>1.00</td><td>2.00</td></tr></table>')
        self.assertEqual(header_rows(cells), [0, 1])
        self.assertEqual(grid(cells)[(1, 0)], '项目')
        self.assertEqual(grid(cells)[(0, 2)], '账面余额')

    def test_unit_in_a_header_cell(self):
        found = stated_unit('(人民币百万元,特别注明除外) 2023年 2022年', 'table header')
        self.assertEqual((found['unit'], found['currency'], found['unit_source']),
                         ('百万元', '人民币', 'table header'))

    def test_document_wide_statements(self):
        for text, unit in (('2019年度财务报表附注(除特别注明外，金额单位为人民币元)', '元'),
                           ('本财务报表以人民币列示，除特别注明外，均四舍五入取整到百万元', '百万元')):
            self.assertEqual(DECLARED.search(text).group(1), unit)
        self.assertIsNone(DECLARED.search('除上述事项外，本年收入人民币100亿元'))

    def test_non_text_blocks_before_a_table_are_ignored(self):
        found = context([text('paragraph_title', '商誉'),
                         {'type': 'equation', 'content': '\\begin{array}{c} 1 \\end{array}'},
                         text('text', '单位：元')])
        self.assertEqual((found['title'], found['unit']), ('商誉', '元'))

    def test_counting_unit_has_no_scale(self):
        found = stated_unit('单位：股', 'line before table')
        self.assertEqual((found['unit'], found['scale'], found['currency']), ('股', None, None))
        self.assertEqual(stated_unit('被投资单位名称', 'line before table'), {})


if __name__ == '__main__':
    unittest.main()
