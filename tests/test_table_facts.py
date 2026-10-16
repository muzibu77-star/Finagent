import unittest

from src.data.table_facts import (DECLARED, STATEMENT_UNIT, column_headings, column_headers,
                                   context, declared_unit, declared_units, grid, header_line,
                                   header_rows, label_unit, note_number, notes_declared_unit,
                                   rows_to_facts, scope, statement_table, statement_title,
                                   stated_unit, text_layer_cells)
from src.data.table_map import NumberSpan, table_cells


def text(kind, content):
    return {'type': kind, 'content': [{'type': 'text', 'content': content}]}


class Page:
    """The part of a PDFium page that the document-wide unit scan reads."""

    def __init__(self, content):
        self.content = content

    def get_textpage(self):
        content = self.content

        class TextPage:
            def get_text_bounded(self):
                return content

            def close(self):
                pass

        return TextPage()


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

    def test_statement_unit_declaration_names_the_financial_statements(self):
        # The sentence that opens the financial statements of 比亚迪 2024 (002594-1222881496, p. 142).
        self.assertEqual(STATEMENT_UNIT.search('财务附注中报表的单位为：千元').group(1), '千元')
        self.assertEqual(STATEMENT_UNIT.search('二、财务报表 财务附注中报表的单位均为人民币元')
                         .group(1), '元')
        self.assertEqual(declared_units([Page('二、财务报表 财务附注中报表的单位为：千元')]),
                         [(1, {'unit': '千元', 'scale': '1000', 'currency': None,
                               'unit_source': 'statement on page 1'}, True)])

    def test_a_declared_statement_unit_binds_only_a_statement_table(self):
        declared = declared_units([Page('财务报表附注 (除特别注明外，金额单位为人民币千元)'),
                                   Page('财务附注中报表的单位为：元')])
        self.assertEqual([(page, fields['unit'], only) for page, fields, only in declared],
                         [(1, '千元', False), (2, '元', True)])
        # The general statement covers every later table, the statement-only one
        # covers a statement whatever its number and no note table.
        self.assertEqual(declared_unit([declared[0]], 9, '（1） 企业集团的构成')['unit'], '千元')
        self.assertIsNone(declared_unit([declared[1]], 9, '（1） 企业集团的构成'))
        self.assertEqual(declared_unit(declared, 9, '7、股东权益变动表')['unit'], '元')

    def test_a_note_states_the_unit_of_the_statements_after_them(self):
        # 比亚迪 2023 (002594-1219412018) declares it on page 153, after the statements
        # that start on page 138 and state no unit of their own.
        unit, clash = notes_declared_unit(
            [Page('本公司记账本位币和编制本财务报表所采用的货币均为人民币，'
                  '除有特别说明外，均以人民币千元为单位表示。')])
        self.assertEqual((unit['unit'], unit['scale'], unit['currency'], unit['unit_source']),
                         ('千元', '1000', '人民币', 'statement unit declared in notes'))
        self.assertFalse(clash)
        # A statement of the same unit twice keeps the declaration readable.
        again, clash = notes_declared_unit(
            [Page('财务附注中报表的单位为：千元'),
             Page('编制本财务报表所采用的货币均为人民币，除有特别说明外，'
                  '均以人民币千元为单位表示。')])
        self.assertEqual(again['unit'], '千元')
        self.assertFalse(clash)

    def test_two_different_statement_units_in_notes_are_left_alone(self):
        unit, clash = notes_declared_unit([Page('编制本财务报表以人民币千元为单位表示。'),
                                           Page('报表的单位为：元')])
        self.assertIsNone(unit)
        self.assertTrue(clash)

    def test_a_unit_statement_that_does_not_name_the_statements_is_not_read(self):
        for sentences in (['除特别注明外，金额单位为人民币元'],
                          ['币报表折算差额及采用远期汇率交易产品对冲汇率风险的美元借款'
                           '和美元应付债券未包括在内'],
                          ['入本集团合并报表及出售汽车之家时带来的一次性损益，本公司发行的'
                           '美元、港元可转换债券的转股权价值重估损益等合']):
            unit, clash = notes_declared_unit([Page(sentence) for sentence in sentences])
            self.assertIsNone(unit)
            self.assertFalse(clash)

    def test_a_preparing_entity_is_not_a_unit(self):
        # "编制单位：" names the issuer, not a unit (迈瑞医疗 2023, 300760-1219856065, p. 224).
        self.assertEqual(stated_unit('编制单位:深圳迈瑞生物医疗电子股份有限公司', 'line before table'), {})
        self.assertEqual(stated_unit('单位:股', 'line before table')['unit'], '股')

    def test_a_statement_title_text_line_keeps_its_unit(self):
        # 京东方A 2023 prints the title and the unit on one line (000725-1219494121, p. 102).
        found = context([text('text', '京东方科技集团股份有限公司合并资产负债表 (续) 2023 年 12 月 31 日'
                                      '(金额单位:人民币元)')])
        self.assertEqual((found['title'], found['unit'], found['scale'], found['currency']),
                         ('合并资产负债表(续)', '元', '1', '人民币'))

    def test_a_summary_row_states_its_own_unit(self):
        # The main-accounting-data rows of 格力电器 2024 (000651-1223330631, p. 7).
        self.assertEqual(label_unit('营业收入（元）'),
                         {'unit': '元', 'scale': '1', 'unit_source': 'row label'})
        self.assertEqual(label_unit('总资产（万元）'),
                         {'unit': '万元', 'scale': '10000', 'unit_source': 'row label'})
        self.assertEqual(label_unit('基本每股收益（元/股）'), {})
        self.assertEqual(label_unit('货币资金'), {})

    def test_only_statement_tables_carry_statement_units(self):
        self.assertTrue(statement_table('六、主要会计数据和财务指标'))
        self.assertTrue(statement_table('1、合并资产负债表'))
        self.assertTrue(statement_table('7、股东权益变动表'))
        self.assertFalse(statement_table('（8） 主要销售客户和主要供应商情况'))
        self.assertFalse(statement_table(None))

    def test_counting_unit_has_no_scale(self):
        found = stated_unit('单位：股', 'line before table')
        self.assertEqual((found['unit'], found['scale'], found['currency']), ('股', None, None))
        self.assertEqual(stated_unit('被投资单位名称', 'line before table'), {})

    def test_statement_title_set_as_plain_text(self):
        found = context([text('paragraph_title', '中国移动有限公司'), text('text', '合并资产负债表'),
                         text('text', '(除特别注明外，金额单位为人民币百万元)')])
        self.assertEqual((found['title'], found['unit']), ('合并资产负债表', '百万元'))

    def test_scope_from_column_in_combined_statements(self):
        self.assertEqual(scope('合并及公司资产负债表', '2025年12月31日 合并'), '合并')
        self.assertEqual(scope('合并及公司资产负债表', '2025年12月31日 公司'), '母公司')
        self.assertEqual(scope('母公司利润表', '本期发生额'), '母公司')
        self.assertEqual(scope('合并利润表', '本期发生额'), '合并')
        self.assertIsNone(scope('应收账款', '期末余额'))

    def test_a_banks_own_column_is_the_parent_scope(self):
        # 兴业银行 2023—2025 (601166) prints "本集团" over the consolidated columns and
        # "本银行" over the bank's own columns of one "合并及银行…表"; without this the
        # bank's own figures were read as consolidated ones.
        self.assertEqual(scope('合并及银行利润表', '本集团 2024年'), '合并')
        self.assertEqual(scope('合并及银行利润表', '本银行 2024年'), '母公司')
        self.assertEqual(scope('合并及银行资产负债表（续）', '本银行 2024年'), '母公司')
        self.assertEqual(scope('', '本银行'), '母公司')
        # A column that heads both the group's and the bank's figures stays consolidated.
        self.assertEqual(scope('28. 其他权益工具', '本集团及本银行 2024年'), '合并')
        placed = {(0, 0): '项目', (0, 1): '本集团 2024年', (0, 2): '本银行 2024年',
                  (1, 0): '资产总计', (1, 1): '1,000', (1, 2): '900'}
        values = {(1, 1): {'start': 0, 'end': 5, 'box': [], 'text': '1,000'},
                  (1, 2): {'start': 6, 'end': 9, 'box': [], 'text': '900'}}
        meta = {'title': '合并及银行资产负债表（续）', 'unit': '元', 'scale': '1',
                'currency': '人民币', 'statement_date': '2024年12月31日',
                'unit_source': 'title'}
        facts, _ = rows_to_facts(placed, [0], column_headers(placed, [0]), values, meta,
                                 None, 66)
        self.assertEqual([(fact['column_header'], fact['scope']) for fact in facts],
                         [('本集团 2024年', '合并'), ('本银行 2024年', '母公司')])

    def test_note_numbers_under_a_merged_header(self):
        self.assertTrue(note_number('2023年12月31日 附注五', '42'))
        self.assertFalse(note_number('2023年12月31日 附注五', '1,234'))
        self.assertFalse(note_number('2023年12月31日', '42'))

    def test_title_as_document_title_with_unit_heading_after_it(self):
        found = context([{'type': 'doc_title', 'content': [{'type': 'text', 'content': '合并资产负债表'}]},
                         text('paragraph_title', '(除另有注明外，金额单位均为人民币元)')])
        self.assertEqual((found['title'], found['unit']), ('合并资产负债表', '元'))

    def test_title_in_running_header_and_group_columns(self):
        found = context([{'type': 'header', 'content': [{'type': 'text', 'content': '资产负债表(续)'}]}])
        self.assertEqual(found['title'], '资产负债表(续)')
        self.assertEqual(scope('资产负债表', '本集团 2025年末'), '合并')
        self.assertEqual(scope('资产负债表', '本公司 2025年末'), '母公司')

    def test_unit_inside_a_long_title(self):
        found = context([text('paragraph_title', '中国神华能源股份有限公司合并资产负债表2025 年 12 月 31 日'
                                                 '(金额单位:人民币百万元)')])
        self.assertEqual((found['unit'], found['unit_source']), ('百万元', 'title'))


# The text layer of 格力电器 2024 (000651-1223330631) page 111, whose table the
# layout parser merged into single cells, and the x of each column on the page:
# the two value columns are printed from 354.9 to 425.7 and 481.9 to 552.7.
CURRENT, PRIOR = 390.0, 500.0
PAGE = ('111\n'
        '合并资产负债表\n'
        '2024 年 12 月 31 日\n'
        '编制单位：珠海格力电器股份有限公司 单位：人民币元\n'
        '项 目 附注 2024 年 12 月 31 日 2024 年 1 月 1 日\n'
        '流动资产：\n'
        '货币资金 五、1 113,900,461,797.94 124,104,987,289.62\n'
        '拆出资金\n'
        '衍生金融资产 五、3 108,919,513.22\n'
        '应收股利 4,325,690.04 19,936,649.83\n'
        '其中：应收利息\n'
        '资产总计 368,031,704,522.86 368,053,902,576.37\n'
        '法定代表人： 主管会计工作负责人： 会计机构负责人：\n')
PAGE_WINDOWS = [(331.7, 393.4), (462.7, 515.5)]
PAGE_NUMBERS = [('113,900,461,797.94', CURRENT), ('124,104,987,289.62', PRIOR),
                ('108,919,513.22', PRIOR), ('4,325,690.04', CURRENT),
                ('19,936,649.83', PRIOR), ('368,031,704,522.86', CURRENT),
                ('368,053,902,576.37', PRIOR)]


def printed(text, values):
    """The number spans of one page, at the x each number is printed on."""
    spans = {}
    for value, left in values:
        start = text.index(value)
        spans[start] = NumberSpan(value, start, start + len(value),
                                  (left, 700.0, left + len(value) * 4.0, 708.0))
    return spans


def text_layer_found(text, numbers, windows, title='合并资产负债表', date='2024年12月31日'):
    header = header_line(text)
    assert header is not None, 'the page text holds no column heading row'
    line, periods = header
    cells, mapped = text_layer_cells(text, line, column_headings(text, periods),
                                     windows, printed(text, numbers))
    placed = grid(cells)
    rows = header_rows(cells)
    meta = {'title': title, 'unit': '元', 'scale': '1', 'currency': '人民币',
            'statement_date': date, 'unit_source': 'text layer line before table'}
    return rows_to_facts(placed, rows, column_headers(placed, rows),
                         {(item['row'], item['column']): item for item in mapped},
                         meta, None, 111)[0]


class TextLayerTest(unittest.TestCase):
    def test_statement_title_heading_row_and_column_headings(self):
        self.assertEqual(statement_title(PAGE), ('合并资产负债表', '2024年12月31日'))
        line, periods = header_line(PAGE)
        self.assertEqual(PAGE[periods[0][0]:periods[0][1]], '2024 年 12 月 31 日')
        self.assertEqual(PAGE[periods[1][0]:periods[1][1]], '2024 年 1 月 1 日')
        # An opening balance column is written as the year-end it equals.
        self.assertEqual(column_headings(PAGE, periods), ['2024年12月31日', '2023年12月31日'])
        self.assertEqual(line, 4)

    def test_a_note_heading_is_not_a_statement(self):
        # Only a title followed by the statement's own date line opens a statement.
        self.assertIsNone(statement_title('（1）资产负债表\n项 目 附注 2024 年 12 月 31 日'))
        self.assertIsNone(statement_title('合并资产负债表\n项 目 附注 2024 年 12 月 31 日'))

    def test_page_text_gives_the_statement_row_by_row(self):
        facts = text_layer_found(PAGE, PAGE_NUMBERS, PAGE_WINDOWS)
        found = {(fact['row_label'], fact['column_header']): fact for fact in facts}
        self.assertEqual((found[('货币资金', '2024年12月31日')]['text'],
                          found[('货币资金', '2024年12月31日')]['unit'],
                          found[('货币资金', '2024年12月31日')]['section']),
                         ('113,900,461,797.94', '元', '流动资产：'))
        self.assertEqual(found[('货币资金', '2023年12月31日')]['text'], '124,104,987,289.62')
        # A row with one figure only: its horizontal position decides the column.
        self.assertEqual(found[('衍生金融资产', '2023年12月31日')]['text'], '108,919,513.22')
        self.assertNotIn(('衍生金融资产', '2024年12月31日'), found)
        # A label of its own without a value is not a fact, and the note reference
        # after a label is neither a value nor part of the label.
        self.assertFalse([fact for fact in facts if fact['row_label'] == '拆出资金'])
        self.assertEqual(sorted(fact['row_label'] for fact in facts),
                         ['应收股利', '应收股利', '衍生金融资产', '货币资金', '货币资金',
                          '资产总计', '资产总计'])

    def test_a_single_number_is_left_alone_without_column_positions(self):
        facts = text_layer_found(PAGE, PAGE_NUMBERS, [None, None])
        self.assertEqual([(fact['row_label'], fact['column_header'], fact['text'])
                          for fact in facts],
                         [('货币资金', '2024年12月31日', '113,900,461,797.94'),
                          ('货币资金', '2023年12月31日', '124,104,987,289.62'),
                          ('应收股利', '2024年12月31日', '4,325,690.04'),
                          ('应收股利', '2023年12月31日', '19,936,649.83'),
                          ('资产总计', '2024年12月31日', '368,031,704,522.86'),
                          ('资产总计', '2023年12月31日', '368,053,902,576.37')])

    def test_a_wrapped_label_continues_on_the_next_line(self):
        # 格力电器 2024 (000651-1223330631) page 113 prints the label of this row on
        # two lines while both its figures sit on the first of them.
        text = ('113\n合并利润表\n2024 年 1-12 月\n'
                '项 目 附注 2024 年度 2023 年度\n'
                '（二）按所有权归属分类：\n'
                '1、归属于母公司股东的净利润（净亏损以“-”号填\n'
                '列）\n'
                '32,184,570,372.28 29,017,387,604.18\n'
                '2、少数股东损益（净亏损以“-”号填列） 186,498,915.07 -1,298,364,690.59\n')
        facts = text_layer_found(text, [('32,184,570,372.28', CURRENT),
                                        ('29,017,387,604.18', PRIOR),
                                        ('186,498,915.07', CURRENT),
                                        ('-1,298,364,690.59', PRIOR)],
                                 [(387.8, 425.2), (482.3, 519.6)])
        self.assertEqual([(fact['row_label'], fact['column_header'], fact['text'])
                          for fact in facts],
                         [('1、归属于母公司股东的净利润（净亏损以“-”号填列）', '2024年度',
                           '32,184,570,372.28'),
                          ('1、归属于母公司股东的净利润（净亏损以“-”号填列）', '2023年度',
                           '29,017,387,604.18'),
                          ('2、少数股东损益（净亏损以“-”号填列）', '2024年度', '186,498,915.07'),
                          ('2、少数股东损益（净亏损以“-”号填列）', '2023年度',
                           '-1,298,364,690.59')])

    def test_a_bracketed_row_label_keeps_its_brackets_and_sign(self):
        # 格力电器 2024 (000651-1223330631) page 113: bracketed row labels are labels,
        # a dash marks a negative, and a bracketed row without figures is not a fact.
        text = ('113\n合并利润表\n2024 年 1-12 月\n'
                '项 目 附注 2024 年度 2023 年度\n'
                '1、不能重分类进损益的其他综合收益 141,296,779.55 -1,419,870,672.10\n'
                '（1）重新计量设定受益计划变动额 -34,144,434.00 -15,837,466.00\n'
                '（2）权益法下不能转损益的其他综合收益 -147,457,182.75 -852,472,209.72\n'
                '（4）企业自身信用风险公允价值变动\n')
        facts = text_layer_found(text, [('141,296,779.55', CURRENT),
                                        ('-1,419,870,672.10', PRIOR),
                                        ('-34,144,434.00', CURRENT),
                                        ('-15,837,466.00', PRIOR),
                                        ('-147,457,182.75', CURRENT),
                                        ('-852,472,209.72', PRIOR)],
                                 [(387.8, 425.2), (482.3, 519.6)])
        self.assertEqual([(fact['row_label'], fact['column_header'], fact['text'])
                          for fact in facts],
                         [('1、不能重分类进损益的其他综合收益', '2024年度', '141,296,779.55'),
                          ('1、不能重分类进损益的其他综合收益', '2023年度', '-1,419,870,672.10'),
                          ('（1）重新计量设定受益计划变动额', '2024年度', '-34,144,434.00'),
                          ('（1）重新计量设定受益计划变动额', '2023年度', '-15,837,466.00'),
                          ('（2）权益法下不能转损益的其他综合收益', '2024年度', '-147,457,182.75'),
                          ('（2）权益法下不能转损益的其他综合收益', '2023年度', '-852,472,209.72')])
        self.assertFalse([fact for fact in facts if '企业自身信用风险' in fact['row_label']])

    def test_a_note_reference_with_a_bracketed_sub_number_is_dropped(self):
        # 格力电器 2024 (000651-1223330631) page 114 writes the note as "五、70（1）";
        # the note is neither part of the label nor a value.
        text = ('114\n合并现金流量表\n2024 年 1-12 月\n'
                '项 目 附注 2024 年度 2023 年度\n'
                '收到其他与经营活动有关的现金 五、70（1） 3,249,319,163.59 2,838,768,145.79\n'
                '支付其他与经营活动有关的现金 五、70（1） 10,924,246,751.62 17,557,320,670.91\n'
                '经营活动产生的现金流量净额 29,369,250,570.66 56,398,426,354.17\n')
        facts = text_layer_found(text, [('3,249,319,163.59', CURRENT), ('2,838,768,145.79', PRIOR),
                                        ('10,924,246,751.62', CURRENT),
                                        ('17,557,320,670.91', PRIOR),
                                        ('29,369,250,570.66', CURRENT),
                                        ('56,398,426,354.17', PRIOR)],
                                 [(387.8, 425.2), (482.3, 519.6)])
        self.assertEqual([(fact['row_label'], fact['column_header'], fact['text'])
                          for fact in facts],
                         [('收到其他与经营活动有关的现金', '2024年度', '3,249,319,163.59'),
                          ('收到其他与经营活动有关的现金', '2023年度', '2,838,768,145.79'),
                          ('支付其他与经营活动有关的现金', '2024年度', '10,924,246,751.62'),
                          ('支付其他与经营活动有关的现金', '2023年度', '17,557,320,670.91'),
                          ('经营活动产生的现金流量净额', '2024年度', '29,369,250,570.66'),
                          ('经营活动产生的现金流量净额', '2023年度', '56,398,426,354.17')])

    def test_two_rows_printed_on_one_line_are_read_apart(self):
        # 格力电器 2025 (000651-1225250396) page 87 prints these two rows on one line.
        text = ('87\n合并资产负债表（续）\n2025 年 12 月 31 日\n'
                '项 目 附注 2025 年 12 月 31 日 2025 年 1 月 1 日\n'
                '股东权益：\n'
                '盈余公积 五、51 1,234,137,135.52 1,789,443,715.25 一般风险准备 五、52 '
                '512,918,996.50 509,245,480.58\n')
        facts = text_layer_found(text, [('1,234,137,135.52', CURRENT),
                                        ('1,789,443,715.25', PRIOR),
                                        ('512,918,996.50', CURRENT),
                                        ('509,245,480.58', PRIOR)],
                                 [(330.5, 392.0), (458.9, 511.4)],
                                 title='合并资产负债表（续）', date='2025年12月31日')
        self.assertEqual([(fact['row_label'], fact['column_header'], fact['text'],
                           fact['section']) for fact in facts],
                         [('盈余公积', '2025年12月31日', '1,234,137,135.52', '股东权益：'),
                          ('盈余公积', '2024年12月31日', '1,789,443,715.25', '股东权益：'),
                          ('一般风险准备', '2025年12月31日', '512,918,996.50', '股东权益：'),
                          ('一般风险准备', '2024年12月31日', '509,245,480.58', '股东权益：')])


if __name__ == '__main__':
    unittest.main()
