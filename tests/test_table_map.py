import unittest

from src.data.table_map import (NumberSpan, horizontal_shift, map_table, map_wrapped,
                                reading_order, table_cells)


def span(text, start, left, bottom):
    return NumberSpan(text, start, start + len(text), (left, bottom, left + 40, bottom + 8))


class TableMapTest(unittest.TestCase):
    def test_spans_place_cells_on_the_grid(self):
        cells = table_cells(
            '<table><tr><td rowspan="2">项目</td><td colspan="2">金额</td></tr>'
            '<tr><td>期末</td><td>期初</td></tr>'
            '<tr><td>货币资金</td><td>1,200.50</td><td>980.00</td></tr></table>')
        placed = {(c.row, c.column): c.text for c in cells}
        self.assertEqual(placed[(1, 1)], '期末')
        self.assertEqual(placed[(2, 2)], '980.00')

    def test_reading_order_is_visual_not_stream_order(self):
        ordered = reading_order([span('3', 20, 100, 500), span('2', 10, 300, 600),
                                 span('1', 30, 100, 601)])
        self.assertEqual([s.text for s in ordered], ['1', '2', '3'])

    def test_repeated_values_are_resolved_by_position(self):
        cells = table_cells('<table><tr><td>a</td><td>100</td><td>100</td></tr>'
                            '<tr><td>b</td><td>(5.2)</td><td>100</td></tr></table>')
        numbers = [span('100', 4, 100, 600), span('100', 9, 200, 600),
                   span('(5.2)', 15, 100, 580), span('100', 22, 200, 580)]
        result = map_table(cells, numbers)
        self.assertEqual([(m['row'], m['column'], m['start']) for m in result['mapped']],
                         [(0, 1, 4), (0, 2, 9), (1, 1, 15), (1, 2, 22)])
        self.assertFalse(result['unmapped_cells'] or result['unmapped_numbers'])

    def test_misread_digit_is_reported_not_mapped(self):
        cells = table_cells('<table><tr><td>营业收入</td><td>1,234.56</td><td>88.10</td></tr></table>')
        numbers = [span('1,284.56', 5, 100, 600), span('88.10', 15, 200, 600)]
        result = map_table(cells, numbers)
        self.assertEqual([m['text'] for m in result['mapped']], ['88.10'])
        self.assertEqual(result['unmapped_cells'][0]['text'], '1,234.56')
        self.assertEqual(result['unmapped_numbers'][0]['text'], '1,284.56')

    def test_grouping_commas_do_not_block_a_match(self):
        cells = table_cells('<table><tr><td>34151434208.68</td></tr></table>')
        result = map_table(cells, [span('34,151,434,208.68', 0, 100, 600)])
        self.assertEqual(result['mapped'][0]['text'], '34,151,434,208.68')

    def test_label_cells_and_note_references_are_ignored(self):
        cells = table_cells('<table><tr><td>货币资金</td><td>七、1</td><td>42</td></tr></table>')
        result = map_table(cells, [span('42', 12, 300, 600)])
        self.assertEqual(len(result['mapped']), 1)
        self.assertFalse(result['unmapped_cells'])

    def test_leftover_numbers_say_where_they_went(self):
        cells = table_cells('<table><tr><td>人民币百万元2022</td><td>2023</td></tr>'
                            '<tr><td>营业收入474,967</td><td>507,843</td></tr>'
                            '<tr><td>毛利率</td><td>增加0.24个百分点</td></tr></table>')
        numbers = [span('2022', 6, 200, 620), span('2023', 11, 300, 620),
                   span('474,967', 20, 200, 600), span('507,843', 28, 300, 600),
                   span('0.24', 40, 300, 580), span('99', 50, 300, 560)]
        result = map_table(cells, numbers)
        self.assertEqual({n['text']: n['where'] for n in result['unmapped_numbers']},
                         {'2022': 'merged', '474,967': 'merged', '0.24': 'embedded',
                          '99': 'missing'})

    def test_number_wrapped_across_lines_is_mapped_with_its_line_breaks(self):
        text = '项目\r\n8\r\n7\r\n2,\r\n4\r\n5\r\n1.\r\n4\r\n0\r\n99\r\n'

        class Page:
            def get_charbox(self, position):
                return (100.0, 600.0 - position, 106.0, 608.0 - position)

        cells = table_cells('<table><tr><td>项目</td><td>872,451.40</td><td>12</td></tr></table>')
        result = map_table(cells, [span('8', 4, 100, 596), span('7', 7, 100, 593),
                                   span('4', 14, 100, 586), span('5', 17, 100, 583),
                                   span('4', 24, 100, 576), span('0', 27, 100, 573),
                                   span('99', 30, 100, 570)])
        map_wrapped(result, Page(), text, (90.0, 500.0, 120.0, 620.0))
        wrapped = [m for m in result['mapped'] if m.get('wrapped')]
        self.assertEqual(len(wrapped), 1)
        self.assertEqual(''.join(text[wrapped[0]['start']:wrapped[0]['end']].split()), '872,451.40')
        self.assertEqual([c['text'] for c in result['unmapped_cells']], ['12'])
        self.assertEqual([n['text'] for n in result['unmapped_numbers']], ['99'])

    def test_text_layer_displaced_by_one_page_width_is_detected(self):
        size, table = (600.0, 800.0), [[0.1, 0.1, 0.9, 0.9]]
        at_page = [span('1', 0, 100, 400), span('2', 2, 300, 400)]
        displaced = [span('1', 0, 700, 400), span('2', 2, 900, 400)]
        self.assertEqual(horizontal_shift(at_page, size, table), 0.0)
        self.assertEqual(horizontal_shift(displaced, size, table), 600.0)
        self.assertEqual(horizontal_shift([], size, table), 0.0)


if __name__ == '__main__':
    unittest.main()
