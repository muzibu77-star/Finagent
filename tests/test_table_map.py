import unittest

from src.data.table_map import NumberSpan, map_table, reading_order, table_cells


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


if __name__ == '__main__':
    unittest.main()
