import unittest

from src.training.extraction_samples import direction, shape


class ShapeTest(unittest.TestCase):
    def test_amounts_changes_growth_and_ratios(self):
        self.assertEqual(shape('(x0)/100000000'), {'kind': 'amount', 'name': 'x0'})
        self.assertEqual(shape('-(x0)/100000000'), {'kind': 'amount', 'name': 'x0'})
        self.assertEqual(shape('((x0)-(x1))/100000000'), {'kind': 'change', 'now': 'x0', 'then': 'x1'})
        for formula in ('(a-b)/abs(b)*100', '((x0)/(x1)-1)*100', '(a - b) / b * 100'):
            self.assertEqual(shape(formula)['kind'], 'growth', formula)
        margin = shape('(rev-cost)/rev*100')
        self.assertEqual((margin['kind'], margin['top'], margin['bottom']),
                         ('ratio', [(1, 'rev'), (-1, 'cost')], [(1, 'rev')]))
        self.assertEqual(shape('a/b*100')['kind'], 'ratio')

    def test_unreadable_forms_are_left_out(self):
        self.assertIsNone(shape('(a+b+c)/100000000'))
        self.assertIsNone(shape('max(a, b)'))
        self.assertIsNone(shape('a*b'))

    def test_direction_words_just_before_the_number(self):
        text = '营业收入同比下降7.5%，净利润增长10.3%'
        self.assertEqual(direction(text, text.index('7.5')), 'down')
        self.assertEqual(direction(text, text.index('10.3')), 'up')
        self.assertIsNone(direction('营业收入为100亿元', 5))


if __name__ == '__main__':
    unittest.main()
