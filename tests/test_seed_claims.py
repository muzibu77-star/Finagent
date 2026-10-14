from decimal import Decimal
import random
import unittest

from src.evaluation.seed_claims import (MEETING, amount, clean_metric, derived, direct, growth,
                                        in_yi, repeated, role, wrong_digit, wrong_growth,
                                        wrong_period, wrong_scope, wrong_source, wrong_unit)

PAIR = {'company': '某公司', 'fiscal_year': 2024, 'statement': '合并利润表', 'metric': '营业收入',
        'current_fact': 1, 'prior_fact': 2, 'current': Decimal('174144069958.25'),
        'prior': Decimal('150560330316.45')}


class SeedClaimsTest(unittest.TestCase):
    def test_column_roles(self):
        self.assertEqual(role('2024年度', 2024), 'current')
        self.assertEqual(role('2023年12月31日', 2024), 'prior')
        self.assertEqual(role('期初余额', 2024), 'prior')
        self.assertIsNone(role('2022年度', 2024))
        self.assertIsNone(role('附注', 2024))

    def test_metric_names_lose_numbering_and_filing_hints(self):
        self.assertEqual(clean_metric('一、营业总收入'), '营业总收入')
        self.assertEqual(clean_metric('四、净利润（净亏损以“－”号填列）'), '净利润')
        self.assertEqual(clean_metric('其中：营业成本'), '营业成本')

    def test_amounts_scale_and_sign(self):
        self.assertEqual(amount('(1,456)', '1000000'), Decimal('-1456000000'))
        self.assertEqual(in_yi(Decimal('174144069958.25')), '1741.44')
        self.assertEqual(growth(PAIR['current'], PAIR['prior']), '15.7')

    def test_correct_claims_mark_the_number_span(self):
        claim = direct(PAIR)
        self.assertEqual(claim['text'], '某公司2024年营业收入为1741.44亿元。')
        self.assertEqual(claim['text'][claim['span'][0]:claim['span'][1]], '1741.44')
        self.assertEqual(derived(PAIR)['text'], '某公司2024年营业收入同比增长15.7%。')
        self.assertIsNone(derived({**PAIR, 'prior': Decimal('-5')}))

    def test_each_injection_changes_exactly_what_it_records(self):
        rng, claim, change = random.Random(1), direct(PAIR), derived(PAIR)
        digit = wrong_digit(claim, rng)
        self.assertEqual((digit['error'], digit['label']), ('E1', 'inconsistent'))
        self.assertEqual(sum(a != b for a, b in zip(digit['stated'], claim['stated'])), 1)
        self.assertGreaterEqual(abs(Decimal(digit['stated']) - Decimal(claim['stated'])), 1)
        unit = wrong_unit(claim, rng)
        self.assertEqual((unit['stated'], unit['error']), (claim['stated'], 'E2'))
        self.assertNotEqual(unit['unit'], '亿元')
        period = wrong_period(claim, PAIR)
        self.assertEqual((period['stated'], period['error']), ('1505.60', 'E3'))
        wrong = wrong_growth(change, PAIR)
        self.assertEqual((wrong['stated'], wrong['error']), ('13.5', 'E5'))
        self.assertEqual(wrong['text'][wrong['span'][0]:wrong['span'][1]], '13.5')
        self.assertIsNone(wrong_growth(claim, PAIR))
        self.assertIsNone(wrong_period(change, PAIR))

    def test_parent_amount_stated_as_group_amount(self):
        pair = {**PAIR, 'document_id': 'd'}
        parents = {('d', '合并利润表', '营业收入'): Decimal('90000000000')}
        scope = wrong_scope(direct(pair), pair, parents)
        self.assertEqual((scope['stated'], scope['error']), ('900.00', 'E4'))
        self.assertIsNone(wrong_scope(direct(pair), pair, {}))

    def test_second_mention_with_another_number(self):
        claim = direct(PAIR)
        twice = repeated(claim, random.Random(2))
        self.assertEqual(twice['error'], 'E9')
        self.assertEqual(twice['text'][twice['first_span'][0]:twice['first_span'][1]], '1741.44')
        self.assertEqual(twice['text'][twice['span'][0]:twice['span'][1]], twice['stated'])
        self.assertNotEqual(twice['stated'], '1741.44')

    def test_meeting_release_pattern(self):
        match = MEETING.search('新华社北京6月7日电 国务院总理李强6月7日主持召开国务院常务会议，'
                               '听取当前房地产市场形势和下一步构建房地产发展新模式有关工作考虑的汇报。')
        self.assertEqual((match.group(1), match.group(2)), ('6', '7'))
        self.assertTrue(match.group(3).startswith('听取当前房地产'))

    def test_source_note_naming_an_earlier_report(self):
        pair = {**PAIR, 'code': '600000', 'current_digits': '174144069958.25'}
        found = wrong_source(direct(pair), pair, {('600000', 2023): {'150560330316.45'}})
        self.assertEqual([(c['label'], c['cited_report'][1]) for c in found],
                         [('consistent', 2024), ('inconsistent', 2023)])
        self.assertTrue(found[1]['text'].endswith('（数据来源：某公司2023年年度报告）。'))
        self.assertEqual(found[1]['text'][found[1]['span'][0]:found[1]['span'][1]], '1741.44')
        later = wrong_source(direct(pair), pair, {('600000', 2023): {'174144069958.25'}})
        self.assertEqual(len(later), 1)


if __name__ == '__main__':
    unittest.main()
