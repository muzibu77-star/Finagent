import copy
import json
import re

from src.agent.runner import Runner
from tests.test_runner import RunnerTests, ScriptedModel, tool


class InlineTests(RunnerTests):
    def test_exact_original_survives_repeated_context_and_read(self):
        messages = []
        calls = self.successful_calls()
        calls.insert(2, tool('get_evidence', evidence_id='e1'))
        model = ScriptedModel(calls)
        def capture(prompt, tools, cancelled, deadline):
            messages.append(copy.deepcopy(prompt))
            return model(prompt, tools, cancelled, deadline)
        row = self.run_model(capture, context_management=True, inline_facts=True)
        self.assertEqual(row['report']['calculations'][0]['value'], 30)
        reads = [json.loads(m['content']) for m in messages[-1]
                 if m.get('name') == 'get_evidence']
        self.assertEqual(reads[1]['already_read'], 'e1')
        original = self.corpus.documents['e1']['evidence']
        encoded = json.dumps(reads[0]['document']['evidence'])
        self.assertEqual(json.loads(re.sub(r'\[f\d+\]', '', encoded)), original)
        self.assertEqual(encoded.count('[f0]'), 1)
        self.assertNotIn('facts', reads[0])
        self.assertEqual(row['state']['facts']['f0']['raw'], '120')

    def test_column_headers_only_where_columns_are_ambiguous(self):
        from src.agent.inline import inline_observation
        from src.tools.benchmark_tools import extract_numeric_facts

        def render(table, **options):
            evidence = {'evidence_id': 'e', 'table': table, 'pre_text': ['sales rose 7 %']}
            facts = extract_numeric_facts(evidence)
            shown = inline_observation({'document': {'evidence': evidence}, 'facts': facts},
                                       facts, **options)
            return shown, evidence

        wide = [['', 'december 31 2013', '2012'], ['revenue 5', '$ 120', '150']]
        default, original = render(wide)
        self.assertNotIn('|', json.dumps(default))
        shown, _ = render(wide, column_headers=True)
        cells = shown['document']['evidence']['table']
        self.assertRegex(cells[1][1], r'^\$ \[f\d+\|december 31 2013\]120$')
        self.assertRegex(cells[1][2], r'^\[f\d+\|2012\]150$')
        # Header-row cells, label-column cells and numbers in text keep the plain tag.
        self.assertRegex(cells[0][2], r'^\[f\d+\]2012$')
        self.assertRegex(cells[1][0], r'^revenue \[f\d+\]5$')
        self.assertRegex(shown['document']['evidence']['pre_text'][0], r'^sales rose \[f\d+\]7 %$')
        encoded = json.dumps(shown['document']['evidence'])
        self.assertEqual(json.loads(re.sub(r'\[f\d+(\|[^\]]*)?\]', '', encoded)), original)
        self.assertEqual(re.sub(r'\|[^\]]*\]', ']', encoded), json.dumps(default['document']['evidence']))
        self.assertIn('column header', shown['fact_reference_format'])
        narrow, _ = render([['balance at december 31 2011', '$ 4045'], ['increase', '299']],
                           column_headers=True)
        self.assertNotIn('|', json.dumps(narrow['document']['evidence']))

    def test_column_headers_require_inline_facts(self):
        with self.assertRaisesRegex(ValueError, 'inline fact tags'):
            Runner(self.store, self.corpus, ScriptedModel([]), context_management=True,
                   column_headers=True)
