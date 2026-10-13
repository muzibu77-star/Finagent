import copy
import json
import unittest

from src.agent.runner import Runner
from tests.test_runner import RunnerTests, ScriptedModel, tool


class ContextTests(RunnerTests):
    def capture(self, calls):
        prompts = []

        class CapturingModel(ScriptedModel):
            def __call__(self, messages, tools, cancelled, deadline):
                prompts.append(copy.deepcopy(messages))
                return super().__call__(messages, tools, cancelled, deadline)

        row = self.run_model(CapturingModel(calls), context_management=True)
        return row, prompts

    @staticmethod
    def reads(messages):
        return [json.loads(m['content']) for m in messages
                if m['role'] == 'tool' and m['name'] == 'get_evidence']

    def test_complete_narrative_is_not_repeated_for_each_number(self):
        evidence = self.corpus.documents['e1']['evidence']
        paragraph = 'Revenue 120 and expenses 90; amounts in millions, unaudited.'
        evidence.update(table=[], pre_text=[paragraph], post_text=[])
        row, prompts = self.capture(self.successful_calls())
        self.assertEqual(row['business'], 'answered')
        content = '\n'.join(m.get('content', '') for m in prompts[-1])
        self.assertEqual(content.count(paragraph), 1)
        self.assertEqual(self.reads(prompts[-1])[0]['document']['evidence'], evidence)
        self.assertEqual(row['state']['facts']['f1']['location'],
                         {'section': 'pre_text', 'paragraph': 0, 'start': 25, 'end': 27})

    def test_pdf_paragraph_and_all_original_coordinates_survive(self):
        evidence = self.corpus.documents['e1']['evidence']
        paragraph = 'Revenue 120 and expenses 90 and margin 30, amounts in millions.'
        evidence.update(table=[], pre_text=[], post_text=[], paragraphs=[{'text': paragraph}])
        row, prompts = self.capture(self.successful_calls())
        content = '\n'.join(m.get('content', '') for m in prompts[-1])
        self.assertEqual(content.count(paragraph), 1)
        self.assertEqual(self.reads(prompts[-1])[0]['document']['evidence'], evidence)
        for fact in row['state']['facts'].values():
            loc = fact['location']
            self.assertEqual(paragraph[loc['start']:loc['end']], fact['value_token'])

    def test_low_overlap_table_denominator_is_not_removed(self):
        evidence = self.corpus.documents['e1']['evidence']
        evidence.update(table=[['', '2012'], ['total', '10877']],
                        pre_text=['unfunded commitments 685\n' * 45], post_text=[])
        calls = [tool('search_documents', query='unfunded commitments'),
                 tool('get_evidence', evidence_id='e1'),
                 tool('calculate', steps=[{'operation': 'divide', 'args': ['f2', 'f1']}]),
                 tool('submit_report', calculation_ids=['c0'], outcome='answered')]
        row, prompts = self.capture(calls)
        observation = self.reads(prompts[-1])[0]
        self.assertEqual(len(observation['facts']), 47)
        self.assertEqual(observation['document']['evidence'], evidence)
        self.assertEqual(row['report']['calculations'][0]['value'], 0.06298)
        self.assertEqual(row['state']['facts']['f1']['location'], {'row': 1, 'column': 1, 'start': 0, 'end': 5})

    def test_context_workflow_and_duplicate_search(self):
        calls = self.successful_calls()
        calls.insert(1, tool('search_documents', query=' revenue '))
        row, prompts = self.capture(calls)
        self.assertEqual(row['business'], 'answered')
        self.assertEqual(row['state']['attempts'], 5)
        self.assertEqual(len(row['state']['searches']), 1)
        actions = [call['function']['name'] for m in prompts[-1]
                   for call in m.get('tool_calls', [])]
        self.assertEqual(actions, ['search_documents', 'search_documents',
                                   'get_evidence', 'calculate'])
        self.assertEqual(json.loads(prompts[-1][-1]['content'])['calculation_id'], 'c0')
        body = json.loads(prompts[-1][1]['content'])
        self.assertEqual(body['calculations']['c0']['value'], 30)
        self.assertEqual(body['read_evidence_ids'], ['e1'])

    def test_repeated_reads_keep_one_complete_source_and_all_actions(self):
        calls = self.successful_calls()
        calls.insert(2, tool('get_evidence', evidence_id='e1'))
        row, prompts = self.capture(calls)
        self.assertEqual(row['state']['attempts'], 5)
        reads = self.reads(prompts[-1])
        self.assertEqual(len(reads), 2)
        self.assertEqual(reads[0]['document'], self.corpus.documents['e1'])
        self.assertEqual(set(reads[0]['facts']), set(row['state']['facts']))
        self.assertEqual(reads[1]['already_read'], 'e1')
        self.assertEqual(row['report']['calculations'][0]['value'], 30)


if __name__ == '__main__':
    unittest.main()
