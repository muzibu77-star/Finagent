"""Opt-in harness feedback: actionable errors, retry guard, official constants, turn memory."""

import json
from pathlib import Path
import tempfile
import unittest

from src.agent.context import linear_program
from src.agent.runner import Runner
from src.data.corpus import Corpus
from src.harness.store import Store
from src.tools.benchmark_tools import CONSTANTS, OFFICIAL_CONSTANTS, explain_calculation_error
from tests.test_runner import ScriptedModel, tool


class RecordingModel(ScriptedModel):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.seen = []

    def __call__(self, messages, tools, cancelled, deadline):
        self.seen.append(json.loads(json.dumps(messages)))
        return super().__call__(messages, tools, cancelled, deadline)


class HarnessFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        corpus = {'e1': {'company': 'A', 'period': '2023', 'source_report_id': 'A/2023', 'evidence': {
            'evidence_id': 'e1', 'table': [['Revenue', '120', '150']]}}}
        (root / 'corpus.json').write_text(json.dumps(corpus))
        self.corpus = Corpus(str(root / 'corpus.json'), 's1')
        self.store = Store(str(root / 'state.sqlite'))
        self.payload = {'question': 'What is the change?', 'company': 'A', 'period': '2023', 'snapshot_id': 's1'}

    def tearDown(self):
        self.corpus.close()
        self.tmp.cleanup()

    def observations(self, request='r1'):
        return [e['body']['observation'] for e in self.store.events(request) if e['kind'] == 'tool_observation']

    def run_model(self, outputs, **options):
        model = RecordingModel(outputs)
        self.store.submit('r1', self.payload)
        row = Runner(self.store, self.corpus, model, context_management=True, **options).run('r1')
        return row, model

    def test_rejected_calculation_names_the_operand_and_escalates_on_identical_retry(self):
        bad = tool('calculate', steps=[{'operation': 'multiply', 'args': ['f0', 'const_1000']}])
        good = [tool('calculate', steps=[{'operation': 'subtract', 'args': ['f1', 'f0']}]),
                tool('submit_report', calculation_ids=['c0'], outcome='answered')]
        row, _ = self.run_model([tool('get_evidence', evidence_id='e1'), bad, bad, *good], error_feedback=True)
        self.assertEqual(row['business'], 'answered')
        first, second = self.observations()[1:3]
        self.assertIn('const_1000 is not an available constant', first['hint'])
        self.assertNotIn('repeated_failure', first)
        self.assertIn('failed 2 times', second['repeated_failure'])
        self.assertEqual(first['error'], second['error'])  # the tool's own verdict is unchanged

    def test_feedback_is_absent_by_default(self):
        bad = tool('calculate', steps=[{'operation': 'multiply', 'args': ['f0', 'const_1000']}])
        self.run_model([tool('get_evidence', evidence_id='e1'), bad, bad], max_calls=3)
        for observation in self.observations()[1:]:
            self.assertEqual(set(observation), {'error'})

    def test_official_constants_execute_only_when_enabled(self):
        calls = [tool('get_evidence', evidence_id='e1'),
                 tool('calculate', steps=[{'operation': 'multiply', 'args': ['f0', 'const_1000']}]),
                 tool('submit_report', calculation_ids=['c0'], outcome='answered')]
        row, model = self.run_model(calls, official_constants=True)
        self.assertEqual(row['report']['calculations'][0]['value'], 120000)
        self.assertIn('const_1000000', model.seen[0][0]['content'])
        self.assertIn('const_1000', OFFICIAL_CONSTANTS - CONSTANTS)

    def test_malformed_reply_gets_the_parser_problem_and_required_form(self):
        plain = {'text': '{"calculation_ids": ["c0"], "outcome": "answered"}', 'stopped_on_eos': True, 'new_tokens': 5}
        _, model = self.run_model([plain, plain], max_calls=2, error_feedback=True)
        correction = model.seen[1][-1]['content']
        self.assertIn('Plain JSON or prose is not a tool call', correction)
        self.assertIn('<function=TOOL_NAME>', correction)

    def test_follow_up_turn_sees_the_previous_program_only_when_enabled(self):
        first = [tool('get_evidence', evidence_id='e1'),
                 tool('calculate', steps=[{'operation': 'subtract', 'args': ['f1', 'f0']}]),
                 tool('submit_report', calculation_ids=['c0'], outcome='answered')]
        for request, enabled in (('with', True), ('without', False)):
            model = RecordingModel(first + first)
            runner = Runner(self.store, self.corpus, model, context_management=True, history_programs=enabled)
            self.store.submit(request, self.payload)
            runner.run(request)
            self.store.submit(request + ':2', {**self.payload, 'question': 'and as a ratio?'}, task_id=request)
            runner.run(request)
            history = json.loads(model.seen[3][1]['content'])['history']
            self.assertEqual(history[0]['calculations'][0].get('program'),
                             'subtract(150, 120)' if enabled else None)

    def test_program_rendering_and_error_explanations(self):
        self.assertEqual(linear_program(['subtract(', '9', '4', ')', 'divide(', '#0', '4', ')', 'EOF']),
                         'subtract(9, 4), divide(#0, 4)')
        facts = {'f0': {'value_token': '10'}, 'f1': {'value_token': '0'}}
        explain = lambda steps, done={}: explain_calculation_error({'steps': steps}, facts, done)
        self.assertIn('not an earlier step', explain([{'operation': 'add', 'args': ['f0', '#0']}]))
        self.assertIn('exactly two args', explain([{'operation': 'add', 'args': ['f0', 'f0', 'f1']}]))
        self.assertIn('f0 to f1', explain([{'operation': 'add', 'args': ['f0', 'f7']}]))
        self.assertIn('whose value is 0', explain([{'operation': 'divide', 'args': ['f0', 'f1']}]))
        self.assertIn('only go to submit_report', explain_calculation_error(
            {'steps': [{'operation': 'add', 'args': ['f0', 'c0']}]}, facts, None))
        self.assertIsNone(explain([{'operation': 'add', 'args': ['f0', 'f0']}]))


if __name__ == '__main__':
    unittest.main()


class SamplingModel(RecordingModel):
    """Greedy outputs come from the script; sampled proposals from a separate list."""

    def __init__(self, outputs, samples):
        super().__init__(outputs)
        self.samples = iter(samples)
        self.sample_requests = []

    def sample(self, messages, tools, cancelled, deadline, temperature, seed):
        self.sample_requests.append((temperature, seed))
        return next(self.samples)


class SelfConsistencyTests(HarnessFeedbackTests):
    def run_voting(self, greedy, samples, **options):
        calls = [tool('get_evidence', evidence_id='e1'), greedy,
                 tool('submit_report', calculation_ids=['c0'], outcome='answered')]
        model = SamplingModel(calls, samples)
        self.store.submit('r1', self.payload)
        row = Runner(self.store, self.corpus, model, context_management=True,
                     self_consistency=3, **options).run('r1')
        return row, model

    def test_two_agreeing_samples_override_the_greedy_calculation(self):
        subtract = tool('calculate', steps=[{'operation': 'subtract', 'args': ['f1', 'f0']}])
        add = tool('calculate', steps=[{'operation': 'add', 'args': ['f1', 'f0']}])
        row, model = self.run_voting(add, [subtract, subtract])
        self.assertEqual(row['report']['calculations'][0]['value'], 30)
        self.assertEqual(model.sample_requests, [(0.7, 1), (0.7, 2)])
        self.assertEqual(row['report']['attempts'], 5)  # both samples are charged as calls
        event = next(e['body'] for e in self.store.events('r1') if e['kind'] == 'self_consistency')
        self.assertEqual((event['values'], event['chosen']), ([270.0, 30.0, 30.0], 1))

    def test_greedy_wins_when_no_two_proposals_agree(self):
        add = tool('calculate', steps=[{'operation': 'add', 'args': ['f1', 'f0']}])
        subtract = tool('calculate', steps=[{'operation': 'subtract', 'args': ['f1', 'f0']}])
        invalid = tool('calculate', steps=[{'operation': 'add', 'args': ['f1', 'f9']}])
        row, _ = self.run_voting(add, [subtract, invalid])
        self.assertEqual(row['report']['calculations'][0]['value'], 270)

    def test_vote_is_skipped_when_it_would_leave_no_call_for_submission(self):
        add = tool('calculate', steps=[{'operation': 'add', 'args': ['f1', 'f0']}])
        row, model = self.run_voting(add, [], max_calls=4)
        self.assertEqual(row['business'], 'answered')
        self.assertEqual(model.sample_requests, [])

    def test_voting_requires_a_sampling_model(self):
        with self.assertRaisesRegex(ValueError, 'can sample'):
            Runner(self.store, self.corpus, ScriptedModel([]), self_consistency=3)


class CalculationRationaleTests(HarnessFeedbackTests):
    def test_rationale_is_required_recorded_and_not_executed(self):
        from src.agent.runner import SCHEMAS, tool_set
        tools, schemas = tool_set(True)
        self.assertEqual(schemas['calculate']['required'], ['reasoning', 'steps'])
        self.assertEqual(SCHEMAS['calculate']['required'], ['steps'])  # the shared definition is untouched
        self.assertEqual(tool_set(False), (__import__('src.agent.runner', fromlist=['TOOLS']).TOOLS, SCHEMAS))
        steps = [{'operation': 'subtract', 'args': ['f1', 'f0']}]
        calls = [tool('get_evidence', evidence_id='e1'),
                 tool('calculate', steps=steps),  # missing reasoning: rejected by the contract
                 tool('calculate', reasoning='f1 = 150: Revenue; f0 = 120: Revenue', steps=steps),
                 tool('submit_report', calculation_ids=['c0'], outcome='answered')]
        row, model = self.run_model(calls, calculation_rationale=True)
        self.assertEqual(row['report']['calculations'][0]['value'], 30)
        rejected = [e for e in self.store.events('r1') if e['kind'] == 'tool_rejected']
        self.assertEqual(len(rejected), 1)  # the call without reasoning never reaches the tool
        self.assertIn('reasoning', json.dumps(model.seen[0]))  # the tool definition sent to the model
        self.assertIn('Every calculate call starts with reasoning', model.seen[0][0]['content'])
        recorded = [e['body']['arguments'] for e in self.store.events('r1')
                    if e['kind'] == 'tool_observation' and e['body']['tool'] == 'calculate']
        self.assertEqual([r['reasoning'] for r in recorded], ['f1 = 150: Revenue; f0 = 120: Revenue'])

    def test_default_contract_rejects_a_reasoning_argument(self):
        steps = [{'operation': 'subtract', 'args': ['f1', 'f0']}]
        self.run_model([tool('get_evidence', evidence_id='e1'),
                        tool('calculate', reasoning='x', steps=steps)], max_calls=2)
        self.assertEqual(self.store.get('r1')['execution'], 'failed')

    def test_teacher_rationale_names_row_and_column_or_phrase(self):
        from src.training.teacher import fact_rationale
        evidence = {'evidence_id': 'e', 'table': [['', '2009', '2008'], ['net revenue', '3460', '3369']],
                    'pre_text': ['the company repaid $ 25 million of notes .'], 'post_text': []}
        steps = [{'operation': 'subtract', 'args': ['f2', 'f3']}, {'operation': 'divide', 'args': ['#0', 'f4']}]
        self.assertEqual(fact_rationale(evidence, steps, 10),
                         'f12 = 3460: net revenue, 2009; f13 = 3369: net revenue, 2008; '
                         'f14 = 25: "the company repaid $"')
