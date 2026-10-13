from src.agent.runner import Runner
from src.tools.benchmark_tools import execute_with_results, extract_numeric_facts
from tests.test_runner import RunnerTests, ScriptedModel, tool


class CalculationReferenceTests(RunnerTests):
    def test_same_turn_result_keeps_sources_but_is_not_carried_to_next_turn(self):
        calls = self.successful_calls()[:3] + [
            tool('calculate', steps=[{'operation': 'divide', 'args': ['c0', 'f0']}]),
            tool('submit_report', calculation_ids=['c1'], outcome='answered')]
        row = self.run_model(ScriptedModel(calls), context_management=True,
                             calculation_references=True, inline_facts=True)
        result = row['report']['calculations'][0]
        self.assertEqual(result['value'], .25)
        self.assertEqual(set(result['facts']), {'f0', 'f1'})
        self.assertEqual(result['dependencies'], ['c0'])
        self.store.submit('next', {**self.payload, 'question': 'Use that result.'}, task_id='r1')
        model = ScriptedModel([tool('calculate', steps=[
            {'operation': 'multiply', 'args': ['c1', 'const_1']}])])
        row = Runner(self.store, self.corpus, model, max_calls=1,
                     context_management=True, calculation_references=True).run('r1')
        self.assertEqual(row['execution'], 'failed')
        self.assertFalse(row['state']['calculations'])

    def test_expanded_program_limit_is_explicit(self):
        facts = extract_numeric_facts({'evidence_id': 'e', 'table': [['1']]})
        calculations = {}
        for index in range(8):
            first = 'f0' if not index else f'c{index - 1}'
            steps = [{'operation': 'multiply', 'args': [first, 'const_1']}]
            steps.extend({'operation': 'multiply', 'args': [f'#{i - 1}', 'const_1']}
                         for i in range(1, 8))
            calculations[f'c{index}'] = execute_with_results({'steps': steps}, facts, calculations)
        self.assertEqual((len(calculations['c7']['program']) - 1) // 4, 64)
        with self.assertRaisesRegex(ValueError, 'expanded calculation step budget'):
            execute_with_results({'steps': [{'operation': 'multiply',
                                  'args': ['c7', 'const_1']}]}, facts, calculations)
