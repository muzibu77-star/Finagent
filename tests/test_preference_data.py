from pathlib import Path
import tempfile
import unittest

from src.tools.benchmark_tools import extract_numeric_facts
from src.training.preference_data import execution_reward, require_full_baseline


class PreferenceDataTests(unittest.TestCase):
    def test_actual_execution_correctness_and_source_required(self):
        facts = extract_numeric_facts({'evidence_id': 'p', 'table': [['Revenue', '120', '150']]})
        text = ('<tool_call>\n<function=calculate>\n<parameter=steps>\n'
                '[{"operation":"subtract","args":["f1","f0"]}]\n'
                '</parameter>\n</function>\n</tool_call>')
        output = {'text': text, 'stopped_on_eos': True}
        self.assertEqual(execution_reward(output, facts, 30)[0], 1)
        self.assertEqual(execution_reward(output, facts, 31)[0], 0)
        self.assertEqual(execution_reward(output, {'f0': facts['f0']}, 30)[0], 0)
        self.assertEqual(execution_reward({**output, 'stopped_on_eos': False}, facts, 30)[0], 0)
        self.assertEqual(execution_reward({'text': '30', 'stopped_on_eos': True}, facts, 30)[0], 0)
        self.assertNotIn('prediction', output)

    def test_incomplete_baseline_blocks_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'P0-2'):
                require_full_baseline(Path(directory))
