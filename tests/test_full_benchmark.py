import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from src.evaluation import select_expanded_adapter

from src.evaluation.full_benchmark import tatqa_prediction, tatqa_score
from src.evaluation.statistics import cluster_interval, paired_interval
from src.evaluation.validate_full_benchmark import validate_rows
from src.evaluation.select_expanded_adapter import development_rows


class FullBenchmarkTests(unittest.TestCase):
    def test_model_selection_rejects_test_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'settings.json').write_text(json.dumps({'split': 'test'}))
            with self.assertRaisesRegex(ValueError, 'never test'):
                development_rows(root)

    def test_model_selection_rejects_mixed_tatqa_prompts(self):
        policy = json.loads(Path('configs/jd_evaluation_protocol.json').read_text())
        settings = {'split': 'dev', 'receipt': {}, 'max_input_tokens': 8192,
                    'max_output_tokens': 256, 'backend': 'vllm', 'workers': 4,
                    'runtime_precision': {}, 'evaluation_policy': policy}
        rows = [{'dataset': 'finqa', 'condition': 'calculator', 'latency_s': 1}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regression = root / 'regression_historical_action'
            regression.mkdir()
            (regression / 'summary.json').write_text('{}')
            reads = [(rows, settings),
                     (rows, {**settings, 'tatqa_system_suffix': 'different prompt'})]
            with patch.object(select_expanded_adapter, 'development_rows', side_effect=reads), \
                 patch('sys.argv', ['select', '--root', str(root),
                                    '--output-dir', str(root / 'output')]):
                with self.assertRaisesRegex(ValueError, 'TAT-QA prompts differ'):
                    select_expanded_adapter.main()
            self.assertFalse((root / 'output').exists())

    def test_saved_wrong_score_and_missing_calls_are_rejected(self):
        tasks = [{'task_id': 't', 'dataset': 'tatqa', 'source_report_id': 'r'}]
        gold = {'t': {'labels_available': True, 'gold': {
            'answer_type': 'count', 'answer': 2, 'scale': ''}}}
        row = {'task_id': 't', 'dataset': 'tatqa', 'source_report_id': 'r',
               'condition': 'direct', 'labels_available': True, 'stopped_on_eos': True,
               'text': '{"answer": 2, "scale": ""}', 'em': 1.0, 'f1': 1.0, 'scale': 1.0}
        self.assertEqual(validate_rows([row], tasks, gold, {}), 0)
        for records in ([], [row, row], [{**row, 'em': 0.0}]):
            with self.assertRaises(ValueError):
                validate_rows(records, tasks, gold, {})

    def test_missing_labels_do_not_become_zero_scores(self):
        tasks = [{'task_id': 't', 'dataset': 'tatqa', 'source_report_id': 'r'}]
        gold = {'t': {'labels_available': False}}
        row = {'task_id': 't', 'dataset': 'tatqa', 'source_report_id': 'r',
               'condition': 'direct', 'labels_available': False}
        self.assertEqual(validate_rows([row], tasks, gold, {}), 0)
        with self.assertRaises(ValueError):
            validate_rows([{**row, 'em': 0.0}], tasks, gold, {})

    def test_multispan_and_zero_are_scored(self):
        gold = {'answer_type': 'multi-span', 'answer': ['North America', 'Asia'], 'scale': ''}
        self.assertEqual(tatqa_score({'answer': ['Asia', 'North America'], 'scale': ''}, gold)['em'], 1)
        zero = {'answer_type': 'count', 'answer': 0, 'scale': ''}
        self.assertEqual(tatqa_score({'answer': 0, 'scale': ''}, zero)['em'], 1)

    def test_bad_contract_is_rejected(self):
        for text in ('{"answer": true, "scale": ""}', '{"answer": 1, "scale": "yuan"}',
                     '{"answer": NaN, "scale": ""}', '{"answer": Infinity, "scale": ""}'):
            with self.assertRaises(ValueError):
                tatqa_prediction(text)

    def test_cluster_interval_keeps_single_report_together(self):
        rows = [{'source_report_id': 'one', 'em': x} for x in (0, 1)]
        self.assertEqual(cluster_interval(rows, 'em', repetitions=50)['ci95'], [.5, .5])

    def test_paired_comparison_rejects_changed_membership(self):
        row = {'task_id': 'one', 'condition': 'direct', 'source_report_id': 'r', 'em': 0}
        self.assertEqual(paired_interval([row], [{**row, 'em': 1}], 'em')['mean'], 1)
        with self.assertRaises(ValueError):
            paired_interval([row], [], 'em')


if __name__ == '__main__':
    unittest.main()
