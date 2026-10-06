"""CPU regressions for M0 parsing and acceptance; no model is loaded."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from src.evaluation import m0_inference_check as check


class ToolParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tools = json.loads(Path('configs/m0_smoke_tools.json').read_text())
        cls.schemas = {t['function']['name']: t['function']['parameters'] for t in tools}

    def call(self, operands='[120, 150]', extra=''):
        return (
            '<tool_call><function=calculate_metric>'
            '<parameter=operation>growth_rate</parameter>'
            f'<parameter=operands>{operands}</parameter>{extra}'
            '</function></tool_call>'
        )

    def test_valid_numeric_array(self):
        calls, errors = check.parse_tool_calls(self.call(), self.schemas)
        self.assertEqual(errors, [])
        self.assertEqual(calls[0]['arguments']['operands'], [120, 150])

    def test_invalid_numeric_array(self):
        for raw in ('[true, 150]', '["120", 150]', '[NaN, 150]', '[Infinity, 150]'):
            with self.subTest(raw=raw):
                self.assertTrue(check.parse_tool_calls(self.call(raw), self.schemas)[1])

    def test_duplicate_parameter(self):
        raw = self.call(extra='<parameter=operation>sum</parameter>')
        self.assertTrue(check.parse_tool_calls(raw, self.schemas)[1])

    def test_malformed_parameter(self):
        raw = self.call(extra='<parameter=bad>unterminated')
        self.assertTrue(check.parse_tool_calls(raw, self.schemas)[1])

    def test_malformed_call(self):
        self.assertTrue(check.parse_tool_calls('<tool_call>broken', self.schemas)[1])

    def test_invalid_enum(self):
        raw = self.call().replace('growth_rate', 'execute_python')
        self.assertTrue(check.parse_tool_calls(raw, self.schemas)[1])

    def test_extra_tool_call_fails_expectation(self):
        calls, _ = check.parse_tool_calls(self.call() * 2, self.schemas)
        self.assertFalse(check.meets_expectation({'tool': 'calculate_metric'}, '', calls))


class RepeatTests(unittest.TestCase):
    def records(self):
        return [{'id': 'a', 'repeat': i, 'text': 'ok'} for i in range(3)]

    def test_complete_repeats(self):
        self.assertTrue(check.repeats_identical(self.records(), ['a'], 3))

    def test_missing_repeat_or_task(self):
        self.assertFalse(check.repeats_identical(self.records()[:2], ['a'], 3))
        self.assertFalse(check.repeats_identical(self.records(), ['a', 'b'], 3))
        self.assertFalse(check.repeats_identical([], ['a'], 3))

    def test_duplicate_repeat(self):
        records = self.records()
        records[2]['repeat'] = 1
        self.assertFalse(check.repeats_identical(records, ['a'], 3))

    def test_different_output(self):
        records = self.records()
        records[2]['text'] = 'different'
        self.assertFalse(check.repeats_identical(records, ['a'], 3))


class AcceptanceTests(unittest.TestCase):
    def test_exit_status_matches_acceptance(self):
        for answer, eos, expected in (
            ("25", True, 0),
            ("26", True, 1),
            ("25 <tool_call>broken", True, 1),
            ("25", False, 1),
        ):
            with self.subTest(answer=answer, eos=eos), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = yaml.safe_load(Path("configs/m0_inference.yaml").read_text())
                task = json.loads(Path(config["paths"]["tasks"]).read_text().splitlines()[0])
                (root / "tasks.jsonl").write_text(json.dumps(task) + "\n")
                config["paths"]["tasks"] = str(root / "tasks.jsonl")
                config["paths"]["output_dir"] = str(root / "output")
                config["run"]["budget_probe_input_tokens"] = []
                (root / "config.yaml").write_text(yaml.safe_dump(config))
                record = {
                    "text": answer, "stopped_on_eos": eos, "latency_s": 1.0,
                    "tokens_per_s": 1.0, "peak_allocated_mib": 1.0,
                    "peak_reserved_mib": 1.0,
                }
                argv = ["check", "--config", str(root / "config.yaml"), "--variant", "bf16"]
                with (
                    patch.object(check.sys, "argv", argv),
                    patch.object(check.AutoTokenizer, "from_pretrained",
                                 return_value=Mock(chat_template="test")),
                    patch.object(check, "load_model"),
                    patch.object(check, "memory_mib", return_value={}),
                    patch.object(check.torch.cuda, "get_device_name", return_value="test"),
                    patch.object(check.torch.cuda, "reset_peak_memory_stats"),
                    patch.object(check, "generate", side_effect=lambda *a: dict(record)),
                    patch.object(check.logger, "info"),
                ):
                    self.assertEqual(check.main(), expected)
                summary = json.loads(next((root / "output").glob("*/*/summary.json")).read_text())
                self.assertEqual(summary["passed"], expected == 0)


if __name__ == '__main__':
    unittest.main()
