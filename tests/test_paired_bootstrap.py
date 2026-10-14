import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.paired_bootstrap import compare


def write(path: Path, rows: list[tuple]) -> Path:
    path.write_text(''.join(json.dumps({'review_id': r, 'start': s, 'truth': t, 'verdict': v}) + '\n'
                            for r, s, t, v in rows))
    return path


class PairedBootstrapTest(unittest.TestCase):
    def test_difference_and_identical_runs(self):
        rows = [('r1', 0, 'correct', 'consistent'), ('r1', 5, 'correct', 'inconsistent'),
                ('r2', 0, 'wrong', 'inconsistent'), ('r2', 9, 'correct', 'consistent')]
        better = [('r1', 0, 'correct', 'consistent'), ('r1', 5, 'correct', 'consistent'),
                  ('r2', 0, 'wrong', 'inconsistent'), ('r2', 9, 'correct', 'consistent')]
        with tempfile.TemporaryDirectory() as tmp:
            a, b = write(Path(tmp) / 'a', rows), write(Path(tmp) / 'b', better)
            result = compare(a, b, samples=200)['rates']
            self.assertAlmostEqual(result['false_alarm_rate']['b_minus_a'], -1 / 3, places=4)
            same = compare(a, a, samples=50)['rates']['confirm_rate']
            self.assertEqual(same['ci95'], [0.0, 0.0])

    def test_runs_must_score_the_same_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write(Path(tmp) / 'a', [('r1', 0, 'correct', 'consistent')])
            b = write(Path(tmp) / 'b', [('r1', 1, 'correct', 'consistent')])
            with self.assertRaises(ValueError):
                compare(a, b, samples=10)


if __name__ == '__main__':
    unittest.main()
