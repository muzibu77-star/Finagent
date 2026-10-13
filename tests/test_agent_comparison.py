import unittest

from src.evaluation.compare_agent_runs import paired_quality


class AgentComparisonTests(unittest.TestCase):
    def test_related_tasks_are_one_cluster_and_pairs_are_preserved(self):
        tasks = {'a': {'company': 'same'}, 'b': {'company': 'same'}}
        left = {'a': {'all_turns_passed': False}, 'b': {'all_turns_passed': True}}
        right = {'a': {'all_turns_passed': True}, 'b': {'all_turns_passed': True}}
        result = paired_quality(left, right, tasks, ['a', 'b'])
        self.assertEqual(result['difference']['clusters'], 1)
        self.assertEqual(result['difference']['mean'], .5)
        self.assertEqual(result['difference']['ci95'], [.5, .5])
        with self.assertRaises(ValueError):
            paired_quality(left, {'a': right['a']}, tasks, ['a', 'b'])
        tasks['b']['company'] = None
        self.assertEqual(paired_quality(left, right, tasks, ['a', 'b'])['difference']['clusters'], 2)
