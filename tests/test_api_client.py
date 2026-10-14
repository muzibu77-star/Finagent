from pathlib import Path
import tempfile
import unittest

from src.model.api_client import ApiClient, BudgetExceeded


def answer(prompt_tokens=1_000_000, completion_tokens=0):
    return {'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens},
            'choices': [{'message': {'content': '好'}, 'finish_reason': 'stop'}]}


class ApiClientTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cache = Path(self.directory.name) / 'answers.jsonl'
        self.sent = []

    def tearDown(self):
        self.directory.cleanup()

    def client(self, budget, payload=None):
        def send(body):
            self.sent.append(body)
            return payload or answer()
        return ApiClient('deepseek-flash', budget, self.cache, send=send)

    def test_cost_uses_recorded_prices_and_thinking_is_off_by_default(self):
        row = self.client(1.0, answer(1_000_000, 500_000)).chat([{'role': 'user', 'content': 'a'}], 8)
        self.assertAlmostEqual(row['cost_usd'], 0.3 + 0.6)
        self.assertEqual(self.sent[0]['thinking'], {'type': 'disabled'})

    def test_repeated_request_is_answered_from_the_cache_across_clients(self):
        messages = [{'role': 'user', 'content': 'a'}]
        self.client(1.0).chat(messages, 8)
        again = self.client(1.0)
        again.chat(messages, 8)
        self.assertEqual(len(self.sent), 1)
        self.assertAlmostEqual(again.spent, 0.3)

    def test_no_new_call_once_the_budget_is_spent(self):
        client = self.client(0.25)
        client.chat([{'role': 'user', 'content': 'a'}], 8)
        with self.assertRaises(BudgetExceeded):
            client.chat([{'role': 'user', 'content': 'b'}], 8)
        self.assertEqual(len(self.sent), 1)

    def test_cache_never_contains_a_key(self):
        self.client(1.0).chat([{'role': 'user', 'content': 'a'}], 8)
        self.assertNotIn('Bearer', self.cache.read_text())

    def test_unpriced_model_is_rejected(self):
        with self.assertRaises(ValueError):
            ApiClient('unknown', 1.0, self.cache, send=lambda body: answer())


if __name__ == '__main__':
    unittest.main()
