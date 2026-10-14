import hashlib
import json
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

    def test_local_model_is_free_and_switches_thinking_through_the_template(self):
        sent = []
        client = ApiClient('qwen3.5-9b', 0.0, self.cache, url='http://127.0.0.1:8000/v1/chat/completions',
                           send=lambda body: sent.append(body) or answer())
        row = client.chat([{'role': 'user', 'content': 'a'}], 8)
        self.assertEqual(row['cost_usd'], 0.0)
        self.assertEqual(sent[0]['chat_template_kwargs'], {'enable_thinking': False})
        self.assertNotIn('thinking', sent[0])
        with self.assertRaises(ValueError):
            ApiClient('qwen3.5-9b', 0.0, self.cache, url='https://example.com/v1', send=lambda b: answer())

    def test_tools_ride_along_and_untooled_requests_keep_the_older_hash(self):
        messages = [{'role': 'user', 'content': 'a'}]
        tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {}}}]
        client = self.client(1.0)
        client.chat(messages, 8, tools=tools)
        self.assertEqual(self.sent[0]['tools'], tools)
        client.chat(messages, 8)
        self.assertNotIn('tools', self.sent[1])
        # A request without tools hashes exactly as it did before tools existed,
        # so the caches written by the extraction runs still answer it.
        body = {'model': 'deepseek-flash', 'messages': messages, 'max_tokens': 8,
                'thinking': {'type': 'disabled'}}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False)
                                .encode()).hexdigest()
        self.assertIn(digest, [json.loads(line)['request_sha256']
                               for line in self.cache.read_text().splitlines()])
        again = self.client(1.0)
        again.chat(messages, 8)
        self.assertEqual(len(self.sent), 2)

    def test_tool_calls_are_kept_with_the_answer(self):
        calls = [{'id': 'call_1', 'type': 'function',
                  'function': {'name': 'f', 'arguments': '{}'}}]
        response = answer()
        response['choices'][0]['message']['tool_calls'] = calls
        row = self.client(1.0, response).chat([{'role': 'user', 'content': 'a'}], 8)
        self.assertEqual(row['tool_calls'], calls)
        self.assertEqual(json.loads(self.cache.read_text().splitlines()[0])['tool_calls'], calls)

    def test_an_answer_cached_before_tools_existed_carries_no_tool_calls(self):
        headless = {'request_sha256': 'x', 'model': 'deepseek-flash', 'content': '好',
                    'finish_reason': 'stop',
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1}, 'cost_usd': 0.0}
        self.cache.write_text(json.dumps(headless) + '\n')
        client = self.client(1.0)
        self.assertEqual(client.spent, 0.0)
        self.assertIsNone(client.answers['x'].get('tool_calls'))

    def test_extract_local_model_is_free(self):
        client = ApiClient('qwen3.5-9b-extract', 0.0, self.cache,
                           url='http://127.0.0.1:8000/v1/chat/completions',
                           send=lambda body: answer())
        row = client.chat([{'role': 'user', 'content': 'a'}], 8)
        self.assertEqual(row['cost_usd'], 0.0)


if __name__ == '__main__':
    unittest.main()
