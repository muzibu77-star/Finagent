import io
import json
import unittest
from unittest.mock import patch

from src.model.http_driver import HttpDriver


class HttpDriverTests(unittest.TestCase):
    def test_public_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            HttpDriver('http://127.0.0.1:80@example.com', 'base')

    def test_raw_token_budget_and_eos_contract(self):
        with patch('src.model.http_driver.AutoTokenizer.from_pretrained') as tokenizer, \
                patch('src.model.http_driver.urllib.request.build_opener') as opener:
            tokenizer.return_value.apply_chat_template.return_value = [1, 2, 3]
            tokenizer.return_value.eos_token_id = 7
            tokenizer.return_value.pad_token_id = 8
            opener.return_value.open.side_effect = [
                io.BytesIO(b'{"data":[{"id":"base"}]}'),
                io.BytesIO(b'{"choices":[{"text":"ok","finish_reason":"stop"}],'
                           b'"usage":{"completion_tokens":1}}')]
            driver = HttpDriver('http://127.0.0.1:8000', 'base')
            self.assertEqual(driver.generate([], max_input_tokens=2)['error'], 'input_over_budget')
            result = driver.generate([], max_input_tokens=3)
            self.assertTrue(result['stopped_on_eos'])
            payload = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertEqual(payload['prompt'], [1, 2, 3])
            self.assertEqual(payload['stop_token_ids'], [7, 8])
            self.assertEqual(payload['temperature'], 0)
            self.assertFalse(tokenizer.return_value.apply_chat_template.call_args.kwargs['return_dict'])


if __name__ == '__main__':
    unittest.main()
