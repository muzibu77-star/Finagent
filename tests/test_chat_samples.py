import json
from pathlib import Path
import tempfile
import unittest

from src.training.m0_lora_check import review_issuer, training_issuers


class ChatSamplesTest(unittest.TestCase):
    def test_issuer_is_read_from_the_review_id(self):
        self.assertEqual(review_issuer('review-flash_closed-600519-1219506510'), '600519')
        with self.assertRaises(ValueError):
            review_issuer('seed-000001')

    def test_training_issuers_come_from_the_frozen_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'splits.json'
            path.write_text(json.dumps({'training': [{'code': '600519'}], 'development': [{'code': '000858'}]}))
            self.assertEqual(training_issuers(path), {'600519'})


if __name__ == '__main__':
    unittest.main()
