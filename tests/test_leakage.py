import random
import unittest

from src.data.leakage import content_text, near_pairs, shingles


class LeakageTests(unittest.TestCase):
    def test_ids_do_not_change_content(self):
        a = {'table': [['Sales', '12']], 'paragraphs': [{'uid': 'a', 'text': 'Total'}]}
        b = {'table': [['Sales', '12']], 'paragraphs': [{'uid': 'b', 'text': 'Total'}]}
        self.assertEqual(content_text(a), content_text(b))

    def test_prefix_join_matches_brute_force(self):
        rng = random.Random(19)
        texts = []
        for _ in range(30):
            words = [str(rng.randrange(10)) for _ in range(30)]
            texts.append(' '.join(words))
            words[rng.randrange(30)] = 'changed'
            texts.append(' '.join(words))
        for threshold in (0.5, 0.9, 1.0):
            values = [shingles(t) for t in texts]
            expected = {(j, i) for i in range(len(values)) for j in range(i)
                        if len(values[i] & values[j]) / len(values[i] | values[j]) >= threshold}
            self.assertEqual({(a, b) for a, b, _ in near_pairs(texts, threshold)}, expected)

    def test_empty_not_a_duplicate(self):
        self.assertEqual(near_pairs(['', '']), [])
