import unittest

from src.evaluation.rag_generation import failure_stage


class AttributionTests(unittest.TestCase):
    def test_earliest_document_pipeline_failure(self):
        relevant = {'a', 'b'}
        self.assertEqual(failure_stage(relevant, {'a'}, relevant, relevant, True), 'parsing')
        self.assertEqual(failure_stage(relevant, relevant, {'a'}, relevant, True), 'recall')
        self.assertEqual(failure_stage(relevant, relevant, relevant, {'a'}, True), 'reranking')
        self.assertEqual(failure_stage(relevant, relevant, relevant, relevant, False), 'generation')
        self.assertIsNone(failure_stage(relevant, relevant, relevant, relevant, True))


if __name__ == '__main__':
    unittest.main()
