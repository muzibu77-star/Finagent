import copy
import json
import unittest

from src.data.leakage import content_text
from src.evaluation.retrieval_run import metrics
from src.evaluation.validate_rag_generation import MODES, validate_retrieval, validate_rows


class RagAuditTests(unittest.TestCase):
    def setUp(self):
        self.tasks = {'q': {'split': 'test', 'company': 'issuer',
                            'source_report_id': 'report', 'relevant': ['p']}}
        self.gold = {'q': {'answer': '30', 'answer_type': 'arithmetic', 'scale': ''}}
        self.retrieval = {'q': {'results': {mode: {'ranking': ['p'],
                            'chunks': [{'page_id': 'p'}]} for mode in MODES}}}
        self.rows = [{'id': 'q', 'mode': mode, 'split': 'test', 'company': 'issuer',
                      'source_report_id': 'report', 'supplied': ['p'],
                      'labels_available': True, 'stopped_on_eos': True,
                      'text': json.dumps({'answer': 30, 'scale': '', 'citations': ['p']}),
                      'em': 1.0, 'f1': 1.0, 'scale': 1.0,
                      'citation_supported': True, 'stage': None} for mode in MODES]

    def audit(self, rows):
        return validate_rows(rows, self.tasks, self.gold, self.retrieval, {'p'})

    def test_complete_raw_output_rescoring(self):
        result = self.audit(self.rows)
        self.assertFalse(result['q/bm25']['answer_failed'])

    def test_missing_duplicate_wrong_score_and_false_citation_rejected(self):
        invalid = [self.rows[:-1], self.rows[:-1] + [self.rows[0]]]
        for field, value in [('em', 0), ('citation_supported', False),
                             ('supplied', ['outside']), ('stage', 'recall')]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            invalid.append(rows)
        for rows in invalid:
            with self.assertRaises(ValueError):
                self.audit(rows)

    def test_unlabeled_relevance_does_not_claim_answer_success(self):
        for row in self.rows:
            row.update(labels_available=False, em=0.0, f1=0.0, scale=0.0)
        result = validate_rows(self.rows, self.tasks, {}, self.retrieval, {'p'})
        self.assertIsNone(result['q/bm25']['answer_failed'])

    def test_excerpt_must_match_original_page(self):
        evidence = {'evidence_id': 'p', 'paragraphs': [{'text': 'Revenue is 30.'}], 'table': []}
        text = content_text(evidence)
        for result in self.retrieval['q']['results'].values():
            result.update(metrics(['p'], ['p']))
            result['chunks'] = [{'page_id': 'p', 'start': 0, 'end': len(text), 'text': text}]
        validate_retrieval(self.retrieval, self.tasks, {'p': {'evidence': evidence}})
        self.retrieval['q']['results']['bm25']['chunks'][0]['text'] = 'Revenue is 300.'
        with self.assertRaisesRegex(ValueError, 'original page span'):
            validate_retrieval(self.retrieval, self.tasks, {'p': {'evidence': evidence}})
