"""Regression checks for conservative report linkage."""

import unittest

from src.data.tatqa_source_audit import link_sources, question_key, summarize


def qa():
    return {"table": {"uid": "t1"}, "questions": [
        {"uid": "q1", "question": "What revenue?", "answer": 123},
        {"uid": "q2", "question": "Which year?", "answer": 2020},
    ]}


def dqa(source="company_2020.pdf"):
    return {"doc": {"uid": "doc1", "source": source, "page": 1},
            "questions": qa()["questions"]}


class SourceAuditTests(unittest.TestCase):
    def test_exact_link_ignores_gold(self):
        row = dqa()
        row["questions"][0]["answer"] = "different answer"
        links = link_sources([("train", qa())], [("train", row)])
        self.assertEqual(links[0]["source_report_id"], "company_2020.pdf")
        self.assertNotIn("answer", links[0])

    def test_order_case_whitespace(self):
        row = qa()
        row["questions"].reverse()
        row["questions"][0]["question"] = " WHICH   YEAR? "
        self.assertEqual(question_key(row), question_key(qa()))

    def test_single_common_question_does_not_link(self):
        row = dqa()
        row["questions"] = row["questions"][:1]
        link = link_sources([("train", qa())], [("train", row)])[0]
        self.assertEqual(link["reason"], "no_exact_bundle")

    def test_ambiguous_source_quarantined(self):
        rows = [("train", dqa()), ("test", dqa("other.pdf"))]
        link = link_sources([("train", qa())], rows)[0]
        self.assertEqual(link["reason"], "ambiguous_reports")
        self.assertIsNone(link["source_report_id"])

    def test_cross_split_report_detected(self):
        links = link_sources([("train", qa())], [("test", dqa())])
        overlaps = summarize(links)["reports_in_multiple_official_splits"]
        self.assertEqual(overlaps, {"company_2020.pdf": ["test", "train"]})

    def test_duplicate_context_fails(self):
        with self.assertRaises(ValueError):
            link_sources([("train", qa()), ("train", qa())], [])

    def test_empty_questions_fail(self):
        with self.assertRaises(ValueError):
            question_key({"questions": []})
