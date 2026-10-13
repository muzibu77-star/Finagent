import unittest
from src.evidence.identity import evidence_identity


class EvidenceIdentityTests(unittest.TestCase):
    def test_aliases_require_same_report_and_structured_content(self):
        a={'evidence_id':'q1','table':[['Revenue','120'],['Profit','30']]}
        b={**a,'evidence_id':'q2'}
        self.assertEqual(evidence_identity(a,'A/2023'),evidence_identity(b,'A/2023'))
        self.assertNotEqual(evidence_identity(a,'A/2023'),evidence_identity(b,'B/2023'))
        b['table']=[['Revenue','120','Profit','30']]
        self.assertNotEqual(evidence_identity(a,'A/2023'),evidence_identity(b,'A/2023'))
