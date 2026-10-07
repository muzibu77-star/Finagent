"""Conservative source identity: same report and identical structured visible content."""

from src.evidence.models import evidence_hash


def evidence_identity(evidence: dict, source_report_id: str) -> tuple[str, str]:
    return source_report_id, evidence_hash({k: v for k, v in evidence.items() if k != 'evidence_id'})
