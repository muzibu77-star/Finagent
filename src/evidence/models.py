"""Traceable evidence and financial facts for deterministic business tools."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json


def number(text: str) -> Decimal:
    """Parse a disclosed scalar; never evaluate an expression."""
    value = text.strip().replace(',', '').replace('$', '').replace('%', '')
    if value.startswith('(') and value.endswith(')'):
        value = '-' + value[1:-1]
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('invalid numeric evidence') from exc
    if not result.is_finite():
        raise ValueError('nonfinite numeric evidence')
    return result


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    document_id: str
    version: str
    text: str
    source_hash: str
    page: int | None = None
    section: str | None = None
    table_id: str | None = None
    row: int | None = None
    column: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class Fact:
    fact_id: str
    entity: str
    metric: str
    value: Decimal
    unit: str
    currency: str | None
    scale: Decimal
    period: str
    scope: str
    evidence_ids: tuple[str, ...]
    status: str = 'candidate'


def validate_fact(fact: Fact, evidence: dict[str, Evidence]) -> None:
    """Verify exact disclosed scalar references; metadata still needs review."""
    if fact.status != 'validated':
        raise ValueError('financial metadata has not been validated')
    if not fact.evidence_ids or any(key not in evidence for key in fact.evidence_ids):
        raise ValueError('invalid evidence reference')
    if not fact.value.is_finite() or not fact.scale.is_finite() or fact.scale <= 0:
        raise ValueError('invalid fact value or scale')
    if any(number(evidence[key].text) != fact.value for key in fact.evidence_ids):
        raise ValueError('fact value does not match disclosed evidence')
    if not all((fact.entity, fact.metric, fact.unit, fact.period, fact.scope)):
        raise ValueError('missing financial metadata')


def evidence_hash(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
