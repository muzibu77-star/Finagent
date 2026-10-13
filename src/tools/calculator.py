"""Bounded Decimal arithmetic over validated, traceable fact IDs."""

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext

from src.evidence.models import Evidence, Fact, validate_fact

OPERATIONS = {'sum', 'subtract', 'multiply', 'divide', 'ratio', 'growth_rate',
              'difference', 'percentage_points'}


@dataclass(frozen=True)
class Calculation:
    calculation_id: str
    operation: str
    fact_ids: tuple[str, ...]
    value: str
    unit: str
    currency: str | None
    version: str = 'decimal-v1'
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def calculate(
    calculation_id: str, operation: str, fact_ids: list[str],
    facts: dict[str, Fact], evidence: dict[str, Evidence],
) -> Calculation:
    """Compute one operation, retaining full Decimal precision up to 28 digits.

    Monetary scales are converted to base units. Negative-base business growth
    returns a difference and explanation. Baseline dataset scoring must use its
    own reference-program semantics instead.
    """
    if operation not in OPERATIONS or not 1 <= len(fact_ids) <= 32:
        raise ValueError('unsupported operation or operand budget')
    if operation != 'sum' and len(fact_ids) != 2:
        raise ValueError('binary operation requires two facts')
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError('duplicate fact reference')
    if any(key not in facts for key in fact_ids):
        raise ValueError('unknown fact reference')
    selected = [facts[key] for key in fact_ids]
    for fact in selected:
        validate_fact(fact, evidence)
    first = selected[0]
    if any((f.entity, f.scope) != (first.entity, first.scope) for f in selected):
        raise ValueError('entity or scope mismatch')
    if any((f.unit, f.currency) != (first.unit, first.currency) for f in selected):
        raise ValueError('unit or currency mismatch')
    if operation in ('growth_rate', 'difference', 'percentage_points'):
        if selected[0].metric != selected[1].metric:
            raise ValueError('change requires the same metric')
        if operation == 'growth_rate' and selected[0].period == selected[1].period:
            raise ValueError('growth requires distinct periods')
    elif any(f.period != first.period for f in selected):
        raise ValueError('period mismatch')
    note = None
    unit, currency = first.unit, first.currency
    with localcontext() as ctx:
        ctx.prec = 28
        ctx.rounding = ROUND_HALF_EVEN
        values = [f.value * f.scale for f in selected]
        if operation == 'sum':
            result = sum(values, Decimal(0))
        elif operation == 'subtract':
            result = values[0] - values[1]
        elif operation == 'difference':
            result = values[1] - values[0]
        elif operation == 'multiply':
            if first.unit not in ('number', 'ratio'):
                raise ValueError('multiplication requires dimensionless operands')
            result = values[0] * values[1]
        elif operation in ('divide', 'ratio'):
            if values[1] == 0:
                raise ValueError('zero denominator')
            result = values[0] / values[1]
            unit, currency = 'ratio', None
        elif operation == 'percentage_points':
            if first.unit not in ('percent', 'ratio'):
                raise ValueError('percentage points require percentages or ratios')
            result = (values[1] - values[0]) * (100 if first.unit == 'ratio' else 1)
            unit, currency = 'percentage_points', None
        else:
            if values[0] == 0:
                raise ValueError('zero growth base')
            if values[0] < 0:
                result = values[1] - values[0]
                operation = 'difference'
                note = 'Negative base: report absolute change, not business growth rate.'
            else:
                result = (values[1] - values[0]) / values[0]
                unit, currency = 'ratio', None
    if not result.is_finite():
        raise ValueError('nonfinite calculation')
    return Calculation(calculation_id, operation, tuple(fact_ids),
                       str(result), unit, currency, note=note)
