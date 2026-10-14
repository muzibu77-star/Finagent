"""Local checks for manually adjudicated review numbers; never infer labels.

Evidence operands carry fact IDs or page spans, an exact Decimal value, and
their units. A formula documents the arithmetic. Semantic selection of the
metric, period, scope and operands remains the reviewer's responsibility.
"""

import ast
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN
import json
import operator
from pathlib import Path
import random
import re
import sqlite3
import struct


LABELS = {'correct', 'wrong', 'unverifiable', 'not_a_claim'}
ERRORS = {'value', 'unit', 'period', 'metric', 'scope', 'calculation', 'rounding'}
FIELDS = {
    'review_id', 'start', 'text', 'metric', 'period', 'scope', 'label',
    'error_type', 'expected', 'evidence', 'note',
}
NUMBER = re.compile(r'(-?\d[\d,]*(?:\.\d+)?)\s*(亿元|万元|百万元|%)')
SCALES = {'亿元': Decimal('1e8'), '万元': Decimal('1e4'),
          '百万元': Decimal('1e6'), '%': Decimal(1)}


def calculate(formula: str, operands: list[dict]) -> Decimal:
    """Evaluate explicit reviewer-authored arithmetic using Decimal only."""
    values = {item['name']: Decimal(item['value']) for item in operands}
    if len(values) != len(operands):
        raise ValueError('duplicate operand name')
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Decimal(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'abs' and len(node.args) == 1
                and not node.keywords):
            return abs(visit(node.args[0]))
        raise ValueError('unsupported calculation')

    result = visit(ast.parse(formula, mode='eval').body)
    if not result.is_finite():
        raise ValueError('nonfinite calculation')
    return result


def fact_operand(db: sqlite3.Connection, fact_id: int, name: str) -> dict:
    """Read an explicitly selected money/percent fact; preserve its sign.

    The reviewer must have checked the fact's metadata against its source page.
    Monetary operand values use yuan; percentage operands use percent units.
    """
    row = db.execute('SELECT text, unit, scale FROM facts WHERE id=?',
                     (fact_id,)).fetchone()
    if row is None:
        raise ValueError('unknown fact')
    text, unit, scale = row
    raw = re.sub(r'\s', '', text).replace(',', '').rstrip('%')
    raw = '-' + raw[1:-1] if raw.startswith('(') and raw.endswith(')') else raw
    if unit == '%':
        scale = '1'
    elif scale is None:
        raise ValueError('money fact has no scale')
    return {'name': name, 'fact_id': fact_id,
            'value': str(Decimal(raw) * Decimal(scale)),
            'unit': '%' if unit == '%' else '元'}


def adjudicate(review: dict, start: int, metric: str, period: str,
               operands: list[dict], formula: str, *, direction: int = 1,
               error_type: str = 'value', note: str = '', scope: str = '合并',
               comparison_operator: str = '=') -> dict:
    """Build a label from manually selected semantics, sources and formula.

    Formula results must already be expressed in the claim's written unit.
    No metric, period, scope, direction or evidence selection is automated.
    """
    number, = [item for item in review['numbers'] if item['start'] == start]
    result = compare(number['text'], calculate(formula, operands), direction,
                     comparison_operator)
    if result['label'] == 'wrong' and result['error_type'] is None:
        if error_type not in ERRORS:
            raise ValueError('invalid error type')
        result['error_type'] = error_type
    return {'review_id': review['id'], 'start': start, 'text': number['text'],
            'metric': metric, 'period': period, 'scope': scope, **result,
            'evidence': {'formula': formula, 'operands': operands,
                         'direction': direction,
                         'comparison_operator': comparison_operator}, 'note': note}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compare(text: str, expected: Decimal, direction: int = 1,
            comparison_operator: str = '=') -> dict:
    """Compare unrounded expected value in the written unit and direction.

    direction=-1 means a positive written magnitude describes a decrease or
    outflow. The returned expected string is also in that direction; a negative
    expected decrease therefore denotes an increase. Never discard signs.
    """
    match = NUMBER.fullmatch(text)
    if match is None or direction not in (-1, 1) or not expected.is_finite():
        raise ValueError('invalid number, direction or expected value')
    number = Decimal(match[1].replace(',', ''))
    step = Decimal(1).scaleb(number.as_tuple().exponent)
    difference = abs(number * direction - expected)
    label = 'correct' if difference <= step / 2 else 'wrong'
    if comparison_operator not in {'=', '<=', '>='}:
        raise ValueError('invalid comparison operator')
    if comparison_operator != '=':
        if direction != 1:
            raise ValueError('inequality requires an explicitly signed operand')
        relation = operator.le if comparison_operator == '<=' else operator.ge
        label = 'correct' if relation(expected, number) else 'wrong'
    return {
        'label': label,
        'error_type': ('rounding' if comparison_operator == '='
                       and step / 2 < difference <= step else None),
        'expected': format((expected * direction).quantize(
            step, rounding=ROUND_HALF_EVEN), 'f') + match[2],
    }


def blind_sample(reviews: list[dict], seed: int = 20261014) -> list[dict]:
    """Sample ceil(10%) in stable order without weak or first-pass labels."""
    items = []
    for review in reviews:
        for number in review['numbers']:
            items.append({
                'review_id': review['id'], 'start': number['start'],
                'text': number['text'], 'unit': number['unit'],
                'review_text': review['text'],
                'document_id': review['document_id'],
                'fiscal_year': review['fiscal_year'], 'code': review['code'],
            })
    items.sort(key=lambda item: (item['review_id'], item['start']))
    return random.Random(seed).sample(items, (len(items) + 9) // 10)


def validate(reviews: list[dict], labels: list[dict], db: sqlite3.Connection,
             *, complete: bool = True) -> dict:
    """Check coverage and resolvable citations, not their semantic correctness.

    Page numbers are one-based PDF pages; character spans are half-open.
    A previous annual report may supply an explicitly identified prior-period
    operand. Later reports and other issuers are always rejected.
    """
    inputs = {}
    for review in reviews:
        for number in review['numbers']:
            key = (review['id'], number['start'])
            if key in inputs:
                raise ValueError(f'duplicate input: {key}')
            start = number['start']
            if review['text'][start:start + len(number['text'])] != number['text']:
                raise ValueError(f'input text mismatch: {key}')
            inputs[key] = (review, number)
    seen = set()
    for label in labels:
        if set(label) != FIELDS:
            raise ValueError('invalid fields')
        key = (label['review_id'], label['start'])
        if key not in inputs or key in seen:
            raise ValueError(f'unknown or duplicate label: {key}')
        seen.add(key)
        review, number = inputs[key]
        if label['text'] != number['text'] or label['label'] not in LABELS:
            raise ValueError(f'invalid text or label: {key}')
        if not all(label[field] for field in ('metric', 'period', 'scope')):
            raise ValueError(f'missing claim semantics: {key}')
        if label['label'] == 'wrong':
            if label['error_type'] not in ERRORS:
                raise ValueError(f'missing error type: {key}')
        elif label['error_type'] is not None:
            raise ValueError(f'unexpected error type: {key}')
        if not isinstance(label['evidence'], dict):
            raise ValueError(f'invalid evidence object: {key}')
        if label['label'] in {'correct', 'wrong'}:
            if not label['expected'] or not label['evidence'].get('operands'):
                raise ValueError(f'missing evidence or expected: {key}')
            if not label['evidence'].get('formula'):
                raise ValueError(f'missing calculation: {key}')
            evidence = label['evidence']
            claim_text = label['text']
            if 'claim_value' in evidence:
                claim = evidence['claim_value']
                full_text = review['text'][claim['start']:claim['end']]
                claim_text = claim['value'] + claim['unit']
                if (not 0 <= claim['start'] < claim['end'] <= len(review['text'])
                        or full_text != claim['quote']
                        or full_text.replace('，', '').replace(',', '') != claim_text
                        or not claim['start'] <= label['start'] < claim['end']):
                    raise ValueError(f'invalid full claim span: {key}')
            result = compare(claim_text, calculate(
                evidence['formula'], evidence['operands']),
                evidence.get('direction', 1), evidence.get('comparison_operator', '='))
            if (result['label'] != label['label']
                    or result['expected'] != label['expected']):
                raise ValueError(f'calculation disagrees with label: {key}')
            if (result['error_type'] == 'rounding'
                    and label['error_type'] != 'rounding'):
                raise ValueError(f'rounding error misclassified: {key}')
        elif not label['note']:
            raise ValueError(f'missing explanation: {key}')
        for operand in label['evidence'].get('operands', []):
            if not Decimal(operand['value']).is_finite():
                raise ValueError(f'invalid operand: {key}')
            if operand.get('source_type') == 'pdf_image':
                doc, page = operand['document_id'], operand['page']
                if doc != review['document_id']:
                    raise ValueError(f'image must be from the target report: {key}')
                page_row = db.execute(
                    'SELECT text FROM pages WHERE document_id=? AND page=?',
                    (doc, page),
                ).fetchone()
                if page_row is None or page_row[0].strip():
                    raise ValueError(f'image exception requires an empty text page: {key}')
                sources = json.loads(Path('configs/cninfo_sources.json').read_text())
                source = next(item for item in sources['sources'] if item['id'] == doc)
                if (operand['pdf_path'] != source['path']
                        or not Path(operand['pdf_path']).is_file()):
                    raise ValueError(f'image PDF binding mismatch: {key}')
                image_path = Path(operand['image_path'])
                root = Path('data/staged/eval_v2_labels/evidence').resolve()
                if not image_path.resolve().is_relative_to(root):
                    raise ValueError(f'image outside evidence directory: {key}')
                header = image_path.read_bytes()[:24]
                if header[:8] != b'\x89PNG\r\n\x1a\n':
                    raise ValueError(f'evidence image is not PNG: {key}')
                width, height = struct.unpack('>II', header[16:24])
                box = operand['bbox']
                if (box['image_width'] != width or box['image_height'] != height
                        or not 0 <= box['x0'] < box['x1'] <= width
                        or not 0 <= box['y0'] < box['y1'] <= height
                        or not operand.get('quote')):
                    raise ValueError(f'invalid image coordinates: {key}')
                continue
            if 'fact_id' in operand:
                row = db.execute(
                    'SELECT document_id, page, start, end, text FROM facts WHERE id=?',
                    (operand['fact_id'],),
                ).fetchone()
                if row is None:
                    raise ValueError(f'unknown fact: {key}')
                doc, page, start, end, quote = row
            else:
                doc, page, start, end, quote = (
                    operand[field] for field in
                    ('document_id', 'page', 'start', 'end', 'quote')
                )
            if doc != review['document_id']:
                source = db.execute(
                    'SELECT code, fiscal_year, kind FROM documents WHERE id=?',
                    (doc,),
                ).fetchone()
                prior = review['fiscal_year'] - 1
                if (source != (review['code'], prior, 'annual')
                        or str(prior) not in operand.get('period', '')
                        or not operand.get('previous_period_reference')
                        or str(prior) not in label['period']):
                    raise ValueError(f'cross-report evidence not justified: {key}')
            row = db.execute('SELECT text FROM pages WHERE document_id=? AND page=?',
                             (doc, page)).fetchone()
            if row is None or not 0 <= start < end <= len(row[0]):
                raise ValueError(f'invalid evidence span: {key}')
            if re.sub(r'\s', '', row[0][start:end]) != re.sub(r'\s', '', quote):
                raise ValueError(f'evidence quote mismatch: {key}')
    if complete and seen != inputs.keys():
        raise ValueError(f'incomplete labels: {len(seen)}/{len(inputs)}')
    return {'input_numbers': len(inputs), 'labeled': len(seen),
            'remaining': len(inputs) - len(seen)}


def agreement(reviews: list[dict], first: list[dict], second: list[dict],
              seed: int = 20261014) -> dict:
    """Measure label agreement after independent review of the fixed sample.

    Independence cannot be proved by code: the second reviewer must not be
    shown first-pass labels. This function neither generates nor repairs them.
    """
    expected = {(item['review_id'], item['start'])
                for item in blind_sample(reviews, seed)}
    indexes = []
    for rows in (first, second):
        index = {(row['review_id'], row['start']): row['label'] for row in rows}
        if len(index) != len(rows) or any(value not in LABELS for value in index.values()):
            raise ValueError('duplicate or invalid adjudication')
        indexes.append(index)
    left, right = indexes
    if right.keys() != expected or not expected <= left.keys() or not expected:
        raise ValueError('second-pass sample does not match the fixed selection')
    matches = sum(left[key] == right[key] for key in expected)
    return {'seed': seed, 'sample_size': len(expected), 'matches': matches,
            'label_agreement': matches / len(expected),
            'passes_95_percent': matches * 100 >= len(expected) * 95}


def counts(reviews: list[dict], labels: list[dict]) -> dict:
    """Count one split, preserving model/condition groups and pending work."""
    inputs = {(r['id'], n['start']): (r, n)
              for r in reviews for n in r['numbers']}
    groups = {}
    for review, _ in inputs.values():
        group = (review['model'], 'closed' if review['closed_book'] else 'open')
        if group not in groups:
            groups[group] = {'input_numbers': 0, 'labeled': 0,
                             'labels': Counter(), 'error_types': Counter(),
                             'supported_true_wrong': 0, 'supported_false_correct': 0}
        groups[group]['input_numbers'] += 1
    seen = set()
    for label in labels:
        key = (label['review_id'], label['start'])
        if key in seen or key not in inputs:
            raise ValueError('duplicate or unknown label')
        seen.add(key)
        review, number = inputs[key]
        group = groups[(review['model'], 'closed' if review['closed_book'] else 'open')]
        group['labeled'] += 1
        group['labels'][label['label']] += 1
        if label['error_type'] is not None:
            group['error_types'][label['error_type']] += 1
        group['supported_true_wrong'] += int(
            number['supported'] and label['label'] == 'wrong')
        group['supported_false_correct'] += int(
            not number['supported'] and label['label'] == 'correct')
    result = []
    for (model, condition), group in sorted(groups.items()):
        group['labels'] = {name: group['labels'][name] for name in sorted(LABELS)}
        group['error_types'] = {name: group['error_types'][name]
                                for name in sorted(ERRORS)}
        result.append({'model': model, 'condition': condition, **group,
                       'remaining': group['input_numbers'] - group['labeled']})
    return {'input_numbers': len(inputs), 'labeled': len(labels),
            'remaining': len(inputs) - len(labels), 'groups': result}


def split_summary(reviews: list[dict], first: list[dict], second: list[dict],
                  seed: int = 20261014, *,
                  first_before_blind: list[dict] | None = None) -> dict:
    """Count final labels, measuring agreement against the pre-blind decisions.

    Supply the preserved first pass when adjudication changed any labels after
    blind review. Corrections must not artificially improve reported agreement.
    """
    result = counts(reviews, first)
    result['labels'] = dict(Counter(item['label'] for item in first))
    result['error_types'] = dict(Counter(item['error_type'] for item in first
                                        if item['error_type'] is not None))
    for name in LABELS:
        result['labels'].setdefault(name, 0)
    for name in ERRORS:
        result['error_types'].setdefault(name, 0)
    result['second_pass'] = None
    result['status'] = 'in_progress' if first else 'not_started'
    if not result['remaining']:
        result['status'] = 'first_pass_complete'
    if second:
        if result['remaining']:
            raise ValueError('blind review cannot precede a complete first pass')
        baseline = first if first_before_blind is None else first_before_blind
        if ({(row['review_id'], row['start']) for row in baseline}
                != {(row['review_id'], row['start']) for row in first}):
            raise ValueError('pre-blind coverage differs from final labels')
        result['second_pass'] = agreement(reviews, baseline, second, seed)
        keys = {(item['review_id'], item['start']) for item in second}
        sample_counts = counts(reviews, [item for item in first
                                        if (item['review_id'], item['start']) in keys])
        for field in ('supported_true_wrong', 'supported_false_correct'):
            result['second_pass'][field] = sum(group[field]
                                               for group in sample_counts['groups'])
        result['status'] = ('complete' if result['second_pass']['passes_95_percent']
                            else 'requires_rework')
    for field in ('supported_true_wrong', 'supported_false_correct'):
        result[field] = sum(group[field] for group in result['groups'])
    return result
