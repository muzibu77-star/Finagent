"""Supervised extraction samples from verified review labels.

For every number in a review the verified label gives the figures it should be
compared with (a formula over operands that point at the report).  That formula
is read back into the extraction schema: an amount names one line item, a
growth or change compares one line item across two years, a ratio divides line
items.  The line item and year come from the referenced figure itself, so the
target says what the sentence refers to, whether its number is right or wrong.
No model output is used as a target.
"""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3

from src.verify.checker import norm, role
from src.verify.extract import PROMPT, line_items

UNITS = ('亿元', '万元', '百万元', '%')
UP = re.compile(r'(增长|增加|提升|上升|增至|扩大|上涨|回升|多流入|少流出)[^，。；]{0,3}$')
DOWN = re.compile(r'(下降|减少|下滑|降至|回落|收窄|下跌|缩减|少流入|多流出)[^，。；]{0,3}$')


class Figures:
    """Resolves an evidence operand to the line item and fiscal year it shows."""

    def __init__(self, library: sqlite3.Connection, labelled: sqlite3.Connection) -> None:
        self.db, self.old = library, labelled

    def operand(self, op: dict) -> tuple[str, int] | None:
        if 'fact_id' in op:
            row = self.old.execute('SELECT document_id, page, start FROM facts WHERE id = ?',
                                   (op['fact_id'],)).fetchone()
            if row is None:
                return None
            document_id, page, start = row
        elif op.get('document_id') and 'start' in op:
            document_id, page, start = op['document_id'], op['page'], op['start']
        else:
            return None
        fact = self.db.execute(
            'SELECT f.row_label, f.column_header, d.fiscal_year FROM facts f JOIN documents d '
            'ON d.id = f.document_id WHERE f.document_id = ? AND f.page = ? AND f.start = ?',
            (document_id, page, start)).fetchone()
        if fact is None or not norm(fact[0]):
            return None
        side = role(fact[1], fact[2])
        if side is None:
            return None
        return norm(fact[0]), fact[2] - (side == 'prior')


def terms(node: ast.AST) -> list[tuple[int, str]] | None:
    """A sum of names as ``[(sign, name)]``, or ``None`` for anything else."""
    if isinstance(node, ast.Name):
        return [(1, node.id)]
    if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'abs' and len(node.args) == 1:
        return terms(node.args[0])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = terms(node.operand)
        return [(-sign, name) for sign, name in inner] if inner else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = terms(node.left), terms(node.right)
        if left is None or right is None:
            return None
        flip = -1 if isinstance(node.op, ast.Sub) else 1
        return left + [(flip * sign, name) for sign, name in right]
    return None


def constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def shape(formula: str) -> dict | None:
    """The kind of comparison a verified formula makes, over operand names."""
    try:
        node = ast.parse(formula.replace(' ', ''), mode='eval').body
    except SyntaxError:
        return None
    percent = False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and constant(node.right) \
            and node.right.value == 100:
        node, percent = node.left, True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and constant(node.right):
        node = node.left  # a change of unit
    if percent and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) \
            and constant(node.right) and node.right.value == 1 and isinstance(node.left, ast.BinOp) \
            and isinstance(node.left.op, ast.Div):
        top, bottom = terms(node.left.left), terms(node.left.right)
        if top and bottom and len(top) == len(bottom) == 1:
            return {'kind': 'growth', 'now': top[0][1], 'then': bottom[0][1]}
        return None
    if percent and isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        top, bottom = terms(node.left), terms(node.right)
        if not top or not bottom:
            return None
        if len(top) == 2 and len(bottom) == 1 and top[0][0] == 1 and top[1] == (-1, bottom[0][1]):
            return {'kind': 'growth', 'now': top[0][1], 'then': bottom[0][1]}
        return {'kind': 'ratio', 'top': top, 'bottom': bottom}
    if percent:
        return None
    flat = terms(node)
    if flat and len(flat) == 1:
        return {'kind': 'amount', 'name': flat[0][1]}
    if flat and len(flat) == 2 and flat[0][0] == 1 and flat[1][0] == -1:
        return {'kind': 'change', 'now': flat[0][1], 'then': flat[1][1]}
    return None


def direction(text: str, start: int) -> str | None:
    before = text[max(0, start - 10):start]
    if DOWN.search(before):
        return 'down'
    return 'up' if UP.search(before) else None


def target(label: dict, review: str, figures: Figures, items: list[str]) -> dict:
    """The extraction answer for one verified number."""
    number_text, unit = label['text'], None
    for name in UNITS:
        if number_text.endswith(name):
            number_text, unit = number_text[:-len(name)], name
    occurrence = len(re.findall(r'(?<![\d.,])' + re.escape(number_text) + r'(?![\d])',
                                review[:label['start']])) + 1
    answer = {'number': number_text, 'unit': unit, 'occurrence': occurrence,
              'phrase': label.get('metric') or None, 'type': 'other', 'label': None,
              'numerator': None, 'denominator': None, 'direction': None, 'year': None,
              'scope': '母公司' if label.get('scope') == '母公司' else '合并'}
    evidence = label.get('evidence') or {}
    form = shape(evidence['formula']) if evidence.get('formula') else None
    if label['label'] not in ('correct', 'wrong') or form is None:
        return answer
    resolved = {op['name']: figures.operand(op) for op in evidence.get('operands', [])}
    if any(value is None for value in resolved.values()):
        return answer
    if form['kind'] == 'amount' and form['name'] in resolved:
        item, year = resolved[form['name']]
        if item in items:
            answer.update(type='amount', label=item, year=year)
    elif form['kind'] in ('growth', 'change') and {form['now'], form['then']} <= resolved.keys():
        (item, year), (other, before) = resolved[form['now']], resolved[form['then']]
        if item == other and before == year - 1 and item in items:
            answer.update(type=form['kind'], label=item, year=year,
                          direction=direction(review, label['start']))
    elif form['kind'] == 'ratio' and all(n in resolved for _, n in form['top'] + form['bottom']):
        years = {resolved[n][1] for _, n in form['top'] + form['bottom']}
        names = [resolved[n][0] for _, n in form['top'] + form['bottom']]
        if len(years) == 1 and all(name in items for name in names):
            answer.update(type='ratio', year=years.pop(),
                          numerator=[('-' if s < 0 else '') + resolved[n][0] for s, n in form['top']],
                          denominator=[('-' if s < 0 else '') + resolved[n][0]
                                       for s, n in form['bottom']])
    return answer


def build(library: Path, labelled: Path, reviews: Path, labels: Path) -> tuple[list[dict], Counter]:
    db = sqlite3.connect(f'file:{library}?mode=ro', uri=True)
    figures = Figures(db, sqlite3.connect(f'file:{labelled}?mode=ro', uri=True))
    by_review: dict[str, list[dict]] = {}
    for line in labels.read_text().splitlines():
        row = json.loads(line)
        by_review.setdefault(row['review_id'], []).append(row)
    samples, counts = [], Counter()
    for line in reviews.read_text().splitlines():
        review = json.loads(line)
        items = line_items(db, review['code'], review['fiscal_year'])
        answers = [target(row, review['text'], figures, items)
                   for row in sorted(by_review.get(review['id'], []), key=lambda r: r['start'])]
        for answer, row in zip(answers, sorted(by_review.get(review['id'], []), key=lambda r: r['start'])):
            counts[(row['label'], answer['type'])] += 1
        samples.append({'review_id': review['id'], 'messages': [
            {'role': 'user', 'content': PROMPT.format(
                company=review['company'], year=review['fiscal_year'],
                labels='\n'.join(items) or '（无）', review=review['text'])},
            {'role': 'assistant', 'content': json.dumps(answers, ensure_ascii=False, indent=1)}]})
    return samples, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--labelled-library', type=Path, required=True,
                        help='the library whose fact ids the labels cite')
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'refusing to overwrite {args.output}')
    samples, counts = build(args.library, args.labelled_library, args.reviews, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in samples))
    print(json.dumps({'samples': len(samples), 'numbers': sum(counts.values()),
                      'by_label_and_type': {f'{a}/{b}': n for (a, b), n in sorted(counts.items())}},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
