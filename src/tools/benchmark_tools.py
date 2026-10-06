"""Fact-reference-only arithmetic under bounded FinQA execution semantics."""

import json
import math
import re

from src.evaluation.vendor.finqa_execution import eval_program, str_to_num

NUMBER = re.compile(r'(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w.])')
OPS = {'add', 'subtract', 'multiply', 'divide', 'exp', 'greater'}
CONSTANTS = {'const_0', 'const_1', 'const_2', 'const_3', 'const_4', 'const_100', 'const_m1'}


def extract_numeric_facts(evidence: dict) -> dict[str, dict]:
    """Register exact scalar spans with table/text coordinates, without gold."""
    facts = {}
    entries = []
    for r, row in enumerate(evidence['table']):
        for c, cell in enumerate(row):
            entries.append((str(cell), {'row': r, 'column': c}))
    for field in ('pre_text', 'post_text'):
        for p, text in enumerate(evidence.get(field, [])):
            entries.append((text, {'section': field, 'paragraph': p}))
    for p, paragraph in enumerate(evidence.get('paragraphs', [])):
        entries.append((paragraph['text'], {'section': 'paragraphs', 'paragraph': p}))
    for text, location in entries:
        for match in NUMBER.finditer(text):
            key = f'f{len(facts)}'
            value = match.group()
            if ('row' in location and text.strip().startswith('(')
                    and text.strip().endswith(')') and len(NUMBER.findall(text)) == 1):
                value = '-' + value.lstrip('-')
            facts[key] = {'raw': match.group(), 'value_token': value,
                          'evidence_id': evidence['evidence_id'],
                          'location': {**location, 'start': match.start(), 'end': match.end()}}
    return facts


def execute_action(action: dict, facts: dict[str, dict]) -> dict:
    """Convert referenced scalars to bounded official evaluator tokens."""
    if set(action) != {'steps'} or not isinstance(action['steps'], list):
        raise ValueError('expected steps only')
    steps = action['steps']
    if not 1 <= len(steps) <= 8:
        raise ValueError('step budget exceeded')
    tokens = []
    cited = []
    results = []
    for i, step in enumerate(steps):
        if set(step) != {'operation', 'args'} or step['operation'] not in OPS:
            raise ValueError('invalid operation')
        if not isinstance(step['args'], list) or len(step['args']) != 2:
            raise ValueError('binary args required')
        args = []
        for ref in step['args']:
            if not isinstance(ref, str):
                raise ValueError('fact references must be strings')
            if ref in facts:
                args.append(facts[ref]['value_token'])
                cited.append(ref)
            elif ref in CONSTANTS:
                args.append(ref)
            elif re.fullmatch(r'#[0-7]', ref) and int(ref[1:]) < i:
                args.append(ref)
            else:
                raise ValueError('unknown fact or forward reference')
        values = [results[int(a[1:])] if a.startswith('#') else str_to_num(a) for a in args]
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v) > 1e15 for v in values):
            raise ValueError('operand exceeds numeric budget')
        if step['operation'] == 'exp' and abs(values[1]) > 10:
            raise ValueError('exponent budget exceeded')
        if step['operation'] == 'divide' and values[1] == 0:
            raise ValueError('zero denominator')
        tokens.extend([step['operation'] + '(', args[0], args[1], ')'])
        invalid, value = eval_program(tokens + ['EOF'], [])
        if invalid:
            raise ValueError('invalid benchmark program')
        results.append(value)
    if not cited:
        raise ValueError('calculation must cite evidence')
    return {'value': results[-1], 'program': tokens + ['EOF'],
            'fact_ids': sorted(set(cited)), 'facts': {key: facts[key] for key in set(cited)},
            'protocol': 'FinQA official execution; not business growth rules'}


def parse_action(text: str) -> dict:
    text = text.split('</think>')[-1].strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    action = json.loads(text)
    if not isinstance(action, dict):
        raise ValueError('action must be an object')
    return action
