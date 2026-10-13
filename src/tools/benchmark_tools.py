"""Fact-reference-only arithmetic under bounded FinQA execution semantics."""

import copy
import json
import math
import operator
import re

from src.evaluation.vendor.finqa_execution import eval_program, str_to_num

NUMBER = re.compile(r'(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w.])')
OPS = {'add', 'subtract', 'multiply', 'divide', 'exp', 'greater'}
CONSTANTS = {'const_0', 'const_1', 'const_2', 'const_3', 'const_4', 'const_100', 'const_m1'}
# Every constant used by the official FinQA/ConvFinQA train programs. The original
# whitelist above rejects about 5% of those reference programs.
OFFICIAL_CONSTANTS = CONSTANTS | {
    'const_5', 'const_6', 'const_7', 'const_8', 'const_9', 'const_10', 'const_1000',
    'const_10000', 'const_100000', 'const_1000000', 'const_1000000000'}


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


def execute_action(action: dict, facts: dict[str, dict],
                   constants: frozenset[str] | set[str] = CONSTANTS) -> dict:
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
            elif ref in constants:
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


def unrounded_program_value(program: list[str]) -> float | str:
    """Replay an already validated stored program for operand prechecks only.

    The official evaluator still determines the final submitted value. Display
    rounding must not turn a valid carried denominator into zero.
    """
    operations = {'add': operator.add, 'subtract': operator.sub,
                  'multiply': operator.mul, 'divide': operator.truediv,
                  'exp': operator.pow,
                  'greater': lambda a, b: 'yes' if a > b else 'no'}
    results = []
    for start in range(0, len(program) - 1, 4):
        operation, left, right, _ = program[start:start + 4]
        values = [results[int(token[1:])] if token.startswith('#')
                  else str_to_num(token) for token in (left, right)]
        results.append(operations[operation.removesuffix('(')](*values))
    return results[-1]


def execute_with_results(action: dict, facts: dict, calculations: dict,
                         constants: frozenset[str] | set[str] = CONSTANTS) -> dict:
    """Inline verified prior programs; round only the final FinQA result.

    #k stays local in model input. Each call still accepts at most eight new
    steps, with at most 64 total expanded steps including prior computations.
    The source closure contains only original facts, never synthetic evidence.
    """
    available = dict(facts)
    for cid, calculation in calculations.items():
        if not re.fullmatch(r'c\d+', cid) or cid in facts:
            raise ValueError('invalid calculation namespace')
        available[cid] = {'value_token': str(unrounded_program_value(calculation['program']))}
    result = execute_action(action, available, constants)
    cited, dependencies = {}, []
    for ref in result['fact_ids']:
        if ref in calculations:
            dependencies.append(ref)
            for fid, fact in calculations[ref]['facts'].items():
                if fid not in facts or facts[fid] != fact:
                    raise ValueError('prior calculation provenance outside current state')
                cited[fid] = facts[fid]
        else:
            cited[ref] = facts[ref]
    if not cited:
        raise ValueError('calculation must cite evidence')
    tokens, endpoints = [], {}
    for cid in sorted(dependencies):
        offset = len(tokens) // 4
        prior = calculations[cid]['program'][:-1]
        tokens.extend(f'#{int(token[1:]) + offset}' if token.startswith('#')
                      else token for token in prior)
        endpoints[cid] = len(tokens) // 4 - 1
    offset = len(tokens) // 4
    for step in action['steps']:
        operands = []
        for ref in step['args']:
            if ref in endpoints:
                operands.append(f'#{endpoints[ref]}')
            elif ref.startswith('#'):
                operands.append(f'#{offset + int(ref[1:])}')
            else:
                operands.append(facts[ref]['value_token'] if ref in facts else ref)
        tokens.extend([step['operation'] + '(', *operands, ')'])
    if len(tokens) // 4 > 64:
        raise ValueError('expanded calculation step budget exceeded')
    invalid, value = eval_program(tokens + ['EOF'], [])
    if invalid:
        raise ValueError('invalid expanded benchmark program')
    result.update(value=value, program=tokens + ['EOF'])
    result.update(facts=cited, fact_ids=sorted(cited),
                  action=copy.deepcopy(action), dependencies=sorted(dependencies))
    return result


def explain_calculation_error(action, facts: dict, calculations: dict | None,
                              constants: frozenset[str] | set[str] = CONSTANTS) -> str | None:
    """Name the first invalid part of a rejected calculate call and what is valid.

    Advisory text for the model only: it never changes whether a call executes.
    ``calculations`` is None when completed c-IDs are not legal operands.
    """
    steps = action.get('steps') if isinstance(action, dict) else None
    if not isinstance(steps, list) or not steps:
        return 'Pass one argument named steps: a non-empty list of {"operation", "args"} objects.'
    if len(steps) > 8:
        return (f'{len(steps)} steps were sent; one call takes at most 8.'
                + (' Finish part of it first and use its c-ID as an operand.'
                   if calculations is not None else ''))
    allowed = ', '.join(sorted(constants, key=lambda name: (len(name), name)))
    for i, step in enumerate(steps):
        where = f'Step #{i}'
        if not isinstance(step, dict) or set(step) != {'operation', 'args'}:
            return f'{where} must have exactly the keys "operation" and "args".'
        if step['operation'] not in OPS:
            return f"{where}: operation {step['operation']!r} does not exist. Use one of {', '.join(sorted(OPS))}."
        if not isinstance(step['args'], list) or len(step['args']) != 2:
            return (f'{where} needs exactly two args. Chain longer sums: '
                    'add(a, b) as one step, then add(#0, c) as the next.')
        for ref in step['args']:
            if not isinstance(ref, str):
                return (f'{where}: {ref!r} is a literal value. Operands are fact IDs such as "f12", '
                        f'earlier steps such as "#0", or constants ({allowed}).')
            if ref in facts or ref in constants or (calculations is not None and ref in calculations):
                continue
            if re.fullmatch(r'#\d+', ref):
                if int(ref[1:]) < i:
                    continue
                valid = ', '.join(f'#{k}' for k in range(i)) or 'none yet'
                return (f'{where} uses {ref}, which is not an earlier step of this call. '
                        f'Earlier steps available here: {valid}. Numbering restarts at #0 in every call.')
            if ref.startswith('const_'):
                return f'{where}: {ref} is not an available constant. Available constants: {allowed}.'
            if re.fullmatch(r'c\d+', ref):
                if calculations is None:
                    return f'{where}: {ref} is a calculation ID. Calculation IDs only go to submit_report.'
                done = ', '.join(sorted(calculations)) or 'none'
                return f'{where}: {ref} is not a completed calculation of this turn. Completed: {done}.'
            if re.fullmatch(r'f\d+', ref):
                return (f'{where}: {ref} is not a fact of any evidence read in this turn. '
                        f'{len(facts)} facts are loaded (f0 to f{len(facts) - 1}); read the evidence that contains it first.'
                        if facts else f'{where}: no evidence has been read in this turn, so {ref} does not exist yet.')
            return (f'{where}: {ref!r} is not a valid operand. Operands are fact IDs such as "f12", '
                    f'earlier steps such as "#0", or constants ({allowed}).')
    values = {ref: str_to_num(facts[ref]['value_token']) for step in steps for ref in step['args'] if ref in facts}
    for i, step in enumerate(steps):
        divisor = step['args'][1]
        if step['operation'] == 'divide' and (values.get(divisor) == 0 or divisor == 'const_0'):
            return f'Step #{i} divides by {divisor}, whose value is 0. Choose the non-zero denominator.'
    return None
