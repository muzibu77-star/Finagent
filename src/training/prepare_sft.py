"""Convert frozen training programs to executable paired supervision."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer
import yaml

from src.evaluation.vendor.finqa_execution import program_tokenization, str_to_num
from src.model.financial import TOOLS, messages_for
from src.tools.benchmark_tools import CONSTANTS, OPS, execute_action, extract_numeric_facts
from src.evidence.compact import compact_messages


def convert_program(program: str, evidence: dict, expected,
                    reference_facts: set[str] | None = None,
                    table_reductions: bool = False) -> tuple[dict, dict]:
    """Reject ambiguous scalar bindings instead of inventing semantic positions."""
    facts = extract_numeric_facts(evidence)
    tokens = program_tokenization(program)
    if tokens[-1] != 'EOF' or (len(tokens) - 1) % 4:
        raise ValueError('invalid reference program')
    steps = []
    result_indices = {}
    for offset in range(0, len(tokens)-1, 4):
        operation, left, right, close = tokens[offset:offset+4]
        operation = operation.removesuffix('(')
        if table_reductions and operation in {'table_sum', 'table_average'}:
            rows = [i for i, row in enumerate(evidence['table']) if row and row[0] == left]
            if len(rows) != 1 or right != 'none' or close != ')':
                raise ValueError('ambiguous table reduction row')
            row_index = rows[0]
            operands = []
            for column in range(1, len(evidence['table'][row_index])):
                matches = [key for key, fact in facts.items()
                           if fact['location'].get('row') == row_index
                           and fact['location'].get('column') == column]
                if len(matches) != 1:
                    raise ValueError('table reduction requires one numeric value per cell')
                operands.append(matches[0])
            if not operands or (operation == 'table_average' and f'const_{len(operands)}' not in CONSTANTS):
                raise ValueError('table reduction exceeds supported constant contract')
            accumulator = operands[0]
            for operand in operands[1:]:
                steps.append({'operation': 'add', 'args': [accumulator, operand]})
                accumulator = f'#{len(steps) - 1}'
            if operation == 'table_average':
                steps.append({'operation': 'divide', 'args': [accumulator, f'const_{len(operands)}']})
            elif len(operands) == 1:
                steps.append({'operation': 'multiply', 'args': [accumulator, 'const_1']})
            result_indices[offset // 4] = len(steps) - 1
            continue
        if operation not in OPS or close != ')':
            raise ValueError('unsupported reference operation')
        args = []
        for operand in (left, right):
            if operand.startswith('#'):
                index = int(operand[1:])
                if index not in result_indices:
                    raise ValueError('invalid reference program result')
                args.append(f'#{result_indices[index]}')
                continue
            if operand in CONSTANTS:
                args.append(operand)
                continue
            value = str_to_num(operand)
            matches = [key for key, fact in facts.items()
                       if str_to_num(fact['value_token']) == value]
            if len(matches) > 1 and reference_facts is not None:
                matches = [key for key in matches if key in reference_facts]
            if len(matches) != 1:
                raise ValueError('missing or ambiguous scalar binding')
            args.append(matches[0])
        steps.append({'operation': operation, 'args': args})
        result_indices[offset // 4] = len(steps) - 1
    action = {'steps': steps}
    result = execute_action(action, facts)
    if result['value'] != expected:
        raise ValueError('reference execution disagrees with gold')
    return action, result


def annotated_facts(evidence: dict, gold_inds: dict) -> set[str]:
    """Resolve training references using released evidence annotations only.

    Text annotations must equal original text; table annotations identify an
    original row. Ambiguity inside the annotated evidence is still rejected.
    This is never called by inference or test-time evidence selection.
    """
    selected = set()
    for key, fact in extract_numeric_facts(evidence).items():
        loc = fact['location']
        if 'row' in loc:
            annotation = gold_inds.get(f"table_{loc['row']}")
            # Verify the released row label before using its original row index.
            row = evidence['table'][loc['row']]
            if annotation and str(row[0]).strip() in annotation:
                selected.add(key)
        elif evidence[loc['section']][loc['paragraph']] in gold_inds.values():
            selected.add(key)
    return selected


def encode_messages(tokenizer, messages: list[dict], tools: list | None) -> dict:
    """Mask all system/user/tool tokens; supervise complete assistant turns."""
    options = {'tools': tools, 'enable_thinking': False, 'tokenize': False}
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    spans = []
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, **options)
        ending = tokenizer.apply_chat_template(messages[:i+1], add_generation_prompt=False, **options)
        if not full.startswith(prefix) or not full.startswith(ending):
            raise ValueError('assistant template boundary mismatch')
        spans.append((len(prefix), len(ending)))
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    labels = [token if any(start >= a and end <= b and end > start for a, b in spans) else -100
              for token, (start, end) in zip(encoded['input_ids'], encoded['offset_mapping'])]
    if not any(label != -100 for label in labels):
        raise ValueError('empty assistant supervision')
    return {'input_ids': encoded['input_ids'], 'labels': labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--candidate-limit', type=int, default=500)
    parser.add_argument('--crop-evidence', action='store_true')
    parser.add_argument('--reference-bindings', action='store_true')
    parser.add_argument('--table-reductions', action='store_true')
    args = parser.parse_args()
    if args.candidate_limit <= 0 or args.max_tokens <= 0:
        parser.error('candidate limit and token budget must be positive')
    root = Path('data/staged/m0_frozen_v1')
    receipt = json.loads((root/'validation.json').read_text())
    if not receipt['passed']:
        raise ValueError('data not validated')
    for name, digest in receipt['sha256'].items():
        if hashlib.sha256((root/f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise ValueError('data hash mismatch')
    def rows(name):
        return list(map(json.loads, (root/f'{name}.jsonl').read_text().splitlines()))
    evidence = {r['evidence_id']: r for r in rows('evidence')}
    questions = {r['task_id']: r['question'] for r in rows('questions')}
    gold = {r['task_id']: r['gold'] for r in rows('gold')}
    candidates = [r for r in rows('manifest') if r['dataset']=='finqa'
                  and r['official_eligible'] and r['original_split']=='train']
    random.Random(31).shuffle(candidates)
    model = yaml.safe_load(Path('configs/m0_inference.yaml').read_text())['model']
    tokenizer = AutoTokenizer.from_pretrained(model['path'], local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    kept, rejected = [], []
    for row in candidates[:args.candidate_limit]:
        task_id = row['task_id']
        context = evidence[row['evidence_id']]
        try:
            references = annotated_facts(context, gold[task_id].get('gold_inds', {})) if args.reference_bindings else None
            action, result = convert_program(gold[task_id]['program'], context,
                                             gold[task_id]['exe_ans'], references, args.table_reductions)
            direct, facts = messages_for(context, questions[task_id], 'direct')
            actions, _ = messages_for(context, questions[task_id], 'calculator')
            if args.crop_evidence:
                direct = compact_messages(direct, context, facts, result['fact_ids'])
                actions = compact_messages(actions, context, facts, result['fact_ids'])
            answer = json.dumps({'answer': result['value']})
            direct.append({'role':'assistant', 'content':answer})
            actions.extend([
                {'role':'assistant', 'tool_calls':[{'type':'function', 'function':{
                    'name':'calculate', 'arguments':action}}]},
                {'role':'tool', 'name':'calculate', 'content':json.dumps({
                    'value':result['value'], 'fact_ids':result['fact_ids']})},
                {'role':'assistant', 'content':answer}])
            encoded = {'answer': encode_messages(tokenizer, direct, None),
                       'action': encode_messages(tokenizer, actions, TOOLS)}
            if max(len(x['input_ids']) for x in encoded.values()) > args.max_tokens:
                raise ValueError('complete trajectory exceeds token budget')
            kept.append({'task_id':task_id, 'source_report_id':row['source_report_id'],
                'evidence_id':row['evidence_id'], 'supervision':'provided gold evidence; no retrieval training',
                'action':action, 'observation':result, 'encoded':encoded})
        except ValueError as exc:
            rejected.append({'task_id':task_id,'reason':str(exc)})
    if not kept:
        raise ValueError('no complete supervision samples')
    for name, data in [('traces',kept), ('rejected',rejected)]:
        (args.output_dir/f'{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in data))
    # Paired arms use identical IDs/order; place longest action first for resource check.
    kept.sort(key=lambda r: len(r['encoded']['action']['input_ids']), reverse=True)
    for arm in ('answer','action'):
        samples=[{'task_id':r['task_id'], **r['encoded'][arm]} for r in kept]
        (args.output_dir/f'{arm}.json').write_text(json.dumps(samples))
    summary={'candidate_limit':args.candidate_limit,'eligible_train':len(candidates),'selected':len(kept),
        'evidence_view': 'reference-selected rows with original coordinates' if args.crop_evidence else 'complete',
        'binding': 'released evidence annotations resolve duplicates' if args.reference_bindings else 'unique in full evidence',
        'table_reductions': args.table_reductions,
        'rejected':dict(Counter(r['reason'] for r in rejected)), 'max_tokens':args.max_tokens,
        'seed':31, 'source_receipt':receipt, 'arms':{arm:{
            'tokens':sum(len(r['encoded'][arm]['input_ids']) for r in kept),
            'supervised_tokens':sum(sum(t!=-100 for t in r['encoded'][arm]['labels']) for r in kept),
            'max_tokens':max(len(r['encoded'][arm]['input_ids']) for r in kept),
            'sha256':hashlib.sha256((args.output_dir/f'{arm}.json').read_bytes()).hexdigest(),
        } for arm in ('answer','action')}}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='source_receipt'}))


if __name__ == '__main__':
    main()
