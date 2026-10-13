"""Source-bound complex training tasks and actual deployment-context supervision."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import zipfile

from src.data.build_manifest import financial_source, visible_evidence
from src.training.autonomous_sft import read_rows, validate_lock


def selected_rows(path: Path, field: str, allowed: set[str]) -> list[dict]:
    """Decode values only after a row's source-bound ID is allowed."""
    pattern = re.compile(r'"' + re.escape(field) + r'"\s*:\s*("(?:[^"\\]|\\.)*")')
    rows = []
    with path.open() as stream:
        for line in stream:
            match = pattern.search(line)
            if match and json.loads(match.group(1)) in allowed:
                rows.append(json.loads(line))
    return rows


def training_conversations(archive_path: Path, sources: set[str]) -> list[dict]:
    """Decode conversation annotations only for allowed original train reports."""
    with zipfile.ZipFile(archive_path) as archive:
        text = archive.read('data/train.json').decode('utf-8')
    filename = re.compile(r'"filename"\s*:\s*("(?:[^"\\]|\\.)*")')
    rows, depth, start = [], 0, None
    quoted, escaped = False, False
    for index, character in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == '{':
            if depth == 0:
                start = index
            depth += 1
        elif character == '}':
            depth -= 1
            if depth == 0:
                raw = text[start:index + 1]
                match = filename.search(raw)
                if match and 'fin:' + financial_source(json.loads(match.group(1))) in sources:
                    rows.append(json.loads(raw))
    if depth or quoted:
        raise ValueError('incomplete conversation archive')
    return rows


def freeze_complex(output: Path, cross_count: int, dialogue_count: int) -> None:
    """Use only official eligible FinQA train reports, with separate labels."""
    root = Path('data/staged/m0_frozen_v1')
    receipt = json.loads((root / 'validation.json').read_text())
    if not receipt['passed']:
        raise ValueError('training source validation failed')
    for name, digest in receipt['sha256'].items():
        if hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise ValueError('training source changed')
    manifest = read_rows(root / 'manifest.jsonl')
    allowed = [r for r in manifest if r['dataset'] == 'finqa'
               and r['original_split'] == 'train' and r['official_eligible']]
    sources = {r['source_report_id'] for r in allowed}
    heldout = {r['source_report_id'] for r in manifest if r['official_eligible']
               and r['original_split'] != 'train'}
    dev = json.loads(Path('data/staged/agent_v2_dev/lock.json').read_text())
    acceptance = json.loads(Path('data/staged/agent_v3_test/lock.json').read_text())
    excluded = heldout | set(dev['source_reports']) | set(acceptance['source_reports'])
    excluded |= set(acceptance['excluded_sources']['exposed_v2_tasks'])
    if sources & excluded:
        raise ValueError('complex training source intersects a held-out boundary')
    parent_ids = {r['task_id'] for r in allowed}
    evidence_ids = {r['evidence_id'] for r in allowed}
    evidence = {r['evidence_id']: r for r in selected_rows(root / 'evidence.jsonl', 'evidence_id', evidence_ids)}
    questions = {r['task_id']: r['question'] for r in selected_rows(root / 'questions.jsonl', 'task_id', parent_ids)}
    gold = {r['task_id']: r['gold'] for r in selected_rows(root / 'gold.jsonl', 'task_id', parent_ids)}
    seed = 143
    random.Random(seed).shuffle(allowed)
    companies = defaultdict(dict)
    for row in allowed:
        companies[row['company']].setdefault(str(row['report_period']), row)
    pairs = [[periods[p] for p in sorted(periods)[:2]]
             for _, periods in sorted(companies.items()) if len(periods) >= 2]
    random.Random(seed).shuffle(pairs)
    archive_path = Path('data/raw/convfinqa/data.zip')
    raw_lock = json.loads(Path('configs/m0_data_lock.json').read_text())
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != raw_lock['sha256'][str(archive_path)]:
        raise ValueError('conversation training archive changed')
    conversations = training_conversations(archive_path, sources)
    random.Random(seed).shuffle(conversations)
    dialogues, seen = [], set()
    for row in conversations:
        source = 'fin:' + financial_source(row['filename'])
        annotation = row['annotation']
        if (source not in sources or source in seen
                or len(annotation['dialogue_break']) < 2
                or len(annotation['dialogue_break']) != len(annotation.get('exe_ans_list', []))):
            continue
        dialogues.append(row)
        seen.add(source)
    if not 1 <= cross_count <= len(pairs) or not 1 <= dialogue_count <= len(dialogues):
        raise ValueError(f'insufficient train pool: {len(pairs)} pairs, {len(dialogues)} dialogues')
    snapshot = 'complex-autonomous-train-v1'
    corpus = {r['evidence_id']: {'evidence': evidence[r['evidence_id']],
              'source_report_id': r['source_report_id'], 'company': r['company'],
              'period': str(r['report_period'])} for r in allowed}
    tasks, labels, provenance = [], {}, {}

    def add(category, company, period, turns, expected, ids, parents):
        key = f'{snapshot}-{len(tasks):03d}'
        tasks.append(dict(task_id=key, category=category, company=company,
                          period=period, turns=turns, snapshot_id=snapshot))
        labels[key] = dict(expected=expected, evidence_ids=ids)
        provenance[key] = parents

    for a, b in pairs[:cross_count]:
        prompt = (f'Answer both questions using a separate calculation for each. '
                  f"Report {a['report_period']}: {questions[a['task_id']]} "
                  f"Report {b['report_period']}: {questions[b['task_id']]}")
        add('cross_evidence', a['company'], [str(a['report_period']), str(b['report_period'])],
            [prompt], [[gold[a['task_id']]['exe_ans'], gold[b['task_id']]['exe_ans']]],
            [[a['evidence_id'], b['evidence_id']]],
            dict(kind='finqa_pair', parent_task_ids=[a['task_id'], b['task_id']],
                 source_report_ids=sorted([a['source_report_id'], b['source_report_id']])))
    for row in dialogues[:dialogue_count]:
        source = 'fin:' + financial_source(row['filename'])
        company, period = source.removeprefix('fin:').split('/')
        eid = 'conv:' + row['id']
        corpus[eid] = dict(evidence={'evidence_id': eid, **visible_evidence('finqa', row)},
                           source_report_id=source, company=company, period=period)
        turns = row['annotation']['dialogue_break']
        add('multi_turn', company, period, turns,
            [[v] for v in row['annotation']['exe_ans_list']], [[eid] for _ in turns],
            dict(kind='convfinqa_train', conversation_id=row['id'], source_report_ids=[source]))
    visible_fields = {'evidence_id', 'table', 'paragraphs', 'pre_text', 'post_text'}
    if any(set(doc['evidence']) - visible_fields for doc in corpus.values()):
        raise ValueError('non-evidence fields in model-visible training corpus')
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('tasks', tasks), ('gold', labels), ('corpus', corpus),
                        ('provenance', provenance)]:
        (output / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
    lock = dict(complex_training=True, split='train', seed=seed, snapshot_id=snapshot,
                builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                candidates=len(tasks), counts=dict(Counter(t['category'] for t in tasks)),
                turns=sum(len(t['turns']) for t in tasks), source_reports=sorted(sources),
                source_intersections=0, available_pairs=len(pairs),
                available_dialogue_reports=len(dialogues), source_receipt=receipt,
                conversation_archive_sha256=raw_lock['sha256'][str(archive_path)],
                sha256={n: hashlib.sha256((output / f'{n}.json').read_bytes()).hexdigest()
                        for n in ('tasks', 'gold', 'corpus', 'provenance')},
                protocol='Actual autonomous trajectories; no gold/target page in model-visible request.')
    (output / 'lock.json').write_text(json.dumps(lock, indent=2))


def validate_complex_parents(samples: list[dict], manifest: list[dict], root: Path,
                             lock_sha256: str) -> None:
    """Bind synthetic parents to frozen original train questions/conversations."""
    if hashlib.sha256((root / 'lock.json').read_bytes()).hexdigest() != lock_sha256:
        raise ValueError('complex supervision source lock changed')
    lock = validate_lock(root)
    if not lock.get('complex_training') or lock['split'] != 'train':
        raise ValueError('complex supervision requires a training lock')
    allowed = {r['task_id']: r['source_report_id'] for r in manifest
               if r['dataset'] == 'finqa' and r['original_split'] == 'train'
               and r['official_eligible']}
    sources = set(allowed.values())
    provenance = json.loads((root / 'provenance.json').read_text())
    archive_path = Path('data/raw/convfinqa/data.zip')
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != lock['conversation_archive_sha256']:
        raise ValueError('original conversation archive differs from training freeze')
    conv = {r['id']: 'fin:' + financial_source(r['filename'])
            for r in training_conversations(archive_path, sources)}
    ids = [s['task_id'] for s in samples]
    if not samples or len(ids) != len(set(ids)):
        raise ValueError('complex supervision must contain unique actions')
    for sample in samples:
        parent = sample.get('parent_task_id')
        meta = provenance.get(parent)
        prefix = f'{parent}:autonomous:'
        suffix = sample['task_id'].removeprefix(prefix)
        if (meta is None or not sample['task_id'].startswith(prefix)
                or not suffix.isdecimal() or str(int(suffix)) != suffix
                or sample.get('source_report_ids') != meta['source_report_ids']
                or not set(meta['source_report_ids']) <= sources):
            raise ValueError('invalid complex training parent or source')
        if meta['kind'] in ('finqa_pair', 'finqa_single'):
            parents = meta['parent_task_ids']
            if (len(parents) != (2 if meta['kind'] == 'finqa_pair' else 1)
                    or any(p not in allowed for p in parents)
                    or sorted(allowed[p] for p in parents) != meta['source_report_ids']):
                raise ValueError('invalid paired FinQA parents')
        elif meta['kind'] == 'convfinqa_train':
            if [conv.get(meta['conversation_id'])] != meta['source_report_ids']:
                raise ValueError('invalid original ConvFinQA training conversation')
        else:
            raise ValueError('unknown complex training parent kind')


def prepare_complex(args) -> None:
    """Retain whole correct trajectories; encode the exact captured model input."""
    from transformers import AutoTokenizer
    import yaml
    from src.agent.runner import tool_set
    from src.evaluation.m0_inference_check import parse_tool_calls
    from src.evaluation.validate_agent_run import audit
    from src.evidence.identity import evidence_identity
    from src.training.autonomous_sft import successful_trace
    from src.training.prepare_sft import encode_messages

    lock = validate_lock(args.tasks_root)
    settings = json.loads((args.run / 'settings.json').read_text())
    if settings['lock'] != lock or not lock.get('complex_training'):
        raise ValueError('complex collection and source lock differ')
    # The tool definitions are part of the rendered prompt and must match the run.
    TOOLS, SCHEMAS = tool_set(bool(settings.get('calculation_rationale')))
    validation = audit(args.run)
    if not validation['passed']:
        raise ValueError('complex collection audit failed')
    records = read_rows(args.run / 'records.jsonl')
    labels = json.loads((args.tasks_root / 'gold.json').read_text())
    provenance = json.loads((args.tasks_root / 'provenance.json').read_text())
    documents = json.loads((args.tasks_root / 'corpus.json').read_text())
    identities = {k: evidence_identity(v['evidence'], v['source_report_id'])
                  for k, v in documents.items()}
    model = yaml.safe_load(Path('configs/m0_inference.yaml').read_text())['model']
    tokenizer = AutoTokenizer.from_pretrained(model['path'], local_files_only=True)
    calls = defaultdict(list)
    for call in read_rows(args.run / 'calls.jsonl'):
        calls[call['task_id']].append(call)
    db = sqlite3.connect('file:' + str(args.run / 'tasks.sqlite') + '?mode=ro', uri=True)
    samples, excluded, lengths = [], [], []
    accepted = Counter()
    final_only = getattr(args, 'final_action_only', False)
    fraction = getattr(args, 'other_action_fraction', 1.0)
    order_seed = getattr(args, 'order_seed', None)
    if not 0 < fraction <= 1:
        raise ValueError('other-action fraction must be in (0, 1]')
    for record in records:
        key = record['task_id']
        expected = labels[key]
        passed = [successful_trace(row, {'expected': expected['expected'][i],
                                        'evidence_ids': expected['evidence_ids'][i]}, identities)
                  for i, row in enumerate(record['turns'])]
        if record['all_turns_passed'] != all(passed):
            raise ValueError('complex trace score differs from unchanged scoring')
        if not all(passed):
            reasons = Counter()
            for row in record['turns']:
                if row['execution'] != 'succeeded': reasons['execution failure'] += 1
                elif not row['numeric_correct']: reasons['incorrect numbers'] += 1
                elif not row['source_supported']: reasons['incorrect source'] += 1
            excluded.append(dict(task_id=key, reason='whole task not correct', failed_turns=dict(reasons)))
            continue
        accepted[record['category']] += 1
        events = [(kind, json.loads(body)) for kind, body in db.execute(
            'SELECT kind,body FROM events WHERE task_id=? ORDER BY seq', (key + ':turn0',))]
        finished = [body for kind, body in events if kind == 'generation_finished']
        trace = calls[key]
        if finished != [call['output'] for call in trace]:
            excluded.append(dict(task_id=key, reason='capture and committed generations differ'))
            continue
        if any(kind == 'tool_rejected' or (kind == 'tool_observation'
                and 'error' in body['observation']) for kind, body in events):
            excluded.append(dict(task_id=key, reason='trajectory contains rejected actions'))
            continue
        tools_used = {body['tool'] for kind, body in events if kind == 'tool_observation'}
        if not {'search_documents', 'get_evidence', 'calculate'} <= tools_used:
            raise ValueError('accepted complex trajectory lacks actual evidence/calculation actions')
        parent_samples, parent_lengths = [], []
        for step, call in enumerate(trace):
            parsed, errors = parse_tool_calls(call['output'].get('text', ''), SCHEMAS)
            if errors or len(parsed) != 1 or not call['output'].get('stopped_on_eos'):
                raise ValueError('accepted complex trace contains invalid generation')
            action = parsed[0]
            messages = call['messages'] + [{'role': 'assistant', 'tool_calls': [
                {'type': 'function', 'function': {'name': action['name'],
                                                 'arguments': action['arguments']}}]}]
            encoded = encode_messages(tokenizer, messages, TOOLS, final_only=final_only)
            components = Counter()
            for message in call['messages']:
                group = ('evidence_observation' if message.get('name') == 'get_evidence'
                         else 'state_and_history' if message['role'] == 'user'
                         else 'system' if message['role'] == 'system' else 'action_history')
                components[group] += len(tokenizer.encode(json.dumps(message, ensure_ascii=False),
                                                         add_special_tokens=False))
            length = dict(task_id=key, category=record['category'], step=step,
                          turn=call['turn'], tokens=len(encoded['input_ids']),
                          component_estimates=dict(components))
            lengths.append(length)
            parent_lengths.append(length)
            # Every calculation is kept; routine actions may be thinned deterministically.
            if (action['name'] != 'calculate' and fraction < 1
                    and random.Random(f'{key}:{step}').random() >= fraction):
                continue
            parent_samples.append(dict(task_id=f'{key}:autonomous:{step}', parent_task_id=key,
                source_report_ids=provenance[key]['source_report_ids'], tool=action['name'],
                turn=call['turn'], **encoded))
        oversized = [row for row in parent_lengths if row['tokens'] > args.max_tokens]
        if oversized:
            excluded.append(dict(task_id=key, reason='full trajectory exceeds token budget',
                                 largest_tokens=max(r['tokens'] for r in oversized)))
        else:
            samples.extend(parent_samples)
    db.close()
    samples.sort(key=lambda row: len(row['input_ids']), reverse=True)
    if order_seed is not None and samples:
        # The longest sample stays first to exercise the memory boundary at update one.
        rest = samples[1:]
        random.Random(order_seed).shuffle(rest)
        samples = samples[:1] + rest
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'action.json').write_text(json.dumps(samples))
    (args.output_dir / 'excluded.json').write_text(json.dumps(excluded, indent=2))
    (args.output_dir / 'lengths.json').write_text(json.dumps(lengths, indent=2))
    long_rows = [r for r in lengths if r['tokens'] > 3072]
    summary = dict(candidates=len(records), accepted_trajectories=sum(accepted.values()),
        accepted_by_category=dict(accepted), selected_parents=len({s['parent_task_id'] for s in samples}),
        selected_parent_ids=sorted({s['parent_task_id'] for s in samples}), samples=len(samples),
        tools=dict(Counter(s['tool'] for s in samples)), max_tokens=max((len(s['input_ids']) for s in samples), default=0),
        excluded=dict(Counter(r['reason'] for r in excluded)), max_sequence_length=args.max_tokens,
        over_3072=dict(actions=len(long_rows), parents=len({r['task_id'] for r in long_rows}),
                      dominant_component=dict(Counter(max(r['component_estimates'],
                            key=r['component_estimates'].get) for r in long_rows))),
        lock=lock, collection_settings=settings,
        sha256=hashlib.sha256((args.output_dir / 'action.json').read_bytes()).hexdigest(),
        final_action_only=final_only, other_action_fraction=fraction, order_seed=order_seed,
        supervision='Unmodified actual deployment messages; only assistant spans supervised; whole-task filtering.')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    if not samples:
        raise ValueError('no complete correct admissible complex trajectories; exclusions retained')


def collect_complex(args) -> None:
    """Use the deployment evaluator with exact action capture and unchanged scoring."""
    import subprocess
    import sys
    from src.evaluation.validate_agent_run import audit

    command = [sys.executable, '-m', 'src.evaluation.agent_run', '--tasks-root',
               str(args.tasks_root.resolve()), '--output-dir', str(args.output_dir.resolve()),
               '--context-management', '--capture-actions']
    if args.server_url:
        command += ['--server-url', args.server_url, '--served-model', args.served_model]
    if args.adapter:
        command += ['--adapter', str(args.adapter.resolve())]
    if args.resume:
        command += ['--resume']
    for option in ('calculation_references', 'inline_facts', 'arithmetic_guidance'):
        if getattr(args, option):
            command.append('--' + option.replace('_', '-'))
    subprocess.run(command, check=True)
    audit(args.output_dir)
    rows = read_rows(args.output_dir / 'records.jsonl')
    summary = dict(candidates=len(rows), accepted=sum(r['all_turns_passed'] for r in rows),
                   categories={c: dict(candidates=sum(r['category'] == c for r in rows),
                       accepted=sum(r['category'] == c and r['all_turns_passed'] for r in rows))
                       for c in {r['category'] for r in rows}}, command=command)
    path = args.output_dir / 'collection_summary.json'
    if not path.exists():
        path.write_text(json.dumps(summary, indent=2))
