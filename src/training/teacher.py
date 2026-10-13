"""Reference-program demonstrations executed through the real agent tools.

A scripted expert replays released FinQA/ConvFinQA train programs as
search/read/calculate/submit calls inside the unchanged ``Runner``. Every
observation is produced by the real tools; labels never enter model-visible
messages. The expert only works on pools frozen by ``freeze_teacher``.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from src.data.build_manifest import financial_source, visible_evidence
from src.evaluation.vendor.finqa_execution import program_tokenization
from src.evidence.identity import evidence_identity
from src.tools.benchmark_tools import CONSTANTS, OFFICIAL_CONSTANTS, extract_numeric_facts
from src.training.autonomous_sft import read_rows, validate_lock
from src.training.complex_autonomous import selected_rows, training_conversations
from src.training.prepare_sft import annotated_facts, convert_program

MAX_INPUT_TOKENS = 8192


def bind_program(program: str, evidence: dict, expected, gold_inds: list[dict],
                 constants: frozenset[str] | set[str] = CONSTANTS) -> list[dict]:
    """Return executable steps, or raise when a reference value is missing or ambiguous."""
    tokens = program_tokenization(program)
    if len(tokens) == 2:  # ConvFinQA lookup turn: a bare literal
        program = f'multiply({tokens[0]}, const_1)'
    references = set()
    for annotation in gold_inds:
        references |= annotated_facts(evidence, annotation)
    action, _ = convert_program(program, evidence, expected, references, table_reductions=True,
                                constants=constants)
    return action['steps']


def freeze_teacher(output: Path, holdout_root: Path, cross_count: int, dialogue_count: int,
                   single_count: int = 0, dialogues_per_report: int = 1, questions_per_pair: int = 1,
                   official_constants: bool = False, snapshot: str = 'teacher-train-v2') -> None:
    """Freeze train-only tasks whose every turn has an unambiguous executable reference.

    The defaults reproduce the v2 pool. Later pools may add single-report questions,
    several conversations per report and the official constant vocabulary.
    """
    constants = OFFICIAL_CONSTANTS if official_constants else CONSTANTS
    root = Path('data/staged/m0_frozen_v1')
    receipt = json.loads((root / 'validation.json').read_text())
    if not receipt['passed']:
        raise ValueError('training source validation failed')
    for name, digest in receipt['sha256'].items():
        if hashlib.sha256((root / f'{name}.jsonl').read_bytes()).hexdigest() != digest:
            raise ValueError('training source changed')
    holdout_path = holdout_root / 'lock.json'
    holdout_sha = hashlib.sha256(holdout_path.read_bytes()).hexdigest()
    if (holdout_root / 'lock.sha256').read_text().strip() != holdout_sha:
        raise ValueError('held-out acceptance lock changed')
    holdout = json.loads(holdout_path.read_text())
    blocked_companies = set(holdout['holdout_companies'])
    blocked_reports = set(holdout['source_reports']) | set(holdout['training_side_near_duplicates'])
    manifest = read_rows(root / 'manifest.jsonl')
    allowed = [r for r in manifest if r['dataset'] == 'finqa' and r['original_split'] == 'train'
               and r['official_eligible'] and r['company'] not in blocked_companies
               and r['source_report_id'] not in blocked_reports]
    sources = {r['source_report_id'] for r in allowed}
    other_splits = {r['source_report_id'] for r in manifest
                    if r['official_eligible'] and r['original_split'] != 'train'}
    if sources & (blocked_reports | other_splits):
        raise ValueError('teacher source intersects a held-out boundary')
    evidence = {r['evidence_id']: r for r in selected_rows(
        root / 'evidence.jsonl', 'evidence_id', {r['evidence_id'] for r in allowed})}
    parent_ids = {r['task_id'] for r in allowed}
    questions = {r['task_id']: r['question'] for r in selected_rows(root / 'questions.jsonl', 'task_id', parent_ids)}
    gold = {r['task_id']: r['gold'] for r in selected_rows(root / 'gold.jsonl', 'task_id', parent_ids)}
    seed = 157
    rejected = Counter()

    def bound(row):
        label = gold[row['task_id']]
        try:
            return bind_program(label['program'], evidence[row['evidence_id']], label['exe_ans'],
                                [label.get('gold_inds', {})], constants)
        except (ValueError, KeyError, IndexError):
            rejected['finqa_question'] += 1
            return None

    rows = sorted(allowed, key=lambda r: r['task_id'])
    random.Random(seed).shuffle(rows)
    by_company = defaultdict(lambda: defaultdict(list))
    for row in rows:
        steps = bound(row)
        if steps is not None:
            by_company[row['company']][str(row['report_period'])].append((row, steps))
    pairs = []
    for _, periods in sorted(by_company.items()):
        ordered = sorted(periods)
        # Disjoint adjacent report years, each question used once, at most two year pairs per issuer.
        for a, b in list(zip(ordered[0::2], ordered[1::2]))[:2]:
            for k in range(min(len(periods[a]), len(periods[b]), questions_per_pair)):
                pairs.append((periods[a][k], periods[b][k]))
    random.Random(seed).shuffle(pairs)
    archive_path = Path('data/raw/convfinqa/data.zip')
    raw_lock = json.loads(Path('configs/m0_data_lock.json').read_text())
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_sha != raw_lock['sha256'][str(archive_path)]:
        raise ValueError('conversation training archive changed')
    conversations = training_conversations(archive_path, sources)
    random.Random(seed).shuffle(conversations)
    dialogues, seen = [], Counter()
    for row in conversations:
        source = 'fin:' + financial_source(row['filename'])
        annotation = row['annotation']
        turns = annotation['dialogue_break']
        if (seen[source] >= dialogues_per_report or len(turns) < 2
                or len(turns) != len(annotation.get('exe_ans_list', []))):
            continue
        document = visible_evidence('finqa', row)
        annotations = [row[key]['gold_inds'] for key in ('qa', 'qa_0', 'qa_1') if key in row]
        try:
            steps = [bind_program(program, {'evidence_id': 'conv:' + row['id'], **document}, value,
                                  annotations, constants)
                     for program, value in zip(annotation['turn_program'], annotation['exe_ans_list'])]
        except (ValueError, KeyError, IndexError):
            rejected['conversation'] += 1
            continue
        dialogues.append((row, steps))
        seen[source] += 1
    paired = {row['task_id'] for pair in pairs[:cross_count] for row, _ in pair}
    singles = [item for periods in by_company.values() for items in periods.values()
               for item in items if item[0]['task_id'] not in paired]
    singles.sort(key=lambda item: item[0]['task_id'])
    random.Random(seed).shuffle(singles)
    if (not 1 <= cross_count <= len(pairs) or not 1 <= dialogue_count <= len(dialogues)
            or not 0 <= single_count <= len(singles)):
        raise ValueError(f'insufficient teacher pool: {len(pairs)} pairs, {len(dialogues)} dialogues, '
                         f'{len(singles)} single questions')
    corpus = {r['evidence_id']: {'evidence': evidence[r['evidence_id']],
              'source_report_id': r['source_report_id'], 'company': r['company'],
              'period': str(r['report_period'])} for r in allowed}
    tasks, labels, provenance, plans = [], {}, {}, {}

    def add(category, company, period, turns, expected, ids, parents, plan):
        key = f'{snapshot}-{len(tasks):03d}'
        tasks.append(dict(task_id=key, category=category, company=company,
                          period=period, turns=turns, snapshot_id=snapshot))
        labels[key] = dict(expected=expected, evidence_ids=ids)
        provenance[key] = parents
        plans[key] = plan

    for (a, steps_a), (b, steps_b) in pairs[:cross_count]:
        prompt = (f'Answer both questions using a separate calculation for each. '
                  f"Report {a['report_period']}: {questions[a['task_id']]} "
                  f"Report {b['report_period']}: {questions[b['task_id']]}")
        add('cross_evidence', a['company'], [str(a['report_period']), str(b['report_period'])],
            [prompt], [[gold[a['task_id']]['exe_ans'], gold[b['task_id']]['exe_ans']]],
            [[a['evidence_id'], b['evidence_id']]],
            dict(kind='finqa_pair', parent_task_ids=[a['task_id'], b['task_id']],
                 source_report_ids=sorted([a['source_report_id'], b['source_report_id']])),
            [[dict(query=f"{r['report_period']} {questions[r['task_id']]}",
                   evidence_id=r['evidence_id'], steps=s) for r, s in ((a, steps_a), (b, steps_b))]])
    for row, steps in singles[:single_count]:
        # Not named 'calculation': that category triggers the model-only fixed-flow comparator.
        add('single_report', row['company'], str(row['report_period']), [questions[row['task_id']]],
            [[gold[row['task_id']]['exe_ans']]], [[row['evidence_id']]],
            dict(kind='finqa_single', parent_task_ids=[row['task_id']],
                 source_report_ids=[row['source_report_id']]),
            [[dict(query=questions[row['task_id']], evidence_id=row['evidence_id'], steps=steps)]])
    for row, steps in dialogues[:dialogue_count]:
        source = 'fin:' + financial_source(row['filename'])
        company, period = source.removeprefix('fin:').split('/')
        # Evaluation sets name conversation pages 'conv:'. Training uses ordinary page
        # names so that the evidence ID never marks the page a conversation is about.
        eid = 'finqa:train:' + row['id'].split('_', 1)[1]
        document = visible_evidence('finqa', row)
        if eid in corpus and evidence_identity(corpus[eid]['evidence'], source) != evidence_identity(document, source):
            eid += '-b'
        corpus.setdefault(eid, dict(evidence={'evidence_id': eid, **document},
                                    source_report_id=source, company=company, period=period))
        turns = row['annotation']['dialogue_break']
        add('multi_turn', company, period, turns,
            [[v] for v in row['annotation']['exe_ans_list']], [[eid] for _ in turns],
            dict(kind='convfinqa_train', conversation_id=row['id'], source_report_ids=[source]),
            [[dict(query=question, evidence_id=eid, steps=s)] for question, s in zip(turns, steps)])
    visible_fields = {'evidence_id', 'table', 'paragraphs', 'pre_text', 'post_text'}
    if any(set(doc['evidence']) - visible_fields for doc in corpus.values()):
        raise ValueError('non-evidence fields in model-visible training corpus')
    output.mkdir(parents=True, exist_ok=False)
    files = [('tasks', tasks), ('gold', labels), ('corpus', corpus),
             ('provenance', provenance), ('teacher', plans)]
    for name, value in files:
        (output / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
    lock = dict(complex_training=True, teacher_training=True, split='train', seed=seed,
                snapshot_id=snapshot,
                builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                candidates=len(tasks), counts=dict(Counter(t['category'] for t in tasks)),
                turns=sum(len(t['turns']) for t in tasks), source_reports=sorted(sources),
                heldout_lock_sha256=holdout_sha, heldout_company_intersection=0,
                available_pairs=len(pairs), available_dialogues=len(dialogues),
                available_singles=len(singles), official_constants=official_constants,
                dialogues_per_report=dialogues_per_report, questions_per_pair=questions_per_pair,
                unbound_references=dict(rejected), source_receipt=receipt,
                conversation_archive_sha256=archive_sha,
                sha256={n: hashlib.sha256((output / f'{n}.json').read_bytes()).hexdigest()
                        for n, _ in files},
                protocol=('Scripted expert actions from released train programs; real tool '
                          'observations; teacher.json and gold.json are never model-visible.'))
    (output / 'lock.json').write_text(json.dumps(lock, indent=2))


def fact_rationale(evidence: dict, steps: list[dict], offset: int) -> str:
    """Name where each referenced fact stands: row label and column header, or its phrase."""
    facts = extract_numeric_facts(evidence)
    parts, seen = [], set()
    for step in steps:
        for arg in step['args']:
            if arg in seen or not re.fullmatch(r'f\d+', arg):
                continue
            seen.add(arg)
            fact = facts[arg]
            location = fact['location']
            if 'row' in location:
                table = evidence['table']
                row, column = location['row'], location['column']
                header = str(table[0][column]).strip() if row and column < len(table[0]) else ''
                label = ', '.join(part for part in (str(table[row][0]).strip() if column else '', header) if part)
            else:
                paragraph = (evidence[location['section']][location['paragraph']]
                             if location['section'] != 'paragraphs'
                             else evidence['paragraphs'][location['paragraph']]['text'])
                label = '"' + ' '.join(paragraph[:location['start']].split()[-8:]) + '"'
            parts.append(f"f{int(arg[1:]) + offset} = {fact['value_token']}: {label or 'table header'}")
    return '; '.join(parts)


def tool_call(name: str, arguments: dict) -> str:
    """Render the raw tool-call text accepted by ``parse_tool_calls``."""
    parameters = ''.join(
        f'<parameter={key}>\n{value if isinstance(value, str) else json.dumps(value)}\n</parameter>\n'
        for key, value in arguments.items())
    return f'<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>'


class TeacherDriver:
    """Drop-in model callable for ``agent_run``; refuses anything but a teacher pool."""

    def __init__(self, root: Path, rationale: bool = False):
        self.rationale = rationale
        lock = validate_lock(root)
        if not lock.get('teacher_training') or lock.get('split') != 'train':
            raise ValueError('the scripted expert only runs on a frozen teacher training pool')
        self.plans = json.loads((root / 'teacher.json').read_text())
        documents = json.loads((root / 'corpus.json').read_text())
        self.identity = {key: evidence_identity(doc['evidence'], doc['source_report_id'])
                         for key, doc in documents.items()}
        self.fact_count = {key: len(extract_numeric_facts(doc['evidence']))
                           for key, doc in documents.items()}
        self.evidence = {key: doc['evidence'] for key, doc in documents.items()}
        self.plan = None

    def __enter__(self):
        from transformers import AutoTokenizer
        import yaml
        model = yaml.safe_load(Path('configs/m0_inference.yaml').read_text())['model']
        self.tokenizer = AutoTokenizer.from_pretrained(model['path'], local_files_only=True)
        return self

    def __exit__(self, *exc):
        return False

    def begin_turn(self, task_id: str, turn: int) -> None:
        self.plan = self.plans[task_id][turn]

    def __call__(self, messages, tools, stop_requested, deadline) -> dict:
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, enable_thinking=False,
            tokenize=True, return_dict=False)
        if len(prompt) > MAX_INPUT_TOKENS:
            return {'prompt_tokens': len(prompt), 'error': 'input_over_budget'}
        name, arguments = self.next_action(messages)
        if name is None:
            return {'prompt_tokens': len(prompt), 'error': arguments}
        return {'text': tool_call(name, arguments), 'prompt_tokens': len(prompt),
                'new_tokens': 0, 'stopped_on_eos': True, 'backend': 'teacher'}

    def next_action(self, messages: list[dict]) -> tuple[str | None, dict | str]:
        history = json.loads(messages[1]['content'])['history']
        reads, known, done, searches, pending = [], [], 0, 0, None
        for row in history:
            for calculation in row['calculations']:
                known.extend(calculation['evidence_ids'])
        for message in messages[2:]:
            if message['role'] == 'assistant':
                pending = message['tool_calls'][0]['function']
            elif message['role'] == 'tool':
                observation = json.loads(message['content'])
                if 'error' in observation:
                    return None, 'teacher action was rejected by a tool'
                if pending['name'] == 'search_documents':
                    known.extend(hit['evidence_id'] for hit in observation['hits'])
                    searches += 1
                elif pending['name'] == 'get_evidence':
                    if pending['arguments']['evidence_id'] not in reads:
                        reads.append(pending['arguments']['evidence_id'])
                elif pending['name'] == 'calculate':
                    done, searches = done + 1, 0
        if done == len(self.plan):
            return 'submit_report', {'calculation_ids': [f'c{i}' for i in range(done)],
                                     'outcome': 'answered'}
        need = self.plan[done]
        target = self.identity[need['evidence_id']]
        read = next((key for key in reads if self.identity[key] == target), None)
        if read is not None:
            # Runtime fact IDs continue across sources in first-read order.
            offset = sum(self.fact_count[key] for key in reads[:reads.index(read)])
            steps = [{'operation': step['operation'], 'args': [
                f'f{int(arg[1:]) + offset}' if re.fullmatch(r'f\d+', arg) else arg
                for arg in step['args']]} for step in need['steps']]
            if self.rationale:
                return 'calculate', {'reasoning': fact_rationale(self.evidence[read], need['steps'], offset),
                                     'steps': steps}
            return 'calculate', {'steps': steps}
        choice = next((key for key in known if self.identity.get(key) == target), None)
        if choice is not None:
            return 'get_evidence', {'evidence_id': choice}
        if not searches:
            return 'search_documents', {'query': need['query']}
        return None, 'teacher: reference evidence was not retrieved'
