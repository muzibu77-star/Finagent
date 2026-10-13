"""Baseline single-source workflow using the same durable task/report contracts."""

import copy
import json
import time

from src.harness.store import Cancelled, Store
from src.model.financial import TOOLS, decode_prediction, messages_for


class FixedRunner:
    def __init__(self, store: Store, corpus, model, seconds: float = 180,
                 *, unicode_context: bool = False):
        self.store, self.corpus, self.model = store, corpus, model
        self.seconds = seconds
        self.unicode_context = unicode_context

    def run(self, task_id: str) -> dict:
        if not self.store.claim(task_id):
            return self.store.get(task_id)
        task = self.store.get(task_id)
        revision, payload = task['revision'], task['payload']
        state = copy.deepcopy(task['state'])
        state.setdefault('deadline_at', time.time() + self.seconds)
        deadline = time.monotonic() + max(0, state['deadline_at'] - time.time())
        try:
            if state.setdefault('corpus_sha256', self.corpus.fingerprint) != self.corpus.fingerprint:
                raise ValueError('task source snapshot changed since submission')
            if (not payload.get('company') or not payload.get('period')
                    or payload['company'] not in self.corpus.companies):
                self.store.stop(task_id, revision, 'succeeded', 'awaiting_input',
                    {'question': '请指定资料库中的公司代码和报告期间。', 'origin': 'scope_guard'})
                return self.store.get(task_id)
            allowed = self.corpus.allowed(payload)
            question = payload['question']
            if state.get('history') or state.get('original_question'):
                question = json.dumps({'question': question, 'history': state.get('history', []),
                                       'original_question': state.get('original_question')},
                                      ensure_ascii=not self.unicode_context)
            hits = ([{'evidence_id': state['selected_evidence_id']}]
                    if state.get('selected_evidence_id') else self.corpus.search(question, allowed))
            self.store.save(task_id, revision, state, 'tool_observation',
                            {'tool': 'search_documents', 'observation': {'hits': hits}})
            if not hits:
                self.store.finish(task_id, revision, 'insufficient_evidence',
                    {'calculations': [], 'reason': 'No evidence matches the scoped query.',
                     'snapshot_id': payload['snapshot_id'], 'corpus_sha256': self.corpus.fingerprint,
                     'request_scope': {k: payload.get(k) for k in ('company', 'period', 'as_of')},
                     'workflow': 'fixed-v1'})
                return self.store.get(task_id)
            key = hits[0]['evidence_id']
            document = self.corpus.read(key, allowed)
            state['selected_evidence_id'] = key
            messages, facts = messages_for(document['evidence'], question, 'calculator')
            if self.unicode_context:
                messages[0]['content'] += (
                    ' For a single-value lookup, use multiply with the listed fact ID '
                    'and const_1. const is not a valid operation.')
                messages[-1]['content'] = json.dumps(
                    json.loads(messages[-1]['content']), ensure_ascii=False)
            state['read'] = {key: list(facts)}
            self.store.save(task_id, revision, state, 'tool_observation',
                            {'tool': 'get_evidence', 'arguments': {'evidence_id': key}})
            if 'prediction_record' not in state:
                if state.get('attempts', 0):
                    raise TimeoutError('generation was interrupted; consumed call is not replayed')
                if time.monotonic() >= deadline:
                    raise TimeoutError('task time budget exceeded')
                state['attempts'] = 1
                self.store.save(task_id, revision, state, 'generation_started',
                                {'attempt': 1, 'reserved_output_tokens': 512})
                record = self.model(messages, TOOLS,
                    lambda: bool(self.store.get(task_id)['cancel']), deadline)
                state['prediction_record'] = record
                state['actual_output_tokens'] = record.get('new_tokens', 0)
                self.store.save(task_id, revision, state, 'generation_finished', record,
                                allow_cancelled=True)
            if self.store.get(task_id)['cancel']:
                raise Cancelled('cancelled during generation')
            if time.monotonic() >= deadline:
                raise TimeoutError('task time budget exceeded')
            record = decode_prediction(copy.deepcopy(state['prediction_record']), facts, 'calculator')
            if 'error' in record:
                if 'out of memory' in record['error'].lower():
                    healthy = self.model.health_check()
                    self.store.save(task_id, revision, state, 'worker_health', {'healthy': healthy})
                raise RuntimeError(record['error'])
            calculation = record['calculation']
            if any(f['evidence_id'] not in allowed for f in calculation['facts'].values()):
                raise ValueError('citation outside current scope')
            self.store.save(task_id, revision, state, 'tool_observation',
                            {'tool': 'calculate', 'observation': calculation})
            self.store.finish(task_id, revision, 'answered', {'question': payload['question'],
                'calculations': [{'calculation_id': 'c0', **calculation}],
                'snapshot_id': payload['snapshot_id'], 'corpus_sha256': self.corpus.fingerprint,
                'request_scope': {k: payload.get(k) for k in ('company', 'period', 'as_of')},
                'workflow': 'fixed-v1', 'attempts': state['attempts'],
                'actual_output_tokens': state['actual_output_tokens'],
                'protocol': 'Single-source FinQA arithmetic; financial metadata not certified'})
        except Cancelled as exc:
            self.store.stop(task_id, revision, 'cancelled', 'blocked', {'error': str(exc)})
        except (ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as exc:
            self.store.stop(task_id, revision, 'failed', 'blocked',
                            {'error': f'{type(exc).__name__}: {exc}'})
        return self.store.get(task_id)
