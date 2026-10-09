"""Compact repeated evidence without removing facts or actual action history."""

import json


def context_messages(state: dict, payload: dict, corpus, system: str) -> list[dict]:
    history = []
    for row in state.get('history', [])[-6:]:
        report = row.get('report') or {}
        history.append({'question': row['question'], 'calculations': [
            {'value': calc['value'], 'evidence_ids': sorted({
                fact['evidence_id'] for fact in calc.get('facts', {}).values()})}
            for calc in report.get('calculations', [])]})
    calculations = {key: {'value': value['value'], 'fact_ids': value['fact_ids']}
                    for key, value in state['calculations'].items()}
    body = {'request': payload, 'history': history,
        'original_question': state.get('original_question'),
        'read_evidence_ids': list(state['read']), 'calculations': calculations,
        'calls_remaining_including_this_call': state.get('calls_remaining')}
    instruction = system.replace('earlier #step results', 'earlier #0, #1, etc. results')
    instruction += (' Complete evidence and fact IDs remain in the first successful '
        'get_evidence observation for each source. Repeated reads refer to that observation. '
        'Do not repeat an unchanged search/read/calculation. '
        'Listed constants: const_0, const_1, const_2, const_3, const_4, const_100, const_m1. '
        'Intermediate results are #0, #1, etc., not #step0. '
        'When all requested results are computed, submit their calculation IDs in question order.')
    messages = [{'role': 'system', 'content': instruction},
        {'role': 'user', 'content': json.dumps(body, ensure_ascii=False, separators=(',', ':'))}]
    seen = set()
    for message in state.get('messages', [])[2:]:
        if message.get('role') == 'tool':
            observation = json.loads(message['content'])
            if message.get('name') == 'get_evidence' and 'document' in observation:
                eid = observation['document']['evidence']['evidence_id']
                if eid in seen:
                    observation = {'already_read': eid,
                                   'reference': 'first successful get_evidence observation'}
                seen.add(eid)
            messages.append({**message, 'content': json.dumps(
                observation, ensure_ascii=False, separators=(',', ':'))})
        else:
            messages.append(message)
    return messages
