"""Evidence views retain original row/paragraph coordinates and fact identities."""

import copy
import json


def evidence_view(evidence: dict, facts: dict, fact_ids: list[str]) -> dict:
    """Keep complete cited rows, table headings and cited source paragraphs.

    Selection is supplied by the caller. Training may use reference executions;
    inference must use retrieved/selected facts and must never consult gold.
    Numeric binding is resolved against the original evidence before this view.
    """
    if not set(fact_ids) <= facts.keys():
        raise ValueError('unknown selected fact')
    rows = set()
    paragraphs = set()
    for key in fact_ids:
        fact = facts[key]
        if fact['evidence_id'] != evidence['evidence_id']:
            raise ValueError('selected fact belongs to another evidence')
        location = fact['location']
        if 'row' in location:
            rows.add(location['row'])
        else:
            paragraphs.add((location['section'], location['paragraph']))
    table = evidence.get('table', [])
    if table and rows:
        rows.add(0)
    view = {'evidence_id': evidence['evidence_id'], 'table_rows': [
        {'original_row': row, 'cells': table[row]} for row in sorted(rows)], 'paragraphs': []}
    for section, index in sorted(paragraphs):
        source = evidence[section][index]
        view['paragraphs'].append({'section': section, 'original_paragraph': index,
                                   'text': source.get('text', '') if isinstance(source, dict) else source})
    # Narrative before/after a table can carry units, periods and caveats.
    if rows:
        for section in ('pre_text', 'post_text'):
            if evidence.get(section):
                view[section] = evidence[section]
    return view


def compact_messages(messages: list[dict], evidence: dict, facts: dict,
                     fact_ids: list[str]) -> list[dict]:
    result = copy.deepcopy(messages)
    body = json.loads(result[1]['content'])
    body['evidence'] = evidence_view(evidence, facts, fact_ids)
    body['facts'] = {key: body['facts'][key] for key in fact_ids}
    result[1]['content'] = json.dumps(body, ensure_ascii=False, separators=(',', ':'))
    return result
