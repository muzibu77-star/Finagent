"""Lossless numeric ID annotations for explicit experimental contexts."""
import copy
from collections import defaultdict


def inline_observation(observation: dict, facts: dict, column_headers: bool = False) -> dict:
    """Insert IDs before exact original spans; retain all original characters.

    With ``column_headers``, a tag below the header row of a table that has more than one
    value column also names its column: ``[f3|december 31 2013]``.
    """
    result = copy.deepcopy(observation)
    evidence = result['document']['evidence']
    table = evidence.get('table') or [[]]
    # One value column needs no disambiguation, and such tables often have no header row.
    headers = ([str(cell).strip() for cell in table[0]]
               if column_headers and len(table[0]) >= 3 else [])
    grouped = defaultdict(list)
    for fid, fact in facts.items():
        if fact['evidence_id'] != evidence['evidence_id']:
            continue
        location = fact['location']
        position = (('table', location['row'], location['column']) if 'row' in location
                    else (location['section'], location['paragraph']))
        grouped[position].append((fid, fact))
    for position, entries in grouped.items():
        section = position[0]
        if section == 'table':
            text = str(evidence['table'][position[1]][position[2]])
        elif section == 'paragraphs':
            text = evidence[section][position[1]]['text']
        else:
            text = evidence[section][position[1]]
        for fid, fact in sorted(entries, key=lambda item: item[1]['location']['start'], reverse=True):
            loc = fact['location']
            if text[loc['start']:loc['end']] != fact['raw']:
                raise ValueError('fact span does not match original evidence')
            header = (headers[position[2]] if section == 'table' and position[1] and position[2]
                      and position[2] < len(headers) else '')
            tag = f'{fid}|{header}' if header else fid
            text = text[:loc['start']] + f'[{tag}]' + text[loc['start']:]
        if section == 'table':
            evidence['table'][position[1]][position[2]] = text
        elif section == 'paragraphs':
            evidence[section][position[1]]['text'] = text
        else:
            evidence[section][position[1]] = text
    result.pop('facts', None)
    result['fact_reference_format'] = '[fN] immediately precedes the original numeric span; use fN in calculate. All original text is preserved; coordinates remain in runtime state.'
    if column_headers:
        result['fact_reference_format'] = (
            '[fN] or [fN|column header] immediately precedes the original numeric span; the text '
            'after | repeats the header of that table column and is not part of the document; '
            'use fN in calculate. All original text is preserved; coordinates remain in runtime state.')
    return result
