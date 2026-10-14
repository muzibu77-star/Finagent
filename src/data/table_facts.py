"""Bind mapped table numbers to their row label, column header, unit and title.

Every number keeps its PDF character span from ``table_map``.  The surrounding
metadata is read from the parsed layout and never guessed: a table without a
unit line, or a continuation whose first part was not parsed, yields facts with
the missing fields left empty and a recorded problem.  All facts are candidates
until their metadata has been checked.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import pypdfium2 as pdfium

from src.data.table_map import NUMBER, Cell, parsed_tables

SCALES = {'元': '1', '千元': '1000', '万元': '10000', '百万元': '1000000', '亿元': '100000000'}
# "单位:元 币种:人民币", "单位：人民币百万元", "(人民币百万元)"; digits are excluded
# so that a sentence such as "收入人民币100亿元" is not read as a unit line.
UNIT = re.compile(r'(?:单位|人民币|港币|美元)[^。\d]{0,12}?(百万元|千元|万元|亿元|元)')
# A statement covering the pages that follow, e.g. "(除特别注明外，金额单位为人民币元)"
# or "除特别注明外，均四舍五入取整到百万元".
DECLARED = re.compile(r'除[^。\n]{0,14}(?:注明|说明|标明|指明|列明)[^。\n\d]{0,24}?(百万元|千元|万元|亿元|元)')
OTHER_UNIT = re.compile(r'单位\s*[:：]\s*([^\s:：，,。)）(（\d]{1,6})')
IDENTIFIER = re.compile(r'\d{15,}')
DATE_COLUMN = re.compile(r'时间|日期|年月|到期日|起始日')
CURRENCY = re.compile(r'人民币|港币|港元|美元')
DATE = re.compile(r'\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?|\s*度|\s*\d{1,2}\s*[-—–~至]\s*\d{1,2}\s*月)')
NOTE_COLUMNS = ('附注', '注释', '附注号', '附注编号')
NO_META = {'title': None, 'unit': None, 'scale': None, 'currency': None,
           'statement_date': None, 'unit_source': None}


def block_text(block: dict) -> str:
    content = block.get('content', [])
    if not isinstance(content, list):  # an equation block holds one LaTeX string
        return ''
    return ''.join(part['content'] for part in content
                   if part.get('type') == 'text' and isinstance(part['content'], str)).strip()


def stated_unit(text: str, source: str) -> dict:
    """Unit fields read from one short piece of text, or nothing."""
    unit = UNIT.search(text)
    if not unit:
        # A counting unit such as "单位：股" or "单位：人" has no monetary scale.
        other = OTHER_UNIT.search(text)
        return ({'unit': other.group(1), 'scale': None, 'currency': None,
                 'unit_source': source} if other else {})
    currency = CURRENCY.search(text)
    return {'unit': unit.group(1), 'scale': SCALES[unit.group(1)],
            'currency': currency.group() if currency else None, 'unit_source': source}


def context(blocks: list[dict]) -> dict:
    """Title, unit and date stated between the nearest title and the table."""
    found = dict(NO_META)
    for block in blocks:
        text = block_text(block)
        if block['type'] == 'paragraph_title':
            found = {**NO_META, 'title': text}
        elif block['type'] == 'text' and len(text) <= 80:
            found.update(stated_unit(text, 'line before table'))
            if DATE.fullmatch(text):
                found['statement_date'] = re.sub(r'\s', '', text)
    return found


def declared_units(document) -> list[tuple[int, dict]]:
    """Document-wide unit statements as ``(pdf_page, unit fields)`` in page order."""
    found = []
    for index in range(len(document)):
        textpage = document[index].get_textpage()
        text = re.sub(r'[ \r\n]', '', textpage.get_text_bounded())
        textpage.close()
        match = DECLARED.search(text)
        if match:
            currency = CURRENCY.search(match.group())
            found.append((index + 1, {
                'unit': match.group(1), 'scale': SCALES[match.group(1)],
                'currency': currency.group() if currency else None,
                'unit_source': f'statement on page {index + 1}'}))
    return found


def grid(cells: list[Cell]) -> dict[tuple[int, int], str]:
    """Cell text at every grid position, repeating spanned cells."""
    return {(cell.row + r, cell.column + c): cell.text for cell in cells
            for r in range(cell.rowspan) for c in range(cell.colspan)}


def header_rows(cells: list[Cell]) -> list[int]:
    """Leading rows that label columns: several labels and no number."""
    rows: dict[int, list[Cell]] = {}
    for cell in cells:
        rows.setdefault(cell.row, []).append(cell)
    header = []
    for row in sorted(rows):
        texts = [cell.text for cell in rows[row] if cell.text]
        if len(texts) < 2 or any(NUMBER.fullmatch(text) for text in texts):
            break
        header.append(row)
    return header


def extract(document, middle: dict) -> dict:
    """Candidate facts for every mapped table number, with unresolved problems.

    A table's unit comes, in order, from the line before it, its own header or
    caption, the table it continues, and the latest document-wide statement on
    or before its page.  ``unit_source`` records which; percentage columns
    carry ``%`` instead of the monetary unit.
    """
    facts, problems = [], []
    declared = declared_units(document)
    previous = None
    for pdf_page, before, block, mapping in parsed_tables(document, middle):
        cells = mapping['cells']
        placed = grid(cells)
        header = header_rows(cells)
        columns = {column: ' '.join(dict.fromkeys(
            placed[(row, column)] for row in header if placed.get((row, column))))
            for column in {c for _, c in placed}}
        meta, section = context(before), None
        flagged = bool(block.get('continues_prev'))
        # A statement that runs onto the next page repeats its header there
        # without repeating the title or the unit line.
        repeated = (previous is not None and meta['title'] is None and meta['unit'] is None
                    and pdf_page == previous['pdf_page'] + 1 and bool(header)
                    and columns == previous['columns'])
        if flagged and previous is None:
            problems.append({'pdf_page': pdf_page, 'problem': 'continuation without its start'})
        elif flagged or repeated:
            meta, section = previous['meta'], previous['section']
            if not header:
                if len(columns) != len(previous['columns']):
                    problems.append({'pdf_page': pdf_page,
                                     'problem': 'continuation changes column count'})
                else:
                    columns = previous['columns']
        if meta['unit'] is None:
            own = ' '.join([*columns.values(), *(part['content'] for part in block['content']
                                                 if part['type'] == 'table_caption'
                                                 and isinstance(part['content'], str))])
            meta = {**meta, **stated_unit(own, 'table header')}
        if meta['unit'] is None:
            # A running header such as "(金额单位：人民币百万元)" or "2025年度 人民币元".
            for running in before:
                if running['type'] == 'header' and len(block_text(running)) <= 60:
                    meta = {**meta, **stated_unit(block_text(running), 'page header')}
        if meta['unit'] is None:
            earlier = [fields for page, fields in declared if page <= pdf_page]
            if earlier:
                meta = {**meta, **earlier[-1]}
            else:
                problems.append({'pdf_page': pdf_page, 'problem': 'no unit for table',
                                 'title': meta['title']})
        values = {(m['row'], m['column']): m for m in mapping['mapped']}
        for row in sorted({r for r, _ in placed}):
            if row in header:
                continue
            label = placed.get((row, 0), '')
            filled = [placed[(row, c)] for c in columns if c and placed.get((row, c))]
            # A heading such as "流动资产:"; an empty line item is not a heading.
            if label.endswith((':', '：')) and not filled:
                section = label
            for column in sorted(columns):
                value = values.get((row, column))
                # The "附注" column holds note numbers, a date column holds values
                # such as "2025.11" and a long ungrouped integer is a registration
                # code; none is a disclosed amount.
                if (value is None or columns[column] in NOTE_COLUMNS
                        or DATE_COLUMN.search(columns[column]) or IDENTIFIER.fullmatch(value['text'])):
                    continue
                title = meta['title'] or ''
                fact = {
                    'pdf_page': pdf_page, 'start': value['start'], 'end': value['end'],
                    'box': value['box'], 'text': value['text'], 'row_label': label,
                    'section': section, 'column_header': columns[column] or None,
                    'scope': '母公司' if '母公司' in title else '合并' if '合并' in title else None,
                    **meta}
                # Only the value or its column says it is a percentage; a row
                # label such as "以适用税率25%计算的所得税费用" names an amount.
                if '%' in value['text'] or re.search(r'[%％]|比例|百分', columns[column]):
                    fact.update(unit='%', scale=None, currency=None, unit_source='percentage')
                facts.append(fact)
        for item in mapping['unmapped_cells']:
            problems.append({'pdf_page': pdf_page, 'problem': 'cell not in PDF text', **item})
        for item in mapping['unmapped_numbers']:
            if item['where'] != 'embedded':
                problems.append({'pdf_page': pdf_page,
                                 'problem': f"PDF number {item['where']}", **item})
        previous = {'pdf_page': pdf_page, 'meta': meta, 'section': section, 'columns': columns}
    return {'facts': facts, 'problems': problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--middle', type=Path, required=True,
                        help='mineru-kit parse -f middle_json output for the same PDF')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'refusing to overwrite {args.output}')
    with pdfium.PdfDocument(args.pdf) as document:
        result = extract(document, json.loads(args.middle.read_text()))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({'facts': len(result['facts']),
                      'problems': dict(Counter(p['problem'] for p in result['problems']))},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
