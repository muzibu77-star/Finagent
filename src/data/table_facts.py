"""Bind mapped table numbers to their row label, column header, unit and title.

Every number keeps its PDF character span from ``table_map``.  The surrounding
metadata is read from the parsed layout and never guessed: a table without a
unit line, or a continuation whose first part was not parsed, yields facts with
the missing fields left empty and a recorded problem.  All facts are candidates
until their metadata has been checked.

A statement page on which the layout parser mapped nothing is read from the PDF
text layer instead, which holds it row by row; those facts say so in
``unit_source``.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import pypdfium2 as pdfium

from src.data.table_map import NUMBER, Cell, NumberSpan, page_numbers, parsed_tables

SCALES = {'元': '1', '千元': '1000', '万元': '10000', '百万元': '1000000', '亿元': '100000000'}
# "单位:元 币种:人民币", "单位：人民币百万元", "(人民币百万元)"; digits are excluded
# so that a sentence such as "收入人民币100亿元" is not read as a unit line.
UNIT = re.compile(r'(?:单位|人民币|港币|美元)[^。\d]{0,12}?(百万元|千元|万元|亿元|元)')
# A statement covering the pages that follow, e.g. "(除特别注明外，金额单位为人民币元)"
# or "除特别注明外，均四舍五入取整到百万元".
DECLARED = re.compile(r'除[^。\n]{0,14}(?:注明|说明|标明|指明|列明)[^。\n\d]{0,24}?(百万元|千元|万元|亿元|元)')
# A unit declared once for the statements that follow, e.g. "财务附注中报表的单位为：千元".
# The sentence scopes itself to the statements, so it is used only for a table whose title
# names one; unlike DECLARED it never binds a note table.
STATEMENT_UNIT = re.compile(r'报表(?:中)?(?:的)?单位(?:均)?为[^。\n\d]{0,8}?(百万元|千元|万元|亿元|元)')
# A note may name the unit of the statements themselves after the statements, e.g.
# "编制本财务报表所采用的货币均为人民币，除有特别说明外，均以人民币千元为单位表示"
# (比亚迪 2023, page 153, for the statements from page 138).  The page break and the
# full stop bound the sentence, so a note about something else keeps its own unit.
NOTES_UNIT = re.compile(r'报表[^。\n]{0,60}?(?:单位(?:均)?为|以)[^。\n\d]{0,16}?(百万元|千元|万元|亿元|元)')
# "单位：股" and "单位：人" are counting units without a monetary scale.  "编制单位：" names
# the entity that prepared the statement, not a unit, and must not be read as one.
OTHER_UNIT = re.compile(r'(?<!编制)(?<!填报)单位\s*[:：]\s*([^\s:：，,。)）(（\d]{1,6})')
IDENTIFIER = re.compile(r'\d{15,}')
DATE_COLUMN = re.compile(r'时间|日期|年月|到期日|起始日')
CURRENCY = re.compile(r'人民币|港币|港元|美元')
DATE = re.compile(r'\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?|\s*度|\s*\d{1,2}\s*[-—–~至]\s*\d{1,2}\s*月)')
STATEMENT_TITLE = re.compile(r'(?:[一二三四五六七八九十\d]+[、.．])?(?:合并(?:及(?:母)?公司)?|母公司|公司)?'
                             r'(?:资产负债表|利润表|现金流量表|所有者权益变动表|股东权益变动表)'
                             r'(?:[（(]续[)）])?')
# A summary row that states its own unit: "营业收入（元）", "总资产（万元）".
LABEL_UNIT = re.compile(r'[（(]\s*(百万元|千元|万元|亿元|元)\s*[)）]$')
# The statements the checker reads, written as a page's own title line.  The
# equity-change statement is left out: its rows hold more than two value columns.
STATEMENT_LINE = re.compile(r'^(?:[（(]?[一二三四五六七八九十\d]+[、.．)）]?)?'
                            r'(?:合并(?:及(?:母)?公司)?|母公司|公司)?'
                            r'(?:资产负债表|利润表|现金流量表)(?:[（(]续[)）])?$')
# The column heading row of a statement, as "项 目 附注 2024 年 12 月 31 日 …".
HEADER_ROW = re.compile(r'^项\s*目')
# A note reference written after a row label, as "五、1", "五、54" or "五、70（1）".
NOTE_REFERENCE = re.compile(r'(?:附注)?[一二三四五六七八九十]+[、.．]\d+'
                            r'(?:[、.．]\d+)*(?:[（(]\d+[)）])?$')
# The opening balance of a year is the year-end it follows, and the checker reads
# the year a column states to tell the current period from the comparative, so an
# opening column is written as that year-end.
YEAR_END = re.compile(r'(\d{4})\s*年\s*12\s*月\s*31\s*日')
OPENING = re.compile(r'(\d{4})\s*年\s*1\s*月\s*1\s*日')
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


def label_unit(label: str) -> dict:
    """The unit a summary row states for itself, or nothing.

    The main-accounting-data table writes amounts in rows labelled "营业收入（元）";
    the row states its own unit, so such a table needs no unit line above it.
    """
    found = LABEL_UNIT.search(label)
    return ({'unit': found.group(1), 'scale': SCALES[found.group(1)],
             'unit_source': 'row label'} if found else {})


def context(blocks: list[dict]) -> dict:
    """Title, unit and date stated between the nearest title and the table."""
    found = dict(NO_META)
    for block in blocks:
        text = block_text(block)
        compact = re.sub(r'\s', '', text)
        if block['type'] in ('text', 'doc_title', 'header') and STATEMENT_TITLE.search(compact) \
                and len(compact) <= 60:
            # "合并资产负债表" set as text, as a document title or in the running header;
            # such a line may carry the unit too ("…合并资产负债表 (续) (金额单位:人民币元)").
            found = {**NO_META, 'title': STATEMENT_TITLE.search(compact).group(),
                     **stated_unit(text, 'title')}
        elif block['type'] == 'paragraph_title' and stated_unit(text, '') and len(text) <= 40:
            # "(除另有注明外，金额单位均为人民币元)" set as a heading is a unit line.
            found.update(stated_unit(text, 'line before table'))
        elif block['type'] == 'paragraph_title':
            # A title may carry the unit too: "合并资产负债表……(金额单位:人民币百万元)".
            found = {**NO_META, 'title': text, **stated_unit(text, 'title')}
        elif block['type'] == 'text' and len(text) <= 80:
            found.update(stated_unit(text, 'line before table'))
            if DATE.fullmatch(text):
                found['statement_date'] = re.sub(r'\s', '', text)
    return found


def declared_units(document) -> list[tuple[int, dict, bool]]:
    """Document-wide unit statements as ``(pdf_page, unit fields, statements only)``.

    In page order; the flag is false for a statement that covers every table that
    follows and true for one that names the financial statements only.
    """
    found = []
    for index in range(len(document)):
        textpage = document[index].get_textpage()
        text = re.sub(r'[ \r\n]', '', textpage.get_text_bounded())
        textpage.close()
        for pattern, statements_only in ((DECLARED, False), (STATEMENT_UNIT, True)):
            match = pattern.search(text)
            if match:
                currency = CURRENCY.search(match.group())
                found.append((index + 1, {
                    'unit': match.group(1), 'scale': SCALES[match.group(1)],
                    'currency': currency.group() if currency else None,
                    'unit_source': f'statement on page {index + 1}'}, statements_only))
    return found


def notes_declared_unit(document) -> tuple[dict | None, bool]:
    """The unit a note declares for the statements, and whether two of them clash.

    The statements may stand before the note that names their unit, so this is read
    for a statement table that states no unit of its own, whatever its page.  Two
    different units leave the statements alone rather than pick one.
    """
    found = []
    for index in range(len(document)):
        textpage = document[index].get_textpage()
        text = re.sub(r'[ \t\r]', '', textpage.get_text_bounded())
        textpage.close()
        for match in NOTES_UNIT.finditer(text):
            currency = CURRENCY.search(match.group())
            found.append({'unit': match.group(1), 'scale': SCALES[match.group(1)],
                          'currency': currency.group() if currency else None,
                          'unit_source': 'statement unit declared in notes'})
    if len({fields['unit'] for fields in found}) == 1:
        return found[0], False
    return None, bool(found)


def statement_table(title: str | None) -> bool:
    """A table whose rows state the reported figures.

    The checker reads the consolidated statements and, in a report that binds no
    statement, the main-accounting-data table; those are the tables a unit named
    for the statements, or in a row label, belongs to.
    """
    title = title or ''
    return bool(STATEMENT_TITLE.search(title)) or '主要会计数据' in title


def declared_unit(declared: list[tuple[int, dict, bool]], pdf_page: int,
                  title: str | None) -> dict | None:
    """The latest document-wide unit statement in force for a table on that page."""
    earlier = [fields for page, fields, statements_only in declared
               if page <= pdf_page and (not statements_only or statement_table(title))]
    return earlier[-1] if earlier else None


def note_number(column: str, text: str) -> bool:
    """A note reference in a column whose merged header mentions "附注"."""
    return '附注' in column and re.fullmatch(r'\d{1,3}', text) is not None


def scope(title: str, column: str) -> str | None:
    """Consolidated or parent figures; a "合并及公司" table says so per column."""
    if re.search(r'合并|本集团', column):
        return '合并'
    if re.search(r'本公司|本行|母公司', column):
        return '母公司'
    if re.search(r'(?<!合并)(?:母公司|本公司|公司)$|(?:母公司|本公司)', column):
        return '母公司'
    if '母公司' in title:
        return '母公司'
    return '合并' if '合并' in title else None


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


def column_headers(placed: dict[tuple[int, int], str], header: list[int]) -> dict[int, str]:
    """The heading text of every column, joined over the header rows."""
    return {column: ' '.join(dict.fromkeys(
        placed[(row, column)] for row in header if placed.get((row, column))))
        for column in {c for _, c in placed}}


def rows_to_facts(placed: dict[tuple[int, int], str], header: list[int],
                  columns: dict[int, str], values: dict, meta: dict,
                  section: str | None, page: int) -> tuple[list[dict], str | None]:
    """The facts of one table, from its placed cells and the numbers inside them.

    Both readers of a page hand over the same grid: the cells of a table the
    layout parser mapped, and the cells and numbers the text layer holds for a
    table whose column the parser merged into a single cell.
    """
    facts = []
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
                    or DATE_COLUMN.search(columns[column]) or IDENTIFIER.fullmatch(value['text'])
                    or note_number(columns[column], value['text'])):
                continue
            title = meta['title'] or ''
            fact = {
                'pdf_page': page, 'start': value['start'], 'end': value['end'],
                'box': value['box'], 'text': value['text'], 'row_label': label,
                'section': section, 'column_header': columns[column] or None,
                'scope': scope(title, columns[column]),
                **meta}
            # Only the value or its column says it is a percentage; a row
            # label such as "以适用税率25%计算的所得税费用" names an amount.
            if '%' in value['text'] or re.search(r'[%％]|比例|百分', columns[column]):
                fact.update(unit='%', scale=None, currency=None, unit_source='percentage')
            elif '每股' in label:
                # Earnings per share are stated in yuan per share whatever the
                # table's amount unit is.
                fact.update(unit='元/股', scale=None, currency='人民币', unit_source='per share')
            else:
                # A summary row states its own unit: "营业收入（元）".
                if statement_table(meta['title']):
                    fact.update(label_unit(label))
            facts.append(fact)
    return facts, section


def open_label(label: str) -> bool:
    """A label the PDF broke inside a bracket is continued on the next line."""
    return (label.count('（') > label.count('）') or label.count('(') > label.count(')')
            or label.count('“') > label.count('”'))


def statement_title(text: str) -> tuple[str, str | None] | None:
    """The statement a page's text layer opens with, and its own date line.

    A title line is taken only when the statement's date follows it; a heading
    that merely names a statement carries no date line and is left alone.
    """
    lines = [re.sub(r'\s', '', line) for line in text.splitlines()]
    for index, compact in enumerate(lines):
        if not compact or not STATEMENT_LINE.match(compact):
            continue
        following = [line for line in lines[index + 1:] if line]
        date = following[0] if following and DATE.fullmatch(following[0]) else None
        return (compact, date) if date else None
    return None


def header_line(text: str) -> tuple[int, list[tuple[int, int]]] | None:
    """The column heading row of the page text and the spans of its two periods."""
    base = 0
    for index, line in enumerate(text.splitlines(keepends=True)):
        if HEADER_ROW.match(line.strip()) and '附注' in line:
            periods = list(DATE.finditer(line))
            if len(periods) >= 2:
                return index, [(base + period.start(), base + period.end())
                               for period in periods[-2:]]
        base += len(line)
    return None


def column_headings(text: str, periods: list[tuple[int, int]]) -> list[str]:
    """The two column headings, with an opening balance written as the year-end it equals."""
    headings = [re.sub(r'\s', '', text[start:end]) for start, end in periods]
    ending, opening = YEAR_END.fullmatch(headings[0]), OPENING.fullmatch(headings[1])
    if ending and opening and ending.group(1) == opening.group(1):
        headings[1] = f'{int(opening.group(1)) - 1}年12月31日'
    return headings


def column_windows(textpage, periods: list[tuple[int, int]]) -> list[tuple[float, float] | None]:
    """The x range each column heading occupies on the page."""
    windows = []
    for start, end in periods:
        drawn = []
        for index in range(start, end):
            try:  # PDFium has no box for a generated character
                left, _, right, _ = textpage.get_charbox(index)
            except Exception:
                continue
            if right > left:
                drawn.append((left, right))
        windows.append((min(left for left, _ in drawn), max(right for _, right in drawn))
                       if drawn else None)
    return windows


def text_layer_cells(text: str, header: int, headings: list[str],
                     windows: list[tuple[float, float] | None],
                     spans: dict[int, NumberSpan]) -> tuple[list[Cell], list[dict]]:
    """Cells and number spans of a statement the layout parser left unread.

    A row's label is the text before its numbers, with a note reference dropped,
    and each number is placed under the column heading it lies below.  A heading
    the PDF wrapped is continued on the next line, two rows printed on one line
    are read apart, and a row whose label or column cannot be read is skipped
    rather than guessed.
    """
    # The two value columns are read in order and their numbers are right-aligned
    # inside them, so a number whose centre lies left of the second column's
    # heading is a figure of the first column.
    boundary = windows[1][0] if all(windows) else None
    lines = text.splitlines(keepends=True)
    base = sum(len(line) for line in lines[:header + 1])
    rows: list[tuple[str, str, list[NumberSpan]]] = []
    label, origin = '', None
    for index, line in enumerate(lines[header + 1:], header + 2):
        matches = list(NUMBER.finditer(line))
        drawn = [spans.get(base + match.start()) for match in matches]
        base += len(line)
        if any(value is None for value in drawn):
            continue  # a number whose position PDFium does not report is not placed
        pieces, cursor = [], 0
        for match, value in zip(matches, drawn):
            pieces.append(('text', line[cursor:match.start()]))
            pieces.append(('number', value))
            cursor = match.end()
        pieces.append(('text', line[cursor:]))
        numbers = []
        for kind, value in pieces:
            if kind == 'number':
                numbers.append(value)
                continue
            written = NOTE_REFERENCE.sub('', value.strip()).strip()
            if not written:
                continue
            if written.endswith(('：', ':')):
                # A section heading ends the row before it and labels what follows.
                rows.append(('heading', written, []))
                numbers, label, origin = [], '', None
            elif not numbers and origin is not None and origin != index and open_label(label):
                label += written
            else:
                if numbers:
                    rows.append(('row', label, numbers))
                numbers, label, origin = [], written, index
        if numbers:
            rows.append(('row', label, numbers))

    cells = [Cell(0, 0, 1, 1, '项 目'), Cell(0, 1, 1, 1, '附注'),
             Cell(0, 2, 1, 1, headings[0]), Cell(0, 3, 1, 1, headings[1])]
    mapped: list[dict] = []
    for number, (kind, written, numbers) in enumerate(rows, 1):
        if kind == 'heading':
            cells.append(Cell(number, 0, 1, 1, written))
            continue
        if boundary is None:
            # One number without column positions: which of the two periods it
            # states cannot be told, and the order of a single number says nothing.
            if len(numbers) != 2:
                continue
            columns = [2, 3]
        else:
            columns = [2 if (value.box[0] + value.box[2]) / 2 < boundary else 3
                       for value in numbers]
        if not written or len(set(columns)) != len(numbers):
            continue
        cells.append(Cell(number, 0, 1, 1, written))
        for column, value in zip(columns, numbers):
            cells.append(Cell(number, column, 1, 1, value.text))
            mapped.append({'row': number, 'column': column, 'text': value.text,
                           'start': value.start, 'end': value.end, 'box': value.box})
    return cells, mapped


def extract(document, middle: dict) -> dict:
    """Candidate facts for every mapped table number, with unresolved problems.

    A table's unit comes, in order, from the line before it, its own header or
    caption, the table it continues, and the latest document-wide statement on
    or before its page; a statement table that still states none takes the unit
    a note declares for the statements.  ``unit_source`` records which;
    percentage columns carry ``%`` instead of the monetary unit.

    A statement page on which the layout parser mapped no number is read from
    the same page's text layer instead, row by row, and those facts record it
    in ``unit_source``.
    """
    facts, problems = [], []
    declared = declared_units(document)
    notes, notes_clash = notes_declared_unit(document)
    previous = None
    for pdf_page, before, block, mapping in parsed_tables(document, middle):
        cells = mapping['cells']
        placed = grid(cells)
        header = header_rows(cells)
        columns = column_headers(placed, header)
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
            stated = declared_unit(declared, pdf_page, meta['title'])
            if stated:
                meta = {**meta, **stated}
            elif statement_table(meta['title']) and notes:
                # A note may state the unit of the statements after the statements.
                meta = {**meta, **notes}
            elif statement_table(meta['title']) and notes_clash:
                problems.append({'pdf_page': pdf_page,
                                 'problem': 'conflicting statement units declared in notes'})
            else:
                problems.append({'pdf_page': pdf_page, 'problem': 'no unit for table',
                                 'title': meta['title']})
        values = {(m['row'], m['column']): m for m in mapping['mapped']}
        table, section = rows_to_facts(placed, header, columns, values, meta, section, pdf_page)
        facts += table
        for item in mapping['unmapped_cells']:
            problems.append({'pdf_page': pdf_page, 'problem': 'cell not in PDF text', **item})
        for item in mapping['unmapped_numbers']:
            if item['where'] != 'embedded':
                problems.append({'pdf_page': pdf_page,
                                 'problem': f"PDF number {item['where']}", **item})
        previous = {'pdf_page': pdf_page, 'meta': meta, 'section': section, 'columns': columns}
    # A statement page the layout parser mapped nothing on: it merged a whole column
    # into a single cell, so no number reached a cell.  The text layer of the same
    # page still holds the statement row by row, so it is read from there instead.
    read = {fact['pdf_page'] for fact in facts}
    carried = None
    for pdf_page in range(1, len(document) + 1):
        if pdf_page in read:
            carried = None
            continue
        textpage = document[pdf_page - 1].get_textpage()
        text = textpage.get_text_bounded()
        found = statement_title(text) or carried
        header = header_line(text)
        if found and header:
            line, periods = header
            cells, mapped_numbers = text_layer_cells(
                text, line, column_headings(text, periods),
                column_windows(textpage, periods),
                {span.start: span for span in page_numbers(textpage, text)})
            if mapped_numbers:
                meta = dict(NO_META, title=found[0], statement_date=found[1])
                for before in text.splitlines()[:line]:
                    stated = stated_unit(before, 'text layer line before table')
                    if stated:
                        meta.update(stated)
                        break
                if meta['unit'] is None and statement_table(found[0]) and notes_clash:
                    problems.append({'pdf_page': pdf_page,
                                     'problem': 'conflicting statement units declared in notes'})
                elif meta['unit'] is None and statement_table(found[0]) and notes:
                    meta.update(notes)
                placed = grid(cells)
                columns = column_headers(placed, header_rows(cells))
                table, _ = rows_to_facts(placed, header_rows(cells), columns,
                                         {(item['row'], item['column']): item
                                          for item in mapped_numbers}, meta, None, pdf_page)
                facts += table
            carried = found
        else:
            carried = None
        textpage.close()
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
