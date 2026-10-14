"""Map numeric table cells from a layout parser back to the PDF text layer.

A layout parser returns each table as HTML plus one bounding box.  Citations in
this project point at character spans of the PDFium page text, so every numeric
cell is aligned to the PDF's own number at that position.  The PDF text layer is
the authority: a cell whose digits differ from it is reported, never corrected.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
import re

# A standalone disclosed number: bounded by whitespace so note references such
# as "七、1" and codes embedded in words are not treated as table values.
NUMBER = re.compile(r'(?<!\S)\(?-?\d[\d,]*(?:\.\d+)?\)?%?(?!\S)')
Box = tuple[float, float, float, float]  # left, bottom, right, top; PDF points


@dataclass(frozen=True)
class NumberSpan:
    """A number in the page text: ``text[start:end]`` and its drawn box."""
    text: str
    start: int
    end: int
    box: Box


@dataclass(frozen=True)
class Cell:
    row: int
    column: int
    rowspan: int
    colspan: int
    text: str


def key(text: str) -> str:
    """Digits as written; only grouping commas and spaces are ignored."""
    return re.sub(r'[,\s]', '', text)


def page_numbers(textpage, text: str, box: Box | None = None,
                 margin: float = 2.0) -> list[NumberSpan]:
    """Numbers of one page in visual reading order, optionally inside ``box``.

    ``text`` must be ``textpage.get_text_bounded()``; its indexes equal PDFium
    character indexes, which is what makes the returned spans citable.
    """
    spans = []
    for match in NUMBER.finditer(text):
        boxes = []
        for index in range(match.start(), match.end()):
            left, bottom, right, top = textpage.get_charbox(index)
            if right > left and top > bottom:
                boxes.append((left, bottom, right, top))
        if not boxes:
            continue
        drawn = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                 max(b[2] for b in boxes), max(b[3] for b in boxes))
        centre_x, centre_y = (drawn[0] + drawn[2]) / 2, (drawn[1] + drawn[3]) / 2
        if box and not (box[0] - margin <= centre_x <= box[2] + margin
                        and box[1] - margin <= centre_y <= box[3] + margin):
            continue
        spans.append(NumberSpan(match.group(), match.start(), match.end(), drawn))
    return reading_order(spans)


def reading_order(spans: list[NumberSpan]) -> list[NumberSpan]:
    """Top-to-bottom lines, left-to-right within a line."""
    lines: list[list[NumberSpan]] = []
    for span in sorted(spans, key=lambda s: -(s.box[1] + s.box[3]) / 2):
        centre = (span.box[1] + span.box[3]) / 2
        height = span.box[3] - span.box[1]
        last = lines[-1][-1] if lines else None
        if last and abs((last.box[1] + last.box[3]) / 2 - centre) <= height / 2:
            lines[-1].append(span)
        else:
            lines.append([span])
    return [span for line in lines for span in sorted(line, key=lambda s: s.box[0])]


class _Table(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cells: list[Cell] = []
        self.taken: set[tuple[int, int]] = set()
        self.row = -1
        self.open: dict | None = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row += 1
        elif tag in ('td', 'th'):
            if self.row < 0:
                raise ValueError('table cell outside a row')
            attributes = dict(attrs)
            column = 0
            while (self.row, column) in self.taken:
                column += 1
            self.open = {'column': column, 'text': [],
                         'rowspan': int(attributes.get('rowspan') or 1),
                         'colspan': int(attributes.get('colspan') or 1)}

    def handle_data(self, data):
        if self.open is not None:
            self.open['text'].append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.open is not None:
            cell, self.open = self.open, None
            for r in range(cell['rowspan']):
                for c in range(cell['colspan']):
                    self.taken.add((self.row + r, cell['column'] + c))
            self.cells.append(Cell(self.row, cell['column'], cell['rowspan'],
                                   cell['colspan'], ''.join(cell['text']).strip()))


def table_cells(html: str) -> list[Cell]:
    """Cells of an HTML table at their grid origin, honouring row/column spans."""
    parser = _Table()
    parser.feed(html)
    parser.close()
    return parser.cells


def map_table(cells: list[Cell], numbers: list[NumberSpan]) -> dict:
    """Align numeric cells with PDF numbers by order and exact digits.

    Returns ``mapped`` (cell with its span), ``unmapped_cells`` (numeric cells
    with no PDF counterpart: a misread value or a scanned page) and
    ``unmapped_numbers`` (PDF numbers in the table area without a numeric cell).
    Each unmapped number carries ``where``: ``merged`` when a text cell ends
    with it (a label and a value column were fused), ``embedded`` when a text
    cell merely contains it (a date in a header, "增加0.24个百分点"), and
    ``missing`` when no cell has it.
    """
    numeric = [cell for cell in sorted(cells, key=lambda c: (c.row, c.column))
               if NUMBER.fullmatch(cell.text)]
    prose = [key(cell.text) for cell in cells if not NUMBER.fullmatch(cell.text)]
    matcher = SequenceMatcher(None, [key(c.text) for c in numeric],
                              [key(n.text) for n in numbers], autojunk=False)
    mapped, used_cells, used_numbers = [], set(), set()
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            cell, span = numeric[block.a + offset], numbers[block.b + offset]
            used_cells.add(block.a + offset)
            used_numbers.add(block.b + offset)
            mapped.append({'row': cell.row, 'column': cell.column, 'text': span.text,
                           'start': span.start, 'end': span.end, 'box': list(span.box)})
    return {
        'mapped': mapped,
        'unmapped_cells': [{'row': c.row, 'column': c.column, 'text': c.text}
                           for i, c in enumerate(numeric) if i not in used_cells],
        'unmapped_numbers': [{'text': n.text, 'start': n.start, 'end': n.end,
                              'where': _placement(key(n.text), prose)}
                             for i, n in enumerate(numbers) if i not in used_numbers],
    }


def _placement(number: str, prose: list[str]) -> str:
    if any(text.endswith(number) for text in prose):
        return 'merged'
    return 'embedded' if any(number in text for text in prose) else 'missing'



def page_box(size: tuple[float, float], bbox: list[float], shift: float = 0.0) -> Box:
    """A MinerU block box in PDFium page coordinates.

    Block boxes are fractions of the page with a top-left origin; PDFium boxes
    are in points with a bottom-left origin.
    """
    width, height = size
    left, top, right, bottom = bbox
    return (left * width + shift, (1 - bottom) * height,
            right * width + shift, (1 - top) * height)


def horizontal_shift(numbers: list[NumberSpan], size: tuple[float, float],
                     bboxes: list[list[float]]) -> float:
    """Offset of the text layer on pages cut from a double-page spread.

    Some reports store each printed page as half of a spread and leave its
    characters at the spread's coordinates, one page width to the side.  The
    offset that places the most numbers inside the page's tables is used; a
    tie keeps the text where it is.
    """
    def inside(shift: float) -> int:
        boxes = [page_box(size, bbox, shift) for bbox in bboxes]
        return sum(any(b[0] <= (n.box[0] + n.box[2]) / 2 <= b[2]
                       and b[1] <= (n.box[1] + n.box[3]) / 2 <= b[3] for b in boxes)
                   for n in numbers)
    return max((0.0, size[0], -size[0]), key=lambda shift: (inside(shift), shift == 0.0))


def map_table_body(textpage, text: str, size: tuple[float, float], part: dict,
                   shift: float = 0.0) -> dict:
    """Map one MinerU ``table_body`` part onto its page."""
    box = page_box(size, part['bbox'], shift)
    cells = table_cells(part['content'])
    result = map_table(cells, page_numbers(textpage, text, box))
    map_wrapped(result, textpage, text, box)
    return {'box': list(box), 'cells': cells, **result}


def map_wrapped(result: dict, textpage, text: str, box: Box, margin: float = 2.0) -> None:
    """Map cells whose number is broken across lines in the PDF text.

    A narrow column wraps "872,451.40" into "8 / 7 / 2, / 4 / ..."; the text
    layer then holds the digits in order but separated by line breaks, so no
    whole number exists to align with.  Such a cell is matched against the
    table's remaining characters with whitespace removed, in reading order and
    only when the digits are long enough to be distinctive.  ``result`` is
    updated in place; the span of a wrapped number includes its line breaks.
    """
    if not result['unmapped_cells']:
        return
    taken = [False] * len(text)
    for item in result['mapped']:
        taken[item['start']:item['end']] = [True] * (item['end'] - item['start'])
    stream, index, boxes = [], [], []
    for position, char in enumerate(text):
        if char.isspace() or taken[position]:
            continue
        try:
            left, bottom, right, top = textpage.get_charbox(position)
        except Exception:  # PDFium has no box for a generated character
            continue
        if not (box[0] - margin <= (left + right) / 2 <= box[2] + margin
                and box[1] - margin <= (bottom + top) / 2 <= box[3] + margin):
            continue
        stream.append(char)
        index.append(position)
        boxes.append((left, bottom, right, top))
    joined, cursor, remaining = ''.join(stream), 0, []
    for cell in result['unmapped_cells']:
        wanted = re.sub(r'\s', '', cell['text'])
        at = joined.find(wanted, cursor) if len(wanted) >= 6 else -1
        if at < 0:
            remaining.append(cell)
            continue
        start, end = index[at], index[at + len(wanted) - 1] + 1
        # Every character between the first and last digit must belong to the
        # match; otherwise two unrelated fragments were joined.
        if sum(not char.isspace() for char in text[start:end]) != len(wanted):
            remaining.append(cell)
            continue
        drawn = boxes[at:at + len(wanted)]
        result['mapped'].append({
            'row': cell['row'], 'column': cell['column'], 'text': cell['text'],
            'start': start, 'end': end, 'wrapped': True,
            'box': [min(b[0] for b in drawn), min(b[1] for b in drawn),
                    max(b[2] for b in drawn), max(b[3] for b in drawn)]})
        taken[start:end] = [True] * (end - start)
        cursor = at + len(wanted)
    result['unmapped_cells'] = remaining
    result['unmapped_numbers'] = [number for number in result['unmapped_numbers']
                                  if not taken[number['start']]]


def parsed_tables(document, middle: dict):
    """Yield ``(page, preceding blocks, table block, mapping)`` for every table.

    ``document`` is an open ``pypdfium2.PdfDocument`` of the file that produced
    the MinerU ``docvortex.middle`` result ``middle``.
    """
    if middle.get('schema') != 'docvortex.middle':
        raise ValueError('unsupported layout parser output')
    for parsed in middle['pages']:
        page = document[parsed['page_idx']]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        blocks = parsed.get('blocks', [])  # a blank page has no blocks
        bodies = [part['bbox'] for block in blocks if block['type'] == 'table'
                  for part in block['content'] if part['type'] == 'table_body']
        shift = (horizontal_shift(page_numbers(textpage, text), page.get_size(), bodies)
                 if bodies else 0.0)
        for position, block in enumerate(blocks):
            if block['type'] != 'table':
                continue
            for part in block['content']:
                if part['type'] == 'table_body':
                    yield (parsed['page_idx'] + 1, blocks[:position], block,
                           map_table_body(textpage, text, page.get_size(), part, shift))
        textpage.close()
        page.close()
