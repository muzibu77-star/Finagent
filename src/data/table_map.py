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
    ``unmapped_numbers`` (PDF numbers in the table area the parser left out).
    """
    numeric = [cell for cell in sorted(cells, key=lambda c: (c.row, c.column))
               if NUMBER.fullmatch(cell.text)]
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
        'unmapped_numbers': [{'text': n.text, 'start': n.start, 'end': n.end}
                             for i, n in enumerate(numbers) if i not in used_numbers],
    }
