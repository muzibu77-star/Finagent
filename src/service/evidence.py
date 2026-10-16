"""The original page of one fact, rendered with its box marked.

The review page shows a number beside the PDF page it was taken from.  The
position is the fact's own character box in the source library and the file is
the download receipt in ``configs/cninfo_sources.json``; this module infers
neither a position nor a value.  A file whose SHA256 no longer matches its
receipt is never rendered, and a fact without a box - a number the layout
parser could not tie to the text layer - is shown as the whole page only.

Coordinates: a library box is ``(left, bottom, right, top)`` in PDF points with
a bottom-left origin, exactly as ``src/data/table_map.py`` stores it, while a
rendering has a top-left origin and ``scale`` pixels per point, so the vertical
axis is flipped around the page height.  A crop removes ``(left, bottom, right,
top)`` points from the rendering, which moves that origin to ``(crop_left,
height - crop_top)`` in page points.
"""

import hashlib
import io
import json
from pathlib import Path
import sqlite3
import threading
from typing import Sequence

from PIL import ImageDraw
import pypdfium2 as pdfium

SCALE = 2.0    # pixels per point of the page image
ZOOM = 5.0     # pixels per point of the crop around the box
MARGIN = 60.0  # points of page around the box kept in a crop
OUTLINE = 3    # outline width in pixels
MARK = (214, 48, 49)
# PDFium is not thread-safe: two renderings at once corrupt the library state of the
# whole process, after which every document fails to load, so the server renders one
# page at a time.
PDFIUM = threading.Lock()
FACT_FIELDS = ('id', 'page', 'box', 'text', 'unit', 'row_label', 'column_header', 'title')


def receipt(sources: Path, document_id: str) -> dict:
    """The download receipt of a document, after checking its file's SHA256.

    Raises ``LookupError`` when the receipt or the file is missing and
    ``ValueError`` when the file no longer matches the recorded hash.
    """
    entries = json.loads(Path(sources).read_text())['sources']
    entry = next((item for item in entries if item['id'] == document_id), None)
    if entry is None:
        raise LookupError(f'下载收据里没有文件 {document_id}')
    path = Path(entry['path'])
    if not path.is_file():
        raise LookupError(f'原件不在本机：{entry["path"]}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'{path.name} 与下载收据的哈希不符（{digest[:12]} ≠ '
                         f'{entry["sha256"][:12]}），拒绝显示')
    return {**entry, 'file': path}


def fact(db: sqlite3.Connection, document_id: str, page: int, start: int,
         end: int) -> dict | None:
    """The library fact at one page and character range of a document, or ``None``.

    That position rather than the fact id identifies the fact: ids are assigned
    when a library is built, so they differ between library versions, while the
    page and the character range follow the PDF text.  The page is part of the
    key because one character range can occur on several pages: a workpaper
    cites 2,615 positions, of which 913 would find another page's fact without
    it.
    """
    row = db.execute(
        'SELECT f.id, f.page, f.box, f.text, f.unit, f.row_label, f.column_header,'
        ' d.title FROM facts f JOIN documents d ON d.id = f.document_id'
        ' WHERE f.document_id = ? AND f.page = ? AND f.start = ? AND f.end = ?',
        (document_id, page, start, end)).fetchone()
    if row is None:
        return None
    return dict(zip(FACT_FIELDS, row))


def box_of(located: dict) -> list[float]:
    """The box a library fact carries, or an empty list when it has none."""
    return json.loads(located['box'] or '[]')


def box_to_pixels(box: Sequence[float], size: Sequence[float], scale: float = 1.0,
                  crop: Sequence[float] = (0.0, 0.0, 0.0, 0.0)) -> tuple[int, int, int, int]:
    """The pixel rectangle ``(left, top, right, bottom)`` of a box in a rendering.

    ``box`` and ``crop`` are PDF points with a bottom-left origin, ``size`` is
    the page size in points and ``scale`` is the pixels per point of the
    rendering, which has a top-left origin.
    """
    left, bottom, right, top = box
    cut_left, *_, cut_top = crop
    edge = size[1] - cut_top  # the top edge of the rendering, in page points
    return (round((left - cut_left) * scale), round((edge - top) * scale),
            round((right - cut_left) * scale), round((edge - bottom) * scale))


def crop_box(box: Sequence[float], size: Sequence[float],
             margin: float = MARGIN) -> tuple[float, float, float, float]:
    """The ``(left, bottom, right, top)`` points to cut so the box is centred."""
    return (max(box[0] - margin, 0.0), max(box[1] - margin, 0.0),
            max(size[0] - box[2] - margin, 0.0), max(size[1] - box[3] - margin, 0.0))


def page_png(path: Path, page: int, box: Sequence[float] | None = None, *,
             scale: float = SCALE, margin: float | None = None) -> bytes:
    """One page of a PDF as PNG, the fact's ``box`` outlined when it has one.

    Without ``margin`` the whole page is rendered; with it the rendering is the
    box plus that many points of page around it.
    """
    with PDFIUM, pdfium.PdfDocument(path) as document:
        drawn = document[page - 1]
        if drawn.get_rotation() % 360:
            raise ValueError(f'第 {page} 页有旋转，暂不显示')
        size = drawn.get_size()
        crop = crop_box(box, size, margin) if margin is not None and box else (0.0,) * 4
        bitmap = drawn.render(scale=scale, crop=crop)
        image = bitmap.to_pil().convert('RGB')
        bitmap.close()
    if box:
        pixels = box_to_pixels(box, size, scale, crop)
        draw = ImageDraw.Draw(image)
        for offset in range(OUTLINE):
            draw.rectangle((pixels[0] - offset, pixels[1] - offset,
                            pixels[2] + offset, pixels[3] + offset), outline=MARK)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()
