"""Tests for the evidence view: box coordinates, the hash check and feedback."""

import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image

from src.data.library import SCHEMA
from src.service import evidence
from src.service.app import ReviewServer

# The page talks to a loopback server; the proxy of this machine must not see it.
os.environ.setdefault('NO_PROXY', '127.0.0.1,localhost')
os.environ.setdefault('no_proxy', '127.0.0.1,localhost')

WIDTH, HEIGHT = 200.0, 300.0
BOX = [20.0, 240.0, 60.0, 260.0]  # left, bottom, right, top; PDF points
WHITE = (255, 255, 255)


def minimal_pdf(width: float = WIDTH, height: float = HEIGHT) -> bytes:
    """A one-page PDF without content, so a known box can be checked."""
    bodies = [b'<< /Type /Catalog /Pages 2 0 R >>',
              b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
              f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:g} {height:g}]'
              ' /Resources << >> >>'.encode()]
    out = bytearray(b'%PDF-1.4\n')
    offsets = []
    for index, body in enumerate(bodies, 1):
        offsets.append(len(out))
        out += f'{index} 0 obj\n'.encode() + body + b'\nendobj\n'
    start = len(out)
    out += f'xref\n0 {len(bodies) + 1}\n'.encode() + b'0000000000 65535 f \n'
    for offset in offsets:
        out += f'{offset:010d} 00000 n \n'.encode()
    out += (f'trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\nstartxref\n{start}\n'
            '%%EOF\n').encode()
    return bytes(out)


def build_library(path: Path) -> None:
    """A library with one text-layer fact, a same-range fact on another page and
    one without a box.

    The character range 10..19 occurs on two pages, so the page is part of the
    position: without it that range would find the other page's fact.
    """
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.execute('INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               ('doc', 'annual', '某公司', '600000', 2024, '年度报告', 'http://x',
                '2025-03-30', 'h', 9))
    for start, end, box, text in ((10, 19, BOX, '1,745.44'),
                                  (30, 39, [], '1,741.44'),
                                  (40, 49, [], '1,700.00')):
        db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, '
                   'row_label, column_header, title, unit, scale, scope) '
                   'VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                   ('doc', start, end, json.dumps(box), text, text.replace(',', ''),
                    '营业收入', '2024年度', '合并利润表', '亿元', '1', '合并'))
    db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, '
               'row_label, column_header, title, unit, scale, scope) '
               'VALUES (?, 4, 10, 19, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               ('doc', json.dumps([100.0, 100.0, 140.0, 120.0]), '1,750.00', '1750.00',
                '未分配利润', '本期增加', '合并利润表', '亿元', '1', '合并'))
    db.execute('INSERT INTO facts (document_id, page, start, end, box, text, digits, '
               'row_label, column_header, title, unit, scale, scope) '
               'VALUES (?, 8, 60, 69, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
               ('doc', json.dumps(BOX), '1,000.00', '1000.00', '营业收入', '2024年度',
                '合并利润表', '亿元', '1', '合并'))
    db.commit()
    db.close()


class CoordinatesTest(unittest.TestCase):
    """The PDF-point to pixel conversion, on a page whose box is known."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.page = Path(self.directory.name) / 'page.pdf'
        self.page.write_bytes(minimal_pdf())

    def tearDown(self):
        self.directory.cleanup()

    def image(self, **kwargs) -> Image.Image:
        return Image.open(io.BytesIO(evidence.page_png(self.page, 1, **kwargs))).convert('RGB')

    def test_a_box_maps_to_its_pixel_rectangle(self):
        self.assertEqual(evidence.box_to_pixels(BOX, (WIDTH, HEIGHT), 2.0), (40, 80, 120, 120))
        # The vertical axis is flipped around the page height: a box near the
        # bottom of the page lands near the bottom of the image.
        self.assertEqual(evidence.box_to_pixels([20.0, 20.0, 60.0, 40.0], (WIDTH, HEIGHT), 2.0),
                         (40, 520, 120, 560))
        # A crop is measured from its own corner.
        self.assertEqual(evidence.box_to_pixels(BOX, (WIDTH, HEIGHT), 2.0, (10.0,) * 4),
                         (20, 60, 100, 100))

    def test_the_mark_is_drawn_where_the_box_is(self):
        image = self.image(box=BOX, scale=2.0)
        self.assertEqual(image.size, (400, 600))
        self.assertEqual(image.getpixel((38, 100)), evidence.MARK)   # left edge
        self.assertEqual(image.getpixel((122, 100)), evidence.MARK)  # right edge
        self.assertEqual(image.getpixel((80, 78)), evidence.MARK)    # top edge
        self.assertEqual(image.getpixel((80, 122)), evidence.MARK)   # bottom edge
        self.assertEqual(image.getpixel((80, 100)), WHITE)           # inside
        self.assertEqual(image.getpixel((80, 400)), WHITE)           # below: no flip error

    def test_a_crop_marks_the_same_box(self):
        # The crop cuts the page 10 points on every side of the box.
        crop = evidence.crop_box(BOX, (WIDTH, HEIGHT), margin=10.0)
        self.assertEqual(crop, (10.0, 230.0, 130.0, 30.0))
        image = self.image(box=BOX, scale=2.0, margin=10.0)
        self.assertEqual(image.size, (120, 80))
        pixels = evidence.box_to_pixels(BOX, (WIDTH, HEIGHT), 2.0, crop)
        self.assertEqual(pixels, (20, 20, 100, 60))
        self.assertEqual(image.getpixel((18, 40)), evidence.MARK)
        self.assertEqual(image.getpixel((102, 40)), evidence.MARK)
        self.assertEqual(image.getpixel((60, 40)), WHITE)

    def test_a_page_without_a_box_is_rendered_whole(self):
        image = self.image(scale=2.0)
        self.assertEqual(image.size, (400, 600))
        self.assertEqual(image.getpixel((40, 80)), WHITE)


REAL_ID = '601166-1222927176'  # 兴业银行 2024 annual report, the T19b demo document


def real_report() -> Path | None:
    """The local copy of a real annual report, or ``None`` on a machine without it."""
    sources = Path('configs/cninfo_sources.json')
    if not sources.is_file():
        return None
    entry = next((item for item in json.loads(sources.read_text())['sources']
                  if item['id'] == REAL_ID), None)
    path = Path(entry['path']) if entry else None
    return path if path and path.is_file() else None


class ConcurrencyTest(unittest.TestCase):
    """PDFium is not thread-safe; unguarded, parallel renderings of a real report fail
    and leave every later rendering in the process failing too."""

    @unittest.skipUnless(real_report(), 'the original annual report is not on this machine')
    def test_renderings_at_once_all_succeed(self):
        path, failures = real_report(), []

        def render(page: int) -> None:
            try:
                evidence.page_png(path, page, (273.4, 556.9, 308.7, 564.9),
                                  scale=evidence.ZOOM, margin=evidence.MARGIN)
            except Exception as error:  # noqa: BLE001 - collected for the assertion
                failures.append(error)

        threads = [threading.Thread(target=render, args=(page,))
                   for page in (24, 210, 211, 24, 210, 211)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(failures, [])
        render(24)
        self.assertEqual(failures, [])


class EvidenceTest(unittest.TestCase):
    """The endpoints of the evidence view and the feedback file."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.library = root / 'library.sqlite'
        build_library(self.library)
        self.pdf = root / 'doc.pdf'
        self.pdf.write_bytes(minimal_pdf())
        self.sources = root / 'sources.json'
        self.write_sources(hashlib.sha256(self.pdf.read_bytes()).hexdigest())
        self.data_dir = root / 'data'
        self.data_dir.mkdir()
        self.server = ReviewServer(('127.0.0.1', 0), library=self.library,
                                   data_dir=self.data_dir, budget=0.0, local_url=None,
                                   sources=self.sources)
        threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.02},
                         daemon=True).start()
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.directory.cleanup()

    def write_sources(self, sha256: str) -> None:
        self.sources.write_text(json.dumps({'sources': [
            {'id': 'doc', 'path': str(self.pdf), 'sha256': sha256}]}))

    def get(self, path: str) -> tuple[int, bytes]:
        try:
            with urlopen(self.base + path, timeout=10) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def evidence_path(self, page: int = 1, start: int = 10, end: int = 19, **extra) -> str:
        return '/api/evidence?' + self.query(page, start, end, **extra)

    def image_path(self, page: int = 1, start: int = 10, end: int = 19, **extra) -> str:
        return '/api/evidence.png?' + self.query(page, start, end, **extra)

    def query(self, page: int = 1, start: int = 10, end: int = 19, **extra) -> str:
        return urlencode({'document_id': 'doc', 'page': page, 'start': start,
                          'end': end, **extra})

    def post(self, path: str, body) -> tuple[int, dict]:
        request = Request(self.base + path, data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_the_fact_of_a_character_range_comes_back_with_its_box(self):
        status, body = self.get(self.evidence_path())
        self.assertEqual(status, 200)
        found = json.loads(body)
        self.assertEqual(found['title'], '年度报告')
        self.assertIsInstance(found['id'], int)
        self.assertEqual(found['page'], 1)
        self.assertEqual(found['box'], BOX)
        self.assertEqual(found['text'], '1,745.44')
        self.assertEqual(found['unit'], '亿元')
        self.assertEqual(found['row_label'], '营业收入')
        self.assertEqual(found['page_url'], self.image_path())
        self.assertEqual(found['crop_url'], self.image_path(crop='1'))
        self.assertEqual(self.get(found['page_url'])[0], 200)
        self.assertEqual(Image.open(io.BytesIO(self.get(found['crop_url'])[1])).width, 600)

    def test_the_page_and_the_crop_are_png_of_the_fact(self):
        status, body = self.get(self.image_path(crop='1'))
        self.assertEqual(status, 200)
        image = Image.open(io.BytesIO(body)).convert('RGB')
        crop = evidence.crop_box(BOX, (WIDTH, HEIGHT))
        pixels = evidence.box_to_pixels(BOX, (WIDTH, HEIGHT), evidence.ZOOM, crop)
        self.assertEqual(image.size, (round((WIDTH - crop[0] - crop[2]) * evidence.ZOOM),
                                      round((HEIGHT - crop[1] - crop[3]) * evidence.ZOOM)))
        self.assertEqual(image.getpixel((pixels[0] - 1, (pixels[1] + pixels[3]) // 2)),
                         evidence.MARK)
        status, body = self.get(self.image_path(crop='0'))
        self.assertEqual(status, 200)
        self.assertEqual(Image.open(io.BytesIO(body)).size,
                         (round(WIDTH * evidence.SCALE), round(HEIGHT * evidence.SCALE)))

    def test_the_page_is_part_of_the_position(self):
        """The same character range on another page is another fact."""
        status, body = self.get(self.evidence_path(1))
        self.assertEqual(json.loads(body)['text'], '1,745.44')
        status, body = self.get(self.evidence_path(4))
        self.assertEqual(status, 200)
        found = json.loads(body)
        self.assertEqual(found['text'], '1,750.00')
        self.assertEqual(found['row_label'], '未分配利润')
        self.assertNotEqual(found['box'], BOX)

    def test_a_fact_without_a_box_shows_the_whole_page_only(self):
        status, body = self.get(self.evidence_path(1, 30, 39))
        self.assertEqual(status, 200)
        found = json.loads(body)
        self.assertEqual(found['box'], [])
        self.assertIsNone(found['crop_url'])
        self.assertIn('无法定位到数字', found['note'])
        self.assertEqual(self.get(found['page_url'])[0], 200)
        status, body = self.get(self.image_path(1, 30, 39, crop='1'))
        self.assertEqual(status, 400)
        self.assertIn('没有位置框', json.loads(body)['error'])

    def test_a_changed_file_is_refused(self):
        self.write_sources('0' * 64)
        status, body = self.get(self.evidence_path())
        self.assertEqual(status, 409)
        self.assertIn('哈希不符', json.loads(body)['error'])
        status, body = self.get(self.image_path(crop='1'))
        self.assertEqual(status, 409)
        self.assertIn('拒绝显示', json.loads(body)['error'])

    def test_a_missing_receipt_or_page_is_refused(self):
        self.sources.write_text(json.dumps({'sources': []}))
        status, body = self.get(self.evidence_path())
        self.assertEqual(status, 404)
        self.assertIn('下载收据里没有文件 doc', json.loads(body)['error'])
        self.write_sources(hashlib.sha256(self.pdf.read_bytes()).hexdigest())
        status, body = self.get(self.evidence_path(1, 99, 108))
        self.assertEqual(status, 404)
        self.assertIn('出处库中没有 doc', json.loads(body)['error'])
        status, body = self.get(self.image_path(8, 60, 69))
        self.assertEqual(status, 409)
        self.assertIn('渲染不了', json.loads(body)['error'])

    def test_missing_parameters_are_refused(self):
        for path in ('/api/evidence', '/api/evidence.png',
                     '/api/evidence?document_id=doc&start=x&end=2',
                     '/api/evidence?document_id=doc&start=10&end=19'):
            status, body = self.get(path)
            self.assertEqual(status, 400)
            self.assertIn('必须齐全且为数字', json.loads(body)['error'])

    def test_a_click_appends_one_line_and_never_changes_the_workpaper(self):
        paper = self.data_dir / 'review-flash_open-600000-1.json'
        paper.write_text('{"summary": {"numbers": 1}}')
        before = paper.read_bytes()
        record = {'review_id': 'review-flash_open-600000-1', 'start': 10, 'answer': '确有错误',
                  'note': '与附注不符'}
        status, body = self.post('/api/feedback', record)
        self.assertEqual(status, 200)
        self.assertEqual(body['answer'], '确有错误')
        self.assertEqual(body['workpaper_version'],
                         hashlib.sha256(before).hexdigest())
        self.post('/api/feedback', {**record, 'start': 30, 'answer': '跳过', 'note': ''})
        lines = [json.loads(line) for line
                 in (self.data_dir / 'feedback.jsonl').read_text().splitlines()]
        self.assertEqual([line['start'] for line in lines], [10, 30])
        self.assertEqual([line['answer'] for line in lines], ['确有错误', '跳过'])
        self.assertEqual(lines[0]['note'], '与附注不符')
        self.assertEqual(sorted(lines[0]), ['answer', 'at', 'note', 'review_id', 'start',
                                            'workpaper_version'])
        self.assertEqual(paper.read_bytes(), before)

    def test_a_bad_feedback_request_is_refused(self):
        good = {'review_id': 'review-flash_open-600000-1', 'start': 10, 'answer': '一致'}
        for change in ({'review_id': '../etc/passwd'}, {'start': -1}, {'start': '10'},
                       {'answer': '对'}, {'note': 'x' * 501}):
            status, _ = self.post('/api/feedback', {**good, **change})
            self.assertEqual(status, 400)
        self.assertFalse((self.data_dir / 'feedback.jsonl').exists())

    def test_the_last_click_on_each_number_is_read_back(self):
        review = {'review_id': 'review-flash_open-600000-1', 'note': ''}
        for start, answer in ((10, '跳过'), (30, '一致'), (10, '确有错误')):
            self.post('/api/feedback', {**review, 'start': start, 'answer': answer})
        self.post('/api/feedback', {**review, 'review_id': 'review-flash_open-600000-2',
                                    'start': 10, 'answer': '一致'})
        status, body = self.get('/api/feedback?review_id=review-flash_open-600000-1')
        self.assertEqual(status, 200)
        latest = json.loads(body)
        self.assertEqual({start: record['answer'] for start, record in latest.items()},
                         {'10': '确有错误', '30': '一致'})
        status, _ = self.get('/api/feedback?review_id=../x')
        self.assertEqual(status, 400)

    def test_the_kept_workpapers_are_listed_with_their_counts(self):
        (self.data_dir / 'review-a.json').write_text('{"summary": {"numbers": 3}}')
        (self.data_dir / 'feedback.jsonl').write_text('')
        (self.data_dir / 'broken.json').write_text('{')
        status, body = self.get('/api/workpapers')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), [{'id': 'review-a', 'summary': {'numbers': 3}}])


if __name__ == '__main__':
    unittest.main()
