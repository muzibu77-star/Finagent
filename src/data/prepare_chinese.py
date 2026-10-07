"""Source-bound Chinese questions, with gold calculated using business Decimal rules."""

import argparse
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium

from src.evidence.models import Evidence, Fact, number
from src.tools.calculator import calculate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config_path = Path('configs/m5_chinese_tasks.json')
    config = json.loads(config_path.read_text())
    sources = json.loads(Path('configs/report_sources.json').read_text())
    source = next(s for s in sources['sources'] if s['id'] == config['source_id'])
    if source['published_at'] != config['published_at']:
        raise ValueError('disclosure date mismatch')
    if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('source changed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_path = args.output_dir / 'page1.png'
    with pdfium.PdfDocument(source['path']) as document:
        page = document[config['pdf_page'] - 1]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        textpage.close()
        bitmap = page.render(scale=1240 / page.get_width())
        bitmap.to_pil().save(image_path)
        bitmap.close()
        page.close()
    facts, evidence, bindings = {}, {}, {}
    for key, item in config['facts'].items():
        if text.count(item['raw']) != 1:
            raise ValueError('reviewed scalar does not bind uniquely to PDF text')
        start = text.index(item['raw'])
        evidence[key] = Evidence(key, source['id'], source['version'], item['raw'],
                                 source['sha256'], page=config['pdf_page'])
        facts[key] = Fact(key, 'CHL', item['metric'], number(item['raw']), item['unit'],
                          item['currency'], Decimal(item['scale']), item['period'], 'group',
                          (key,), 'validated')
        bindings[key] = {'start': start, 'end': start + len(item['raw']), **item}
    tasks, gold = [], {}
    for task in config['tasks']:
        result = calculate(task['id'], task['operation'], task['facts'], facts, evidence)
        value = Decimal(result.value) / Decimal(task['display_divisor'])
        expected = value.quantize(Decimal(1).scaleb(-task['decimals']), rounding=ROUND_HALF_EVEN)
        tasks.append({'id': task['id'], 'question': task['question'], 'unit': task['unit'],
                      'text': text, 'image_path': str(image_path), 'source_id': source['id'],
                      'published_at': source['published_at'], 'pdf_page': config['pdf_page'],
                      'url': source['url'], 'source_sha256': source['sha256']})
        gold[task['id']] = {'answer': str(expected), 'unit': task['unit'],
                            'calculation': result.to_dict(), 'decimals': task['decimals']}
    for name, obj in [('tasks', tasks), ('gold', gold), ('bindings', bindings)]:
        (args.output_dir / f'{name}.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    lock = {'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'source_sha256': source['sha256'], 'review': config['review'],
            'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in args.output_dir.iterdir() if p.is_file()}}
    (args.output_dir / 'lock.json').write_text(json.dumps(lock, indent=2))
    print(json.dumps({'tasks': len(tasks), 'gold': gold}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
