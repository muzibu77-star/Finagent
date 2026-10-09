"""Freeze source-bound Chinese arithmetic families from original disclosures."""

import argparse
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/jd_chinese_sources.json'))
    args = parser.parse_args()
    config_path = args.config
    config = json.loads(config_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tasks, gold = [], {}
    for source in config['sources']:
        raw = Path(source['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError('source hash changed')
        image_path = args.output_dir / (source['id'] + '.png')
        with pdfium.PdfDocument(raw) as document:
            page = document[source['page'] - 1]
            textpage = page.get_textpage()
            text = textpage.get_text_bounded()
            textpage.close()
            bitmap = page.render(scale=1600 / page.get_width())
            bitmap.to_pil().save(image_path)
            bitmap.close()
            page.close()
            context_texts = []
            for page_number in source.get('context_pages', []):
                page = document[page_number - 1]
                tp = page.get_textpage()
                context_texts.append(tp.get_text_bounded())
                tp.close()
                page.close()
            text = '\n'.join(context_texts + [text])
        old_year, year = source['years']
        prefix = source.get('qualifier', '') + source['company']
        for index, metric in enumerate(source['metrics']):
            name, old_raw, new_raw = metric[:3]
            if old_raw not in text or new_raw not in text:
                raise ValueError(f'original page lacks declared numbers: {name}')
            old, new = [Decimal(v.replace(',', '').removesuffix('%')) for v in (old_raw, new_raw)]
            unit = metric[3] if len(metric) > 3 else source['unit']
            scale = Decimal(metric[4] if len(metric) > 4 else source['scale'])
            delta_unit = '个百分点' if unit == '%' else unit
            cases = [
                ('old', f'{old_year}年{name}是多少{unit}？', old, unit),
                ('new', f'{year}年{name}是多少{unit}？', new, unit),
                ('delta', f'{year}年{name}比{old_year}年增加多少{delta_unit}？', new-old, delta_unit),
                ('growth', f'按原始数值计算{year}年{name}比{old_year}年的增长率是多少（百分数）？',
                 (new/old-1)*100, '%'),
                ('mean', f'{old_year}年和{year}年{name}的算术平均值是多少{unit}？', (old+new)/2, unit),
                ('ratio', f'{year}年{name}是{old_year}年的多少倍？', new/old, '倍'),
                ('sum', f'{old_year}年和{year}年{name}两个披露数值相加是多少{unit}？', old+new, unit),
            ]
            if unit.startswith('人民币') and unit != '人民币元/股':
                cases.append(('conversion', f'{year}年{name}换算为人民币亿元是多少？',
                              new*scale/Decimal('100000000'), '人民币亿元'))
                cases.append(('wan', f'{year}年{name}换算为人民币万元是多少？',
                              new*scale/Decimal('10000'), '人民币万元'))
            for operation, question, value, answer_unit in cases:
                key = f"{source['id']}-{index}-{operation}"
                tasks.append({'id': key, 'question': prefix+question+'保留四位小数。',
                    'unit': answer_unit, 'text': text, 'image_path': str(image_path),
                    'source_id': source['id'], 'source_report_id': source['path'],
                    'company': source['company'], 'pdf_page': source['page'],
                    'context_pages': source.get('context_pages', []) + [source['page']],
                    'published_at': source['published_at'], 'url': source['url'],
                    'source_sha256': source['sha256'], 'family': f"{source['id']}-{index}"})
                tasks[-1]['split'] = 'dev' if source['company'] == '中国电信' else 'test'
                gold[key] = {'answer': str(value.quantize(Decimal('0.0001'), rounding=ROUND_HALF_EVEN)),
                    'unit': answer_unit, 'decimals': 4, 'operation': operation,
                    'source_values': [old_raw, new_raw], 'original_scale': str(scale)}
    # Preserve the historical eight as explicitly marked development probes.
    old_root = Path('data/staged/m5_chinese_v1')
    old_gold = json.loads((old_root/'gold.json').read_text())
    for task in json.loads((old_root/'tasks.json').read_text()):
        task.update(company='中国移动', split='historical_dev',
                    source_report_id=task['source_id'], family=task['source_id'])
        tasks.append(task)
        gold[task['id']] = old_gold[task['id']]
    for name, value in [('tasks', tasks), ('gold', gold)]:
        (args.output_dir/f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
    lock = {'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'tasks': len(tasks), 'companies': sorted({t['company'] for t in tasks}),
        'limits': config['review'], 'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output_dir.iterdir() if p.is_file()}}
    (args.output_dir/'lock.json').write_text(json.dumps(lock, ensure_ascii=False, indent=2))
    print(json.dumps({'tasks': len(tasks), 'companies': lock['companies']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
