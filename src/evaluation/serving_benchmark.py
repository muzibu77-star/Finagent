"""Fixed-length local serving measurements, separate from task quality scores."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

from src.model.http_driver import HttpDriver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--served-model', default='base')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    driver = HttpDriver(args.server_url, args.served_model)
    settings = {'model': args.served_model, 'url': args.server_url,
        'input_tokens': [256, 2048], 'output_tokens': 128, 'concurrency': [1, 4],
        'requests_per_condition': 8, 'seed': 0, 'temperature': 0,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits': 'Warm service measurement; forced output length, not task quality. '
                  'TTFT is first nonempty streamed text event. Memory is whole-device usage; '
                  'requires otherwise idle GPU. Cold-start time is recorded in server launch logs.'}
    (args.output_dir/'settings.json').write_text(json.dumps(settings, indent=2))
    memory = []
    stopped = threading.Event()
    def sample_memory():
        while not stopped.wait(.2):
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
                                     '--id=0'], capture_output=True, text=True, check=True)
            memory.append(int(result.stdout.strip()))
    sampler = threading.Thread(target=sample_memory)
    sampler.start()
    records, groups = [], []
    try:
        tokens = driver.tokenizer('Financial statement revenue and operating income. '*500,
                                  add_special_tokens=False)['input_ids']
        for size in settings['input_tokens']:
            prompt = tokens[:size]
            if len(prompt) != size:
                raise ValueError('insufficient fixed input tokens')
            def request(index):
                payload = {'model': args.served_model, 'prompt': prompt, 'max_tokens': 128,
                    'temperature': 0, 'seed': 0, 'ignore_eos': True, 'stream': True,
                    'stream_options': {'include_usage': True}}
                start = time.monotonic()
                row = {'input_tokens': size, 'request': index, 'ttft_s': None, 'output_tokens': 0}
                req = urllib.request.Request(args.server_url.rstrip('/')+'/v1/completions',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                try:
                    with driver.opener.open(req, timeout=180) as response:
                        for line in response:
                            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                                continue
                            event = json.loads(line[6:])
                            if row['ttft_s'] is None and any(c.get('text') for c in event['choices']):
                                row['ttft_s'] = time.monotonic()-start
                            if event.get('usage'):
                                row['output_tokens'] = event['usage']['completion_tokens']
                    if row['output_tokens'] != 128 or row['ttft_s'] is None:
                        row['error'] = 'incomplete stream or unexpected output token count'
                except (OSError, ValueError, KeyError) as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                row['latency_s'] = time.monotonic()-start
                return row
            warm = request(-1)
            if warm.get('error'):
                (args.output_dir/'warmup_failure.json').write_text(json.dumps(warm, indent=2))
                raise RuntimeError('warmup failed')
            for concurrency in settings['concurrency']:
                start = time.monotonic()
                with ThreadPoolExecutor(concurrency) as pool:
                    group = list(pool.map(request, range(8)))
                elapsed = time.monotonic()-start
                for row in group:
                    row['concurrency'] = concurrency
                    records.append(row)
                (args.output_dir/'records.json').write_text(json.dumps(records, indent=2))
                successful = [r for r in group if not r.get('error')]
                times = sorted(r['ttft_s'] for r in successful)
                groups.append({'input_tokens': size, 'concurrency': concurrency, 'requests': len(group),
                    'errors': len(group)-len(successful), 'wall_s': elapsed,
                    'successful_requests_per_s': len(successful)/elapsed,
                    'output_tokens_per_s': sum(r['output_tokens'] for r in successful)/elapsed,
                    'ttft_p50_s': statistics.median(times) if times else None,
                    'ttft_p95_s': times[-1] if times else None,
                    'latency_p50_s': statistics.median(r['latency_s'] for r in successful) if successful else None})
    finally:
        stopped.set()
        sampler.join()
    (args.output_dir/'summary.json').write_text(json.dumps({'groups': groups,
        'peak_device_memory_mib': max(memory) if memory else None, 'settings': settings}, indent=2))


if __name__ == '__main__':
    main()
