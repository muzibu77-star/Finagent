"""Download only pinned, checksum-verified M0 JSON and ConvFinQA ZIP assets."""

import hashlib
import json
from pathlib import Path
import urllib.request


def main() -> None:
    lock = json.loads(Path('configs/m0_data_lock.json').read_text())
    tat = json.loads(Path('configs/tatqa_sources.json').read_text())
    for name, expected in lock['sha256'].items():
        path = Path(name)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'existing file has wrong hash: {path}')
            continue
        dataset = path.parent.name
        if dataset in tat:
            url = tat[dataset]['base_url'] + path.name
        else:
            repo = 'FinQA' if dataset == 'finqa' else 'ConvFinQA'
            prefix = 'dataset/' if dataset == 'finqa' else ''
            url = (f'https://raw.githubusercontent.com/czyssrs/{repo}/'
                   f'{lock["revisions"][dataset]}/{prefix}{path.name}')
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'download checksum mismatch: {url}')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(data)
        print(f'verified {path}')


if __name__ == '__main__':
    main()
