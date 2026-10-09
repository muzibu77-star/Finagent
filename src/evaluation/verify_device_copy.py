"""Fail closed if an original model tensor changes during a CUDA round trip."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = Path(yaml.safe_load(Path('configs/m0_inference.yaml').read_text())['model']['path'])
    index = json.loads((model/'model.safetensors.index.json').read_text())
    weight_name = 'lm_head.weight'
    shard = model/index['weight_map'][weight_name]
    result = {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'weight': weight_name,
        'shard': str(shard), 'shard_sha256': hashlib.sha256(shard.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'trials': []}
    with safe_open(shard, framework='pt', device='cpu') as handle:
        original = handle.get_tensor(weight_name)
        for repeat in range(3):
            device = original.to('cuda')
            torch.cuda.synchronize()
            mismatch = 0
            bad_rows = []
            for start in range(0, len(original), 4096):
                expected = original[start:start+4096]
                actual = device[start:start+4096].cpu()
                # Bit comparison catches all differences, including NaN payloads.
                count = int((expected.view(torch.int16) != actual.view(torch.int16)).sum())
                mismatch += count
                if count:
                    bad_rows.append({'row_start': start, 'different_elements': count})
            result['trials'].append({'repeat': repeat, 'different_elements': mismatch,
                                      'bad_blocks': bad_rows})
            del device
            torch.cuda.empty_cache()
    result['passed'] = all(row['different_elements'] == 0 for row in result['trials'])
    (args.output_dir/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
