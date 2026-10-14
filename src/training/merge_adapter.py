"""Merge a trained LoRA adapter into the base model for the serving stack.

``src.training.m0_lora_check`` saves PEFT adapters; evaluation serves full
weights through vLLM, which avoids depending on multimodal LoRA support.  The
base is loaded on CPU in BF16, the adapter is merged and written to a new
directory together with copies of the base model's companion files, and a
receipt records what went in.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
import yaml

from peft import PeftModel
from transformers import AutoModelForMultimodalLM

# Formats a merged model may be saved in; everything else is a companion file.
WEIGHT_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def is_weight(name: str) -> bool:
    """A weight shard or its index; ``config.json`` is rewritten, not copied."""
    return name.endswith('.index.json') or Path(name).suffix in WEIGHT_SUFFIXES


def merge(model_config: Path, adapter: Path, output: Path) -> dict:
    """Write the merged model to a new ``output`` and return the receipt."""
    if output.exists():
        raise SystemExit(f'refusing to overwrite {output}: it must not exist')
    model = yaml.safe_load(model_config.read_text())['model']
    base_path = Path(model['path'])
    base = AutoModelForMultimodalLM.from_pretrained(base_path, dtype=torch.bfloat16,
                                                    local_files_only=True)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    output.mkdir(parents=True)
    merged.save_pretrained(output)
    copied = []
    for source in sorted(base_path.iterdir()):
        if (not source.is_file() or is_weight(source.name)
                or source.name == 'config.json'):
            continue
        shutil.copy2(source, output / source.name)
        copied.append(source.name)
    receipt = {
        'base_model_path': str(base_path),
        'base_model_revision': model['revision'],
        'adapter': str(adapter),
        'adapter_sha256': {p.name: sha256(p) for p in sorted(adapter.iterdir())
                           if p.is_file() and p.name.startswith('adapter_')},
        'weight_files': sorted(p.name for p in output.iterdir() if is_weight(p.name)),
        'copied_files': copied,
    }
    (output / 'merge_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config', type=Path, required=True,
                        help='YAML holding the base model path and revision '
                             '(configs/base_model.yaml)')
    parser.add_argument('--adapter', type=Path, required=True,
                        help='saved PEFT adapter directory (a training checkpoint)')
    parser.add_argument('--output', type=Path, required=True,
                        help='directory to create for the merged model')
    args = parser.parse_args()
    print(json.dumps(merge(args.model_config, args.adapter, args.output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
