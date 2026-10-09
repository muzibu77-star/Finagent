"""Local vLLM completions using the same tokenizer and raw tool-call contract."""

import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit

from transformers import AutoTokenizer
import yaml


class HttpDriver:
    """A client only: server GPU ownership and process lifecycle stay explicit."""

    def __init__(self, url: str, model: str):
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('research serving client requires a loopback server')
        self.url, self.served_model = url.rstrip('/'), model
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.config = yaml.safe_load(Path('configs/m0_inference.yaml').read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['path'], local_files_only=True)
        with self.opener.open(self.url+'/v1/models', timeout=10) as response:
            models = json.load(response)
        if model not in {row['id'] for row in models['data']}:
            raise ValueError('requested served model is not loaded')

    def generate(self, messages: list, tools: list | None = None,
                 max_new_tokens: int = 512, max_input_tokens: int = 8192,
                 timeout: float = 180, *, temperature: float = 0,
                 seed: int = 0) -> dict:
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools,
            add_generation_prompt=True, enable_thinking=False, tokenize=True,
            return_dict=False)
        if len(prompt) > max_input_tokens:
            return {'prompt_tokens': len(prompt), 'error': 'input_over_budget'}
        payload = {'model': self.served_model, 'prompt': prompt, 'max_tokens': max_new_tokens,
            'temperature': temperature, 'seed': seed, 'stop_token_ids':
            [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id], 'skip_special_tokens': True}
        request = urllib.request.Request(self.url+'/v1/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=max(0.1, timeout)) as response:
                result = json.load(response)
        except (OSError, ValueError) as exc:
            return {'prompt_tokens': len(prompt), 'latency_s': time.monotonic()-started,
                    'error': f'{type(exc).__name__}: {exc}', 'backend': 'vllm'}
        choice = result['choices'][0]
        return {'text': choice['text'], 'prompt_tokens': len(prompt),
            'new_tokens': result['usage']['completion_tokens'],
            'stopped_on_eos': choice['finish_reason'] == 'stop',
            'latency_s': time.monotonic()-started, 'backend': 'vllm',
            'served_model': self.served_model}

    def __call__(self, messages, tools, stop_requested, deadline):
        if stop_requested() or time.monotonic() >= deadline:
            return {'error': 'cancelled_before_request'}
        return self.generate(messages, tools, timeout=deadline-time.monotonic())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None
