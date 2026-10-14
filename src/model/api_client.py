"""A metered client for an OpenAI-compatible chat API.

Every answer is stored next to its request so a rerun costs nothing, token use
is priced at the provider's peak rate, and a call that would start after the
budget is spent is refused.  The key is read from the environment file and is
never written to a log, cache or receipt.
"""

import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# US dollars per million tokens at peak hours (provider's pricing page, 2026-10-14).
PRICES = {'deepseek-flash': {'input': 0.3, 'output': 1.2},
          'deepseek-v4-pro': {'input': 1.32, 'output': 3.96}}
# Models served on this machine cost nothing per token.
LOCAL = {'qwen3.5-9b': {'input': 0.0, 'output': 0.0},
         'qwen3.5-9b-extract': {'input': 0.0, 'output': 0.0}}


class BudgetExceeded(RuntimeError):
    pass


def read_key(path: Path, name: str) -> str:
    for line in path.read_text().splitlines():
        if line.startswith(name + '='):
            return line.split('=', 1)[1].strip()
    raise KeyError(f'{name} is not set in {path}')


class ApiClient:
    def __init__(self, model: str, budget_usd: float, cache: Path,
                 key_file: Path = Path('.env'), key_name: str = 'DEEPSEEK_API_KEY',
                 url: str = 'https://api.deepseek.com/chat/completions', send=None) -> None:
        if model not in PRICES and model not in LOCAL:
            raise ValueError(f'no price recorded for {model}')
        self.model, self.budget, self.cache, self.url = model, budget_usd, cache, url
        self.local = model in LOCAL
        if self.local and not url.startswith(('http://127.0.0.1', 'http://localhost')):
            raise ValueError('a local model must be served on this machine')
        self.key = None if send or self.local else read_key(key_file, key_name)
        self.send = send or self._post
        self.spent = 0.0
        self.answers: dict[str, dict] = {}
        self.lock = threading.Lock()  # callers may send requests from several threads
        if cache.exists():
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                self.answers[row['request_sha256']] = row
                # Money already paid for cached answers still counts.
                self.spent += row['cost_usd']

    def _post(self, body: dict) -> dict:
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        request = Request(self.url, data=json.dumps(body).encode(), headers=headers)
        for attempt in range(1, 5):
            try:
                # A local server generates long answers slowly under concurrent load.
                with urlopen(request, timeout=1800 if self.local else 300) as response:
                    return json.loads(response.read())
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503) or attempt == 4:
                    raise
            except OSError:
                if attempt == 4:
                    raise
            time.sleep(10 * attempt)
        raise AssertionError('unreachable')

    def chat(self, messages: list[dict], max_tokens: int, thinking: bool = False) -> dict:
        """One completion: ``{'content', 'finish_reason', 'usage', 'cost_usd'}``."""
        body = {'model': self.model, 'messages': messages, 'max_tokens': max_tokens}
        if self.local:
            # Qwen chat templates switch thinking on and off through the template.
            body.update(temperature=0, chat_template_kwargs={'enable_thinking': thinking})
        else:
            body['thinking'] = {'type': 'enabled' if thinking else 'disabled'}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False)
                                .encode()).hexdigest()
        if digest in self.answers:
            return self.answers[digest]
        if not self.local and self.spent >= self.budget:
            raise BudgetExceeded(f'spent ${self.spent:.2f} of ${self.budget:.2f}')
        payload = self.send(body)
        usage, choice = payload['usage'], payload['choices'][0]
        price = (LOCAL if self.local else PRICES)[self.model]
        cost = (usage['prompt_tokens'] * price['input']
                + usage['completion_tokens'] * price['output']) / 1_000_000
        row = {'request_sha256': digest, 'model': self.model,
               'content': choice['message'].get('content') or '',
               'finish_reason': choice['finish_reason'],
               'usage': {k: usage[k] for k in ('prompt_tokens', 'completion_tokens')},
               'cost_usd': cost}
        with self.lock:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            with self.cache.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            self.answers[digest] = row
            self.spent += cost
        return row
