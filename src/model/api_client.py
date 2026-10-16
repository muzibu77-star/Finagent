"""A metered client for an OpenAI-compatible chat API.

Every answer is stored next to its request so a rerun costs nothing, token use
is priced at the provider's peak rate, and a call that would start after the
budget is spent is refused.  The key is read from the environment file and is
never written to a log, cache or receipt.

A caller may ask for the text of the answer as it is written (``on_delta``).
Only a local model can stream, and only without tools, so the requests of the
flash model and of the tool agents are the ones of before; the cache key is the
request body without its streaming fields, so a streamed answer and a plain one
are the same request, the same cache row and the same charge.
"""

import hashlib
from http.client import HTTPException
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
         'qwen3.5-9b-extract': {'input': 0.0, 'output': 0.0},
         'qwen3.6-27b': {'input': 0.0, 'output': 0.0}}
# An already-cached answer is reported to a streaming caller in pieces of this
# many characters, so the page still sees text arriving instead of one lump.
DELTA_CHARS = 24


def pieces(text: str) -> list[str]:
    """One text as the segments a streaming caller receives."""
    return [text[at:at + DELTA_CHARS] for at in range(0, len(text), DELTA_CHARS)]


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
                 url: str = 'https://api.deepseek.com/chat/completions', send=None,
                 stream=None) -> None:
        if model not in PRICES and model not in LOCAL:
            raise ValueError(f'no price recorded for {model}')
        self.model, self.budget, self.cache, self.url = model, budget_usd, cache, url
        self.local = model in LOCAL
        if self.local and not url.startswith(('http://127.0.0.1', 'http://localhost')):
            raise ValueError('a local model must be served on this machine')
        self.key = None if send or self.local else read_key(key_file, key_name)
        self.send = send or self._post
        self.stream = stream or self._post_stream
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
                    payload = json.loads(response.read())
                # Under load the provider occasionally answers 200 with an error body.
                if 'choices' in payload and 'usage' in payload:
                    return payload
                if attempt == 4:
                    raise RuntimeError(f"no answer in reply: {str(payload.get('error'))[:200]}")
            except HTTPError as error:
                # 405 comes from the outbound proxy under load, not from the provider.
                if error.code not in (405, 429, 500, 502, 503) or attempt == 4:
                    raise
            except (OSError, HTTPException):  # dropped connections, timeouts, truncated replies
                if attempt == 4:
                    raise
            time.sleep(10 * attempt)
        raise AssertionError('unreachable')

    def _post_stream(self, body: dict, on_delta) -> dict:
        """Send one streaming request and answer in the shape of a plain reply.

        Every text segment of the stream goes to ``on_delta`` as it arrives; the
        assembled answer is the same payload a non-streaming reply would carry,
        so the caller prices, caches and reads it exactly the same way.  A
        dropped connection is not retried: part of the text has already been
        reported and replaying it would show the same words twice.
        """
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        request = Request(self.url, data=json.dumps(body).encode(), headers=headers)
        text: list[str] = []
        usage, finish = None, 'stop'
        with urlopen(request, timeout=1800) as response:
            for line in response:
                line = line.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[len('data:'):].strip()
                if data == '[DONE]':
                    break
                chunk = json.loads(data)
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices') or []:
                    delta = (choice.get('delta') or {}).get('content')
                    if delta:
                        text.append(delta)
                        on_delta(delta)
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
        return {'usage': usage or {'prompt_tokens': 0, 'completion_tokens': 0},
                'choices': [{'message': {'content': ''.join(text)}, 'finish_reason': finish}]}

    def chat(self, messages: list[dict], max_tokens: int, thinking: bool = False,
             tools: list | None = None, on_delta=None) -> dict:
        """One completion: ``{'content', 'tool_calls', 'finish_reason', 'usage', 'cost_usd'}``.

        ``tools`` is an OpenAI-format tool list; it is left out of the request
        body entirely when empty or omitted, so requests made without tools
        keep the same hash and their answers keep hitting the existing cache.

        ``on_delta`` asks for the text of the answer as it is written.  A local
        model without tools is asked to stream; any other request is sent
        unchanged and its whole answer is reported in segments afterwards, as a
        cached answer is.  The callback changes nothing else: the request body
        without streaming keeps the same cache key, and a caller that passes no
        callback sends the very same request and reads the very same row.
        """
        body = {'model': self.model, 'messages': messages, 'max_tokens': max_tokens}
        if self.local:
            # Qwen chat templates switch thinking on and off through the template.
            body.update(temperature=0, chat_template_kwargs={'enable_thinking': thinking})
        else:
            body['thinking'] = {'type': 'enabled' if thinking else 'disabled'}
        if tools:
            body['tools'] = tools
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False)
                                .encode()).hexdigest()
        if digest in self.answers:
            if on_delta is not None:
                for piece in pieces(self.answers[digest]['content']):
                    on_delta(piece)
            return self.answers[digest]
        if not self.local and self.spent >= self.budget:
            raise BudgetExceeded(f'spent ${self.spent:.2f} of ${self.budget:.2f}')
        if on_delta is not None and self.local and not tools:
            payload = self.stream({**body, 'stream': True,
                                   'stream_options': {'include_usage': True}}, on_delta)
        else:
            payload = self.send(body)
            if on_delta is not None:
                for piece in pieces(payload['choices'][0]['message'].get('content') or ''):
                    on_delta(piece)
        usage, choice = payload['usage'], payload['choices'][0]
        price = (LOCAL if self.local else PRICES)[self.model]
        cost = (usage['prompt_tokens'] * price['input']
                + usage['completion_tokens'] * price['output']) / 1_000_000
        row = {'request_sha256': digest, 'model': self.model,
               'content': choice['message'].get('content') or '',
               'tool_calls': choice['message'].get('tool_calls'),
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
