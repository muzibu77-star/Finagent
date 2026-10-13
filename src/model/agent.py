"""Exclusive single-GPU model lifetime with cancellation reaching token generation."""

import fcntl
import gc
from pathlib import Path
import time
import threading

import torch
from peft import PeftModel
from transformers import AutoTokenizer
import yaml

from src.evaluation.m0_inference_check import generate, load_model


class ModelDriver:
    def __init__(self, adapter: str | None = None):
        Path('artifacts').mkdir(exist_ok=True)
        self.mutex=threading.RLock()
        self.lock=Path('/tmp/finagent-cuda0.lock').open('a')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        try:
            self.config=yaml.safe_load(Path('configs/m0_inference.yaml').read_text())
            torch.manual_seed(0)
            self.tokenizer=AutoTokenizer.from_pretrained(self.config['model']['path'],local_files_only=True)
            self.model=load_model(self.config['model']['path'],None,self.config['model']['device'])
            if adapter:
                self.model=PeftModel.from_pretrained(self.model,adapter)
                self.model.eval()
            self.healthy=True
        except BaseException:
            self.lock.close()
            raise

    def __call__(self, messages: list, tools: list, stop_requested, deadline: float) -> dict:
        with self.mutex:
            if not self.healthy:
                raise RuntimeError('worker requires successful health check after OOM')
            row=generate(self.model,self.tokenizer,messages,tools,self.config['generation'],
                512,8192,self.config['model']['device'],
                stop_requested=lambda: stop_requested() or time.monotonic()>=deadline)
            if 'out of memory' in row.get('error','').lower():
                self.healthy=False
            return row

    def health_check(self) -> bool:
        with self.mutex:
            gc.collect()
            torch.cuda.empty_cache()
            row=generate(self.model,self.tokenizer,[{'role':'user','content':'Reply OK.'}],None,
                self.config['generation'],1,64,self.config['model']['device'])
            self.healthy='error' not in row
            return self.healthy

    def close(self) -> None:
        with self.mutex:
            if hasattr(self,'model'):
                del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
