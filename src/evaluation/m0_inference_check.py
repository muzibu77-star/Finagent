"""M0 step 1: check text inference and tool calling for one model variant.

Loads one variant from the config (plain BF16 weights or a bitsandbytes
quantization), runs the fixed smoke tasks several times plus a prompt-length
probe, and writes per-generation records and a summary. One variant per process
keeps peak-memory numbers clean. Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("m0_inference_check")

# Tool-call format defined by the model's chat template:
# <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
JSON_TYPES = {"number": (int, float), "array": list, "object": dict, "boolean": bool}
PROBE_SENTENCE = (
    "Revenue for the period was reported in the consolidated statement of income. "
)
PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "peft", "tokenizers")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_tool_calls(
    text: str, schemas: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse tool calls from model text and validate them against the schemas.

    The template writes string arguments verbatim and everything else as JSON,
    so non-string parameters are decoded with ``json.loads``. Returns the parsed
    calls and a list of problems; an empty problem list means every call parsed.
    """
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    matches = TOOL_CALL_RE.findall(text)
    if len(matches) != text.count("<tool_call>"):
        errors.append("malformed tool_call block")
    for name, body in matches:
        schema = schemas.get(name)
        if schema is None:
            errors.append(f"unknown tool: {name}")
            continue
        properties = schema["properties"]
        arguments: dict[str, Any] = {}
        if PARAMETER_RE.sub("", body).strip():
            errors.append(f"{name}: malformed parameter block")
        seen: set[str] = set()
        for key, raw in PARAMETER_RE.findall(body):
            if key in seen:
                errors.append(f"{name}: duplicate parameter {key}")
                continue
            seen.add(key)
            if key not in properties:
                errors.append(f"{name}: unknown parameter {key}")
                continue
            expected_type = properties[key]["type"]
            if expected_type == "string":
                value: Any = raw
            else:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    errors.append(f"{name}.{key}: not valid JSON: {raw!r}")
                    continue
                if not isinstance(value, JSON_TYPES[expected_type]):
                    errors.append(f"{name}.{key}: expected {expected_type}")
                    continue
            if expected_type == "array" and properties[key].get("items") == {"type": "number"}:
                if any(
                    type(item) not in (int, float)
                    or (isinstance(item, float) and not math.isfinite(item))
                    for item in value
                ):
                    errors.append(f"{name}.{key}: expected finite numeric items")
                    continue
            allowed = properties[key].get("enum")
            if allowed and value not in allowed:
                errors.append(f"{name}.{key}: {value!r} not in enum")
                continue
            arguments[key] = value
        missing = [key for key in schema["required"] if key not in arguments]
        if missing:
            errors.append(f"{name}: missing or invalid required {missing}")
        calls.append({"name": name, "arguments": arguments})
    return calls, errors


def meets_expectation(
    expect: dict[str, Any], answer: str, calls: list[dict[str, Any]]
) -> bool:
    """Return whether the answer and parsed calls satisfy a task expectation."""
    if expect.get("no_tool") and calls:
        return False
    if "tool" in expect:
        if len(calls) != 1 or calls[0]["name"] != expect["tool"]:
            return False
        arguments = calls[0]["arguments"]
        for key, value in expect.get("args_equal", {}).items():
            if arguments.get(key) != value:
                return False
        for key, fragment in expect.get("args_contains", {}).items():
            if fragment not in str(arguments.get(key, "")).lower():
                return False
    wanted = expect.get("contains_any")
    if wanted and not any(item.lower() in answer.lower() for item in wanted):
        return False
    return True


def repeats_identical(
    records: list[dict[str, Any]], task_ids: list[str], repeats: int
) -> bool:
    """Require every requested repeat exactly once, with identical output."""
    if not task_ids or repeats < 2 or len(records) != len(task_ids) * repeats:
        return False
    for task_id in task_ids:
        rows = [r for r in records if r["id"] == task_id]
        if len(rows) != repeats or {r["repeat"] for r in rows} != set(range(repeats)):
            return False
        if any("error" in r for r in rows) or len({r["text"] for r in rows}) != 1:
            return False
    return True


def load_model(path: str, quantization: str | None, device: str) -> Any:
    """Load the model on one device; a failure to fit is raised, not offloaded."""
    kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": {"": device}}
    if quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization is not None:
        raise ValueError(f"unsupported quantization: {quantization}")
    model = AutoModelForMultimodalLM.from_pretrained(path, **kwargs)
    model.eval()
    return model


def memory_mib(device: str) -> dict[str, float]:
    """Peak torch memory since the last reset and current device-wide usage."""
    free, total = torch.cuda.mem_get_info(device)
    return {
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
        "device_used_mib": round((total - free) / 2**20, 1),
    }


def generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    generation: dict[str, Any],
    max_new_tokens: int,
    max_input_tokens: int | None,
    device: str,
) -> dict[str, Any]:
    """Run one generation and return its text, token counts, latency and memory.

    A prompt above ``max_input_tokens`` is reported as an error instead of being
    truncated, so evidence is never silently dropped.
    """
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=generation["enable_thinking"],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_tokens = encoded["input_ids"].shape[1]
    record: dict[str, Any] = {"prompt_tokens": prompt_tokens}
    if max_input_tokens is not None and prompt_tokens > max_input_tokens:
        record["error"] = "input_over_budget"
        return record

    encoded = encoded.to(device)
    stop_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=generation["do_sample"],
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
    except RuntimeError as exc:  # includes torch.OutOfMemoryError
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        record.update(memory_mib(device))
        torch.cuda.empty_cache()
        return record
    torch.cuda.synchronize(device)
    latency = time.perf_counter() - started

    new_ids = output[0, prompt_tokens:]
    record.update(memory_mib(device))
    record.update(
        {
            "new_tokens": int(new_ids.shape[0]),
            "latency_s": round(latency, 3),
            "tokens_per_s": round(new_ids.shape[0] / latency, 2),
            "stopped_on_eos": bool(new_ids[-1].item() in stop_ids),
            "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        }
    )
    return record


def probe_messages(tokenizer: Any, target_tokens: int) -> list[dict[str, str]]:
    """Build a filler prompt of roughly ``target_tokens`` tokens."""
    per_sentence = len(tokenizer(PROBE_SENTENCE)["input_ids"])
    filler = PROBE_SENTENCE * (target_tokens // per_sentence)
    return [{"role": "user", "content": filler + "\nSummarize the text in one line."}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m0_inference.yaml")
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    config_text = Path(args.config).read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    quantization = config["variants"][args.variant]["quantization"]
    model_cfg, generation, run = config["model"], config["generation"], config["run"]
    device = model_cfg["device"]

    tools = json.loads(Path(config["paths"]["tools"]).read_text(encoding="utf-8"))
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    task_lines = Path(config["paths"]["tasks"]).read_text(encoding="utf-8").splitlines()
    tasks = [json.loads(line) for line in task_lines if line.strip()]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(config["paths"]["output_dir"]) / args.variant / stamp
    out_dir.mkdir(parents=True)
    logger.info("variant=%s output=%s", args.variant, out_dir)

    torch.manual_seed(run["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"])
    summary: dict[str, Any] = {
        "variant": args.variant,
        "quantization": quantization,
        "model": model_cfg,
        "generation": generation,
        "run": run,
        "config_sha256": sha256_text(config_text),
        "source_sha256": sha256_text(Path(__file__).read_text(encoding="utf-8")),
        "tasks_sha256": sha256_text(Path(config["paths"]["tasks"]).read_text()),
        "tools_sha256": sha256_text(Path(config["paths"]["tools"]).read_text()),
        "chat_template_sha256": sha256_text(tokenizer.chat_template),
        "versions": {name: metadata.version(name) for name in PACKAGES},
        "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(device),
        "started_utc": stamp,
    }

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        model = load_model(model_cfg["path"], quantization, device)
    except (RuntimeError, ValueError, ImportError) as exc:
        summary["load_error"] = f"{type(exc).__name__}: {exc}"[:1000]
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.error("model load failed: %s", summary["load_error"])
        return 1
    summary["load_seconds"] = round(time.perf_counter() - started, 1)
    summary["memory_after_load"] = memory_mib(device)
    logger.info("loaded in %ss %s", summary["load_seconds"], summary["memory_after_load"])

    records: list[dict[str, Any]] = []
    total = run["repeats"] * len(tasks)
    for repeat in range(run["repeats"]):
        for task in tasks:
            record = generate(
                model,
                tokenizer,
                task["messages"],
                tools if task["use_tools"] else None,
                generation,
                generation["max_new_tokens"],
                generation["max_input_tokens"],
                device,
            )
            record.update({"kind": "task", "id": task["id"], "repeat": repeat})
            if "error" not in record:
                answer = record["text"].split("</think>")[-1].strip()
                calls, parse_errors = parse_tool_calls(answer, schemas)
                record["tool_calls"] = calls
                record["parse_errors"] = parse_errors
                record["task_ok"] = (
                    not parse_errors
                    and record["stopped_on_eos"]
                    and meets_expectation(task["expect"], answer, calls)
                )
            records.append(record)
            logger.info(
                "[%d/%d] %s repeat=%d task_ok=%s error=%s latency=%s",
                len(records),
                total,
                task["id"],
                repeat,
                record.get("task_ok"),
                record.get("error"),
                record.get("latency_s"),
            )

    probes: list[dict[str, Any]] = []
    for target in run["budget_probe_input_tokens"]:
        record = generate(
            model,
            tokenizer,
            probe_messages(tokenizer, target),
            None,
            generation,
            run["budget_probe_new_tokens"],
            None,
            device,
        )
        record.update({"kind": "probe", "target_tokens": target})
        record.pop("text", None)
        probes.append(record)
        logger.info("probe %s", record)

    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records + probes:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    completed = [r for r in records if "error" not in r]
    summary.update(
        {
            "generations": len(records),
            "errors": [
                {"id": r["id"], "repeat": r["repeat"], "error": r["error"]}
                for r in records
                if "error" in r
            ],
            "task_ok": sum(r["task_ok"] for r in completed),
            "task_failed_ids": sorted({r["id"] for r in completed if not r["task_ok"]}),
            "tool_parse_error_ids": sorted(
                {r["id"] for r in completed if r["parse_errors"]}
            ),
            "hit_max_new_tokens_ids": sorted(
                {r["id"] for r in completed if not r["stopped_on_eos"]}
            ),
            "identical_across_repeats": repeats_identical(
                records, [task["id"] for task in tasks], run["repeats"]
            ),
            "probes": probes,
        }
    )
    if completed:
        summary.update(
            {
                "latency_s_median": statistics.median(r["latency_s"] for r in completed),
                "latency_s_max": max(r["latency_s"] for r in completed),
                "tokens_per_s_median": statistics.median(
                    r["tokens_per_s"] for r in completed
                ),
                "peak_allocated_mib": max(r["peak_allocated_mib"] for r in completed),
                "peak_reserved_mib": max(r["peak_reserved_mib"] for r in completed),
            }
        )
    summary["passed"] = (
        summary["task_ok"] == total
        and not summary["errors"]
        and summary["identical_across_repeats"]
        and not any("error" in probe for probe in probes)
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("summary written to %s", out_dir / "summary.json")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
