"""Audit TAT-QA report links using exact TAT-DQA question bundles.

Answers are never used for linking. This does not assign training splits or
verify original PDF contents. Run from the project root with python -m.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

SPLITS = {"train": "train", "dev": "dev", "test": "test"}


def question_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Normalize whole bundles, retaining duplicate questions and punctuation."""
    if not row["questions"]:
        raise ValueError("empty question bundle")
    result = []
    for question in row["questions"]:
        value = question["question"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("question must be a nonempty string")
        result.append(" ".join(value.casefold().split()))
    return tuple(sorted(result))


def link_sources(
    qa_rows: list[tuple[str, dict[str, Any]]],
    dqa_rows: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Link only exact bundles with one source report; retain ambiguous matches."""
    index = defaultdict(list)
    for split, row in dqa_rows:
        doc = row["doc"]
        if not isinstance(doc["source"], str) or not doc["source"].strip():
            raise ValueError("missing TAT-DQA source")
        index[question_key(row)].append({
            "split": split, "document_id": doc["uid"],
            "source": doc["source"], "page": doc["page"],
        })
    records = []
    seen = set()
    for split, row in qa_rows:
        context_id = row["table"]["uid"]
        if (split, context_id) in seen:
            raise ValueError(f"duplicate context ID: {split}/{context_id}")
        seen.add((split, context_id))
        key = question_key(row)
        hits = index.get(key, [])
        sources = sorted({h["source"] for h in hits})
        matched = len(sources) == 1
        records.append({
            "dataset": "tatqa", "original_split": split,
            "context_id": context_id,
            "question_ids": [q["uid"] for q in row["questions"]],
            "question_count": len(row["questions"]),
            "source_report_id": sources[0] if matched else None,
            "status": "linked" if matched else "quarantined",
            "reason": "exact_question_bundle" if matched else (
                "ambiguous_reports" if sources else "no_exact_bundle"
            ),
            "bundle_sha256": hashlib.sha256(
                json.dumps(key, ensure_ascii=False).encode()
            ).hexdigest(),
            "matches": hits,
        })
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose report overlap; official splits do not imply source isolation."""
    report_splits = defaultdict(set)
    for row in records:
        if row["status"] == "linked":
            report_splits[row["source_report_id"]].add(row["original_split"])
            report_splits[row["source_report_id"]].update(
                match["split"] for match in row["matches"]
            )
    return {
        "contexts": len(records),
        "questions": sum(row["question_count"] for row in records),
        "statuses": dict(Counter(row["status"] for row in records)),
        "by_split": {
            split: dict(Counter(
                row["status"] for row in records if row["original_split"] == split
            )) for split in SPLITS
        },
        "unique_linked_reports": len(report_splits),
        "reports_in_multiple_official_splits": {
            source: sorted(splits) for source, splits in sorted(report_splits.items())
            if len(splits) > 1
        },
        "split_protocol_frozen": False,
        "limitations": [
            "Dataset record linkage only; original PDFs not verified.",
            "Unmatched/ambiguous records require quarantine before source splitting.",
            "No approximate matching or answer-based linkage is used.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(Path("configs/tatqa_sources.json").read_text())
    rows = {}
    assets = []
    for dataset in ("tatqa", "tatdqa"):
        rows[dataset] = []
        for split, suffix in SPLITS.items():
            name = f"{dataset}_dataset_{suffix}.json"
            path = args.raw_dir / dataset / name
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != sources[dataset]["sha256"][name]:
                raise ValueError(f"source checksum mismatch: {path}")
            parsed = json.loads(data)
            if not isinstance(parsed, list):
                raise ValueError(f"expected a list: {path}")
            rows[dataset].extend((split, row) for row in parsed)
            assets.append({
                "dataset": dataset, "split": split, "path": str(path),
                "url": sources[dataset]["base_url"] + name,
                "revision": sources[dataset]["revision"],
                "sha256": hashlib.sha256(data).hexdigest(), "contexts": len(parsed),
            })
    records = link_sources(rows["tatqa"], rows["tatdqa"])
    summary = summarize(records)
    summary["assets"] = assets
    summary["sources"] = sources
    summary["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "source_links.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({k: summary[k] for k in (
        "contexts", "questions", "statuses", "unique_linked_reports"
    )}))


if __name__ == "__main__":
    main()
