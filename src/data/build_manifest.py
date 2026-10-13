"""Stage visible evidence, questions and gold separately; audit report splits.

Conservative protocol: holdout source precedence is test > dev > train. Unknown
TAT sources and FinQA 2019 cross-dataset aliases remain quarantined. This does
uses a frozen 5-token Jaccard protocol and isolates unresolved report aliases.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

from src.data.tatqa_source_audit import SPLITS, link_sources
from src.data.leakage import content_text, near_pairs

PRIORITY = {"train": 0, "dev": 1, "test": 2}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def financial_source(filename: str) -> str:
    parts = filename.split("/")
    if len(parts) != 3 or not parts[1].isdigit() or len(parts[1]) != 4:
        raise ValueError(f"unrecognized source filename: {filename}")
    return "/".join(parts[:2])


def visible_evidence(dataset: str, row: dict) -> dict:
    """Use an explicit allowlist; never copy model_input/retrieval/gold fields."""
    if dataset == "finqa":
        return {"table": row["table"], "pre_text": row["pre_text"],
                "post_text": row["post_text"]}
    return {"table": row["table"]["table"], "paragraphs": [
        {"uid": p["uid"], "order": p["order"], "text": p["text"]}
        for p in row["paragraphs"]
    ]}


def source_owner(splits: set[str]) -> str:
    return max(splits, key=PRIORITY.__getitem__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_dir
    assets = []
    lock = json.loads(Path("configs/m0_data_lock.json").read_text())
    for name, expected in lock["sha256"].items():
        path = raw / Path(name).relative_to("data/raw")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"raw input checksum mismatch: {path}")

    def read(path: Path) -> list:
        data = path.read_bytes()
        assets.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        return json.loads(data)

    tat = [(split, row) for split, suffix in SPLITS.items()
           for row in read(raw / "tatqa" / f"tatqa_dataset_{suffix}.json")]
    dqa = [(split, row) for split, suffix in SPLITS.items()
           for row in read(raw / "tatdqa" / f"tatdqa_dataset_{suffix}.json")]
    links = {(r["original_split"], r["context_id"]): r
             for r in link_sources(tat, dqa)}
    reports = defaultdict(set)
    for split, row in dqa:
        reports["tat:" + row["doc"]["source"]].add(split)
    zip_path = raw / "convfinqa" / "data.zip"
    assets.append({"path": str(zip_path),
                   "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest()})
    with zipfile.ZipFile(zip_path) as archive:
        for split, name in (("train", "train"), ("dev", "dev"),
                            ("test", "test_private")):
            for row in json.loads(archive.read(f"data/{name}.json")):
                reports["fin:" + financial_source(row["filename"])].add(split)

    pending = []
    for dataset in ("finqa", "tatqa"):
        rows = tat if dataset == "tatqa" else [
            (split, row) for split in SPLITS
            for row in read(raw / "finqa" / f"{split}.json")
        ]
        for split, row in rows:
            if dataset == "finqa":
                source = "fin:" + financial_source(row["filename"])
                context_id = row["id"]
                questions = [{**row["qa"], "uid": row["id"]}]
                reason = "unresolved_2019_company_alias" if source.endswith("/2019") else None
            else:
                context_id = row["table"]["uid"]
                link = links[(split, context_id)]
                source = "tat:" + link["source_report_id"] if link["source_report_id"] else None
                questions = row["questions"]
                reason = None if source else link["reason"]
            if source:
                reports[source].add(split)
            evidence = visible_evidence(dataset, row)
            pending.append({
                "dataset": dataset, "original_split": split,
                "context_id": context_id, "source_report_id": source,
                "quarantine_reason": reason, "evidence": evidence,
                "questions": questions, "content_hash": digest(content_text(evidence)),
            })
    # Exact visible context plus question, never answers, reconciles released gold.
    released = read(raw / "tatqa" / "tatqa_dataset_test_gold.json")
    label_index = defaultdict(list)
    for row in released:
        key = content_text(visible_evidence("tatqa", row))
        for question in row["questions"]:
            label_index[(key, " ".join(question["question"].casefold().split()))].append(question)
    aligned_gold = 0
    for row in pending:
        if row["dataset"] == "tatqa" and row["original_split"] == "test":
            for question in row["questions"]:
                key = (content_text(row["evidence"]),
                       " ".join(question["question"].casefold().split()))
                hits = label_index.get(key, [])
                if len(hits) == 1:
                    for field in ("answer", "derivation", "answer_type", "scale",
                                  "answer_from", "rel_paragraphs"):
                        if field in hits[0]:
                            question[field] = hits[0][field]
                    aligned_gold += 1
    print("Auditing exact and near duplicates", flush=True)
    pairs = near_pairs([content_text(row["evidence"]) for row in pending])
    near_sources = set()
    cross_pairs = []
    for a, b, similarity in pairs:
        first, second = pending[a], pending[b]
        if first["source_report_id"] != second["source_report_id"]:
            near_sources.update(s for s in (
                first["source_report_id"], second["source_report_id"]
            ) if s)
            cross_pairs.append({"first": first["context_id"],
                                "second": second["context_id"], "jaccard": similarity})
    hash_sources = defaultdict(set)
    for row in pending:
        if row["source_report_id"]:
            hash_sources[row["content_hash"]].add(row["source_report_id"])
    duplicate_sources = {source for sources in hash_sources.values()
                         if len(sources) > 1 for source in sources}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    seen_ids = set()
    with (
        (args.output_dir / "evidence.jsonl").open("w") as evidence_file,
        (args.output_dir / "questions.jsonl").open("w") as questions_file,
        (args.output_dir / "gold.jsonl").open("w") as gold_file,
        (args.output_dir / "manifest.jsonl").open("w") as manifest_file,
    ):
        def write(handle, record: dict) -> None:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        for row in pending:
            source = row["source_report_id"]
            reason = row["quarantine_reason"]
            if source in duplicate_sources:
                reason = reason or "identical_evidence_across_reports"
            if source in near_sources:
                reason = reason or "near_duplicate_across_reports"
            group = None if reason else source_owner(reports[source])
            context_id = f'{row["dataset"]}:{row["original_split"]}:{row["context_id"]}'
            write(evidence_file, {"evidence_id": context_id, **row["evidence"]})
            for question in row["questions"]:
                task_id = f'{row["dataset"]}:{row["original_split"]}:{question["uid"]}'
                if task_id in seen_ids:
                    raise ValueError(f"duplicate task: {task_id}")
                seen_ids.add(task_id)
                write(questions_file, {"task_id": task_id,
                    "evidence_id": context_id, "question": question["question"]})
                gold = {k: question[k] for k in (
                    "answer", "exe_ans", "program", "derivation", "answer_type",
                    "answer_from", "scale", "gold_inds", "rel_paragraphs"
                ) if k in question}
                write(gold_file, {"task_id": task_id, "labels_available": "answer" in gold,
                                  "gold": gold})
                eligible = bool(group and group == row["original_split"])
                write(manifest_file, {"task_id": task_id, "evidence_id": context_id,
                    "dataset": row["dataset"], "original_split": row["original_split"],
                    "source_report_id": source, "content_hash": row["content_hash"],
                    "group_split": group, "official_eligible": eligible,
                    "revision": lock["revisions"][row["dataset"]],
                    "language": "en", "modality": "table_text", "parent_ids": [],
                    "license": "MIT (repository)" if row["dataset"] == "finqa" else "CC-BY-4.0",
                    "company": (source.split(":", 1)[1].split("/")[0] if source and source.startswith("fin:")
                                else source[4:].rsplit("_", 1)[0] if source else None),
                    "report_period": (source.rsplit("/", 1)[1] if source and source.startswith("fin:")
                                      else source.rsplit("_", 1)[1].removesuffix(".pdf") if source else None),
                    "exclusion_reason": reason or (None if eligible else "heldout_report_overlap"),
                })
                counts[f'{row["dataset"]}/{row["original_split"]}/total'] += 1
                counts[f'{row["dataset"]}/{row["original_split"]}/' + (
                    "eligible" if eligible else "excluded")] += 1
    summary = {"counts": dict(sorted(counts.items())), "assets": assets,
        "split_protocol_frozen": True,
        "protocols": {
            "official_filtered_v1": "Keep original split, exclude report conflicts; not full official leaderboard.",
            "report_group_v1": "Assign retained reports to highest original holdout priority test > dev > train.",
            "near_duplicates": "5-token shingles, Jaccard >= 0.9; quarantine cross-report pairs' reports.",
            "unresolved_sources": "Quarantine unknown TAT sources and FinQA 2019 aliases.",
            "test_labels": "Exact context+question match only; unaligned questions are unscored.",
        },
        "aligned_tatqa_test_labels": aligned_gold,
        "cross_report_near_pairs": cross_pairs,
        "remaining_checks": [],
        "revisions": lock["revisions"],
        "licenses": {"finqa": "MIT (repository)", "convfinqa": "MIT (repository)",
                     "tatqa": "CC-BY-4.0", "tatdqa": "CC-BY-4.0"},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
