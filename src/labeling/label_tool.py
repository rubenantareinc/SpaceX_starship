"""
label_tool.py
-------------
CLI helper to label incidents in JSONL with multi-labels + evidence sentences.

Usage:
  python -m src.labeling.label_tool --data data/processed/incidents.jsonl --schema data/schema.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from src.utils import split_sentences


def prompt_list(name: str, options: list, existing: list | None = None) -> list:
    existing = existing or []
    print(f"\n{name} options:")
    print(", ".join(options))
    prompt = f"Enter {name} labels (comma-separated, blank to keep {existing}): "
    raw = input(prompt).strip()
    if not raw:
        return existing
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    labels = [x for x in labels if x in options]
    return labels


def prompt_evidence(label: str, sentences: list[str], existing: list[int] | None = None) -> list[int]:
    existing = existing or []
    if not sentences:
        return []
    print(f"Select evidence sentence indices for label '{label}'.")
    raw = input(f"Indices (comma-separated, blank to keep {existing}): ").strip()
    if not raw:
        return existing
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(sentences):
                indices.append(idx)
    return sorted(set(indices))


def show_sentences(sentences: list[str]) -> None:
    print("\nSentences:")
    for idx, sent in enumerate(sentences):
        print(f"  [{idx}] {sent}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL with incidents")
    ap.add_argument("--schema", default="data/schema.yaml")
    ap.add_argument("--out", default=None, help="Write labeled JSONL here (default overwrites --data)")
    args = ap.parse_args()

    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    labels = schema["labels"]

    data_path = Path(args.data)
    out_path = Path(args.out) if args.out else data_path

    records = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(records)} records from {data_path}")

    for i, rec in enumerate(records, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(records)}] {rec.get('incident_id')} — {rec.get('incident_name','')}")
        sources = rec.get("sources", [])
        if sources:
            print(f"Source: {sources[0].get('url','')}  Retrieved: {sources[0].get('retrieved_date','')}")
        print("-" * 80)
        text = rec.get("text", "")
        if not text:
            print("[No text available for this incident. Skipping labeling.]")
            continue
        print(text[:900] + ("..." if len(text) > 900 else ""))

        sentences = split_sentences(text)
        show_sentences(sentences)

        rec.setdefault("labels", {})
        rec.setdefault("evidence_gold", {})

        for field in ["subsystem", "failure_mode", "impact", "cause"]:
            rec["labels"][field] = prompt_list(field, labels[field], rec["labels"].get(field, []))
            rec["evidence_gold"].setdefault(field, {})
            for label in rec["labels"][field]:
                existing = rec["evidence_gold"][field].get(label, [])
                rec["evidence_gold"][field][label] = prompt_evidence(label, sentences, existing)

        cont = input("Continue? [Y/n]: ").strip().lower()
        if cont == "n":
            break

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved labeled data to {out_path}")


if __name__ == "__main__":
    main()
