"""
label_tool.py
-------------
Tiny CLI helper to label incidents in JSONL.

Usage:
  python -m src.labeling.label_tool --data data/processed/incidents.jsonl --schema data/schema.yaml

It will iterate through records and let you type comma-separated labels.

Tip:
  Start by labeling only subsystem/failure_mode/impact. Add cause later.
"""

import argparse
import json
from pathlib import Path

import yaml


def prompt_list(name: str, options: list) -> list:
    print(f"\n{name} options:")
    print(", ".join(options))
    raw = input(f"Enter {name} labels (comma-separated, blank for none): ").strip()
    if not raw:
        return []
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    # keep only known options
    labels = [x for x in labels if x in options]
    return labels


def main():
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
        print(f"[{i}/{len(records)}] {rec.get('incident_id')} — {rec.get('title','')}")
        print(f"Source: {rec.get('source_type')}  URL: {rec.get('source_url')}")
        print("-" * 80)
        print(rec.get("text","")[:900] + ("..." if len(rec.get("text","")) > 900 else ""))

        rec.setdefault("labels", {})
        for field in ["subsystem", "failure_mode", "impact", "cause"]:
            rec["labels"][field] = prompt_list(field, labels[field])

        cont = input("Continue? [Y/n]: ").strip().lower()
        if cont == "n":
            break

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved labeled data to {out_path}")


if __name__ == "__main__":
    main()
