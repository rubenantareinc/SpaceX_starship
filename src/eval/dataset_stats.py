"""
dataset_stats.py
----------------
Compute dataset size, label distribution, and summary stats.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/incidents.jsonl")
    ap.add_argument("--stats-out", default="outputs/dataset_stats.json")
    ap.add_argument("--labels-out", default="outputs/label_distribution.json")
    args = ap.parse_args()

    records = load_jsonl(args.data)
    labeled = [r for r in records if r.get("labels") and r.get("text")]
    with_text = [r for r in records if r.get("text")]

    lengths = [len(r.get("text", "").split()) for r in with_text]
    avg_length = sum(lengths) / len(lengths) if lengths else 0.0

    label_counts: Dict[str, Dict[str, int]] = {}
    label_cardinality = []
    for rec in labeled:
        total_labels = 0
        for field, labels in rec.get("labels", {}).items():
            label_counts.setdefault(field, {})
            for label in labels:
                label_counts[field][label] = label_counts[field].get(label, 0) + 1
            total_labels += len(labels)
        label_cardinality.append(total_labels)

    stats = {
        "n_total": len(records),
        "n_with_text": len(with_text),
        "n_labeled": len(labeled),
        "avg_length_tokens": avg_length,
        "avg_label_cardinality": sum(label_cardinality) / len(label_cardinality) if label_cardinality else 0.0,
    }

    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    Path(args.labels_out).write_text(json.dumps(label_counts, indent=2), encoding="utf-8")

    print(f"Saved dataset stats -> {args.stats_out}")
    print(f"Saved label distribution -> {args.labels_out}")


if __name__ == "__main__":
    main()
