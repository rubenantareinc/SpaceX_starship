"""
split.py
--------
Create deterministic train/test split and save IDs to outputs/split.json.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional


DATE_FORMAT = "%Y-%m-%d"


def load_jsonl(path: str) -> List[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, DATE_FORMAT)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/incidents.jsonl")
    ap.add_argument("--out", default="outputs/split.json")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    records = load_jsonl(args.data)
    labeled = [r for r in records if r.get("labels") and r.get("text")]

    dated = []
    undated = []
    for rec in labeled:
        date_value = parse_date(rec.get("date"))
        if date_value:
            dated.append((date_value, rec))
        else:
            undated.append(rec)

    test_ids: List[str] = []
    train_ids: List[str] = []

    strategy = "random"
    if dated:
        strategy = "time"
        dated.sort(key=lambda x: x[0])
        ordered = [rec for _, rec in dated]
        n = len(ordered)
        test_n = max(1, int(round(n * args.test_size)))
        test_slice = ordered[-test_n:]
        train_slice = ordered[:-test_n]
        test_ids.extend([r["incident_id"] for r in test_slice])
        train_ids.extend([r["incident_id"] for r in train_slice])
        train_ids.extend([r["incident_id"] for r in undated])
    else:
        rng = __import__("random").Random(args.seed)
        ids = [r["incident_id"] for r in labeled]
        rng.shuffle(ids)
        test_n = max(1, int(round(len(ids) * args.test_size)))
        test_ids = ids[:test_n]
        train_ids = ids[test_n:]

    out = {
        "strategy": strategy,
        "seed": args.seed,
        "test_size": args.test_size,
        "n_labeled": len(labeled),
        "train": train_ids,
        "test": test_ids,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved split to {out_path}")


if __name__ == "__main__":
    main()
