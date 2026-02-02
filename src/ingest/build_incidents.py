"""
build_incidents.py
------------------
Build processed incidents.jsonl from raw_text/*.txt and sources.csv.

Schema (per line):
  incident_id, incident_name, text, sources:[{url, retrieved_date}],
  labels(optional), evidence_gold(optional), date(optional), missing_text(optional)
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def load_labels(path: Optional[Path]) -> Dict[str, dict]:
    if not path or not path.exists():
        return {}
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {rec.get("incident_id"): rec for rec in records if rec.get("incident_id")}


def infer_date(incident_id: str, fallback: Optional[str]) -> Optional[str]:
    if fallback:
        return fallback
    match = DATE_RE.search(incident_id)
    return match.group(1) if match else None


def read_sources(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw_text", help="Directory of raw incident text files")
    ap.add_argument("--sources", default="data/sources.csv", help="CSV listing incident sources")
    ap.add_argument("--out", default="data/processed/incidents.jsonl")
    ap.add_argument(
        "--labels-from",
        default="data/processed/incidents.jsonl",
        help="Existing JSONL to copy labels/evidence from",
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    sources_path = Path(args.sources)
    out_path = Path(args.out)

    label_map = load_labels(Path(args.labels_from)) if args.labels_from else {}
    sources = read_sources(sources_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for row in sources:
        incident_id = row.get("incident_id", "").strip()
        incident_name = row.get("incident_name", "").strip() or incident_id
        text_path = raw_dir / f"{incident_id}.txt"
        text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""

        record = {
            "incident_id": incident_id,
            "incident_name": incident_name,
            "text": text,
            "sources": [],
        }

        url = (row.get("url") or "").strip()
        retrieved_date = (row.get("retrieved_date") or "").strip()
        if url:
            source_entry = {"url": url}
            if retrieved_date:
                source_entry["retrieved_date"] = retrieved_date
            record["sources"].append(source_entry)

        labels_rec = label_map.get(incident_id, {})
        if labels_rec.get("labels"):
            record["labels"] = labels_rec["labels"]
        if labels_rec.get("evidence_gold"):
            record["evidence_gold"] = labels_rec["evidence_gold"]

        date_value = infer_date(incident_id, labels_rec.get("date"))
        if date_value:
            record["date"] = date_value

        if not text:
            record["missing_text"] = True

        records.append(record)

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
