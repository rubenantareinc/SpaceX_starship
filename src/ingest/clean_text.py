"""
clean_text.py
-------------
Normalize raw scraped text into a model-friendly format.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict


def normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Input JSONL from data/raw/")
    ap.add_argument("--out", required=True, help="Output JSONL to data/processed/")
    ap.add_argument("--min_chars", type=int, default=200)
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec: Dict = json.loads(line)
            rec["text"] = normalize(rec.get("text", ""))
            if len(rec["text"]) < args.min_chars:
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} records -> {out}")


if __name__ == "__main__":
    main()
