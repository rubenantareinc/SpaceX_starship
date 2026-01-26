"""
predict.py
----------
Run trained models on a JSONL file and emit predictions + evidence sentences.

Evidence strategy:
- Split into sentences
- For each predicted label, take the top-k sentences containing any keyword from a small label->keyword map
  (you can replace with attention/gradient rationales later)

This keeps the demo grounded.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


EVIDENCE_KWS = {
    "heat_shield": ["heat shield", "tiles", "thermal"],
    "raptor_engine": ["raptor", "engine"],
    "stage_separation": ["separation", "staging", "hot-staging"],
    "range_safety": ["FTS", "flight termination", "range safety"],
    "loss_of_control": ["tumble", "lost control", "attitude"],
    "comms_loss": ["telemetry", "communications", "signal"],
    "leak": ["leak"],
    "fire": ["fire", "flames"],
    "explosion": ["explosion"],
}


def split_sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def pick_evidence(text: str, labels: List[str], k: int = 3) -> Dict[str, List[str]]:
    sents = split_sentences(text)
    out = {}
    for lab in labels:
        kws = EVIDENCE_KWS.get(lab, [lab.replace("_", " ")])
        scored = []
        for s in sents:
            score = sum(1 for kw in kws if kw.lower() in s.lower())
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        out[lab] = [s for _, s in scored[:k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL incidents")
    ap.add_argument("--schema", default="data/schema.yaml")
    ap.add_argument("--model_dir", required=True, help="Root output_dir from train script (contains per-field folders)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    label_space = schema["labels"]

    records = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    model_root = Path(args.model_dir)

    pred_out = Path(args.out)
    pred_out.parent.mkdir(parents=True, exist_ok=True)

    # load per-field models
    models = {}
    tokenizers = {}
    for field in ["subsystem", "failure_mode", "impact", "cause"]:
        path = model_root / field / "best"
        if path.exists():
            models[field] = AutoModelForSequenceClassification.from_pretrained(str(path))
            tokenizers[field] = AutoTokenizer.from_pretrained(str(path))
            models[field].eval()

    with pred_out.open("w", encoding="utf-8") as f:
        for rec in records:
            out = {"incident_id": rec["incident_id"], "pred": {}, "confidence": {}, "evidence": {}}
            text = rec["text"]

            for field, model in models.items():
                tokenizer = tokenizers[field]
                enc = tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**enc).logits.squeeze(0).cpu().numpy()
                probs = 1 / (1 + np.exp(-logits))
                labels = label_space[field]
                picked = [lab for lab, p in zip(labels, probs) if p >= args.threshold]
                conf = {lab: float(p) for lab, p in zip(labels, probs) if lab in picked}

                out["pred"][field] = picked
                out["confidence"][field] = conf
                out["evidence"][field] = pick_evidence(text, picked)

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote predictions -> {pred_out}")


if __name__ == "__main__":
    main()
