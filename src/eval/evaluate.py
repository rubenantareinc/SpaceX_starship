"""
evaluate.py
-----------
Evaluate predictions vs gold labels.

Gold file: incidents.jsonl with labels.* fields
Pred file: predictions JSONL with pred.* fields (from baselines or transformer)

Outputs:
- metrics.json
- optional confusion matrix per field (binary) is non-trivial for multi-label;
  we export per-label precision/recall/f1 instead.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score


def load_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def binarize(label_list: List[str], labels: List[List[str]]):
    idx = {l: i for i, l in enumerate(label_list)}
    Y = np.zeros((len(labels), len(label_list)), dtype=int)
    for r, labs in enumerate(labels):
        for lab in labs:
            if lab in idx:
                Y[r, idx[lab]] = 1
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--schema", default="data/schema.yaml")
    ap.add_argument("--out", default="outputs/metrics.json")
    args = ap.parse_args()

    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    label_space = schema["labels"]

    gold = {r["incident_id"]: r for r in load_jsonl(args.gold)}
    pred = {r["incident_id"]: r for r in load_jsonl(args.pred)}

    fields = ["subsystem", "failure_mode", "impact", "cause"]
    report = {"n": len(gold), "fields": {}}

    for field in fields:
        labels = label_space[field]
        y_true = []
        y_pred = []
        for inc_id, g in gold.items():
            if inc_id not in pred:
                continue
            y_true.append(g.get("labels", {}).get(field, []))
            y_pred.append(pred[inc_id].get("pred", {}).get(field, []))

        if not y_true:
            continue

        Yt = binarize(labels, y_true)
        Yp = binarize(labels, y_pred)

        micro_f1 = f1_score(Yt, Yp, average="micro", zero_division=0)
        macro_f1 = f1_score(Yt, Yp, average="macro", zero_division=0)
        micro_p = precision_score(Yt, Yp, average="micro", zero_division=0)
        micro_r = recall_score(Yt, Yp, average="micro", zero_division=0)

        per_label = {}
        for i, lab in enumerate(labels):
            per_label[lab] = {
                "precision": float(precision_score(Yt[:, i], Yp[:, i], zero_division=0)),
                "recall": float(recall_score(Yt[:, i], Yp[:, i], zero_division=0)),
                "f1": float(f1_score(Yt[:, i], Yp[:, i], zero_division=0)),
                "support": int(Yt[:, i].sum()),
            }

        report["fields"][field] = {
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "per_label": per_label,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
