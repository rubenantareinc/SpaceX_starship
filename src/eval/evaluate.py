"""
evaluate.py
-----------
Evaluate predictions vs gold labels (multi-label).

Outputs:
- metrics.json
- metrics.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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


def filter_ids(records: Dict[str, dict], ids: Optional[List[str]]) -> Dict[str, dict]:
    if not ids:
        return records
    return {k: v for k, v in records.items() if k in ids}


def load_split(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics_to_markdown(report: dict) -> str:
    lines = ["# Classification Metrics", ""]
    lines.append(f"Total evaluated incidents: {report.get('n', 0)}")
    lines.append("")

    for field, metrics in report.get("fields", {}).items():
        lines.append(f"## {field}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Micro precision | {metrics['micro_precision']:.3f} |")
        lines.append(f"| Micro recall | {metrics['micro_recall']:.3f} |")
        lines.append(f"| Micro F1 | {metrics['micro_f1']:.3f} |")
        lines.append(f"| Macro F1 | {metrics['macro_f1']:.3f} |")
        lines.append("")
        lines.append("| Label | Precision | Recall | F1 | Support |")
        lines.append("| --- | --- | --- | --- | --- |")
        for label, label_metrics in metrics["per_label"].items():
            lines.append(
                f"| {label} | {label_metrics['precision']:.3f} | {label_metrics['recall']:.3f} | "
                f"{label_metrics['f1']:.3f} | {label_metrics['support']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--schema", default="data/schema.yaml")
    ap.add_argument("--split", default=None, help="Optional split.json to evaluate on test IDs")
    ap.add_argument("--out", default="outputs/metrics.json")
    ap.add_argument("--md-out", default="outputs/metrics.md")
    args = ap.parse_args()

    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    label_space = schema["labels"]

    gold = {r["incident_id"]: r for r in load_jsonl(args.gold)}
    pred = {r["incident_id"]: r for r in load_jsonl(args.pred)}

    split = load_split(args.split)
    test_ids = split.get("test") if split else None

    gold = filter_ids(gold, test_ids)
    pred = filter_ids(pred, test_ids)

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

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(metrics_to_markdown(report), encoding="utf-8")

    print(f"Saved metrics -> {out_path}")
    print(f"Saved metrics markdown -> {md_path}")


if __name__ == "__main__":
    main()
