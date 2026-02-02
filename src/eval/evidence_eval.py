"""
evidence_eval.py
----------------
Compare predicted evidence sentences against evidence_gold.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils import split_sentences


def load_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_split(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_sentence(sent: str) -> str:
    return " ".join(sent.strip().split())


def map_sentence_to_index(sentences: List[str]) -> Dict[str, int]:
    return {normalize_sentence(s): i for i, s in enumerate(sentences)}


def extract_pred_indices(pred: dict, field: str, label: str, sentences: List[str]) -> List[int]:
    evidence = pred.get("evidence", {}).get(field, {})
    if isinstance(evidence, dict):
        values = evidence.get(label, [])
        if all(isinstance(v, int) for v in values):
            return values
        if all(isinstance(v, str) for v in values):
            mapping = map_sentence_to_index(sentences)
            return [mapping[s] for s in values if normalize_sentence(s) in mapping]
    return []


def compute_metrics(gold_records: Dict[str, dict], pred_records: Dict[str, dict]) -> Tuple[dict, dict]:
    fields = ["subsystem", "failure_mode", "impact", "cause"]
    overall = {"precision@1": [], "precision@3": [], "recall@1": [], "recall@3": []}
    coverage_hits = 0
    coverage_total = 0

    per_field = {}
    for field in fields:
        per_field[field] = {"precision@1": [], "precision@3": [], "recall@1": [], "recall@3": []}

    for incident_id, gold in gold_records.items():
        gold_evidence = gold.get("evidence_gold", {})
        if not gold_evidence:
            continue
        text = gold.get("text", "")
        sentences = split_sentences(text)
        pred = pred_records.get(incident_id, {})

        any_predicted = False
        for field, label_map in gold_evidence.items():
            for label, gold_indices in label_map.items():
                if not gold_indices:
                    continue
                pred_indices = extract_pred_indices(pred, field, label, sentences)
                if pred_indices:
                    any_predicted = True
                top1 = pred_indices[:1]
                top3 = pred_indices[:3]
                gold_set = set(gold_indices)
                if not gold_set:
                    continue

                def score_list(pred_list: List[int], k: int) -> Tuple[float, float]:
                    if not pred_list:
                        return 0.0, 0.0
                    hit = len(gold_set.intersection(pred_list))
                    recall = hit / len(gold_set)
                    precision = hit / k
                    return precision, recall

                p1, r1 = score_list(top1, 1)
                p3, r3 = score_list(top3, 3)
                overall["precision@1"].append(p1)
                overall["recall@1"].append(r1)
                overall["precision@3"].append(p3)
                overall["recall@3"].append(r3)
                per_field[field]["precision@1"].append(p1)
                per_field[field]["recall@1"].append(r1)
                per_field[field]["precision@3"].append(p3)
                per_field[field]["recall@3"].append(r3)

        coverage_total += 1
        if any_predicted:
            coverage_hits += 1

    def avg(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    report = {
        "overall": {
            "precision@1": avg(overall["precision@1"]),
            "precision@3": avg(overall["precision@3"]),
            "recall@1": avg(overall["recall@1"]),
            "recall@3": avg(overall["recall@3"]),
            "coverage": float(coverage_hits / coverage_total) if coverage_total else 0.0,
        },
        "per_field": {},
        "n_incidents": coverage_total,
    }

    for field, values in per_field.items():
        report["per_field"][field] = {
            "precision@1": avg(values["precision@1"]),
            "precision@3": avg(values["precision@3"]),
            "recall@1": avg(values["recall@1"]),
            "recall@3": avg(values["recall@3"]),
        }

    return report, per_field


def to_markdown(report: dict) -> str:
    lines = ["# Evidence Grounding Metrics", ""]
    lines.append(f"Evaluated incidents: {report.get('n_incidents', 0)}")
    lines.append("")
    overall = report["overall"]
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Precision@1 | {overall['precision@1']:.3f} |")
    lines.append(f"| Precision@3 | {overall['precision@3']:.3f} |")
    lines.append(f"| Recall@1 | {overall['recall@1']:.3f} |")
    lines.append(f"| Recall@3 | {overall['recall@3']:.3f} |")
    lines.append(f"| Coverage | {overall['coverage']:.3f} |")
    lines.append("")

    lines.append("## Per-field")
    lines.append("")
    lines.append("| Field | Precision@1 | Precision@3 | Recall@1 | Recall@3 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for field, metrics in report["per_field"].items():
        lines.append(
            f"| {field} | {metrics['precision@1']:.3f} | {metrics['precision@3']:.3f} | "
            f"{metrics['recall@1']:.3f} | {metrics['recall@3']:.3f} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default=None, help="Optional split.json to evaluate on test IDs")
    ap.add_argument("--out", default="outputs/evidence_metrics.json")
    ap.add_argument("--md-out", default="outputs/evidence_metrics.md")
    args = ap.parse_args()

    gold_records = {r["incident_id"]: r for r in load_jsonl(args.gold)}
    pred_records = {r["incident_id"]: r for r in load_jsonl(args.pred)}

    split = load_split(args.split)
    test_ids = split.get("test") if split else None
    if test_ids:
        gold_records = {k: v for k, v in gold_records.items() if k in test_ids}
        pred_records = {k: v for k, v in pred_records.items() if k in test_ids}

    report, _ = compute_metrics(gold_records, pred_records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print(f"Saved evidence metrics -> {out_path}")
    print(f"Saved evidence metrics markdown -> {md_path}")


if __name__ == "__main__":
    main()
