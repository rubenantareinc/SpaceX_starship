import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.baselines.keyword_baseline import KEYWORDS, score_labels

SKIP_LABELS = {"", "TODO", "TBD"}


def normalize_label(label: str) -> str:
    return label.strip()


def predict_label(text: str, field: str) -> list[str]:
    labels, _, _ = score_labels(text, KEYWORDS.get(field, {}))
    return labels


def update_confusion(confusion: dict, gold: str, pred: str) -> None:
    confusion[gold][pred] += 1


def evaluate_rows(rows: list[dict]) -> dict:
    results = {
        "subsystem": {"correct": 0, "total": 0, "confusion": defaultdict(Counter)},
        "incident_type": {"correct": 0, "total": 0, "confusion": defaultdict(Counter)},
        "skipped": 0,
    }

    for row in rows:
        text = row["text"].strip()
        subsystem_label = normalize_label(row["subsystem_label"])
        incident_type_label = normalize_label(row["incident_type_label"])

        if subsystem_label in SKIP_LABELS or incident_type_label in SKIP_LABELS:
            results["skipped"] += 1
            continue

        subsystem_pred = predict_label(text, "subsystem")
        incident_type_pred = predict_label(text, "failure_mode")

        subsystem_top = subsystem_pred[0] if subsystem_pred else "none"
        incident_type_top = incident_type_pred[0] if incident_type_pred else "none"

        results["subsystem"]["total"] += 1
        if subsystem_label in subsystem_pred:
            results["subsystem"]["correct"] += 1
        update_confusion(results["subsystem"]["confusion"], subsystem_label, subsystem_top)

        results["incident_type"]["total"] += 1
        if incident_type_label in incident_type_pred:
            results["incident_type"]["correct"] += 1
        update_confusion(results["incident_type"]["confusion"], incident_type_label, incident_type_top)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate keyword baseline against labels.csv")
    parser.add_argument("--labels", default="data/labels.csv", help="Path to labeled CSV")
    parser.add_argument("--out", default="outputs/eval_results.json", help="Output JSON path")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    rows = []
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    results = evaluate_rows(rows)

    subsystem_total = results["subsystem"]["total"]
    incident_total = results["incident_type"]["total"]

    subsystem_accuracy = (
        results["subsystem"]["correct"] / subsystem_total if subsystem_total else 0.0
    )
    incident_accuracy = (
        results["incident_type"]["correct"] / incident_total if incident_total else 0.0
    )

    print(
        f"Subsystem accuracy: {results['subsystem']['correct']}/{subsystem_total} "
        f"({subsystem_accuracy:.3f})"
    )
    print(
        f"Incident type accuracy: {results['incident_type']['correct']}/{incident_total} "
        f"({incident_accuracy:.3f})"
    )
    print(f"Skipped rows (TODO/blank labels): {results['skipped']}")

    output = {
        "subsystem": {
            "correct": results["subsystem"]["correct"],
            "total": subsystem_total,
            "accuracy": subsystem_accuracy,
            "confusion": {
                gold: dict(preds) for gold, preds in results["subsystem"]["confusion"].items()
            },
        },
        "incident_type": {
            "correct": results["incident_type"]["correct"],
            "total": incident_total,
            "accuracy": incident_accuracy,
            "confusion": {
                gold: dict(preds)
                for gold, preds in results["incident_type"]["confusion"].items()
            },
        },
        "skipped": results["skipped"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
