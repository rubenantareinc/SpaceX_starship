import argparse
import csv
import json
from pathlib import Path

from src.eval.keyword_eval import evaluate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate keyword baseline against labels.csv")
    parser.add_argument("--labels", default="data/labels.csv", help="Path to labeled CSV")
    parser.add_argument("--out", default="outputs/eval_results.json", help="Output JSON path")
    parser.add_argument(
        "--summary-out",
        default="outputs/eval_summary.txt",
        help="Output text summary path",
    )
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

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"Subsystem accuracy: {results['subsystem']['correct']}/{subsystem_total} ({subsystem_accuracy:.3f})",
        f"Incident type accuracy: {results['incident_type']['correct']}/{incident_total} ({incident_accuracy:.3f})",
        f"Skipped rows (TODO/blank labels): {results['skipped']}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
