from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from src.baselines.keyword_baseline import KEYWORDS, score_labels

SKIP_LABELS = {"", "TBD"}


def normalize_label(label: str) -> str:
    return label.strip()


def predict_label(text: str, field: str) -> list[str]:
    labels, _, _ = score_labels(text, KEYWORDS.get(field, {}))
    return labels


def update_confusion(confusion: dict, gold: str, pred: str) -> None:
    confusion[gold][pred] += 1


def evaluate_rows(rows: Iterable[dict]) -> dict:
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
