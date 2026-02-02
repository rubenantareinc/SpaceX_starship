"""
tfidf_baseline.py
-----------------
TF–IDF + One-vs-Rest Logistic Regression multi-label baseline.

Trains one classifier per field (subsystem, failure_mode, impact, cause).
Emits predictions + probabilities + simple evidence (top tfidf tokens, not spans).

This baseline is strong enough to beat keywords on small datasets and is a standard NLP reference point.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.utils import split_sentences


def load_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_split(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fit_field(texts: List[str], y: List[List[str]], all_labels: List[str]):
    label_counts = {label: 0 for label in all_labels}
    for row in y:
        for label in row:
            if label in label_counts:
                label_counts[label] += 1

    total = len(y)
    active_labels = [label for label, count in label_counts.items() if 0 < count < total]
    always_on = [label for label, count in label_counts.items() if count == total and total > 0]

    mlb = MultiLabelBinarizer(classes=active_labels)
    y_active = [[label for label in row if label in active_labels] for row in y]
    Y = mlb.fit_transform(y_active)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    X = vec.fit_transform(texts)

    if not active_labels:
        return vec, None, mlb, active_labels, always_on

    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
    clf.fit(X, Y)
    return vec, clf, mlb, active_labels, always_on


def predict_field(vec, clf, mlb, texts: List[str], threshold: float = 0.5):
    if clf is None:
        empty = [[] for _ in texts]
        probs = np.zeros((len(texts), 0))
        return empty, probs, list(mlb.classes_)

    X = vec.transform(texts)
    # decision_function works for LR; fallback to predict_proba
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2 and len(mlb.classes_) == 1:
            probs = probs[:, 1:2]
    else:
        scores = clf.decision_function(X)
        probs = 1 / (1 + np.exp(-scores))

    pred_bin = (probs >= threshold).astype(int)
    pred_labels = mlb.inverse_transform(pred_bin)
    return pred_labels, probs, list(mlb.classes_)


def top_sentence_indices(vec: TfidfVectorizer, text: str, top_k: int = 3) -> List[int]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    X = vec.transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [int(i) for i in top_idx if scores[i] > 0]


def filter_records(records: List[dict], ids: Optional[set]) -> List[dict]:
    if not ids:
        return records
    return [r for r in records if r.get("incident_id") in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Optional path used for both train/test")
    ap.add_argument("--train", default=None, help="Training JSONL (defaults to --data)")
    ap.add_argument("--test", default=None, help="Test JSONL (defaults to --data)")
    ap.add_argument("--schema", default="data/schema.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--split", default=None, help="Optional split.json with train/test ids")
    args = ap.parse_args()

    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    label_space = schema["labels"]

    data_path = args.data
    train_path = args.train or data_path
    test_path = args.test or data_path
    if not train_path or not test_path:
        raise ValueError("Provide --data or both --train/--test paths.")

    train_records = load_jsonl(train_path)
    test_records = load_jsonl(test_path)

    split = load_split(args.split)
    train_ids = set(split.get("train", [])) if split else None
    test_ids = set(split.get("test", [])) if split else None

    train_records = filter_records(train_records, train_ids)
    test_records = filter_records(test_records, test_ids)

    train_texts = [r["text"] for r in train_records]

    models = {}
    for field in ["subsystem", "failure_mode", "impact", "cause"]:
        y = [r.get("labels", {}).get(field, []) for r in train_records]
        vec, clf, mlb, active_labels, always_on = fit_field(train_texts, y, label_space[field])
        models[field] = (vec, clf, mlb, active_labels, always_on)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in test_records:
            text = rec["text"]
            pred = {"incident_id": rec["incident_id"], "pred": {}, "confidence": {}, "evidence": {}}
            sentence_indices = {}
            for field, (vec, clf, mlb, active_labels, always_on) in models.items():
                labels, probs, classes = predict_field(vec, clf, mlb, [text], threshold=args.threshold)
                labels = list(labels[0]) if labels else []
                labels = sorted(set(labels + always_on))
                # confidences: map label -> prob
                conf = {cls: float(p) for cls, p in zip(classes, probs[0])} if probs.size else {}
                for label in always_on:
                    conf[label] = 1.0
                pred["pred"][field] = labels
                pred["confidence"][field] = {k: v for k, v in conf.items() if k in labels}
                # evidence: top sentence indices by tf-idf weight (shared across labels)
                if field not in sentence_indices:
                    sentence_indices[field] = top_sentence_indices(vec, text, top_k=3)
                pred["evidence"][field] = {
                    label: sentence_indices[field] for label in labels
                }
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Wrote tfidf predictions to {out_path}")


if __name__ == "__main__":
    main()
