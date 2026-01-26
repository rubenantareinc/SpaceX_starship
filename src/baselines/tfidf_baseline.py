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
from typing import Dict, List

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def load_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def fit_field(texts: List[str], y: List[List[str]], all_labels: List[str]):
    mlb = MultiLabelBinarizer(classes=all_labels)
    Y = mlb.fit_transform(y)

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    X = vec.fit_transform(texts)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
    clf.fit(X, Y)
    return vec, clf, mlb


def predict_field(vec, clf, mlb, texts: List[str], threshold: float = 0.5):
    X = vec.transform(texts)
    # decision_function works for LR; fallback to predict_proba
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
    else:
        scores = clf.decision_function(X)
        probs = 1 / (1 + np.exp(-scores))

    pred_bin = (probs >= threshold).astype(int)
    pred_labels = mlb.inverse_transform(pred_bin)
    return pred_labels, probs, mlb.classes_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--schema", default="data/schema.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    label_space = schema["labels"]

    records = load_jsonl(args.data)
    texts = [r["text"] for r in records]

    models = {}
    for field in ["subsystem", "failure_mode", "impact", "cause"]:
        y = [r.get("labels", {}).get(field, []) for r in records]
        vec, clf, mlb = fit_field(texts, y, label_space[field])
        models[field] = (vec, clf, mlb)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec, text in zip(records, texts):
            pred = {"incident_id": rec["incident_id"], "pred": {}, "confidence": {}, "evidence": {}}
            for field, (vec, clf, mlb) in models.items():
                labels, probs, classes = predict_field(vec, clf, mlb, [text], threshold=args.threshold)
                labels = list(labels[0]) if labels else []
                # confidences: map label -> prob
                conf = {cls: float(p) for cls, p in zip(classes, probs[0])}
                pred["pred"][field] = labels
                pred["confidence"][field] = {k: v for k, v in conf.items() if k in labels}
                # evidence: top weighted terms (not spans)
                # take top 8 tfidf features in the doc
                X = vec.transform([text])
                if X.nnz:
                    idx = np.argsort(X.data)[-8:][::-1]
                    terms = [vec.get_feature_names_out()[X.indices[i]] for i in idx]
                else:
                    terms = []
                pred["evidence"][field] = {"top_terms": terms}
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Wrote tfidf predictions to {out_path}")


if __name__ == "__main__":
    main()
