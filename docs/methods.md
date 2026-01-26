# Methods

This document describes the pipeline and experimental plan.

## 1. Task

Given an incident narrative (news/report/community summary), predict:

- Subsystem(s) involved (multi-label)
- Failure mode(s) (multi-label)
- Impact (multi-label)
- Cause hypothesis (optional)

Additionally, produce **evidence snippets**: sentences from the input that justify each predicted label.

## 2. Data + schema

- Schema: `data/schema.yaml`
- Records: JSONL in `data/processed/incidents.jsonl`
- Metadata: `source_type` = {official, news, community}

Recommended: expand to 30–100 documents. Keep `data/raw/` uncommitted.

## 3. Models

### Baseline A: Keyword rules
- Transparent keyword maps per label.
- Confidence based on hit count.
- Evidence: sentences containing matched keywords.

### Baseline B: TF–IDF + One-vs-Rest Logistic Regression
- Vectorize with TF–IDF (1–2 grams)
- One-vs-Rest logistic regression per field
- Evidence: top TF–IDF terms (upgrade later to spans)

### Strong model: Transformer encoder (DeBERTa)
- Fine-tune for multi-label classification with BCEWithLogitsLoss
- One model per field (simplifies training on small datasets)
- Evidence: heuristic sentence selection (upgrade path: attention/gradients, sentence reranker)

## 4. Evaluation

Report:
- Micro / Macro F1
- Precision / Recall
- Per-label metrics + support (label imbalance visibility)

Recommended splits:
- random 80/20
- time split (train earlier, test later)
- optional source split (train news, test community) for domain shift

## 5. Error analysis checklist

- ambiguous language ("anomaly", "issue")
- speculative claims ("may have", "likely")
- evolving narratives (early reports corrected later)
- concurrent failures in a single narrative
- label sparsity / imbalance
