# Paper outline (2–4 pages)

## Title
Starship Anomaly Explainer: Evidence-grounded incident understanding from open-source narratives

## Abstract
1 paragraph: motivation, approach, datasets, results, and limitations.

## 1. Introduction
- Why incident narratives are abundant but structured data is scarce.
- Goal: extract structured failure records with evidence, not hallucinated explanations.

## 2. Dataset + schema
- Source types
- Label space (subsystem, failure mode, impact, cause)
- Annotation procedure + inter-annotator notes (if you can)

## 3. Methods
- Keyword baseline
- TF–IDF baseline
- Transformer fine-tuning (DeBERTa)
- Evidence selection strategy

## 4. Experiments
- Splits (random + time split)
- Metrics (micro/macro F1, per-label)

## 5. Results
- Table: baseline vs transformer
- Figure: per-label F1 (optional)

## 6. Error analysis
- 5–10 qualitative examples with explanations

## 7. Limitations + ethics
- Source reliability + speculation
- Not for operational safety decisions
- Encourage transparency: source_type flag, “hypothesis” wording

## 8. Conclusion + future work
- sentence-level rationale model
- joint extraction + classification
- add timeline extraction / event arguments
