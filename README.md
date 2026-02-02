
# Starship Anomaly Explainer 

An end-to-end NLP system that converts unstructured public incident narratives about **SpaceX Starship**
(news articles, official statements, community reports) into **structured, evidence-grounded
“What happened?” cards**.

The goal is to study how reliably NLP can extract **engineering-relevant failure information**
from noisy, evolving, real-world text.

---

## What the system produces

For each incident narrative, the pipeline outputs a structured card containing:

- **Subsystem(s)** — multi-label (e.g. raptor_engine, avionics, heat_shield)
- **Failure mode(s)** — multi-label (e.g. loss_of_control, fire, debris, fts_triggered)
- **Impact** — vehicle loss, pad damage, delay, minor anomaly, etc.
- **Cause hypothesis** — optional, explicitly labeled as *hypothesis*
- **Evidence snippets** — supporting sentences extracted from the text (no hallucination)
- **Confidence score** per field

All predictions are tied to **verbatim text spans** from the input.

---

## Demo screenshots

### Ascent anomaly (engine shutdown + FTS)
![Ascent anomaly](docs/demo_ascent.png)

### Ground systems anomaly (pad fire and debris)
![Pad anomaly](docs/demo_pad.png)

### Reentry anomaly (thermal protection failure)
![Reentry anomaly](docs/demo_reentry.png)

---

## Why this project matters

Spaceflight incidents generate large volumes of unstructured text but little standardized data.
Analysts must manually translate scattered reports into structured failure timelines.

This project demonstrates how NLP can:
- Perform **multi-label incident classification**
- Extract **evidence-grounded explanations**
- Handle **speculative and evolving narratives**
- Support **engineering analysis**, not text generation

---

## System overview

Pipeline stages:

1. **Input**: raw incident narrative
2. **Sentence segmentation**
3. **Baseline inference**
   - Keyword-based matching (fast, interpretable)
   - TF–IDF + One-vs-Rest Logistic Regression
4. **(Optional)** Transformer fine-tuning (DeBERTa, multi-label)
5. **Evidence selection** (sentence-level grounding)
6. **Structured output card**

---

## Dataset

The repository now includes a **40-row incident roster** in `data/sources.csv` plus matching
raw text files under `data/raw_text/`. Only three incidents currently include text and labels;
the remaining entries are templates to be filled with real sources and narratives.

Current dataset statistics (from `outputs/dataset_stats.json`):

| Metric | Value |
| --- | --- |
| Total incidents (roster size) | 40 |
| Incidents with narratives | 3 |
| Incidents with labels | 3 |
| Avg. length (tokens) | 43.0 |
| Avg. label cardinality | 12.0 |

## Reproducibility (3 commands)

```bash
python -m src.ingest.build_incidents --raw-dir data/raw_text --sources data/sources.csv --out data/processed/incidents.jsonl --labels-from data/processed/incidents.jsonl
python -m src.eval.split --data data/processed/incidents.jsonl
python scripts/smoke_end_to_end.py
```

`scripts/smoke_end_to_end.py` rebuilds the processed dataset, generates predictions, and writes
all metrics to `outputs/`.

## Quantitative evaluation

All metrics below are computed on the deterministic **test split** (currently 1 incident due to
3 labeled narratives total). See `outputs/split.json` for the split IDs.

### Keyword baseline (test split)

| Field | Micro P | Micro R | Micro F1 | Macro F1 |
| --- | --- | --- | --- | --- |
| subsystem | 0.750 | 0.750 | 0.750 | 0.214 |
| failure_mode | 1.000 | 0.667 | 0.800 | 0.182 |
| impact | 0.000 | 0.000 | 0.000 | 0.000 |
| cause | 1.000 | 1.000 | 1.000 | 0.286 |

Source: `outputs/keyword_metrics.json` / `outputs/keyword_metrics.md`.

### TF–IDF + Logistic Regression baseline (test split)

| Field | Micro P | Micro R | Micro F1 | Macro F1 |
| --- | --- | --- | --- | --- |
| subsystem | 0.200 | 0.250 | 0.222 | 0.071 |
| failure_mode | 0.200 | 0.333 | 0.250 | 0.091 |
| impact | 0.667 | 1.000 | 0.800 | 0.400 |
| cause | 0.000 | 0.000 | 0.000 | 0.000 |

Source: `outputs/tfidf_metrics.json` / `outputs/tfidf_metrics.md`.

## Evidence grounding evaluation

Evidence is evaluated using sentence-level Precision@k and Recall@k (k=1,3), with coverage
defined as the percent of incidents that return at least one evidence sentence.

Current evidence metrics are **0.000** because `evidence_gold` is not yet populated in the
labeled incidents (see `docs/LABEL_GUIDE.md` for the annotation workflow). Source:
`outputs/evidence_metrics.json` / `outputs/evidence_metrics.md`.

## Limitations and next steps

- The roster includes 40 template incidents, but only 3 have narratives and labels today.
  Populate `data/raw_text/` and `data/sources.csv` with real incidents to scale evaluation.
- Evidence-grounding metrics will remain at 0 until `evidence_gold` is filled via the labeling CLI.
- Add inter-annotator agreement and larger train/test splits as the labeled dataset grows.

### Incident schema

The label taxonomy is captured in both YAML and JSON for convenience:

- `data/schema.yaml`
- `data/schema.json`
