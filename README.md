# Starship Anomaly Explainer 🚀🧠

An end-to-end NLP pipeline that reads public incident narratives about SpaceX Starship (news, reports, community summaries) and outputs **structured “What happened?” cards**:

- **Subsystem(s)** (multi-label)
- **Failure mode(s)** (multi-label)
- **Cause hypothesis** (optional multi-label)
- **Impact** (multi-label)
- **Evidence snippets** (grounded spans, no hallucination)
- **Confidence** per field

This repo is designed to be **portfolio-grade**: baselines + transformer, proper evaluation splits, and a simple Streamlit demo.

---

## Quickstart

### 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Data (sample included)

A small sample is committed at:

- `data/processed/incidents.jsonl`

You can add more records (same schema) and keep large raw text in `data/raw/` (ignored by git).

### 3) Baselines

**Keyword baseline**
```bash
python -m src.baselines.keyword_baseline --data data/processed/incidents.jsonl --out outputs/keyword_preds.jsonl
```

**TF–IDF + One-vs-Rest Logistic Regression**
```bash
python -m src.baselines.tfidf_baseline --data data/processed/incidents.jsonl --out outputs/tfidf_preds.jsonl
```

### 4) Transformer (DeBERTa, multi-label)

```bash
python -m src.models.train_multilabel_deberta \
  --data data/processed/incidents.jsonl \
  --task subsystem failure_mode impact cause \
  --model microsoft/deberta-v3-base \
  --output_dir outputs/deberta_multilabel
```

### 5) Evaluate

```bash
python -m src.eval.evaluate --gold data/processed/incidents.jsonl --pred outputs/tfidf_preds.jsonl
```

### 6) Demo UI (Streamlit)

```bash
streamlit run src/demo/app.py
```

---

## Data format

Each line in `incidents.jsonl` is one JSON record:

```json
{
  "incident_id": "ift3-2024-03-14",
  "date": "2024-03-14",
  "title": "Starship IFT-3 anomaly during reentry",
  "source_url": "https://example.com",
  "source_type": "news|official|community",
  "text": "...",
  "labels": {
    "subsystem": ["heat_shield", "avionics"],
    "failure_mode": ["loss_of_control"],
    "impact": ["vehicle_loss", "delay"],
    "cause": ["unknown"]
  }
}
```

Label space is defined in `data/schema.yaml`.

---

## Project layout

```
starship-anomaly-explainer/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ schema.yaml
│  ├─ raw/                 # not committed
│  └─ processed/
│     └─ incidents.jsonl   # small sample committed
├─ src/
│  ├─ ingest/              # scrape + clean
│  ├─ labeling/            # optional CLI label helper
│  ├─ baselines/           # keyword + tfidf baselines
│  ├─ models/              # DeBERTa multi-label fine-tune + predict
│  ├─ eval/                # metrics + plots
│  └─ demo/                # Streamlit “What happened?” cards
├─ outputs/                # predictions + metrics + artifacts
└─ docs/                   # methods + paper outline
```

---

## Notes on grounding (“evidence snippets”)

This project does **not** generate free-form explanations. It selects **supporting sentences** from the input text (heuristics or model-based scoring) to justify each predicted label. The demo highlights those spans.

---

## License

MIT (see `LICENSE`).
