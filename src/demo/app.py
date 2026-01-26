"""
Streamlit demo: "What happened?" cards
-------------------------------------

Run:
  streamlit run src/demo/app.py

It can:
- Accept raw incident text
- Run either: keyword baseline OR load model predictions from a JSONL file
- Render an incident card with evidence snippets

For quick demo, use keyword baseline on-the-fly (no training needed).
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import streamlit as st
import yaml

from src.baselines.keyword_baseline import score_labels, KEYWORDS, split_sentences


def load_schema(path: str = "data/schema.yaml") -> Dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def card_section(title: str, labels: List[str], evidence: Dict[str, List[str]], confidence: Dict[str, float]):
    st.subheader(title)
    if not labels:
        st.write("_None_")
        return
    for lab in labels:
        conf = confidence.get(lab, None)
        badge = f"**{lab}**" + (f"  (p={conf:.2f})" if conf is not None else "")
        st.markdown(badge)
        ev = evidence.get(lab, [])
        if ev:
            for s in ev:
                st.markdown(f"> {s}")


def run_keyword(text: str):
    pred = {"pred": {}, "confidence": {}, "evidence": {}}
    for field in ["subsystem", "failure_mode", "impact", "cause"]:
        labels, conf, evid = score_labels(text, KEYWORDS.get(field, {}))
        pred["pred"][field] = labels
        # flatten conf dict for labels only
        pred["confidence"][field] = {k: v for k, v in conf.items() if k in labels}
        pred["evidence"][field] = evid
    return pred


st.set_page_config(page_title="Starship Anomaly Explainer", layout="wide")
st.title("Starship Anomaly Explainer 🚀")
st.caption("Paste an incident narrative and get a structured, evidence-grounded card.")

schema = load_schema()
label_space = schema["labels"]

col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area(
        "Incident text",
        height=220,
        value="During ascent, several engines shut down and the vehicle began to tumble. Telemetry dropped and the flight termination system was activated.",
    )

with col2:
    mode = st.radio("Mode", ["Keyword baseline (instant)"], index=0)

if st.button("Generate card"):
    pred = run_keyword(text)

    st.divider()
    st.header("What happened?")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card_section("Subsystem", pred["pred"]["subsystem"], pred["evidence"]["subsystem"], pred["confidence"]["subsystem"])
    with c2:
        card_section("Failure mode", pred["pred"]["failure_mode"], pred["evidence"]["failure_mode"], pred["confidence"]["failure_mode"])
    with c3:
        card_section("Impact", pred["pred"]["impact"], pred["evidence"]["impact"], pred["confidence"]["impact"])
    with c4:
        card_section("Cause (hyp.)", pred["pred"]["cause"], pred["evidence"]["cause"], pred["confidence"]["cause"])

    st.divider()
    st.subheader("Raw JSON")
    st.code(json.dumps(pred, indent=2), language="json")
