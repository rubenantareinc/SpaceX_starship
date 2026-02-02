"""
keyword_baseline.py
-------------------
Simple rules/keywords multi-label baseline.

It:
- loads schema.yaml label space
- maps keywords -> labels
- emits predictions + evidence sentences

This is intentionally transparent for error analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from src.utils import split_sentences


KEYWORDS = {
    "subsystem": {
        "raptor_engine": ["raptor", "engine"],
        "propulsion": ["propellant", "methane", "oxygen", "thrust"],
        "avionics": ["avionics", "computer", "telemetry"],
        "gnc": ["guidance", "navigation", "control", "attitude", "tumble"],
        "stage_separation": ["separation", "hot-staging", "staging"],
        "structures": ["structural", "airframe", "buckling"],
        "heat_shield": ["heat shield", "tiles", "thermal protection", "reentry heating"],
        "flaps": ["flap", "aero surface"],
        "tanks": ["tank", "header tank", "pressurization"],
        "ground_systems": ["ground system", "ground equipment", "tower", "deluge"],
        "launch_pad": ["pad", "launch mount", "flame trench"],
        "range_safety": ["flight termination", "FTS", "range safety"],
        "communications": ["comms", "communications", "signal", "telemetry dropped"],
        "software": ["software", "bug", "update", "algorithm"],
    },
    "failure_mode": {
        "engine_shutdown": ["engine shutdown", "failed to ignite", "shutdown"],
        "explosion": ["explosion", "detonation", "blew up"],
        "leak": ["leak", "leaking"],
        "fire": ["fire", "flames", "burning"],
        "loss_of_control": ["loss of control", "tumble", "uncontrolled", "lost attitude"],
        "structural_failure": ["structural failure", "broke apart", "disintegrated"],
        "fts_triggered": ["flight termination", "FTS activated", "FTS was activated"],
        "pad_damage": ["pad damage", "launch pad damage", "crater"],
        "comms_loss": ["lost telemetry", "communications dropped", "signal lost"],
        "debris": ["debris", "fragments"],
        "reentry_breakup": ["reentry breakup", "broke up during reentry"],
    },
    "impact": {
        "vehicle_loss": ["was lost", "vehicle was lost", "destroyed", "broke apart"],
        "pad_damage": ["pad damage", "launch pad damage"],
        "delay": ["delay", "delayed", "postponed"],
        "minor_anomaly": ["minor anomaly", "small issue"],
        "mission_success_with_anomaly": ["mission success with anomaly", "completed but"],
    },
    "cause": {
        "propellant_leak": ["propellant leak", "leak"],
        "engine_rich_shutdown": ["rich shutdown"],
        "control_authority_loss": ["lost control authority"],
        "software_fault": ["software fault", "bug"],
        "debris_strike": ["debris strike", "hit by debris"],
        "thermal_protection_failure": ["thermal protection", "tiles", "heat shield"],
        "unknown": ["unknown", "unclear"],
    },
}


def score_labels(
    text: str, field_map: Dict[str, List[str]]
) -> Tuple[List[str], Dict[str, float], Dict[str, List[int]]]:
    """Return labels, confidences, and evidence sentence indices."""
    text_l = text.lower()
    sents = split_sentences(text)
    picked = []
    conf = {}
    evidence = {}

    for label, kws in field_map.items():
        hits = 0
        evid = []
        for kw in kws:
            if kw.lower() in text_l:
                hits += 1
                # evidence: sentences containing keyword
                for idx, s in enumerate(sents):
                    if kw.lower() in s.lower():
                        evid.append(idx)
        if hits > 0:
            picked.append(label)
            conf[label] = min(0.95, 0.3 + 0.2 * hits)
            evidence[label] = list(dict.fromkeys(evid))[:3]
    return picked, conf, evidence


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_split(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--schema", default="data/schema.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None, help="Optional split.json to filter to test IDs")
    args = ap.parse_args()

    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    label_space = schema["labels"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(Path(args.data))
    if args.split:
        split = load_split(Path(args.split))
        test_ids = set(split.get("test", []))
        records = [r for r in records if r.get("incident_id") in test_ids]

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            pred = {"incident_id": rec["incident_id"], "pred": {}, "confidence": {}, "evidence": {}}
            for field in ["subsystem", "failure_mode", "impact", "cause"]:
                labels, conf, evid = score_labels(rec["text"], KEYWORDS.get(field, {}))
                # keep only labels in schema
                labels = [x for x in labels if x in label_space[field]]
                pred["pred"][field] = labels
                pred["confidence"][field] = conf
                pred["evidence"][field] = evid
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Wrote keyword predictions to {out_path}")


if __name__ == "__main__":
    main()
