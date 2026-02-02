# Labeling Guide

This guide defines the label taxonomy and evidence selection rules for Starship incident narratives.

## Label fields

The label taxonomy is defined in `data/schema.yaml`.

### subsystem
Identify which part(s) of the vehicle or ground system are implicated.
Examples: `raptor_engine`, `gnc`, `heat_shield`, `launch_pad`.

### failure_mode
Describe the failure mechanism or observable anomaly.
Examples: `engine_shutdown`, `loss_of_control`, `fts_triggered`.

### impact
Describe the operational impact of the incident.
Examples: `vehicle_loss`, `pad_damage`, `delay`.

### cause
Root cause hypotheses or contributing factors.
Examples: `thermal_protection_failure`, `software_fault`, `unknown`.

## Evidence grounding rules

Evidence is recorded at the **sentence level** and stored in `evidence_gold` per incident:

```json
"evidence_gold": {
  "subsystem": {"raptor_engine": [0, 2]},
  "failure_mode": {"engine_shutdown": [0]},
  "impact": {"vehicle_loss": [2]},
  "cause": {"unknown": [1]}
}
```

**Rules:**

1. Evidence must quote a sentence that explicitly supports the label.
2. Evidence should be **verbatim** from the narrative text.
3. If multiple sentences support a label, select all relevant sentences.
4. If no sentence explicitly supports a label, leave the evidence list empty.
5. Use the sentence indices displayed by `src/labeling/label_tool.py`.

## Recommended workflow

1. Run the labeling CLI on `data/processed/incidents.jsonl`.
2. Select multi-labels per field, then record the sentence indices for evidence.
3. Save the updated JSONL and re-run evaluation scripts.
