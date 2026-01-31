import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.eval.keyword_eval import evaluate_rows


def main() -> None:
    rows = [
        {
            "incident_id": "smoke-001",
            "text": "The engine shutdown triggered a fire in the engine bay.",
            "subsystem_label": "raptor_engine",
            "incident_type_label": "engine_shutdown",
        },
        {
            "incident_id": "smoke-002",
            "text": "TODO placeholder.",
            "subsystem_label": "TODO",
            "incident_type_label": "TODO",
        },
    ]

    results = evaluate_rows(rows)
    assert results["subsystem"]["total"] == 1, "Expected one evaluated row"
    assert results["incident_type"]["total"] == 1, "Expected one evaluated row"
    assert results["subsystem"]["correct"] == 1, "Expected subsystem match"
    assert results["incident_type"]["correct"] == 1, "Expected incident type match"
    assert results["skipped"] == 1, "Expected one skipped row"

    print(json.dumps(results, indent=2))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
