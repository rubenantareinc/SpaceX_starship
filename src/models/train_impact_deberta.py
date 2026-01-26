"""
train_impact_deberta.py
-----------------------
Convenience wrapper to train only the impact classifier.

Internally calls train_multilabel_deberta with task=impact.
"""

import sys
from src.models.train_multilabel_deberta import main

if __name__ == "__main__":
    # Inject default args if user didn't pass --task
    if "--task" not in sys.argv:
        sys.argv.extend(["--task", "impact"])
    main()
