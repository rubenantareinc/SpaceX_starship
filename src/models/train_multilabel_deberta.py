"""
train_multilabel_deberta.py
---------------------------
Fine-tune a transformer encoder (default: DeBERTa) for multi-label classification.

This script supports training per-field heads by training a separate model per task field.
For small datasets, this is simpler and often more stable than a multi-head architecture.

Example:
  python -m src.models.train_multilabel_deberta \
    --data data/processed/incidents.jsonl \
    --task subsystem failure_mode impact \
    --model microsoft/deberta-v3-base \
    --output_dir outputs/deberta_multilabel
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch


def load_jsonl(path: str) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def make_dataset(records: List[Dict], field: str, label_list: List[str]) -> Dataset:
    texts = [r["text"] for r in records]
    y = [r.get("labels", {}).get(field, []) for r in records]

    label2id = {l: i for i, l in enumerate(label_list)}
    Y = []
    for labs in y:
        vec = np.zeros(len(label_list), dtype=np.float32)
        for lab in labs:
            if lab in label2id:
                vec[label2id[lab]] = 1.0
        Y.append(vec.tolist())

    return Dataset.from_dict({"text": texts, "labels": Y})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    # micro F1 across all labels
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"micro_f1": float(f1), "precision": float(precision), "recall": float(recall)}


class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--schema", default="data/schema.yaml")
    ap.add_argument("--task", nargs="+", default=["subsystem", "failure_mode", "impact"])
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--output_dir", default="outputs/deberta_multilabel")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    records = load_jsonl(args.data)
    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    label_space = schema["labels"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # naive split: 80/20
    n = len(records)
    split = max(1, int(0.8 * n))
    train_records = records[:split]
    eval_records = records[split:] if split < n else records[:1]

    for field in args.task:
        if field not in label_space:
            raise ValueError(f"Unknown field: {field}")

        labels = label_space[field]
        train_ds = make_dataset(train_records, field, labels).map(tokenize, batched=True)
        eval_ds = make_dataset(eval_records, field, labels).map(tokenize, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=len(labels),
            problem_type="multi_label_classification",
        )

        out_dir = out_root / field
        training_args = TrainingArguments(
            output_dir=str(out_dir),
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
            report_to=[],
        )

        trainer = MultiLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(str(out_dir / "best"))
        print(f"Saved best model for {field} -> {out_dir/'best'}")


if __name__ == "__main__":
    main()
