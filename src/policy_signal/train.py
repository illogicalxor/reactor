from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LABEL_TO_ID = {"bearish": 0, "bullish": 1}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an English bullish/bearish US policy-statement classifier"
    )
    parser.add_argument("--train", required=True, help="CSV: text,label[,asset_context]")
    parser.add_argument("--validation", help="Optional validation CSV with the same columns")
    parser.add_argument("--output", default="models/policy-signal-v1")
    parser.add_argument("--base-model", default="FacebookAI/roberta-base")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume optimizer and scheduler state from a Trainer checkpoint directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        from datasets import DatasetDict, load_dataset
        from sklearn.metrics import accuracy_score, f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install '.[train]'") from exc

    files = {"train": args.train}
    if args.validation:
        files["validation"] = args.validation
    dataset = load_dataset("csv", data_files=files)
    required = {"text", "label"}
    missing = required - set(dataset["train"].column_names)
    if missing:
        raise SystemExit(f"Missing CSV columns: {', '.join(sorted(missing))}")
    invalid = set(dataset["train"].unique("label")) - set(LABEL_TO_ID)
    if invalid:
        raise SystemExit(f"Invalid labels: {', '.join(sorted(invalid))}")
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.2, seed=args.seed)
        dataset = DatasetDict(train=split["train"], validation=split["test"])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        contexts = batch.get("asset_context") or ["US equities"] * len(batch["text"])
        texts = [
            f"[ASSET] {context or 'US equities'} [STATEMENT] {text}"
            for context, text in zip(contexts, batch["text"], strict=True)
        ]
        encoded = tokenizer(texts, truncation=True, max_length=384)
        encoded["labels"] = [LABEL_TO_ID[label] for label in batch["label"]]
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "bearish", 1: "bullish"},
        label2id=LABEL_TO_ID,
    )

    def metrics(prediction):
        labels = prediction.label_ids
        predicted = np.argmax(prediction.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predicted),
            "f1_macro": f1_score(labels, predicted, average="macro"),
        }

    output = Path(args.output)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    evaluation = trainer.evaluate()
    serializable_evaluation = {key: round(float(value), 6) for key, value in evaluation.items()}
    manifest = {
        "base_model": args.base_model,
        "labels": LABEL_TO_ID,
        "asset_scope": "US markets",
        "language": "en",
        "training_parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "data_sha256": {
            "train": file_sha256(args.train),
            "validation": file_sha256(args.validation) if args.validation else None,
        },
        "evaluation": serializable_evaluation,
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(serializable_evaluation)


if __name__ == "__main__":
    main()
