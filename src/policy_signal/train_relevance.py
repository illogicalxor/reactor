from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from .train import file_sha256

LABEL_TO_ID = {"irrelevant": 0, "neutral": 1, "directional": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the irrelevant/neutral/directional policy relevance gate"
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--output", default="models/policy-relevance-gate-v1")
    parser.add_argument("--base-model", default="models/policy-signal-baseline-v2-large")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--class-weight-cap", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from datasets import load_dataset
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            log_loss,
        )
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    files = {"train": args.train, "validation": args.validation}
    if args.test:
        files["test"] = args.test
    dataset = load_dataset("csv", data_files=files)
    required = {"text", "label"}
    for split in dataset:
        missing = required - set(dataset[split].column_names)
        if missing:
            raise SystemExit(f"Missing CSV columns in {split}: {', '.join(sorted(missing))}")
        invalid = set(dataset[split].unique("label")) - set(LABEL_TO_ID)
        if invalid:
            raise SystemExit(f"Invalid labels in {split}: {', '.join(sorted(invalid))}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        encoded = tokenizer(
            [f"[STATEMENT] {text}" for text in batch["text"]],
            truncation=True,
            max_length=384,
        )
        encoded["labels"] = [LABEL_TO_ID[label] for label in batch["label"]]
        return encoded

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label={value: key for key, value in LABEL_TO_ID.items()},
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    class_counts = Counter(dataset["train"]["label"])
    total = len(dataset["train"])
    class_weights = torch.tensor(
        [
            min(total / (len(LABEL_TO_ID) * class_counts[label]), args.class_weight_cap)
            for label in LABEL_TO_ID
        ],
        dtype=torch.float32,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            del num_items_in_batch
            labels = inputs["labels"]
            outputs = model(**inputs)
            loss = functional.cross_entropy(
                outputs.logits,
                labels,
                weight=class_weights.to(outputs.logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    def metrics(prediction):
        predicted = np.argmax(prediction.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(prediction.label_ids, predicted),
            "balanced_accuracy": balanced_accuracy_score(prediction.label_ids, predicted),
            "f1_macro": f1_score(prediction.label_ids, predicted, average="macro"),
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
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )
    trainer = WeightedTrainer(
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
    validation_evaluation = {
        key: round(float(value), 6) for key, value in trainer.evaluate().items()
    }

    manifest = {
        "task": "policy relevance gate",
        "base_model": args.base_model,
        "labels": LABEL_TO_ID,
        "class_counts": dict(class_counts),
        "class_weights": {
            label: round(float(class_weights[index]), 6) for label, index in LABEL_TO_ID.items()
        },
        "training_parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "class_weight_cap": args.class_weight_cap,
            "seed": args.seed,
        },
        "completed_epochs": round(float(trainer.state.epoch or 0.0), 6),
        "selected_checkpoint": (
            Path(trainer.state.best_model_checkpoint).name
            if trainer.state.best_model_checkpoint
            else None
        ),
        "best_validation_f1_macro": (
            round(float(trainer.state.best_metric), 6)
            if trainer.state.best_metric is not None
            else None
        ),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.cuda.is_available(),
        "data_sha256": {
            "train": file_sha256(args.train),
            "validation": file_sha256(args.validation),
            "test": file_sha256(args.test) if args.test else None,
        },
        "validation": {
            key: value for key, value in validation_evaluation.items() if key != "epoch"
        },
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dataset_manifest:
        shutil.copyfile(args.dataset_manifest, output / "dataset_manifest.json")

    if "test" in tokenized:
        prediction = trainer.predict(tokenized["test"])
        probabilities = torch.softmax(torch.from_numpy(prediction.predictions), dim=-1).numpy()
        predicted = np.argmax(probabilities, axis=-1)
        target_names = list(LABEL_TO_ID)
        test_evaluation = {
            "examples": len(prediction.label_ids),
            "accuracy": round(float(accuracy_score(prediction.label_ids, predicted)), 6),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(prediction.label_ids, predicted)), 6
            ),
            "f1_macro": round(float(f1_score(prediction.label_ids, predicted, average="macro")), 6),
            "log_loss": round(
                float(log_loss(prediction.label_ids, probabilities, labels=[0, 1, 2])), 6
            ),
            "confusion_matrix": confusion_matrix(
                prediction.label_ids,
                predicted,
                labels=[0, 1, 2],
            ).tolist(),
            "labels": target_names,
            "per_class": classification_report(
                prediction.label_ids,
                predicted,
                labels=[0, 1, 2],
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            ),
            "limitations": [
                "The chronological test split has very few irrelevant examples.",
                "Labels measure relevance/stance rather than asset returns.",
            ],
        }
        (output / "test_evaluation.json").write_text(
            json.dumps(test_evaluation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(test_evaluation, indent=2, sort_keys=True))
    else:
        print(json.dumps(validation_evaluation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
