from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .domain import TranscriptSegment
from .relevance import PROCEDURAL_OVERRIDE_CONFIDENCE, FineTunedPolicyRelevanceGate

LABELS = ("irrelevant", "neutral", "directional")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"text", "label"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing relevance columns: {', '.join(sorted(missing))}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            text = row["text"].strip()
            label = row["label"].strip()
            if not text:
                raise ValueError(f"Empty text on line {line_number}")
            if label not in LABELS:
                raise ValueError(f"Invalid label on line {line_number}: {label}")
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "event_id": (row.get("event_id") or "ungrouped").strip() or "ungrouped",
                    "source": (row.get("source") or "unknown").strip() or "unknown",
                }
            )
    if not rows:
        raise ValueError("Relevance evaluation dataset is empty")
    return rows


def evaluate_probabilities(
    rows: list[dict[str, str]],
    probabilities: list[dict[str, float]],
    *,
    ece_bins: int = 10,
) -> dict[str, object]:
    if len(rows) != len(probabilities) or not rows:
        raise ValueError("rows and probabilities must have the same non-zero length")
    confusion = {expected: Counter() for expected in LABELS}
    predictions = []
    correctness = []
    confidences = []
    log_losses = []
    brier_scores = []
    by_event: dict[str, list[bool]] = defaultdict(list)

    for row, scores in zip(rows, probabilities, strict=True):
        if set(scores) != set(LABELS):
            raise ValueError("Each prediction must contain all relevance labels")
        predicted = max(LABELS, key=scores.__getitem__)
        expected = row["label"]
        correct = predicted == expected
        confidence = scores[predicted]
        confusion[expected][predicted] += 1
        correctness.append(correct)
        confidences.append(confidence)
        log_losses.append(-math.log(max(scores[expected], 1e-12)))
        brier_scores.append(
            sum(
                (scores[label] - float(label == expected)) ** 2
                for label in LABELS
            )
        )
        by_event[row["event_id"]].append(correct)
        predictions.append(
            {
                **row,
                "predicted": predicted,
                "confidence": round(confidence, 6),
                "probabilities": {label: round(scores[label], 6) for label in LABELS},
            }
        )

    per_class = {}
    recalls = []
    f1_scores = []
    for label in LABELS:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[expected][label] for expected in LABELS)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            recalls.append(recall)
        f1_scores.append(f1)
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    calibration_error = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        indexes = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bin_index == ece_bins - 1 and confidence == 1.0)
        ]
        if not indexes:
            continue
        bin_accuracy = sum(correctness[index] for index in indexes) / len(indexes)
        bin_confidence = sum(confidences[index] for index in indexes) / len(indexes)
        calibration_error += len(indexes) / len(rows) * abs(bin_accuracy - bin_confidence)

    return {
        "examples": len(rows),
        "accuracy": round(sum(correctness) / len(rows), 6),
        "balanced_accuracy": round(sum(recalls) / len(recalls), 6),
        "f1_macro": round(sum(f1_scores) / len(f1_scores), 6),
        "log_loss": round(sum(log_losses) / len(log_losses), 6),
        "multiclass_brier_score": round(sum(brier_scores) / len(brier_scores), 6),
        "expected_calibration_error_10_bins": round(calibration_error, 6),
        "labels": list(LABELS),
        "confusion_matrix": [
            [confusion[expected][predicted] for predicted in LABELS] for expected in LABELS
        ],
        "per_class": per_class,
        "per_event": {
            event_id: {
                "examples": len(values),
                "accuracy": round(sum(values) / len(values), 6),
            }
            for event_id, values in sorted(by_event.items())
        },
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned relevance gate without training dependencies"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="models/policy-relevance-gate-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-procedural-override",
        action="store_true",
        help="Measure raw checkpoint probabilities without the inference guardrail",
    )
    args = parser.parse_args()

    rows = load_rows(args.data)
    segments = [TranscriptSegment(0.0, 1.0, row["text"], "en") for row in rows]
    gate = FineTunedPolicyRelevanceGate(
        args.model,
        procedural_override=not args.no_procedural_override,
    )
    probabilities = gate.predict_probabilities(segments)
    evaluation = {
        "dataset": str(args.data),
        "model": args.model,
        "temperature": gate.temperature,
        "procedural_override": gate.procedural_override,
        "procedural_override_confidence": PROCEDURAL_OVERRIDE_CONFIDENCE,
        "metrics": evaluate_probabilities(rows, probabilities),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
