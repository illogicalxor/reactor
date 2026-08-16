# Relevance-gate validation

## What was measured

The inference-only machine evaluated `models/policy-relevance-gate-v1` on 16 August
2026 using `data/relevance_regression.example.csv`. The set contains 44
repository-authored English statements: 28 procedural `irrelevant` examples, eight
economic `neutral` examples, and eight policy `directional` examples.

Two runs are versioned in the model directory:

- `repository_relevance_regression.baseline.json`: raw checkpoint, temperature 1.0;
- `repository_relevance_regression.json`: checkpoint plus the narrowly anchored
  procedural inference guardrail.

| Metric | Raw checkpoint | Inference policy |
|---|---:|---:|
| Accuracy | 0.8636 | 0.8864 |
| Balanced accuracy | 0.7798 | 0.7917 |
| Macro F1 | 0.7709 | 0.7909 |
| Log loss | 0.3801 | 0.3407 |
| Multiclass Brier score | 0.2031 | 0.1774 |
| ECE, 10 bins | 0.1164 | 0.1123 |
| `irrelevant` recall | 0.9643 | 1.0000 |

The single corrected example was “Let us go to the next question.” The guardrail
matches only complete procedural utterances, so a substantive answer containing
the words “next question” is left to the checkpoint.

## Calibration and the next GPU run

The current v1 checkpoint was trained before temperature calibration was added and
has no `relevance_calibration.json`; inference therefore correctly uses temperature
1.0. The trainer now fits one temperature on held-out validation logits and writes
the value, validation hash, and before/after negative log likelihood into the model
artifact and training manifest. It also evaluates the test split using the fitted
temperature.

Data preparation now accepts repeatable `--supplemental-train` CSVs for
human-reviewed examples. They are restricted to training, deduplicated, checked
against chronological held-out statements, and recorded by SHA-256. The regression
CSV must not be supplied as supplemental training data.

## Limitations

This is a small, synthetic, repository-authored regression set. It demonstrates
deterministic behavior and catches regressions in procedural filtering, but it is
not evidence of production generalization and must not be used to tune then claim
independent performance. Promotion still requires human annotations from multiple
real events, event-level held-out splits, probability calibration on validation
only, and final measurement on an untouched out-of-time test set.
