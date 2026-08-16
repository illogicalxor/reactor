from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

FEDERAL_DATASET = "gtfintechlab/federal_reserve_system"
FOMC_DATASET = "gtfintechlab/fomc_communication"
FEDERAL_LABELS = {
    "hawkish": "directional",
    "dovish": "directional",
    "neutral": "neutral",
    "irrelevant": "irrelevant",
}
FOMC_LABELS = {0: "directional", 1: "directional", 2: "neutral"}


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w]+", " ", text).strip()


def deduplicate(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], int, int]:
    """Remove exact duplicates and all texts carrying conflicting labels."""
    labels_by_text: dict[str, set[str]] = {}
    for row in rows:
        key = normalized_text(str(row["text"]))
        labels_by_text.setdefault(key, set()).add(str(row["label"]))
    conflicts = {key for key, labels in labels_by_text.items() if len(labels) > 1}

    unique = []
    seen: set[str] = set()
    for row in rows:
        key = normalized_text(str(row["text"]))
        if not key or key in conflicts or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    conflicting_rows = sum(normalized_text(str(row["text"])) in conflicts for row in rows)
    removed_duplicates = len(rows) - len(unique) - conflicting_rows
    return unique, max(removed_duplicates, 0), len(conflicts)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["text", "label", "year", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _counts(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "examples": len(rows),
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "sources": dict(sorted(Counter(str(row["source"]) for row in rows).items())),
    }


def prepare(output_directory: Path) -> dict[str, object]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install training dependencies with: pip install -e '.[train]'") from exc

    federal = load_dataset(FEDERAL_DATASET, "5768")
    federal_rows = [
        {
            "text": row["sentences"].strip(),
            "label": FEDERAL_LABELS[row["stance_label"]],
            "year": int(row["year"]),
            "source": FEDERAL_DATASET,
        }
        for split in federal.values()
        for row in split
        if row["sentences"].strip()
    ]
    # Keep the newest copy so a repeated held-out statement can never leak backward.
    federal_rows, federal_duplicates, federal_conflicts = deduplicate(
        sorted(federal_rows, key=lambda row: int(row["year"]), reverse=True)
    )
    test = [row for row in federal_rows if int(row["year"]) >= 2023]
    validation = [row for row in federal_rows if 2019 <= int(row["year"]) <= 2022]
    federal_train = [row for row in federal_rows if int(row["year"]) <= 2018]

    fomc = load_dataset(FOMC_DATASET)
    fomc_rows = [
        {
            "text": row["sentence"].strip(),
            "label": FOMC_LABELS[int(row["label"])],
            "year": int(row["year"]),
            "source": FOMC_DATASET,
        }
        for split in fomc.values()
        for row in split
        if row["sentence"].strip()
    ]
    fomc_rows, fomc_duplicates, fomc_conflicts = deduplicate(fomc_rows)

    held_out_keys = {normalized_text(str(row["text"])) for row in validation + test}
    training_candidates = [
        row
        for row in federal_train + fomc_rows
        if normalized_text(str(row["text"])) not in held_out_keys
    ]
    train, train_duplicates, train_conflicts = deduplicate(training_candidates)

    paths = {
        "train": output_directory / "relevance_train.csv",
        "validation": output_directory / "relevance_validation.csv",
        "test": output_directory / "relevance_test.csv",
    }
    for split, rows in (("train", train), ("validation", validation), ("test", test)):
        _write_csv(paths[split], rows)

    manifest = {
        "task": "irrelevant | neutral | directional policy relevance",
        "mapping": {
            "federal_reserve_system": FEDERAL_LABELS,
            "fomc_communication": {str(key): value for key, value in FOMC_LABELS.items()},
        },
        "splits": {
            "train": "Federal Reserve years <=2018 plus all FOMC Communication rows",
            "validation": "Federal Reserve years 2019-2022",
            "test": "Federal Reserve years 2023-2024",
        },
        "counts": {
            "train": _counts(train),
            "validation": _counts(validation),
            "test": _counts(test),
        },
        "deduplication": {
            "federal_duplicates": federal_duplicates,
            "federal_conflicts": federal_conflicts,
            "fomc_duplicates": fomc_duplicates,
            "fomc_conflicts": fomc_conflicts,
            "combined_training_duplicates": train_duplicates,
            "combined_training_conflicts": train_conflicts,
            "held_out_exact_overlaps_removed": True,
        },
        "data_sha256": {split: file_sha256(path) for split, path in paths.items()},
        "sources": [
            {"dataset": FEDERAL_DATASET, "license": "CC BY-NC-SA 4.0"},
            {"dataset": FOMC_DATASET, "license": "CC BY-NC 4.0"},
        ],
        "limitations": [
            "The held-out irrelevant class has very few examples.",
            "FOMC Communication has random published splits and is used only for training.",
            "Labels measure topical/stance relevance, not observed market returns.",
        ],
    }
    manifest_path = output_directory / "relevance_dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chronological relevance-gate CSVs")
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
