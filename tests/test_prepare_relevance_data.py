from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from policy_signal.prepare_relevance_data import (
    deduplicate,
    load_supplemental_rows,
    normalized_text,
)


class PrepareRelevanceDataTests(unittest.TestCase):
    def test_normalization_is_unicode_and_punctuation_insensitive(self) -> None:
        self.assertEqual(normalized_text(" RATES—Higher! "), "rates higher")

    def test_deduplicate_removes_duplicates_and_every_conflicting_row(self) -> None:
        rows = [
            {"text": "Same statement", "label": "neutral"},
            {"text": "same statement!", "label": "neutral"},
            {"text": "Conflicting statement", "label": "neutral"},
            {"text": "conflicting statement", "label": "directional"},
            {"text": "Unique statement", "label": "directional"},
        ]

        unique, duplicate_count, conflict_count = deduplicate(rows)

        self.assertEqual([row["text"] for row in unique], ["Same statement", "Unique statement"])
        self.assertEqual(duplicate_count, 1)
        self.assertEqual(conflict_count, 1)

    def test_loads_human_reviewed_supplemental_training_rows(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.csv"
            path.write_text(
                "text,label,source\n"
                "Thank you and good night.,irrelevant,human-review\n",
                encoding="utf-8",
            )

            rows = load_supplemental_rows(path)

        self.assertEqual(rows[0]["label"], "irrelevant")
        self.assertEqual(rows[0]["source"], "human-review")

    def test_rejects_invalid_supplemental_label(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reviewed.csv"
            path.write_text("text,label\nHello,bullish\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid supplemental label"):
                load_supplemental_rows(path)


if __name__ == "__main__":
    unittest.main()
