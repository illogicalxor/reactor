from __future__ import annotations

import unittest

from policy_signal.prepare_relevance_data import deduplicate, normalized_text


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


if __name__ == "__main__":
    unittest.main()
