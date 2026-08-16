from __future__ import annotations

import unittest

from policy_signal.evaluate_relevance import evaluate_probabilities


class RelevanceEvaluationTests(unittest.TestCase):
    def test_reports_per_class_and_per_event_metrics(self) -> None:
        rows = [
            {"text": "Thanks.", "label": "irrelevant", "event_id": "a", "source": "test"},
            {"text": "Growth held.", "label": "neutral", "event_id": "b", "source": "test"},
            {
                "text": "Rates may rise.",
                "label": "directional",
                "event_id": "b",
                "source": "test",
            },
        ]
        probabilities = [
            {"irrelevant": 0.8, "neutral": 0.1, "directional": 0.1},
            {"irrelevant": 0.1, "neutral": 0.7, "directional": 0.2},
            {"irrelevant": 0.1, "neutral": 0.2, "directional": 0.7},
        ]

        result = evaluate_probabilities(rows, probabilities)

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1_macro"], 1.0)
        self.assertEqual(result["per_class"]["irrelevant"]["recall"], 1.0)
        self.assertEqual(result["per_event"]["b"]["examples"], 2)


if __name__ == "__main__":
    unittest.main()
