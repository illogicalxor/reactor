from __future__ import annotations

import unittest

from policy_signal.domain import TranscriptSegment
from policy_signal.relevance import FineTunedPolicyRelevanceGate, HeuristicPolicyRelevanceGate


class HeuristicPolicyRelevanceGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = HeuristicPolicyRelevanceGate()

    def test_rejects_procedural_closing_comment(self) -> None:
        result = self.gate.assess(
            [TranscriptSegment(0, 5, "Thank you all. I have no news to make today.")]
        )[0]
        self.assertEqual(result.label, "irrelevant")

    def test_marks_undirected_economic_context_as_neutral(self) -> None:
        result = self.gate.assess([TranscriptSegment(0, 5, "Growth has remained broadly stable.")])[
            0
        ]
        self.assertEqual(result.label, "neutral")

    def test_passes_monetary_policy_content(self) -> None:
        result = self.gate.assess(
            [TranscriptSegment(0, 5, "Interest rates may need to remain higher.")]
        )[0]
        self.assertEqual(result.label, "directional")


class FineTunedPolicyRelevanceGateTests(unittest.TestCase):
    def test_maps_checkpoint_labels(self) -> None:
        gate = FineTunedPolicyRelevanceGate("stub")

        def model(texts, **kwargs):
            del texts, kwargs
            return [
                [
                    {"label": "irrelevant", "score": 0.05},
                    {"label": "neutral", "score": 0.15},
                    {"label": "directional", "score": 0.8},
                ]
            ]

        gate._pipeline = model
        result = gate.assess([TranscriptSegment(0, 5, "Interest rates may rise.")])[0]
        self.assertEqual(result.label, "directional")
        self.assertAlmostEqual(result.confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
