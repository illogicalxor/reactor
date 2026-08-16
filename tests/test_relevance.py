from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from policy_signal.domain import TranscriptSegment
from policy_signal.relevance import (
    FineTunedPolicyRelevanceGate,
    HeuristicPolicyRelevanceGate,
    RelevanceError,
)


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
    @staticmethod
    def _neutral_model(texts, **kwargs):
        del kwargs
        return [
            [
                {"label": "irrelevant", "score": 0.25},
                {"label": "neutral", "score": 0.7},
                {"label": "directional", "score": 0.05},
            ]
            for _ in texts
        ]

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

    def test_overrides_complete_question_management_turn(self) -> None:
        gate = FineTunedPolicyRelevanceGate("stub")
        gate._pipeline = self._neutral_model

        result = gate.assess([TranscriptSegment(0, 5, "Let us go to the next question.")])[0]

        self.assertEqual(result.label, "irrelevant")
        self.assertAlmostEqual(result.confidence, 0.98)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)

    def test_does_not_override_substantive_answer_mentioning_question(self) -> None:
        gate = FineTunedPolicyRelevanceGate("stub")
        gate._pipeline = self._neutral_model
        text = "The next question is whether interest rates should remain restrictive."

        result = gate.assess([TranscriptSegment(0, 5, text)])[0]

        self.assertEqual(result.label, "neutral")

    def test_loads_checkpoint_temperature_calibration(self) -> None:
        with TemporaryDirectory() as directory:
            Path(directory, "relevance_calibration.json").write_text(
                '{"temperature": 2.0}',
                encoding="utf-8",
            )
            gate = FineTunedPolicyRelevanceGate(directory)
            gate._pipeline = self._neutral_model

            result = gate.assess([TranscriptSegment(0, 5, "Growth remained stable.")])[0]

        self.assertEqual(gate.temperature, 2.0)
        self.assertLess(result.confidence, 0.7)

    def test_rejects_non_finite_temperature(self) -> None:
        with self.assertRaises(RelevanceError):
            FineTunedPolicyRelevanceGate("stub", temperature=float("nan"))


if __name__ == "__main__":
    unittest.main()
