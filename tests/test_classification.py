from __future__ import annotations

import unittest

from policy_signal.classification import (
    FineTunedSignalClassifier,
    HeuristicSignalClassifier,
    ZeroShotSignalClassifier,
)
from policy_signal.domain import TranscriptSegment


class HeuristicClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = HeuristicSignalClassifier()

    def test_classifies_fed_easing_as_bullish_for_us_equities(self) -> None:
        segment = TranscriptSegment(0, 4, "There is room to cut rates.", "en")
        prediction = self.classifier.classify([segment], "US equities")[0]
        self.assertEqual(prediction.label, "bullish")
        self.assertGreater(prediction.bullish_probability, 0.8)

    def test_classifies_higher_for_longer_as_bearish(self) -> None:
        segment = TranscriptSegment(0, 5, "Rates must remain higher for longer.", "en")
        prediction = self.classifier.classify([segment], "equities")[0]
        self.assertEqual(prediction.label, "bearish")
        self.assertGreater(prediction.bearish_probability, 0.8)

    def test_negated_cut_is_not_bullish(self) -> None:
        segment = TranscriptSegment(0, 5, "We will not cut rates this year.", "en")
        prediction = self.classifier.classify([segment], "equities")[0]
        self.assertEqual(prediction.label, "bearish")

    def test_preserves_speaker_and_question_evidence(self) -> None:
        segment = TranscriptSegment(
            0,
            5,
            "Rates must remain higher for longer.",
            "en",
            speaker_id="SPEAKER_00",
            speaker_role="policy_maker",
            role_confidence=0.95,
            role_source="audio",
            turn_id="answer_0002",
            question_turn_id="question_0001",
            question_context="Could rates remain elevated?",
        )
        prediction = self.classifier.classify([segment], "equities")[0]
        self.assertEqual(prediction.speaker_id, "SPEAKER_00")
        self.assertEqual(prediction.role_source, "audio")
        self.assertEqual(prediction.turn_id, segment.turn_id)
        self.assertEqual(prediction.question_turn_id, segment.question_turn_id)
        self.assertEqual(prediction.question_context, segment.question_context)


class TransformerAdapterTests(unittest.TestCase):
    def test_zero_shot_maps_scores_to_contract_labels(self) -> None:
        classifier = ZeroShotSignalClassifier("stub")

        def model(texts, *, candidate_labels, **kwargs):
            del texts, kwargs
            return [{"labels": candidate_labels, "scores": [0.7, 0.3]}]

        classifier._pipeline = model
        prediction = classifier.classify(
            [TranscriptSegment(0, 5, "Policy statement", "en")], "equities"
        )[0]
        self.assertEqual(prediction.label, "bullish")
        self.assertAlmostEqual(prediction.bullish_probability, 0.7)

    def test_fine_tuned_maps_checkpoint_labels(self) -> None:
        classifier = FineTunedSignalClassifier("stub")

        def model(texts, **kwargs):
            del texts, kwargs
            return [
                [
                    {"label": "bearish", "score": 0.8},
                    {"label": "bullish", "score": 0.2},
                ]
            ]

        classifier._pipeline = model
        prediction = classifier.classify(
            [TranscriptSegment(0, 5, "Policy statement", "en")], "equities"
        )[0]
        self.assertEqual(prediction.label, "bearish")
        self.assertAlmostEqual(prediction.bearish_probability, 0.8)


if __name__ == "__main__":
    unittest.main()
