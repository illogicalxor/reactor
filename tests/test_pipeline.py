from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from policy_signal.classification import HeuristicSignalClassifier
from policy_signal.domain import SegmentPrediction, TranscriptSegment
from policy_signal.pipeline import (
    VideoAnalysisPipeline,
    aggregate_predictions,
    merge_transcript_segments,
)


class StubMedia:
    def extract_audio(self, source: Path, destination: Path) -> float:
        del source
        destination.touch()
        return 12.0


class StubTranscriber:
    name = "stub-asr"

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio_path: Path, language: str | None = None):
        del audio_path, language
        return [
            TranscriptSegment(0, 4, "Inflation is declining.", "en"),
            TranscriptSegment(4, 9, "We are ready to cut rates.", "en"),
        ]


class PressConferenceTranscriber(StubTranscriber):
    def transcribe(self, audio_path: Path, language: str | None = None):
        del audio_path, language
        return [
            TranscriptSegment(0, 5, "Inflation remains elevated.", "en"),
            TranscriptSegment(
                5,
                12,
                "Reuters. Chair Warsh, will you raise interest rates?",
                "en",
            ),
            TranscriptSegment(12, 20, "Interest rates will depend on the data.", "en"),
            TranscriptSegment(45, 50, "Thank you all. There is no news to make.", "en"),
        ]


class ProceduralTranscriber(StubTranscriber):
    def transcribe(self, audio_path: Path, language: str | None = None):
        del audio_path, language
        return [TranscriptSegment(0, 5, "Thank you all. There is no news to make.", "en")]


class PipelineTests(unittest.TestCase):
    def test_merges_short_adjacent_segments(self) -> None:
        segments = [
            TranscriptSegment(0, 3, "Inflation is declining.", "en"),
            TranscriptSegment(3, 8, "We can cut rates.", "en"),
        ]
        merged = merge_transcript_segments(segments)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "Inflation is declining. We can cut rates.")

    def test_does_not_merge_across_speaker_roles(self) -> None:
        segments = [
            TranscriptSegment(0, 3, "Will rates rise?", "en", speaker_role="journalist"),
            TranscriptSegment(3, 8, "We will assess that.", "en", speaker_role="policy_maker"),
        ]
        merged = merge_transcript_segments(segments)
        self.assertEqual(len(merged), 2)

    def test_recent_language_has_more_weight(self) -> None:
        predictions = [
            SegmentPrediction(0, 10, "Earlier easing signal", "bullish", 0.9, 0.1, 0.9),
            SegmentPrediction(190, 200, "Latest tightening signal", "bearish", 0.1, 0.9, 0.9),
        ]
        result = aggregate_predictions(
            predictions,
            asset_context="equities",
            transcript="test",
            language="en",
            duration_seconds=200,
            model="stub",
            recency_half_life_seconds=30,
        )
        self.assertEqual(result.label, "bearish")
        self.assertGreater(result.bearish_probability, 0.8)

    def test_offline_aggregation_does_not_favor_arbitrary_closing_remarks(self) -> None:
        predictions = [
            SegmentPrediction(0, 10, "Earlier tightening signal", "bearish", 0.1, 0.9, 0.9),
            SegmentPrediction(190, 200, "Closing optimistic remark", "bullish", 0.9, 0.1, 0.9),
        ]
        result = aggregate_predictions(
            predictions,
            asset_context="equities",
            transcript="test",
            language="en",
            duration_seconds=200,
            model="stub",
        )
        self.assertAlmostEqual(result.bullish_probability, 0.5)
        self.assertTrue(result.low_signal)

    def test_marks_ambiguous_result_as_low_signal(self) -> None:
        prediction = SegmentPrediction(0, 5, "Ambiguous", "bullish", 0.52, 0.48, 0.52)
        result = aggregate_predictions(
            [prediction],
            asset_context="equities",
            transcript="Ambiguous",
            language="en",
            duration_seconds=5,
            model="stub",
        )
        self.assertTrue(result.low_signal)
        self.assertEqual(result.label, "bullish")

    def test_video_pipeline_end_to_end_with_test_doubles(self) -> None:
        pipeline = VideoAnalysisPipeline(
            media=StubMedia(),
            transcriber=StubTranscriber(),
            classifier=HeuristicSignalClassifier(),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = pipeline.analyze_video(Path(directory) / "speech.mp4")
        self.assertEqual(result.label, "bullish")
        self.assertEqual(result.language, "en")
        self.assertIn("cut rates", result.transcript)
        self.assertEqual(result.duration_seconds, 12.0)

    def test_pipeline_excludes_reporter_and_irrelevant_segments(self) -> None:
        pipeline = VideoAnalysisPipeline(
            media=StubMedia(),
            transcriber=PressConferenceTranscriber(),
            classifier=HeuristicSignalClassifier(),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = pipeline.analyze_video(Path(directory) / "press-conference.mp4")

        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.excluded_segment_count, 2)
        self.assertEqual(result.directional_segment_count, 2)
        self.assertEqual(result.journalist_segment_count, 1)
        self.assertEqual(result.irrelevant_segment_count, 1)
        self.assertTrue(all(item.speaker_role == "policy_maker" for item in result.segments))
        self.assertFalse(any("Reuters" in item.text for item in result.segments))
        answer = next(item for item in result.segments if item.start == 12)
        self.assertIn("Reuters", answer.question_context or "")
        self.assertLess(result.directional_coverage, 1.0)

    def test_pipeline_abstains_when_nothing_is_directional(self) -> None:
        pipeline = VideoAnalysisPipeline(
            media=StubMedia(),
            transcriber=ProceduralTranscriber(),
            classifier=HeuristicSignalClassifier(),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = pipeline.analyze_video(Path(directory) / "procedural.mp4")

        self.assertEqual(result.bullish_probability, 0.5)
        self.assertEqual(result.bearish_probability, 0.5)
        self.assertTrue(result.low_signal)
        self.assertEqual(result.directional_segment_count, 0)
        self.assertEqual(result.irrelevant_segment_count, 1)

    def test_warmup_loads_inference_weights(self) -> None:
        transcriber = StubTranscriber()
        classifier = HeuristicSignalClassifier()
        pipeline = VideoAnalysisPipeline(
            media=StubMedia(),
            transcriber=transcriber,
            classifier=classifier,
        )
        pipeline.warmup()
        self.assertTrue(transcriber.loaded)


if __name__ == "__main__":
    unittest.main()
