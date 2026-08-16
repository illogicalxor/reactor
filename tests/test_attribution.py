from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from policy_signal.attribution import PressConferenceRoleAssigner, PyannoteRoleAssigner
from policy_signal.domain import TranscriptSegment, TranscriptWord


class PressConferenceRoleAssignerTests(unittest.TestCase):
    def test_keeps_reporter_question_as_context_but_not_policy_speech(self) -> None:
        segments = [
            TranscriptSegment(0, 8, "Good afternoon. Inflation remains elevated.", "en"),
            TranscriptSegment(
                8,
                16,
                "Steve Liesman, CNBC. Chair Warsh, could rates rise again?",
                "en",
            ),
            TranscriptSegment(16, 25, "We will decide meeting by meeting.", "en"),
        ]

        result = PressConferenceRoleAssigner().assign(Path("speech.wav"), segments)

        self.assertEqual(result[0].speaker_role, "policy_maker")
        self.assertEqual(result[1].speaker_role, "journalist")
        self.assertEqual(result[2].speaker_role, "policy_maker")
        self.assertEqual(result[2].question_context, segments[1].text)
        self.assertEqual(result[2].role_source, "text")
        self.assertEqual(
            [segment.turn_id for segment in result],
            ["answer_0001", "question_0001", "answer_0002"],
        )
        self.assertEqual(result[2].question_turn_id, result[1].turn_id)

    def test_preserves_question_and_answer_turns_across_stream_windows(self) -> None:
        assigner = PressConferenceRoleAssigner()
        state = assigner.create_state()

        first = assigner.assign(
            Path("window-1.wav"),
            [
                TranscriptSegment(0, 4, "Inflation remains elevated.", "en"),
                TranscriptSegment(4, 8, "Reuters. Chair, could interest rates", "en"),
            ],
            state=state,
        )
        second = assigner.assign(
            Path("window-2.wav"),
            [
                TranscriptSegment(0, 3, "remain high for longer?", "en"),
                TranscriptSegment(3, 8, "We will decide meeting by meeting.", "en"),
            ],
            state=state,
        )
        third = assigner.assign(
            Path("window-3.wav"),
            [TranscriptSegment(0, 4, "Incoming data will guide us.", "en")],
            state=state,
        )

        self.assertEqual(first[0].turn_id, "answer_0001")
        self.assertEqual(first[1].turn_id, "question_0001")
        self.assertEqual(second[0].turn_id, "question_0001")
        self.assertEqual(second[1].turn_id, "answer_0002")
        self.assertEqual(third[0].turn_id, "answer_0002")
        self.assertEqual(second[1].question_turn_id, "question_0001")
        self.assertEqual(
            second[1].question_context,
            "Reuters. Chair, could interest rates remain high for longer?",
        )


@dataclass(frozen=True)
class FakeTurn:
    start: float
    end: float


class FakeAnnotation:
    def itertracks(self, *, yield_label: bool):
        self.yield_label = yield_label
        yield FakeTurn(0, 20), None, "SPEAKER_CHAIR"
        yield FakeTurn(20, 30), None, "SPEAKER_REPORTER"
        yield FakeTurn(30, 50), None, "SPEAKER_CHAIR"


class FakeOutput:
    exclusive_speaker_diarization = FakeAnnotation()


class PyannoteRoleAssignerTests(unittest.TestCase):
    def test_maps_opening_speaker_and_attaches_reporter_question(self) -> None:
        assigner = PyannoteRoleAssigner(opening_anchor_seconds=25)
        assigner._pipeline = lambda _: FakeOutput()
        segments = [
            TranscriptSegment(0, 20, "Opening statement.", "en"),
            TranscriptSegment(20, 30, "Will you raise rates?", "en"),
            TranscriptSegment(30, 50, "We will assess incoming data.", "en"),
        ]

        result = assigner.assign(Path("speech.wav"), segments)

        self.assertEqual(result[0].speaker_role, "policy_maker")
        self.assertEqual(result[1].speaker_role, "journalist")
        self.assertEqual(result[2].speaker_role, "policy_maker")
        self.assertEqual(result[2].question_context, "Will you raise rates?")
        self.assertEqual(result[2].question_turn_id, result[1].turn_id)
        self.assertEqual(result[2].role_source, "audio")

    def test_splits_asr_segment_when_speaker_changes_between_words(self) -> None:
        assigner = PyannoteRoleAssigner(opening_anchor_seconds=25)
        assigner._pipeline = lambda _: FakeOutput()
        segments = [
            TranscriptSegment(
                0,
                20,
                "Opening statement.",
                "en",
                words=(TranscriptWord(0, 10, " Opening statement."),),
            ),
            TranscriptSegment(
                18,
                32,
                "We are ready. Will rates rise? We will assess it.",
                "en",
                words=(
                    TranscriptWord(18, 20, " We are ready."),
                    TranscriptWord(20, 30, " Will rates rise?"),
                    TranscriptWord(30, 32, " We will assess it."),
                ),
            ),
        ]

        result = assigner.assign(Path("speech.wav"), segments)

        self.assertEqual(
            [segment.speaker_role for segment in result],
            ["policy_maker", "policy_maker", "journalist", "policy_maker"],
        )
        self.assertEqual(
            [segment.turn_id for segment in result],
            ["answer_0001", "answer_0001", "question_0001", "answer_0002"],
        )
        self.assertEqual(result[-1].question_context, "Will rates rise?")


if __name__ == "__main__":
    unittest.main()
