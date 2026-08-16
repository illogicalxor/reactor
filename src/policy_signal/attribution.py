from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from .domain import SpeakerRole, TranscriptSegment, TranscriptWord


class SpeakerAttributionError(RuntimeError):
    pass


@dataclass(slots=True)
class SpeakerAttributionState:
    """Mutable attribution state owned by one streaming connection."""

    detected_role: SpeakerRole = "policy_maker"
    detected_speaker: str = "policy_maker_0"
    detected_journalist_index: int = 0
    active_role: SpeakerRole | None = None
    active_turn_id: str | None = None
    question_index: int = 0
    answer_index: int = 0
    unknown_index: int = 0
    question_parts: list[str] = field(default_factory=list)
    latest_question: str | None = None
    latest_question_turn_id: str | None = None


class SpeakerRoleAssigner(Protocol):
    @property
    def name(self) -> str: ...

    def load(self) -> None: ...

    def create_state(self) -> SpeakerAttributionState: ...

    def assign(
        self,
        audio_path: Path,
        segments: Sequence[TranscriptSegment],
        *,
        state: SpeakerAttributionState | None = None,
    ) -> list[TranscriptSegment]: ...


def _attach_question_context(
    segments: Sequence[TranscriptSegment],
    state: SpeakerAttributionState | None = None,
) -> list[TranscriptSegment]:
    """Assign stable turn IDs and attach each reporter turn to its answer."""
    state = state or SpeakerAttributionState()
    result: list[TranscriptSegment] = []

    for segment in segments:
        if segment.speaker_role != state.active_role:
            state.active_role = segment.speaker_role
            if state.active_role == "journalist":
                state.question_index += 1
                state.active_turn_id = f"question_{state.question_index:04d}"
                state.question_parts = []
            elif state.active_role == "policy_maker":
                state.answer_index += 1
                state.active_turn_id = f"answer_{state.answer_index:04d}"
                if state.question_parts:
                    state.latest_question = " ".join(state.question_parts).strip()
                    state.latest_question_turn_id = f"question_{state.question_index:04d}"
                    state.question_parts = []
            else:
                state.unknown_index += 1
                state.active_turn_id = f"unknown_{state.unknown_index:04d}"

        if segment.speaker_role == "journalist":
            state.question_parts.append(segment.text.strip())
            result.append(
                replace(
                    segment,
                    turn_id=state.active_turn_id,
                    question_turn_id=None,
                    question_context=None,
                )
            )
            continue
        if segment.speaker_role == "policy_maker":
            result.append(
                replace(
                    segment,
                    turn_id=state.active_turn_id,
                    question_turn_id=state.latest_question_turn_id,
                    question_context=state.latest_question,
                )
            )
            continue
        result.append(
            replace(
                segment,
                turn_id=state.active_turn_id,
                question_turn_id=None,
                question_context=None,
            )
        )
    return result


class PassthroughRoleAssigner:
    """Treat all input speech as trusted policy-maker audio when filtering is disabled."""

    @property
    def name(self) -> str:
        return "speaker-role:none"

    def load(self) -> None:
        return None

    def create_state(self) -> SpeakerAttributionState:
        return SpeakerAttributionState()

    def assign(
        self,
        audio_path: Path,
        segments: Sequence[TranscriptSegment],
        *,
        state: SpeakerAttributionState | None = None,
    ) -> list[TranscriptSegment]:
        del audio_path
        attributed = [
            replace(
                segment,
                speaker_id=segment.speaker_id or "unattributed_policy_maker",
                speaker_role="policy_maker",
            )
            for segment in segments
        ]
        return _attach_question_context(attributed, state)


class PressConferenceRoleAssigner:
    """Conservative text fallback for the regular policy press-conference format.

    This is not acoustic diarization. It identifies explicit reporter introductions,
    carries the question until its final question mark, and attributes the following
    turn to the policy maker. The interface is intentionally replaceable by an
    audio-visual diarizer without changing downstream classification.
    """

    _OUTLET = re.compile(
        r"\b(?:reuters|bloomberg|cnbc|nbc(?: news)?|cbs(?: news)?|abc(?: news)?|"
        r"fox(?: business| news)?|associated press|wall street journal|new york times|"
        r"washington post|financial times|marketplace|yahoo finance|axios|politico)\b",
        re.IGNORECASE,
    )
    _GREETING = re.compile(
        r"\b(?:hi|hello|thank you|thanks|good afternoon)\b.{0,45}"
        r"\b(?:chair|chairman|chairwoman)\b",
        re.IGNORECASE,
    )

    @property
    def name(self) -> str:
        return "speaker-role:press-conference-text-v1"

    def load(self) -> None:
        return None

    def create_state(self) -> SpeakerAttributionState:
        return SpeakerAttributionState()

    @classmethod
    def _starts_reporter_turn(cls, text: str) -> bool:
        return bool(cls._OUTLET.search(text) or cls._GREETING.search(text))

    def assign(
        self,
        audio_path: Path,
        segments: Sequence[TranscriptSegment],
        *,
        state: SpeakerAttributionState | None = None,
    ) -> list[TranscriptSegment]:
        del audio_path
        state = state or self.create_state()
        attributed: list[TranscriptSegment] = []

        for segment in segments:
            text = segment.text.strip()
            if self._starts_reporter_turn(text):
                state.detected_journalist_index += 1
                state.detected_role = "journalist"
                state.detected_speaker = f"journalist_{state.detected_journalist_index}"

            if state.detected_role == "journalist":
                attributed.append(
                    replace(
                        segment,
                        speaker_id=state.detected_speaker,
                        speaker_role="journalist",
                        role_confidence=0.75,
                        role_source="text",
                        question_context=None,
                    )
                )
                if "?" in text:
                    state.detected_role = "policy_maker"
                    state.detected_speaker = "policy_maker_0"
                continue

            attributed.append(
                replace(
                    segment,
                    speaker_id=state.detected_speaker,
                    speaker_role="policy_maker",
                    role_confidence=0.65,
                    role_source="text",
                    question_context=None,
                )
            )

        return _attach_question_context(attributed, state)


@dataclass(frozen=True, slots=True)
class _DiarizationTurn:
    start: float
    end: float
    speaker: str


class PyannoteRoleAssigner:
    """Map diarized speaker clusters to Chair/reporter roles.

    FOMC press conferences begin with a long statement from the Chair. The speaker
    with the greatest overlap in that opening window is therefore used as the
    policy-maker anchor; all other clusters are treated as journalists. This solves
    speaker changes acoustically but does not establish a person's legal identity.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-community-1",
        *,
        token: str | None = None,
        device: str = "auto",
        opening_anchor_seconds: float = 600.0,
    ) -> None:
        self.model_name = model_name
        self.token = token
        self.device = device
        self.opening_anchor_seconds = opening_anchor_seconds
        self._pipeline = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"speaker-role:pyannote-anchor:{self.model_name}"

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._load_lock:
                if self._pipeline is None:
                    try:
                        import torch
                        from pyannote.audio import Pipeline
                    except ImportError as exc:
                        raise SpeakerAttributionError(
                            "pyannote.audio is not installed; run `pip install -e '.[diarization]'`"
                        ) from exc
                    try:
                        pipeline = Pipeline.from_pretrained(
                            self.model_name,
                            token=self.token,
                        )
                        target = self.device
                        if target == "auto":
                            target = "cuda" if torch.cuda.is_available() else "cpu"
                        if target != "cpu":
                            pipeline.to(torch.device(target))
                    except Exception as exc:
                        raise SpeakerAttributionError(
                            f"Could not load diarization model `{self.model_name}`"
                        ) from exc
                    self._pipeline = pipeline
        return self._pipeline

    def load(self) -> None:
        self._get_pipeline()

    def create_state(self) -> SpeakerAttributionState:
        return SpeakerAttributionState()

    @staticmethod
    def _turns(output: object) -> list[_DiarizationTurn]:
        # Community-1 exposes an exclusive timeline designed to reconcile diarization
        # with ASR timestamps. Older pyannote versions fall back to the regular one.
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", output)
        if not hasattr(annotation, "itertracks"):
            raise SpeakerAttributionError("Unsupported pyannote diarization output")
        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(_DiarizationTurn(float(turn.start), float(turn.end), str(speaker)))
        return turns

    @staticmethod
    def _overlap(start: float, end: float, turn: _DiarizationTurn) -> float:
        return max(min(end, turn.end) - max(start, turn.start), 0.0)

    @classmethod
    def _speaker_for_interval(
        cls,
        start: float,
        end: float,
        turns: Sequence[_DiarizationTurn],
    ) -> tuple[str | None, float]:
        by_speaker: dict[str, float] = {}
        for turn in turns:
            overlap = cls._overlap(start, end, turn)
            by_speaker[turn.speaker] = by_speaker.get(turn.speaker, 0.0) + overlap
        speaker = max(by_speaker, key=by_speaker.get) if by_speaker else None
        return speaker, by_speaker.get(speaker, 0.0) if speaker else 0.0

    @staticmethod
    def _role(speaker: str | None, policy_speaker: str) -> SpeakerRole:
        if speaker is None:
            return "unknown"
        return "policy_maker" if speaker == policy_speaker else "journalist"

    @classmethod
    def _split_word_aligned_segment(
        cls,
        segment: TranscriptSegment,
        turns: Sequence[_DiarizationTurn],
        policy_speaker: str,
    ) -> list[TranscriptSegment]:
        groups: list[tuple[str | None, list[TranscriptWord], float]] = []
        for word in segment.words:
            speaker, overlap = cls._speaker_for_interval(word.start, word.end, turns)
            if groups and groups[-1][0] == speaker:
                groups[-1][1].append(word)
                previous_speaker, previous_words, previous_overlap = groups[-1]
                groups[-1] = (
                    previous_speaker,
                    previous_words,
                    previous_overlap + overlap,
                )
            else:
                groups.append((speaker, [word], overlap))

        attributed = []
        for speaker, words, overlap in groups:
            text = "".join(word.text for word in words).strip()
            if not text:
                continue
            duration = sum(word.duration for word in words)
            attributed.append(
                replace(
                    segment,
                    start=words[0].start,
                    end=words[-1].end,
                    text=text,
                    speaker_id=speaker,
                    speaker_role=cls._role(speaker, policy_speaker),
                    role_confidence=min(overlap / max(duration, 0.01), 1.0),
                    role_source="audio",
                    turn_id=None,
                    question_turn_id=None,
                    question_context=None,
                    words=tuple(words),
                )
            )
        return attributed

    def assign(
        self,
        audio_path: Path,
        segments: Sequence[TranscriptSegment],
        *,
        state: SpeakerAttributionState | None = None,
    ) -> list[TranscriptSegment]:
        try:
            with self._inference_lock:
                output = self._get_pipeline()(str(audio_path))
            turns = self._turns(output)
        except SpeakerAttributionError:
            raise
        except Exception as exc:
            raise SpeakerAttributionError("Speaker diarization failed") from exc
        if not turns:
            raise SpeakerAttributionError("Speaker diarization detected no speakers")

        opening_overlap: dict[str, float] = {}
        for turn in turns:
            overlap = self._overlap(0.0, self.opening_anchor_seconds, turn)
            opening_overlap[turn.speaker] = opening_overlap.get(turn.speaker, 0.0) + overlap
        policy_speaker = max(opening_overlap, key=opening_overlap.get)

        attributed = []
        for segment in segments:
            if segment.words:
                attributed.extend(self._split_word_aligned_segment(segment, turns, policy_speaker))
                continue
            speaker, overlap = self._speaker_for_interval(
                segment.start,
                segment.end,
                turns,
            )
            confidence = min(overlap / segment.duration, 1.0)
            if overlap <= 0:
                role: SpeakerRole = "unknown"
                speaker_id = None
            else:
                role = self._role(speaker, policy_speaker)
                speaker_id = speaker
            attributed.append(
                replace(
                    segment,
                    speaker_id=speaker_id,
                    speaker_role=role,
                    role_confidence=confidence,
                    role_source="audio",
                    turn_id=None,
                    question_turn_id=None,
                    question_context=None,
                )
            )
        return _attach_question_context(attributed, state)
