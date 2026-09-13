from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

SignalLabel = Literal["bullish", "bearish"]
SpeakerRole = Literal["policy_maker", "journalist", "unknown"]
RoleSource = Literal["none", "text", "audio", "video", "combined"]
RelevanceLabel = Literal["irrelevant", "neutral", "directional"]


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    start: float
    end: float
    text: str
    probability: float | None = None

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.01)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    language: str | None = None
    speaker_id: str | None = None
    speaker_role: SpeakerRole = "unknown"
    role_confidence: float = 0.0
    role_source: RoleSource = "none"
    question_context: str | None = None
    words: tuple[TranscriptWord, ...] = ()
    turn_id: str | None = None
    question_turn_id: str | None = None

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.01)


@dataclass(frozen=True, slots=True)
class SegmentPrediction:
    start: float
    end: float
    text: str
    label: SignalLabel
    bullish_probability: float
    bearish_probability: float
    confidence: float
    weight: float = 1.0
    speaker_id: str | None = None
    speaker_role: SpeakerRole = "unknown"
    role_confidence: float = 0.0
    role_source: RoleSource = "none"
    question_context: str | None = None
    relevance_label: RelevanceLabel = "directional"
    relevance_confidence: float = 1.0
    turn_id: str | None = None
    question_turn_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TurnSignal:
    """Auditable stance summary for one policy-maker answer or statement turn."""

    turn_id: str
    question_turn_id: str | None
    start: float
    end: float
    text: str
    question_context: str | None
    speaker_id: str | None
    role_confidence: float
    label: SignalLabel
    confidence: float
    bullish_probability: float
    bearish_probability: float
    signal_strength: float
    low_signal: bool
    directional_coverage: float
    segment_count: int
    directional_segment_count: int
    neutral_segment_count: int
    irrelevant_segment_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    label: SignalLabel
    confidence: float
    bullish_probability: float
    bearish_probability: float
    signal_strength: float
    low_signal: bool
    asset_context: str
    transcript: str
    language: str | None
    duration_seconds: float
    model: str
    directional_coverage: float = 1.0
    excluded_segment_count: int = 0
    directional_segment_count: int = 0
    policy_maker_segment_count: int = 0
    journalist_segment_count: int = 0
    unknown_speaker_segment_count: int = 0
    neutral_segment_count: int = 0
    irrelevant_segment_count: int = 0
    # Consensus / surprise layer. Stance alone is not market impact.
    consensus_mode: str = "none"
    priced_bullish_probability: float | None = None
    surprise_score: float = 0.0
    surprise_confidence: float = 0.0
    actionable: bool = False
    low_signal_reason: str | None = None
    neutral_margin_applied: float = 0.12
    information_effect_risk: bool = False
    segments: list[SegmentPrediction] = field(default_factory=list)
    turns: list[TurnSignal] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["turns"] = [turn.to_dict() for turn in self.turns]
        payload["segments"] = [segment.to_dict() for segment in self.segments]
        return payload
