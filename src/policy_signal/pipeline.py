from __future__ import annotations

import math
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .attribution import (
    PressConferenceRoleAssigner,
    SpeakerAttributionState,
    SpeakerRoleAssigner,
)
from .classification import SignalClassifier
from .consensus import ConsensusContext, assess_surprise
from .domain import AnalysisResult, SegmentPrediction, TranscriptSegment, TurnSignal
from .media import MediaProcessor
from .relevance import HeuristicPolicyRelevanceGate, PolicyRelevanceGate, RelevanceAssessment
from .transcription import Transcriber


class EmptySpeechError(ValueError):
    pass


def merge_transcript_segments(
    segments: Sequence[TranscriptSegment],
    max_duration: float = 30.0,
    max_characters: int = 600,
) -> list[TranscriptSegment]:
    """Join tiny ASR fragments so the classifier sees enough policy context."""
    merged: list[TranscriptSegment] = []
    for segment in segments:
        if not segment.text.strip():
            continue
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        combined_text = f"{previous.text} {segment.text}".strip()
        combined_duration = segment.end - previous.start
        same_language = previous.language == segment.language
        same_speaker = (
            previous.speaker_id == segment.speaker_id
            and previous.speaker_role == segment.speaker_role
            and previous.turn_id == segment.turn_id
            and previous.question_turn_id == segment.question_turn_id
            and previous.question_context == segment.question_context
        )
        if (
            combined_duration <= max_duration
            and len(combined_text) <= max_characters
            and same_language
            and same_speaker
        ):
            merged[-1] = TranscriptSegment(
                start=previous.start,
                end=segment.end,
                text=combined_text,
                language=previous.language,
                speaker_id=previous.speaker_id,
                speaker_role=previous.speaker_role,
                role_confidence=min(previous.role_confidence, segment.role_confidence),
                role_source=previous.role_source,
                turn_id=previous.turn_id,
                question_turn_id=previous.question_turn_id,
                question_context=previous.question_context,
                words=previous.words + segment.words,
            )
        else:
            merged.append(segment)
    return merged


def _aggregate_prediction_probabilities(
    predictions: Sequence[SegmentPrediction],
    recency_half_life_seconds: float | None,
) -> tuple[float, float, list[SegmentPrediction]]:
    if not predictions:
        raise EmptySpeechError("No speech was detected in the audio track")

    latest_time = max(item.end for item in predictions)
    weighted_logits = 0.0
    total_weight = 0.0
    weighted_predictions: list[SegmentPrediction] = []

    for item in predictions:
        if recency_half_life_seconds is None:
            recency_weight = 1.0
        else:
            age = max(latest_time - item.end, 0.0)
            recency_weight = 2 ** (-age / max(recency_half_life_seconds, 0.01))
        content_weight = max(math.sqrt(max(len(item.text.split()), 1)), 1.0)
        weight = recency_weight * content_weight
        probability = min(max(item.bullish_probability, 1e-6), 1 - 1e-6)
        weighted_logits += weight * math.log(probability / (1 - probability))
        total_weight += weight
        weighted_predictions.append(replace(item, weight=round(weight, 6)))

    # Averaging logits avoids confidence increasing merely because a recording is long.
    bullish = 1.0 / (1.0 + math.exp(-(weighted_logits / total_weight)))
    bearish = 1.0 - bullish
    return bullish, bearish, weighted_predictions


def aggregate_predictions(
    predictions: Sequence[SegmentPrediction],
    *,
    asset_context: str,
    transcript: str,
    language: str | None,
    duration_seconds: float,
    model: str,
    neutral_margin: float = 0.12,
    recency_half_life_seconds: float | None = None,
) -> AnalysisResult:
    bullish, bearish, weighted_predictions = _aggregate_prediction_probabilities(
        predictions,
        recency_half_life_seconds,
    )
    label = "bullish" if bullish >= bearish else "bearish"
    signal_strength = abs(bullish - bearish)
    return AnalysisResult(
        label=label,
        confidence=max(bullish, bearish),
        bullish_probability=bullish,
        bearish_probability=bearish,
        signal_strength=signal_strength,
        low_signal=signal_strength < neutral_margin,
        asset_context=asset_context,
        transcript=transcript,
        language=language,
        duration_seconds=duration_seconds,
        model=model,
        directional_segment_count=len(predictions),
        policy_maker_segment_count=sum(item.speaker_role == "policy_maker" for item in predictions),
        journalist_segment_count=sum(item.speaker_role == "journalist" for item in predictions),
        unknown_speaker_segment_count=sum(item.speaker_role == "unknown" for item in predictions),
        segments=weighted_predictions,
    )


def build_turn_signals(
    assessments: Sequence[RelevanceAssessment],
    predictions: Sequence[SegmentPrediction],
    *,
    neutral_margin: float = 0.12,
) -> list[TurnSignal]:
    """Summarize every policy-maker turn, including neutral/irrelevant answers."""
    grouped: dict[str, list[tuple[RelevanceAssessment, SegmentPrediction | None]]] = {}
    prediction_iterator = iter(predictions)

    for index, assessment in enumerate(assessments, start=1):
        prediction = None
        if assessment.label == "directional":
            try:
                prediction = next(prediction_iterator)
            except StopIteration as exc:
                raise ValueError("Missing prediction for a directional segment") from exc
        turn_id = assessment.segment.turn_id or f"answer_{index:04d}"
        grouped.setdefault(turn_id, []).append((assessment, prediction))

    try:
        next(prediction_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("Received more predictions than directional segments")

    turns: list[TurnSignal] = []
    for turn_id, entries in grouped.items():
        segments = [assessment.segment for assessment, _ in entries]
        directional_predictions = [
            prediction for _, prediction in entries if prediction is not None
        ]
        if directional_predictions:
            bullish, bearish, _ = _aggregate_prediction_probabilities(
                directional_predictions,
                recency_half_life_seconds=None,
            )
        else:
            bullish = bearish = 0.5

        candidate_words = sum(len(segment.text.split()) for segment in segments)
        directional_words = sum(
            len(assessment.segment.text.split())
            for assessment, _ in entries
            if assessment.label == "directional"
        )
        signal_strength = abs(bullish - bearish)
        turns.append(
            TurnSignal(
                turn_id=turn_id,
                question_turn_id=next(
                    (
                        segment.question_turn_id
                        for segment in segments
                        if segment.question_turn_id is not None
                    ),
                    None,
                ),
                start=min(segment.start for segment in segments),
                end=max(segment.end for segment in segments),
                text=" ".join(segment.text for segment in segments),
                question_context=next(
                    (
                        segment.question_context
                        for segment in segments
                        if segment.question_context is not None
                    ),
                    None,
                ),
                speaker_id=segments[0].speaker_id,
                role_confidence=min(segment.role_confidence for segment in segments),
                label="bullish" if bullish >= bearish else "bearish",
                confidence=max(bullish, bearish),
                bullish_probability=bullish,
                bearish_probability=bearish,
                signal_strength=signal_strength,
                low_signal=signal_strength < neutral_margin,
                directional_coverage=(
                    directional_words / candidate_words if candidate_words else 0.0
                ),
                segment_count=len(entries),
                directional_segment_count=sum(
                    assessment.label == "directional" for assessment, _ in entries
                ),
                neutral_segment_count=sum(
                    assessment.label == "neutral" for assessment, _ in entries
                ),
                irrelevant_segment_count=sum(
                    assessment.label == "irrelevant" for assessment, _ in entries
                ),
            )
        )
    return turns


class VideoAnalysisPipeline:
    def __init__(
        self,
        media: MediaProcessor,
        transcriber: Transcriber,
        classifier: SignalClassifier,
        *,
        neutral_margin: float = 0.12,
        recency_half_life_seconds: float | None = None,
        role_assigner: SpeakerRoleAssigner | None = None,
        relevance_gate: PolicyRelevanceGate | None = None,
        require_consensus_for_action: bool = True,
        min_surprise: float = 0.15,
    ) -> None:
        self.media = media
        self.transcriber = transcriber
        self.classifier = classifier
        self.neutral_margin = neutral_margin
        self.recency_half_life_seconds = recency_half_life_seconds
        self.role_assigner = role_assigner or PressConferenceRoleAssigner()
        self.relevance_gate = relevance_gate or HeuristicPolicyRelevanceGate()
        self.require_consensus_for_action = require_consensus_for_action
        self.min_surprise = min_surprise

    def warmup(self) -> None:
        """Load immutable ASR and classifier weights before accepting live traffic."""
        self.transcriber.load()
        self.role_assigner.load()
        self.relevance_gate.load()
        self.classifier.load()

    def create_attribution_state(self) -> SpeakerAttributionState:
        """Create isolated speaker/turn state for one streaming connection."""
        return self.role_assigner.create_state()

    def analyze_video(
        self,
        video_path: Path,
        *,
        asset_context: str = "US equities",
        language: str | None = "en",
        consensus: ConsensusContext | None = None,
    ) -> AnalysisResult:
        with tempfile.TemporaryDirectory(prefix="policy-signal-audio-") as temporary_directory:
            audio_path = Path(temporary_directory) / "audio.wav"
            duration = self.media.extract_audio(video_path, audio_path)
            return self.analyze_normalized_audio(
                audio_path,
                duration_seconds=duration,
                asset_context=asset_context,
                language=language,
                consensus=consensus,
            )

    def analyze_normalized_audio(
        self,
        audio_path: Path,
        *,
        duration_seconds: float,
        asset_context: str = "US equities",
        language: str | None = "en",
        attribution_state: SpeakerAttributionState | None = None,
        consensus: ConsensusContext | None = None,
    ) -> AnalysisResult:
        raw_segments = self.transcriber.transcribe(audio_path, language)
        attributed_segments = self.role_assigner.assign(
            audio_path,
            raw_segments,
            state=attribution_state,
        )
        segments = merge_transcript_segments(attributed_segments)
        if not segments:
            raise EmptySpeechError("No speech was detected in the audio track")

        policy_maker_segments = [
            segment for segment in segments if segment.speaker_role == "policy_maker"
        ]
        assessments = self.relevance_gate.assess(policy_maker_segments)
        directional = [item for item in assessments if item.label == "directional"]
        directional_segments = [item.segment for item in directional]

        transcript = " ".join(segment.text for segment in raw_segments)
        detected_language = next(
            (segment.language for segment in raw_segments if segment.language), language
        )
        candidate_words = sum(len(segment.text.split()) for segment in policy_maker_segments)
        directional_words = sum(len(segment.text.split()) for segment in directional_segments)
        coverage = directional_words / candidate_words if candidate_words else 0.0
        excluded_count = len(segments) - len(directional_segments)
        journalist_count = sum(segment.speaker_role == "journalist" for segment in segments)
        unknown_count = sum(segment.speaker_role == "unknown" for segment in segments)
        neutral_count = sum(item.label == "neutral" for item in assessments)
        irrelevant_count = sum(item.label == "irrelevant" for item in assessments)
        audit_counts = {
            "directional_segment_count": len(directional_segments),
            "policy_maker_segment_count": len(policy_maker_segments),
            "journalist_segment_count": journalist_count,
            "unknown_speaker_segment_count": unknown_count,
            "neutral_segment_count": neutral_count,
            "irrelevant_segment_count": irrelevant_count,
        }
        model_name = (
            f"{self.transcriber.name} + {self.classifier.name} + "
            f"{self.role_assigner.name} + {self.relevance_gate.name}"
        )

        if not directional_segments:
            turns = build_turn_signals(
                assessments,
                [],
                neutral_margin=self.neutral_margin,
            )
            surprise = assess_surprise(
                0.5,
                consensus,
                transcript,
                base_margin=self.neutral_margin,
                min_surprise=self.min_surprise,
                require_consensus_for_action=self.require_consensus_for_action,
            )
            ctx = consensus or ConsensusContext()
            return AnalysisResult(
                label="bullish",
                confidence=0.5,
                bullish_probability=0.5,
                bearish_probability=0.5,
                signal_strength=0.0,
                low_signal=True,
                asset_context=asset_context,
                transcript=transcript,
                language=detected_language,
                duration_seconds=duration_seconds,
                model=model_name,
                directional_coverage=0.0,
                excluded_segment_count=excluded_count,
                turns=turns,
                consensus_mode=ctx.mode,
                priced_bullish_probability=ctx.priced_bullish,
                surprise_score=surprise.surprise_score,
                surprise_confidence=surprise.surprise_confidence,
                actionable=False,
                low_signal_reason=surprise.reason or "no_directional_content",
                neutral_margin_applied=surprise.neutral_margin_applied,
                information_effect_risk=surprise.information_effect_risk,
                **audit_counts,
                segments=[],
            )

        raw_predictions = self.classifier.classify(directional_segments, asset_context)
        predictions = [
            replace(
                prediction,
                relevance_label=assessment.label,
                relevance_confidence=assessment.confidence,
            )
            for prediction, assessment in zip(raw_predictions, directional, strict=True)
        ]
        turns = build_turn_signals(
            assessments,
            predictions,
            neutral_margin=self.neutral_margin,
        )
        result = aggregate_predictions(
            predictions,
            asset_context=asset_context,
            transcript=transcript,
            language=detected_language,
            duration_seconds=duration_seconds,
            model=model_name,
            neutral_margin=self.neutral_margin,
            recency_half_life_seconds=self.recency_half_life_seconds,
        )
        surprise = assess_surprise(
            result.bullish_probability,
            consensus,
            transcript,
            base_margin=self.neutral_margin,
            min_surprise=self.min_surprise,
            require_consensus_for_action=self.require_consensus_for_action,
        )
        ctx = consensus or ConsensusContext()
        label = "bullish" if surprise.adjusted_bullish >= surprise.adjusted_bearish else "bearish"
        final_low_signal = result.low_signal or not surprise.actionable
        reason = result.to_dict().get("low_signal_reason")
        if not surprise.actionable and surprise.reason:
            reason = surprise.reason
        elif result.low_signal and reason is None:
            reason = "weak_stance"
        return replace(
            result,
            label=label,
            confidence=max(surprise.adjusted_bullish, surprise.adjusted_bearish),
            bullish_probability=surprise.adjusted_bullish,
            bearish_probability=surprise.adjusted_bearish,
            signal_strength=surprise.signal_strength,
            low_signal=final_low_signal,
            directional_coverage=coverage,
            excluded_segment_count=excluded_count,
            turns=turns,
            consensus_mode=ctx.mode,
            priced_bullish_probability=ctx.priced_bullish,
            surprise_score=surprise.surprise_score,
            surprise_confidence=surprise.surprise_confidence,
            actionable=surprise.actionable and not result.low_signal,
            low_signal_reason=reason,
            neutral_margin_applied=surprise.neutral_margin_applied,
            information_effect_risk=surprise.information_effect_risk,
            **audit_counts,
        )
