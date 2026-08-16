from __future__ import annotations

import math
import re
import threading
from collections.abc import Sequence
from typing import Protocol

from .domain import SegmentPrediction, TranscriptSegment


class ClassificationError(RuntimeError):
    pass


class SignalClassifier(Protocol):
    @property
    def name(self) -> str: ...

    def classify(
        self,
        segments: Sequence[TranscriptSegment],
        asset_context: str,
    ) -> list[SegmentPrediction]: ...

    def load(self) -> None: ...


class ZeroShotSignalClassifier:
    """English NLI cold-start classifier for US policy statements.

    For production, replace the checkpoint with one fine-tuned on labelled policy
    statements using ``policy-signal-train``.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"zero-shot:{self.model_name}"

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._load_lock:
                if self._pipeline is None:
                    try:
                        import torch
                        from transformers import pipeline
                    except ImportError as exc:
                        raise ClassificationError(
                            "torch/transformers are not installed; run `pip install .`"
                        ) from exc
                    device = 0 if torch.cuda.is_available() else -1
                    try:
                        self._pipeline = pipeline(
                            task="zero-shot-classification",
                            model=self.model_name,
                            device=device,
                        )
                    except Exception as exc:
                        raise ClassificationError(
                            f"Could not load classifier `{self.model_name}`"
                        ) from exc
        return self._pipeline

    def load(self) -> None:
        self._get_pipeline()

    def classify(
        self,
        segments: Sequence[TranscriptSegment],
        asset_context: str,
    ) -> list[SegmentPrediction]:
        if not segments:
            return []
        model = self._get_pipeline()
        labels = [f"bullish for {asset_context}", f"bearish for {asset_context}"]
        try:
            with self._inference_lock:
                outputs = model(
                    [segment.text for segment in segments],
                    candidate_labels=labels,
                    hypothesis_template="The market impact of this policy-maker statement is {}.",
                    multi_label=False,
                )
        except Exception as exc:
            raise ClassificationError("NLP classification failed") from exc

        if isinstance(outputs, dict):
            outputs = [outputs]
        predictions: list[SegmentPrediction] = []
        for segment, output in zip(segments, outputs, strict=True):
            probabilities = dict(zip(output["labels"], output["scores"], strict=True))
            bullish = float(probabilities[labels[0]])
            bearish = float(probabilities[labels[1]])
            total = bullish + bearish
            bullish, bearish = bullish / total, bearish / total
            label = "bullish" if bullish >= bearish else "bearish"
            predictions.append(
                SegmentPrediction(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    label=label,
                    bullish_probability=bullish,
                    bearish_probability=bearish,
                    confidence=max(bullish, bearish),
                    speaker_id=segment.speaker_id,
                    speaker_role=segment.speaker_role,
                    role_confidence=segment.role_confidence,
                    role_source=segment.role_source,
                    turn_id=segment.turn_id,
                    question_turn_id=segment.question_turn_id,
                    question_context=segment.question_context,
                )
            )
        return predictions


class FineTunedSignalClassifier:
    """Sequence classifier produced by the bundled supervised training command."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"fine-tuned:{self.model_name}"

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._load_lock:
                if self._pipeline is None:
                    try:
                        import torch
                        from transformers import pipeline
                    except ImportError as exc:
                        raise ClassificationError(
                            "torch/transformers are not installed; run `pip install .`"
                        ) from exc
                    device = 0 if torch.cuda.is_available() else -1
                    self._pipeline = pipeline(
                        task="text-classification",
                        model=self.model_name,
                        tokenizer=self.model_name,
                        device=device,
                    )
        return self._pipeline

    def load(self) -> None:
        self._get_pipeline()

    def classify(
        self,
        segments: Sequence[TranscriptSegment],
        asset_context: str,
    ) -> list[SegmentPrediction]:
        if not segments:
            return []
        inputs = [f"[ASSET] {asset_context} [STATEMENT] {segment.text}" for segment in segments]
        try:
            with self._inference_lock:
                outputs = self._get_pipeline()(inputs, top_k=None, truncation=True)
        except Exception as exc:
            raise ClassificationError("Supervised classification failed") from exc

        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
        predictions: list[SegmentPrediction] = []
        for segment, output in zip(segments, outputs, strict=True):
            by_label = {str(item["label"]).casefold(): float(item["score"]) for item in output}
            bullish = by_label.get("bullish", by_label.get("label_1"))
            bearish = by_label.get("bearish", by_label.get("label_0"))
            if bullish is None or bearish is None:
                raise ClassificationError(
                    "The checkpoint must define `bullish` and `bearish` labels"
                )
            total = bullish + bearish
            bullish, bearish = bullish / total, bearish / total
            label = "bullish" if bullish >= bearish else "bearish"
            predictions.append(
                SegmentPrediction(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    label=label,
                    bullish_probability=bullish,
                    bearish_probability=bearish,
                    confidence=max(bullish, bearish),
                    speaker_id=segment.speaker_id,
                    speaker_role=segment.speaker_role,
                    role_confidence=segment.role_confidence,
                    role_source=segment.role_source,
                    turn_id=segment.turn_id,
                    question_turn_id=segment.question_turn_id,
                    question_context=segment.question_context,
                )
            )
        return predictions


class HeuristicSignalClassifier:
    """Small offline baseline for smoke tests; it is not the production model."""

    _BULLISH = {
        "cut rates": 2.5,
        "rate cut": 2.5,
        "lower rates": 2.0,
        "easing": 1.8,
        "accommodative": 1.5,
        "soft landing": 1.5,
        "inflation declining": 1.5,
        "inflation is declining": 1.5,
        "disinflation": 1.3,
        "pause rate hikes": 1.8,
        "slower balance sheet runoff": 1.5,
        "liquidity support": 1.5,
        "strong growth": 1.2,
    }
    _BEARISH = {
        "raise rates": 2.5,
        "rate hike": 2.5,
        "higher for longer": 2.2,
        "tightening": 1.8,
        "inflation persistent": 1.5,
        "downside risks": 1.2,
        "recession": 1.4,
        "balance sheet reduction": 1.5,
        "quantitative tightening": 1.8,
        "labor market overheating": 1.3,
    }

    @property
    def name(self) -> str:
        return "heuristic-baseline:v1"

    def load(self) -> None:
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return text.casefold()

    def classify(
        self,
        segments: Sequence[TranscriptSegment],
        asset_context: str,
    ) -> list[SegmentPrediction]:
        del asset_context
        predictions = []
        for segment in segments:
            text = self._normalize(segment.text)
            bullish_score = sum(
                weight for phrase, weight in self._BULLISH.items() if phrase in text
            )
            bearish_score = sum(
                weight for phrase, weight in self._BEARISH.items() if phrase in text
            )
            # Negations invert a nearby phrase in this intentionally simple baseline.
            if re.search(r"\b(no|not)\b.{0,24}\b(cut|easing)\b", text):
                bearish_score += 2.0
                bullish_score = max(0.0, bullish_score - 2.0)
            logit = bullish_score - bearish_score
            bullish = 1.0 / (1.0 + math.exp(-logit))
            bearish = 1.0 - bullish
            label = "bullish" if bullish >= bearish else "bearish"
            predictions.append(
                SegmentPrediction(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    label=label,
                    bullish_probability=bullish,
                    bearish_probability=bearish,
                    confidence=max(bullish, bearish),
                    speaker_id=segment.speaker_id,
                    speaker_role=segment.speaker_role,
                    role_confidence=segment.role_confidence,
                    role_source=segment.role_source,
                    turn_id=segment.turn_id,
                    question_turn_id=segment.question_turn_id,
                    question_context=segment.question_context,
                )
            )
        return predictions
