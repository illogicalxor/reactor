from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .domain import RelevanceLabel, TranscriptSegment


class RelevanceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RelevanceAssessment:
    segment: TranscriptSegment
    label: RelevanceLabel
    confidence: float


class PolicyRelevanceGate(Protocol):
    @property
    def name(self) -> str: ...

    def load(self) -> None: ...

    def assess(self, segments: Sequence[TranscriptSegment]) -> list[RelevanceAssessment]: ...


class PassthroughRelevanceGate:
    @property
    def name(self) -> str:
        return "relevance:none"

    def load(self) -> None:
        return None

    def assess(self, segments: Sequence[TranscriptSegment]) -> list[RelevanceAssessment]:
        return [RelevanceAssessment(segment, "directional", 1.0) for segment in segments]


class HeuristicPolicyRelevanceGate:
    """High-precision baseline until a neutral/irrelevant checkpoint is trained."""

    _DIRECTIONAL = re.compile(
        r"\b(?:federal funds|interest rates?|rate hikes?|rate cuts?|inflation|disinflation|"
        r"price stability|employment|unemployment|labor market|monetary policy|"
        r"balance sheet|financial conditions|tighten(?:ing)?|eas(?:e|ing)|accommodati(?:on|ve)|"
        r"yield curve|treasur(?:y|ies)|liquidity|recession)\b",
        re.IGNORECASE,
    )
    _CONTEXTUAL = re.compile(
        r"\b(?:economy|economic|growth|productivity|investment|tariffs?|oil|energy|"
        r"wages?|prices?|demand|supply|risks?|outlook|activity|job gains?)\b",
        re.IGNORECASE,
    )
    _PROCEDURAL = re.compile(
        r"\b(?:press conference|reporters?|news to make|people in this room|"
        r"viewers|readers|thank you all|questions? today|commentaries)\b",
        re.IGNORECASE,
    )

    @property
    def name(self) -> str:
        return "relevance:heuristic-policy-v1"

    def load(self) -> None:
        return None

    def assess(self, segments: Sequence[TranscriptSegment]) -> list[RelevanceAssessment]:
        assessments: list[RelevanceAssessment] = []
        for segment in segments:
            text = segment.text.strip()
            directional = bool(self._DIRECTIONAL.search(text))
            contextual_count = len(self._CONTEXTUAL.findall(text))
            procedural = bool(self._PROCEDURAL.search(text))

            if directional:
                label: RelevanceLabel = "directional"
                confidence = 0.9
            elif contextual_count >= 2 and not procedural:
                label = "directional"
                confidence = 0.7
            elif contextual_count == 1 and not procedural:
                label = "neutral"
                confidence = 0.65
            else:
                label = "irrelevant"
                confidence = 0.85 if procedural else 0.7
            assessments.append(RelevanceAssessment(segment, label, confidence))
        return assessments


class FineTunedPolicyRelevanceGate:
    """Three-class sequence classifier produced by policy-signal-train-relevance."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"relevance:fine-tuned:{self.model_name}"

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._load_lock:
                if self._pipeline is None:
                    try:
                        import torch
                        from transformers import pipeline
                    except ImportError as exc:
                        raise RelevanceError(
                            "torch/transformers are not installed; run `pip install .`"
                        ) from exc
                    device = 0 if torch.cuda.is_available() else -1
                    try:
                        self._pipeline = pipeline(
                            task="text-classification",
                            model=self.model_name,
                            tokenizer=self.model_name,
                            device=device,
                        )
                    except Exception as exc:
                        raise RelevanceError(
                            f"Could not load relevance model `{self.model_name}`"
                        ) from exc
        return self._pipeline

    def load(self) -> None:
        self._get_pipeline()

    def assess(
        self,
        segments: Sequence[TranscriptSegment],
    ) -> list[RelevanceAssessment]:
        if not segments:
            return []
        inputs = [f"[STATEMENT] {segment.text}" for segment in segments]
        try:
            with self._inference_lock:
                outputs = self._get_pipeline()(inputs, top_k=None, truncation=True)
        except Exception as exc:
            raise RelevanceError("Policy relevance classification failed") from exc
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]

        assessments = []
        allowed = {"irrelevant", "neutral", "directional"}
        for segment, output in zip(segments, outputs, strict=True):
            probabilities = {str(item["label"]).casefold(): float(item["score"]) for item in output}
            if not allowed.issubset(probabilities):
                raise RelevanceError(
                    "The relevance checkpoint must define irrelevant, neutral, and "
                    "directional labels"
                )
            label = max(allowed, key=probabilities.__getitem__)
            assessments.append(
                RelevanceAssessment(
                    segment=segment,
                    label=label,
                    confidence=probabilities[label],
                )
            )
        return assessments
