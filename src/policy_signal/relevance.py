from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from .calibration import scale_probabilities
from .domain import RelevanceLabel, TranscriptSegment

RELEVANCE_LABELS = ("irrelevant", "neutral", "directional")
PROCEDURAL_OVERRIDE_CONFIDENCE = 0.98

# Only match complete housekeeping/question-management turns.  Keeping these
# patterns anchored prevents a substantive answer that merely mentions a
# question, website, or press conference from being discarded.
_PROCEDURAL_ONLY = re.compile(
    r"^\s*(?:"
    r"(?:could|can|would) you (?:please )?repeat (?:the|your) question(?: please)?|"
    r"(?:let us|let's|we(?: will|'ll)) (?:go|move) (?:on )?to (?:the )?next question|"
    r"we(?: will|'ll) take (?:a|the next|one (?:more|final)) question(?: from .+)?|"
    r"there is time for (?:one|a) (?:more|final) question|"
    r"please (?:identify yourself(?: and your organization)?|speak (?:directly )?into "
    r"the microphone|silence your phones(?: before we begin)?)|"
    r"(?:can|could) everyone hear me(?: clearly)?|"
    r"(?:that|this) concludes (?:our|the) press conference|"
    r"the transcript will be published (?:later )?today"
    r")[.!?]*\s*$",
    re.IGNORECASE,
)


class RelevanceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RelevanceAssessment:
    segment: TranscriptSegment
    label: RelevanceLabel
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


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
        return [
            RelevanceAssessment(
                segment,
                "directional",
                1.0,
                {"irrelevant": 0.0, "neutral": 0.0, "directional": 1.0},
            )
            for segment in segments
        ]


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
            remainder = (1.0 - confidence) / 2.0
            probabilities = {
                candidate: confidence if candidate == label else remainder
                for candidate in RELEVANCE_LABELS
            }
            assessments.append(RelevanceAssessment(segment, label, confidence, probabilities))
        return assessments


class FineTunedPolicyRelevanceGate:
    """Three-class sequence classifier produced by policy-signal-train-relevance."""

    def __init__(
        self,
        model_name: str,
        *,
        temperature: float | None = None,
        procedural_override: bool = True,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature if temperature is not None else self._load_temperature()
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise RelevanceError(f"Invalid relevance calibration temperature: {self.temperature}")
        self.procedural_override = procedural_override
        self._pipeline = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load_temperature(self) -> float:
        calibration_path = Path(self.model_name) / "relevance_calibration.json"
        if not calibration_path.is_file():
            return 1.0
        try:
            payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            temperature = float(payload["temperature"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RelevanceError(f"Invalid relevance calibration: {calibration_path}") from exc
        if temperature <= 0 or not math.isfinite(temperature):
            raise RelevanceError(f"Invalid relevance calibration temperature: {temperature}")
        return temperature

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
        probabilities_by_segment = self.predict_probabilities(segments)
        assessments = []
        for segment, probabilities in zip(segments, probabilities_by_segment, strict=True):
            label = max(probabilities, key=probabilities.__getitem__)
            assessments.append(
                RelevanceAssessment(
                    segment=segment,
                    label=cast(RelevanceLabel, label),
                    confidence=probabilities[label],
                    probabilities=probabilities,
                )
            )
        return assessments

    def predict_probabilities(
        self,
        segments: Sequence[TranscriptSegment],
    ) -> list[dict[str, float]]:
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

        results = []
        allowed = set(RELEVANCE_LABELS)
        for segment, output in zip(segments, outputs, strict=True):
            probabilities = {str(item["label"]).casefold(): float(item["score"]) for item in output}
            if not allowed.issubset(probabilities):
                raise RelevanceError(
                    "The relevance checkpoint must define irrelevant, neutral, and "
                    "directional labels"
                )
            selected = {label: probabilities[label] for label in RELEVANCE_LABELS}
            if self.temperature != 1.0:
                selected = scale_probabilities(selected, self.temperature)
            if self.procedural_override and _PROCEDURAL_ONLY.fullmatch(segment.text):
                remaining = 1.0 - PROCEDURAL_OVERRIDE_CONFIDENCE
                other_total = selected["neutral"] + selected["directional"]
                if other_total:
                    selected["neutral"] = remaining * selected["neutral"] / other_total
                    selected["directional"] = remaining * selected["directional"] / other_total
                else:
                    selected["neutral"] = remaining / 2.0
                    selected["directional"] = remaining / 2.0
                selected["irrelevant"] = PROCEDURAL_OVERRIDE_CONFIDENCE
            results.append(selected)
        return results
