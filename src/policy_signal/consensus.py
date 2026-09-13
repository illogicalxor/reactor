"""Consensus and surprise handling.

The text classifier measures *stance-implied* direction (dovish -> bullish,
hawkish -> bearish for US equities). Real market impact is stance *minus*
what was already priced in. Without an explicit consensus input the pipeline
cannot know the surprise, so the event must be non-actionable.

Modes:
- ``none``: no consensus supplied. Adjusted probabilities shrink to 50/50 and
  the result is never actionable when ``require_consensus_for_action`` is set.
- ``explicit``: caller supplies what the market priced (e.g. from Fed Funds
  futures, OIS, options, or a survey median) as ``priced_bullish`` in [0, 1].
- ``proxy``: a weak prior supplied without a named source. Treated like
  explicit but with capped confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ConsensusMode = Literal["none", "explicit", "proxy"]

# Dovish language amid stress can be bearish ("information effect"). When the
# stance leans bullish but the transcript carries risk-off language, we flag
# the event instead of emitting an actionable bullish signal.
_INFORMATION_EFFECT = re.compile(
    r"\b(?:recession|severe downside|financial stress|credit stress|banking stress|"
    r"systemic risk|disorderly|tail risk|severe risk|crisis)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ConsensusContext:
    """What the market already expected before the event."""

    mode: ConsensusMode = "none"
    priced_bullish: float | None = None
    confidence: float = 0.0
    source: str | None = None
    horizon_seconds: int = 300
    expected_move_bps: float | None = None

    def validated(self) -> ConsensusContext:
        if self.mode == "none":
            return ConsensusContext(
                mode="none",
                priced_bullish=None,
                confidence=0.0,
                source=None,
                horizon_seconds=self.horizon_seconds,
                expected_move_bps=self.expected_move_bps,
            )
        if self.priced_bullish is None:
            raise ValueError("priced_bullish is required for explicit/proxy consensus")
        priced = min(max(float(self.priced_bullish), 0.0), 1.0)
        confidence = min(max(float(self.confidence), 0.0), 1.0)
        if self.mode == "proxy":
            # Weak priors must never drive full conviction.
            confidence = min(confidence, 0.5)
        return ConsensusContext(
            mode=self.mode,
            priced_bullish=priced,
            confidence=confidence,
            source=self.source,
            horizon_seconds=self.horizon_seconds,
            expected_move_bps=self.expected_move_bps,
        )


@dataclass(frozen=True, slots=True)
class SurpriseAssessment:
    surprise_score: float
    surprise_confidence: float
    adjusted_bullish: float
    adjusted_bearish: float
    signal_strength: float
    actionable: bool
    reason: str | None
    information_effect_risk: bool
    neutral_margin_applied: float


def build_consensus(
    priced_bullish: float | None,
    confidence: float = 1.0,
    source: str | None = None,
    horizon_seconds: int = 300,
    expected_move_bps: float | None = None,
    mode: ConsensusMode | None = None,
) -> ConsensusContext:
    """Build a validated consensus context from API inputs."""
    if priced_bullish is None:
        return ConsensusContext(
            mode="none",
            horizon_seconds=horizon_seconds,
            expected_move_bps=expected_move_bps,
        )
    resolved_mode: ConsensusMode = mode or ("explicit" if source else "proxy")
    return ConsensusContext(
        mode=resolved_mode,
        priced_bullish=priced_bullish,
        confidence=confidence,
        source=source,
        horizon_seconds=horizon_seconds,
        expected_move_bps=expected_move_bps,
    ).validated()


def dynamic_neutral_margin(base_margin: float, expected_move_bps: float | None) -> float:
    """Widen the abstention margin when the expected noise move is large."""
    if expected_move_bps is None:
        return base_margin
    try:
        move = max(float(expected_move_bps), 0.0)
    except (TypeError, ValueError):
        return base_margin
    # 50 bps of expected noise widens the probability margin by +0.05.
    widening = min(0.15, (move / 10000.0) * 1.0)
    return min(0.45, base_margin + widening)


def detect_information_effect_risk(text: str, stance_bullish: float) -> bool:
    """Flag dovish-leaning language co-occurring with stress language."""
    if stance_bullish < 0.6:
        return False
    return bool(_INFORMATION_EFFECT.search(text or ""))


def assess_surprise(
    stance_bullish: float,
    consensus: ConsensusContext | None,
    transcript: str = "",
    *,
    base_margin: float = 0.12,
    min_surprise: float = 0.15,
    require_consensus_for_action: bool = True,
) -> SurpriseAssessment:
    """Convert stance probabilities into surprise-adjusted, actionable output."""
    stance = min(max(float(stance_bullish), 0.0), 1.0)
    ctx = (consensus or ConsensusContext()).validated()
    margin = dynamic_neutral_margin(base_margin, ctx.expected_move_bps)
    information_risk = detect_information_effect_risk(transcript, stance)

    if ctx.mode == "none" or ctx.priced_bullish is None:
        # No surprise can be measured: shrink fully toward 50/50.
        adjusted = 0.5
        strength = 0.0
        reason: str | None = (
            "consensus_unknown" if require_consensus_for_action else None
        )
        actionable = not require_consensus_for_action and not information_risk
        if information_risk:
            reason = "information_effect_risk"
        return SurpriseAssessment(
            surprise_score=0.0,
            surprise_confidence=0.0,
            adjusted_bullish=adjusted,
            adjusted_bearish=1.0 - adjusted,
            signal_strength=strength,
            actionable=actionable,
            reason=reason,
            information_effect_risk=information_risk,
            neutral_margin_applied=margin,
        )

    priced = ctx.priced_bullish
    surprise = stance - priced
    # Shrink the stance toward what was priced by (1 - confidence).
    adjusted = priced + surprise * ctx.confidence
    adjusted = min(max(adjusted, 0.0), 1.0)
    strength = abs(adjusted - (1.0 - adjusted))

    if information_risk:
        return SurpriseAssessment(
            surprise_score=surprise,
            surprise_confidence=ctx.confidence,
            adjusted_bullish=adjusted,
            adjusted_bearish=1.0 - adjusted,
            signal_strength=strength,
            actionable=False,
            reason="information_effect_risk",
            information_effect_risk=True,
            neutral_margin_applied=margin,
        )
    if abs(surprise) < min_surprise:
        return SurpriseAssessment(
            surprise_score=surprise,
            surprise_confidence=ctx.confidence,
            adjusted_bullish=adjusted,
            adjusted_bearish=1.0 - adjusted,
            signal_strength=strength,
            actionable=False,
            reason="consensus_priced_in",
            information_effect_risk=False,
            neutral_margin_applied=margin,
        )
    if strength < margin:
        return SurpriseAssessment(
            surprise_score=surprise,
            surprise_confidence=ctx.confidence,
            adjusted_bullish=adjusted,
            adjusted_bearish=1.0 - adjusted,
            signal_strength=strength,
            actionable=False,
            reason="weak_surprise",
            information_effect_risk=False,
            neutral_margin_applied=margin,
        )
    return SurpriseAssessment(
        surprise_score=surprise,
        surprise_confidence=ctx.confidence,
        adjusted_bullish=adjusted,
        adjusted_bearish=1.0 - adjusted,
        signal_strength=strength,
        actionable=True,
        reason=None,
        information_effect_risk=False,
        neutral_margin_applied=margin,
    )
