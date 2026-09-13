from __future__ import annotations

import unittest

from policy_signal.consensus import assess_surprise, build_consensus, dynamic_neutral_margin
from policy_signal.impact_labels import (
    CanonicalRow,
    forward_return_bps,
    label_row,
    neutral_band_bps,
)


class ConsensusTests(unittest.TestCase):
    def test_unknown_consensus_is_not_actionable(self) -> None:
        assessment = assess_surprise(0.85, build_consensus(None), "We can cut rates.")
        self.assertFalse(assessment.actionable)
        self.assertEqual(assessment.reason, "consensus_unknown")
        self.assertAlmostEqual(assessment.adjusted_bullish, 0.5)

    def test_priced_in_stance_is_not_actionable(self) -> None:
        ctx = build_consensus(0.8, confidence=0.9, source="fed_funds_futures")
        assessment = assess_surprise(0.82, ctx, "We can cut rates.")
        self.assertFalse(assessment.actionable)
        self.assertEqual(assessment.reason, "consensus_priced_in")

    def test_genuine_surprise_is_actionable(self) -> None:
        ctx = build_consensus(0.5, confidence=0.9, source="fed_funds_futures")
        assessment = assess_surprise(0.9, ctx, "We can cut rates sharply.")
        self.assertTrue(assessment.actionable)
        self.assertIsNone(assessment.reason)
        self.assertGreater(assessment.adjusted_bullish, 0.7)

    def test_low_confidence_shrinks_toward_priced(self) -> None:
        ctx = build_consensus(0.5, confidence=0.2, source="survey")
        assessment = assess_surprise(0.9, ctx, "We can cut rates.")
        self.assertAlmostEqual(assessment.adjusted_bullish, 0.58, places=6)

    def test_information_effect_blocks_bullish(self) -> None:
        ctx = build_consensus(0.3, confidence=0.9, source="futures")
        assessment = assess_surprise(
            0.9, ctx, "We will cut rates because recession risks are severe."
        )
        self.assertTrue(assessment.information_effect_risk)
        self.assertFalse(assessment.actionable)
        self.assertEqual(assessment.reason, "information_effect_risk")

    def test_dynamic_margin_widens_with_vol(self) -> None:
        self.assertGreater(dynamic_neutral_margin(0.12, 50.0), 0.12)
        self.assertEqual(dynamic_neutral_margin(0.12, None), 0.12)


class ImpactLabelTests(unittest.TestCase):
    def test_forward_return(self) -> None:
        self.assertAlmostEqual(forward_return_bps(100.0, 101.0), 99.50, places=1)

    def test_neutral_band_uses_costs(self) -> None:
        band = neutral_band_bps(spread_bps=2.0, transaction_cost_bps=2.0)
        self.assertGreaterEqual(band, 5.0)

    def test_confounder_is_excluded(self) -> None:
        row = CanonicalRow(
            event_id="e1",
            text="statement",
            asset_context="US equities",
            price_before=100.0,
            price_after=102.0,
            horizon_seconds=300,
            confounded=True,
        )
        label, _, _, reason = label_row(row)
        self.assertEqual(label, "neutral")
        self.assertEqual(reason, "confounded_event")

    def test_small_surprise_is_neutral(self) -> None:
        row = CanonicalRow(
            event_id="e2",
            text="statement",
            asset_context="US equities",
            price_before=100.0,
            price_after=101.0,
            horizon_seconds=300,
            surprise_bps=0.1,
        )
        label, _, _, reason = label_row(row, min_abs_surprise_bps=2.0)
        self.assertEqual(reason, "surprise_too_small")


if __name__ == "__main__":
    unittest.main()
