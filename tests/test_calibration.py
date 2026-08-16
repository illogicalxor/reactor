from __future__ import annotations

import unittest

from policy_signal.calibration import (
    fit_temperature,
    negative_log_likelihood,
    scale_probabilities,
)


class TemperatureCalibrationTests(unittest.TestCase):
    def test_softens_probabilities_above_unit_temperature(self) -> None:
        scaled = scale_probabilities(
            {"irrelevant": 0.9, "neutral": 0.08, "directional": 0.02},
            temperature=2.0,
        )

        self.assertLess(scaled["irrelevant"], 0.9)
        self.assertAlmostEqual(sum(scaled.values()), 1.0)

    def test_fitted_temperature_does_not_increase_validation_log_loss(self) -> None:
        logits = [
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
        targets = [0, 1, 1, 2]

        temperature, before, after = fit_temperature(logits, targets)

        self.assertGreater(temperature, 1.0)
        self.assertLessEqual(after, before)
        self.assertAlmostEqual(
            after,
            negative_log_likelihood(logits, targets, temperature),
        )


if __name__ == "__main__":
    unittest.main()
