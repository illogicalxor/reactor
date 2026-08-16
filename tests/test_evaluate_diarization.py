from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from policy_signal.evaluate_diarization import (
    TimelineSegment,
    evaluate_timelines,
    load_timeline_csv,
)


class DiarizationEvaluationTests(unittest.TestCase):
    def test_computes_role_confusion_and_duration_metrics(self) -> None:
        reference = [
            TimelineSegment(0, 10, "policy_maker"),
            TimelineSegment(10, 20, "journalist"),
        ]
        predicted = [
            TimelineSegment(0, 8, "policy_maker"),
            TimelineSegment(8, 20, "journalist"),
        ]

        result = evaluate_timelines(reference, predicted)

        self.assertEqual(result["diarization_error_rate"], 0.1)
        self.assertEqual(result["speaker_confusion_seconds"], 2.0)
        self.assertEqual(result["per_role"]["policy_maker"]["precision"], 1.0)
        self.assertEqual(result["per_role"]["policy_maker"]["recall"], 0.8)
        self.assertEqual(result["per_role"]["journalist"]["precision"], 0.833333)

    def test_loads_reference_timeline_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "reference.csv"
            path.write_text(
                "start,end,speaker_role,speaker_id\n"
                "0,5,policy_maker,chair\n"
                "5,8,journalist,reporter\n",
                encoding="utf-8",
            )

            result = load_timeline_csv(path)

            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].speaker_id, "chair")
            self.assertEqual(result[1].speaker_role, "journalist")


if __name__ == "__main__":
    unittest.main()
