from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from .attribution import PyannoteRoleAssigner, SpeakerAttributionError

TimelineRole = Literal["policy_maker", "journalist", "unknown"]
ALLOWED_ROLES = {"policy_maker", "journalist", "unknown"}


@dataclass(frozen=True, slots=True)
class TimelineSegment:
    start: float
    end: float
    speaker_role: TimelineRole
    speaker_id: str | None = None


def load_timeline_csv(path: Path) -> list[TimelineSegment]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"start", "end", "speaker_role"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing timeline columns: {', '.join(sorted(missing))}")
        segments = []
        for line_number, row in enumerate(reader, start=2):
            start = float(row["start"])
            end = float(row["end"])
            role = row["speaker_role"].strip()
            if end <= start:
                raise ValueError(f"Invalid interval on line {line_number}: end must exceed start")
            if role not in ALLOWED_ROLES:
                raise ValueError(f"Invalid speaker_role on line {line_number}: {role}")
            segments.append(
                TimelineSegment(
                    start=start,
                    end=end,
                    speaker_role=cast(TimelineRole, role),
                    speaker_id=(row.get("speaker_id") or "").strip() or None,
                )
            )
    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def _role_at(segments: list[TimelineSegment], timestamp: float) -> TimelineRole | None:
    return next(
        (
            segment.speaker_role
            for segment in segments
            if segment.start <= timestamp < segment.end
        ),
        None,
    )


def evaluate_timelines(
    reference: list[TimelineSegment],
    predicted: list[TimelineSegment],
) -> dict[str, object]:
    boundaries = sorted(
        {value for segment in reference + predicted for value in (segment.start, segment.end)}
    )
    durations = {
        "reference": 0.0,
        "missed_speech": 0.0,
        "false_alarm": 0.0,
        "speaker_confusion": 0.0,
        "correct_role": 0.0,
    }
    role_counts = {
        role: {"true_positive": 0.0, "predicted": 0.0, "reference": 0.0}
        for role in ("policy_maker", "journalist")
    }

    for start, end in zip(boundaries, boundaries[1:], strict=False):
        duration = end - start
        if duration <= 0:
            continue
        midpoint = start + duration / 2
        expected = _role_at(reference, midpoint)
        actual = _role_at(predicted, midpoint)
        if expected is not None:
            durations["reference"] += duration
        if expected is not None and actual is None:
            durations["missed_speech"] += duration
        elif expected is None and actual is not None:
            durations["false_alarm"] += duration
        elif expected is not None and actual is not None:
            if expected == actual:
                durations["correct_role"] += duration
            else:
                durations["speaker_confusion"] += duration

        for role, counts in role_counts.items():
            if expected == role:
                counts["reference"] += duration
            if actual == role:
                counts["predicted"] += duration
            if expected == role and actual == role:
                counts["true_positive"] += duration

    reference_duration = durations["reference"]
    error_duration = (
        durations["missed_speech"]
        + durations["false_alarm"]
        + durations["speaker_confusion"]
    )
    per_role = {}
    for role, counts in role_counts.items():
        precision = (
            counts["true_positive"] / counts["predicted"] if counts["predicted"] else 0.0
        )
        recall = (
            counts["true_positive"] / counts["reference"] if counts["reference"] else 0.0
        )
        per_role[role] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "reference_seconds": round(counts["reference"], 6),
            "predicted_seconds": round(counts["predicted"], 6),
        }

    return {
        "reference_speech_seconds": round(reference_duration, 6),
        "missed_speech_seconds": round(durations["missed_speech"], 6),
        "false_alarm_seconds": round(durations["false_alarm"], 6),
        "speaker_confusion_seconds": round(durations["speaker_confusion"], 6),
        "diarization_error_rate": round(
            error_duration / reference_duration if reference_duration else 0.0,
            6,
        ),
        "role_accuracy": round(
            durations["correct_role"] / reference_duration if reference_duration else 0.0,
            6,
        ),
        "per_role": per_role,
    }


def predict_timeline(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    opening_anchor_seconds: float,
    token: str | None,
) -> tuple[list[TimelineSegment], str]:
    assigner = PyannoteRoleAssigner(
        model_name,
        token=token,
        device=device,
        opening_anchor_seconds=opening_anchor_seconds,
    )
    try:
        with assigner._inference_lock:
            output = assigner._get_pipeline()(str(audio_path))
        turns = assigner._turns(output)
    except SpeakerAttributionError:
        raise
    except Exception as exc:
        raise SpeakerAttributionError("Speaker diarization evaluation failed") from exc
    if not turns:
        raise SpeakerAttributionError("Speaker diarization detected no speakers")

    opening_overlap: dict[str, float] = {}
    for turn in turns:
        overlap = assigner._overlap(0.0, opening_anchor_seconds, turn)
        opening_overlap[turn.speaker] = opening_overlap.get(turn.speaker, 0.0) + overlap
    policy_speaker = max(opening_overlap, key=opening_overlap.get)
    predicted = [
        TimelineSegment(
            start=turn.start,
            end=turn.end,
            speaker_id=turn.speaker,
            speaker_role=assigner._role(turn.speaker, policy_speaker),
        )
        for turn in turns
    ]
    return predicted, policy_speaker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pyannote speaker-role diarization against an annotated timeline"
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-community-1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--opening-anchor-seconds", type=float, default=600.0)
    args = parser.parse_args()

    token = os.getenv("POLICY_SIGNAL_SPEAKER_DIARIZATION_TOKEN")
    reference = load_timeline_csv(args.reference)
    predicted, policy_speaker = predict_timeline(
        args.audio,
        model_name=args.model,
        device=args.device,
        opening_anchor_seconds=args.opening_anchor_seconds,
        token=token,
    )
    result = {
        "audio": str(args.audio),
        "reference": str(args.reference),
        "model": args.model,
        "device": args.device,
        "opening_anchor_seconds": args.opening_anchor_seconds,
        "policy_speaker_cluster": policy_speaker,
        "metrics": evaluate_timelines(reference, predicted),
        "predicted_timeline": [asdict(segment) for segment in predicted],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
