from __future__ import annotations

import os
from pathlib import Path

from .attribution import (
    PassthroughRoleAssigner,
    PressConferenceRoleAssigner,
    PyannoteRoleAssigner,
)
from .classification import (
    FineTunedSignalClassifier,
    HeuristicSignalClassifier,
    ZeroShotSignalClassifier,
)
from .config import Settings
from .media import MediaProcessor
from .pipeline import VideoAnalysisPipeline
from .relevance import (
    FineTunedPolicyRelevanceGate,
    HeuristicPolicyRelevanceGate,
    PassthroughRelevanceGate,
)
from .transcription import FasterWhisperTranscriber


def _require_model_bundle(
    directory: Path,
    component: str,
    *,
    required: tuple[str, ...],
    alternatives: tuple[tuple[str, ...], ...] = (),
) -> None:
    if not directory.is_dir():
        raise ValueError(f"Local {component} directory does not exist: {directory}")
    missing = [name for name in required if not (directory / name).is_file()]
    for group in alternatives:
        if not any((directory / name).is_file() for name in group):
            missing.append(" or ".join(group))
    if missing:
        raise ValueError(
            f"Local {component} directory is incomplete ({', '.join(missing)}): {directory}"
        )


def build_pipeline(settings: Settings) -> VideoAnalysisPipeline:
    if settings.local_models_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        _require_model_bundle(
            Path(settings.whisper_model),
            "Whisper",
            required=("config.json", "model.bin", "tokenizer.json"),
        )
        if settings.classifier_backend != "heuristic":
            _require_model_bundle(
                Path(settings.classifier_model),
                "classifier",
                required=("config.json",),
                alternatives=(
                    ("model.safetensors", "pytorch_model.bin"),
                    ("tokenizer.json", "vocab.json"),
                ),
            )
        if settings.speaker_role_backend == "pyannote":
            _require_model_bundle(
                Path(settings.speaker_diarization_model),
                "diarization",
                required=("config.yaml",),
            )
        if settings.relevance_backend == "fine-tuned":
            _require_model_bundle(
                Path(settings.relevance_model),
                "relevance model",
                required=("config.json",),
                alternatives=(
                    ("model.safetensors", "pytorch_model.bin"),
                    ("tokenizer.json", "vocab.json"),
                ),
            )

    if settings.classifier_backend == "zero-shot":
        classifier = ZeroShotSignalClassifier(settings.classifier_model)
    elif settings.classifier_backend == "fine-tuned":
        classifier = FineTunedSignalClassifier(settings.classifier_model)
    elif settings.classifier_backend == "heuristic":
        classifier = HeuristicSignalClassifier()
    else:
        raise ValueError(
            "POLICY_SIGNAL_CLASSIFIER_BACKEND must be zero-shot, fine-tuned, or heuristic"
        )

    if settings.speaker_role_backend == "press-conference-text":
        role_assigner = PressConferenceRoleAssigner()
    elif settings.speaker_role_backend == "pyannote":
        role_assigner = PyannoteRoleAssigner(
            settings.speaker_diarization_model,
            token=settings.speaker_diarization_token,
            device=settings.speaker_diarization_device,
            opening_anchor_seconds=settings.speaker_opening_anchor_seconds,
        )
    elif settings.speaker_role_backend == "none":
        role_assigner = PassthroughRoleAssigner()
    else:
        raise ValueError(
            "POLICY_SIGNAL_SPEAKER_ROLE_BACKEND must be press-conference-text, pyannote, or none"
        )

    if settings.relevance_backend == "heuristic":
        relevance_gate = HeuristicPolicyRelevanceGate()
    elif settings.relevance_backend == "fine-tuned":
        relevance_gate = FineTunedPolicyRelevanceGate(settings.relevance_model)
    elif settings.relevance_backend == "none":
        relevance_gate = PassthroughRelevanceGate()
    else:
        raise ValueError("POLICY_SIGNAL_RELEVANCE_BACKEND must be fine-tuned, heuristic, or none")

    return VideoAnalysisPipeline(
        media=MediaProcessor(settings.max_duration_seconds),
        transcriber=FasterWhisperTranscriber(
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
            settings.local_models_only,
        ),
        classifier=classifier,
        role_assigner=role_assigner,
        relevance_gate=relevance_gate,
        neutral_margin=settings.neutral_margin,
        recency_half_life_seconds=settings.offline_recency_half_life_seconds,
    )
