from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="POLICY_SIGNAL_",
        env_file=".env",
        extra="ignore",
    )

    classifier_backend: str = "zero-shot"
    classifier_model: str = "facebook/bart-large-mnli"
    speaker_role_backend: str = "press-conference-text"
    speaker_diarization_model: str = "pyannote/speaker-diarization-community-1"
    speaker_diarization_device: str = "auto"
    speaker_diarization_token: str | None = None
    speaker_opening_anchor_seconds: float = 600.0
    relevance_backend: str = "heuristic"
    relevance_model: str = "models/policy-relevance-gate-v1"
    whisper_model: str = "small.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    default_asset_context: str = "US equities"
    local_models_only: bool = False
    warm_models_on_startup: bool = False
    max_upload_mb: int = 500
    max_duration_seconds: int = 7200
    neutral_margin: float = 0.12
    # Stance without consensus is not tradable. Require explicit consensus
    # for an actionable signal; otherwise the pipeline abstains.
    require_consensus_for_action: bool = True
    min_surprise: float = 0.15
    impact_horizon_seconds: int = 300
    # Full recordings should not be dominated by arbitrary closing remarks.
    # Recency remains useful for the rolling streaming signal below.
    offline_recency_half_life_seconds: float | None = None
    recency_half_life_seconds: float = 90.0
    stream_window_seconds: float = 8.0

    @classmethod
    def from_env(cls) -> Settings:
        return cls()
