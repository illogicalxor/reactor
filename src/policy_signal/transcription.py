from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

from .domain import TranscriptSegment, TranscriptWord


class TranscriptionError(RuntimeError):
    pass


class Transcriber(Protocol):
    @property
    def name(self) -> str: ...

    def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> list[TranscriptSegment]: ...

    def load(self) -> None: ...


class FasterWhisperTranscriber:
    """Lazy faster-whisper adapter so process startup remains quick."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "int8",
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.local_files_only = local_files_only
        self._model = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"faster-whisper:{self.model_name}"

    def _get_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from faster_whisper import WhisperModel
                    except ImportError as exc:
                        raise TranscriptionError(
                            "faster-whisper is not installed; run `pip install .`"
                        ) from exc
                    try:
                        self._model = WhisperModel(
                            self.model_name,
                            device=self.device,
                            compute_type=self.compute_type,
                            local_files_only=self.local_files_only,
                        )
                    except Exception as exc:
                        raise TranscriptionError(
                            f"Could not load Whisper model `{self.model_name}`"
                        ) from exc
        return self._model

    def load(self) -> None:
        self._get_model()

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[TranscriptSegment]:
        model = self._get_model()
        selected_language = None if not language or language == "auto" else language
        try:
            with self._inference_lock:
                raw_segments, info = model.transcribe(
                    str(audio_path),
                    language=selected_language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 350},
                    condition_on_previous_text=True,
                    word_timestamps=True,
                )
                detected_language = info.language
                segments = [
                    TranscriptSegment(
                        start=float(segment.start),
                        end=float(segment.end),
                        text=segment.text.strip(),
                        language=detected_language,
                        words=tuple(
                            TranscriptWord(
                                start=float(word.start),
                                end=float(word.end),
                                text=word.word,
                                probability=(
                                    float(word.probability)
                                    if word.probability is not None
                                    else None
                                ),
                            )
                            for word in (segment.words or [])
                        ),
                    )
                    for segment in raw_segments
                    if segment.text.strip()
                ]
        except Exception as exc:
            raise TranscriptionError("Audio transcription failed") from exc
        return segments
