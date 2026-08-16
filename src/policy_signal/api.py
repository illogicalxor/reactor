from __future__ import annotations

import asyncio
import json
import tempfile
import wave
from collections import deque
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from .attribution import SpeakerAttributionError, SpeakerAttributionState
from .classification import ClassificationError
from .config import Settings
from .domain import SegmentPrediction
from .factory import build_pipeline
from .media import MediaError
from .pipeline import EmptySpeechError, VideoAnalysisPipeline, aggregate_predictions
from .relevance import RelevanceError
from .transcription import TranscriptionError

settings = Settings.from_env()
app = FastAPI(
    title="Policy Signal API",
    version="0.1.0",
    description="Bullish/bearish language signals from US policy-maker video and audio.",
)


@lru_cache(maxsize=1)
def get_pipeline() -> VideoAnalysisPipeline:
    return build_pipeline(settings)


if settings.local_models_only or settings.warm_models_on_startup:
    configured_pipeline = get_pipeline()
    if settings.warm_models_on_startup:
        configured_pipeline.warmup()


@app.get("/healthz")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "market_scope": "US markets",
        "language": "en",
        "classifier_backend": settings.classifier_backend,
        "classifier_model": settings.classifier_model,
        "speaker_role_backend": settings.speaker_role_backend,
        "speaker_diarization_model": settings.speaker_diarization_model,
        "relevance_backend": settings.relevance_backend,
        "relevance_model": settings.relevance_model,
        "whisper_model": settings.whisper_model,
        "local_models_only": settings.local_models_only,
        "warmed_on_startup": settings.warm_models_on_startup,
    }


async def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    total = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="File exceeds the configured limit")
            output.write(chunk)
    if total == 0:
        raise HTTPException(status_code=422, detail="The uploaded file is empty")
    return total


@app.post("/v1/analyze-video")
async def analyze_video(
    file: Annotated[UploadFile, File(description="US policy-maker video or audio")],
    asset_context: Annotated[str, Form(min_length=2, max_length=120)] = (
        settings.default_asset_context
    ),
) -> dict[str, object]:
    if file.content_type and not (
        file.content_type.startswith("video/")
        or file.content_type.startswith("audio/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(status_code=415, detail="Upload a video or audio file")

    raw_suffix = Path(file.filename or "upload.media").suffix.lower()
    suffix = raw_suffix if raw_suffix and len(raw_suffix) <= 10 else ".media"
    try:
        with tempfile.TemporaryDirectory(prefix="policy-signal-upload-") as temporary_directory:
            input_path = Path(temporary_directory) / f"input{suffix}"
            await _save_upload(file, input_path, settings.max_upload_mb * 1024 * 1024)
            result = await asyncio.to_thread(
                get_pipeline().analyze_video,
                input_path,
                asset_context=asset_context,
                language="en",
            )
            return result.to_dict()
    except HTTPException:
        raise
    except (MediaError, EmptySpeechError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (
        TranscriptionError,
        ClassificationError,
        SpeakerAttributionError,
        RelevanceError,
    ) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()


def _validate_stream_configuration(payload: dict[str, object]) -> tuple[int, float, str]:
    if payload.get("format", "pcm_s16le") != "pcm_s16le":
        raise ValueError("The stream only supports signed 16-bit little-endian PCM")
    sample_rate = int(payload.get("sample_rate", 16000))
    if sample_rate not in {8000, 16000, 32000, 44100, 48000}:
        raise ValueError("sample_rate must be 8000, 16000, 32000, 44100, or 48000")
    window_seconds = float(payload.get("window_seconds", settings.stream_window_seconds))
    if not 2 <= window_seconds <= 30:
        raise ValueError("window_seconds must be between 2 and 30")
    if payload.get("language", "en") != "en":
        raise ValueError("This service only supports English-language US market coverage")
    asset_context = str(payload.get("asset_context", settings.default_asset_context)).strip()
    if not 2 <= len(asset_context) <= 120:
        raise ValueError("asset_context must contain between 2 and 120 characters")
    return sample_rate, window_seconds, asset_context


def _analyze_pcm(
    pcm: bytes,
    sample_rate: int,
    asset_context: str,
    attribution_state: SpeakerAttributionState,
):
    duration = len(pcm) / (sample_rate * 2)
    with tempfile.TemporaryDirectory(prefix="policy-signal-stream-") as temporary_directory:
        audio_path = Path(temporary_directory) / "window.wav"
        with wave.open(str(audio_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return get_pipeline().analyze_normalized_audio(
            audio_path,
            duration_seconds=duration,
            asset_context=asset_context,
            language="en",
            attribution_state=attribution_state,
        )


@app.websocket("/v1/analyze-stream")
async def analyze_stream(websocket: WebSocket) -> None:
    """Analyze mono PCM windows extracted from a live video's audio track."""
    await websocket.accept()
    try:
        configuration_message = await websocket.receive_text()
        configuration = json.loads(configuration_message)
        if not isinstance(configuration, dict):
            raise ValueError("The initial configuration must be a JSON object")
        sample_rate, window_seconds, asset_context = _validate_stream_configuration(configuration)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close(code=1003)
        return

    bytes_per_window = int(sample_rate * 2 * window_seconds)
    buffer = bytearray()
    rolling_predictions: list[SegmentPrediction] = []
    rolling_transcript: deque[str] = deque(maxlen=40)
    stream_offset = 0.0
    attribution_state = get_pipeline().create_attribution_state()

    async def process_window(pcm: bytes) -> None:
        nonlocal stream_offset, rolling_predictions
        duration = len(pcm) / (sample_rate * 2)
        try:
            result = await asyncio.to_thread(
                _analyze_pcm,
                pcm,
                sample_rate,
                asset_context,
                attribution_state,
            )
        except EmptySpeechError:
            await websocket.send_json(
                {"type": "silence", "start": stream_offset, "end": stream_offset + duration}
            )
            stream_offset += duration
            return

        shifted = [
            replace(
                item,
                start=item.start + stream_offset,
                end=item.end + stream_offset,
            )
            for item in result.segments
        ]
        rolling_predictions.extend(shifted)
        rolling_transcript.append(result.transcript)
        stream_offset += duration
        # Bound memory and make the rolling signal represent approximately the last five minutes.
        cutoff = max(stream_offset - 300, 0)
        rolling_predictions = [item for item in rolling_predictions if item.end >= cutoff]
        rolling = aggregate_predictions(
            rolling_predictions,
            asset_context=asset_context,
            transcript=" ".join(rolling_transcript),
            language=result.language,
            duration_seconds=stream_offset,
            model=result.model,
            neutral_margin=settings.neutral_margin,
            recency_half_life_seconds=settings.recency_half_life_seconds,
        )
        await websocket.send_json(
            {
                "type": "signal",
                "window": result.to_dict(),
                "rolling": rolling.to_dict(),
                "stream_seconds": stream_offset,
            }
        )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if binary := message.get("bytes"):
                buffer.extend(binary)
                if len(buffer) > bytes_per_window * 4:
                    await websocket.send_json({"type": "error", "detail": "Buffer is too large"})
                    await websocket.close(code=1009)
                    return
                while len(buffer) >= bytes_per_window:
                    window = bytes(buffer[:bytes_per_window])
                    del buffer[:bytes_per_window]
                    await process_window(window)
            elif text := message.get("text"):
                command = json.loads(text).get("command")
                if command in {"flush", "end"} and buffer:
                    await process_window(bytes(buffer))
                    buffer.clear()
                if command == "end":
                    await websocket.close(code=1000)
                    return
    except WebSocketDisconnect:
        return
    except (TranscriptionError, ClassificationError, ValueError, json.JSONDecodeError) as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close(code=1011)


def run() -> None:
    import uvicorn

    uvicorn.run("policy_signal.api:app", host="0.0.0.0", port=8000)
