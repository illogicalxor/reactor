from __future__ import annotations

import json
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    """Raised when a media file cannot be decoded or violates a configured limit."""


class MediaProcessor:
    def __init__(self, max_duration_seconds: float = 7200) -> None:
        self.max_duration_seconds = max_duration_seconds

    def probe_duration(self, source: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(source),
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            duration = float(json.loads(completed.stdout)["format"]["duration"])
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            KeyError,
            ValueError,
        ) as exc:
            raise MediaError("Could not read the media file duration") from exc

        if duration <= 0:
            raise MediaError("The media file does not have a valid duration")
        if duration > self.max_duration_seconds:
            raise MediaError(f"Video exceeds the {self.max_duration_seconds:.0f}-second limit")
        return duration

    def extract_audio(self, source: Path, destination: Path) -> float:
        """Normalize a video/audio track to mono, 16 kHz, signed 16-bit WAV."""
        duration = self.probe_duration(source)
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=max(duration * 2, 60))
        except FileNotFoundError as exc:
            raise MediaError("FFmpeg is not installed or was not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            details = exc.stderr.decode(errors="replace")[-500:]
            raise MediaError(f"FFmpeg could not extract the audio track: {details}") from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaError("Audio extraction timed out") from exc
        return duration
