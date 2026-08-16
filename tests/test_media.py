from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from policy_signal.media import MediaProcessor


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class MediaProcessorTests(unittest.TestCase):
    def test_normalizes_audio_to_16khz_mono(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            destination = Path(directory) / "normalized.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b"\x00\x00" * 2 * 8000)

            duration = MediaProcessor().extract_audio(source, destination)

            self.assertAlmostEqual(duration, 1.0, places=2)
            with wave.open(str(destination), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getsampwidth(), 2)

    def test_extracts_audio_from_video_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "speech.mkv"
            destination = Path(directory) / "normalized.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=16x16:d=1",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=1",
                    "-c:v",
                    "ffv1",
                    "-c:a",
                    "pcm_s16le",
                    str(source),
                ],
                check=True,
                capture_output=True,
            )

            duration = MediaProcessor().extract_audio(source, destination)

            self.assertAlmostEqual(duration, 1.0, places=2)
            self.assertTrue(destination.is_file())


if __name__ == "__main__":
    unittest.main()
