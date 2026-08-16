from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from policy_signal.config import Settings
from policy_signal.factory import build_pipeline


def _touch_files(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


class LocalModelValidationTests(unittest.TestCase):
    def test_rejects_incomplete_whisper_bundle_before_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            whisper = Path(temporary_directory) / "whisper"
            _touch_files(whisper, "config.json", "tokenizer.json")
            settings = Settings(
                local_models_only=True,
                whisper_model=str(whisper),
                classifier_backend="heuristic",
                relevance_backend="none",
            )

            with self.assertRaisesRegex(ValueError, "model.bin"):
                build_pipeline(settings)

    def test_accepts_complete_local_runtime_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            whisper = root / "whisper"
            classifier = root / "classifier"
            relevance = root / "relevance"
            _touch_files(whisper, "config.json", "model.bin", "tokenizer.json")
            _touch_files(classifier, "config.json", "model.safetensors", "tokenizer.json")
            _touch_files(relevance, "config.json", "model.safetensors", "tokenizer.json")
            settings = Settings(
                local_models_only=True,
                whisper_model=str(whisper),
                classifier_backend="fine-tuned",
                classifier_model=str(classifier),
                relevance_backend="fine-tuned",
                relevance_model=str(relevance),
            )

            pipeline = build_pipeline(settings)

            self.assertEqual(pipeline.classifier.model_name, str(classifier))
            self.assertEqual(pipeline.relevance_gate.model_name, str(relevance))


if __name__ == "__main__":
    unittest.main()
