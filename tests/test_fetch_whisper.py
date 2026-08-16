from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy_signal.fetch_whisper import write_manifest


class WhisperManifestTests(unittest.TestCase):
    def test_records_file_size_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Path(temporary_directory)
            (model / "model.bin").write_bytes(b"weights")
            cache = model / ".cache"
            cache.mkdir()
            (cache / "metadata").write_text("temporary", encoding="utf-8")

            manifest_path = write_manifest(model, "small.en")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["requested_model"], "small.en")
            self.assertEqual(manifest["files"][0]["path"], "model.bin")
            self.assertEqual(manifest["files"][0]["size"], 7)
            self.assertEqual(len(manifest["files"][0]["sha256"]), 64)
            self.assertEqual(len(manifest["files"]), 1)


if __name__ == "__main__":
    unittest.main()
