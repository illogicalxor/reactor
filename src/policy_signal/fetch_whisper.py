from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(directory: Path, requested_model: str) -> Path:
    files = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative_path = path.relative_to(directory)
        if path.name == "download_manifest.json" or ".cache" in relative_path.parts:
            continue
        files.append(
            {
                "path": relative_path.as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "artifact_type": "faster-whisper CTranslate2 checkpoint",
        "requested_model": requested_model,
        "files": files,
    }
    manifest_path = directory / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an English faster-whisper checkpoint for offline deployment"
    )
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--output", default="models/whisper-small.en")
    args = parser.parse_args()

    try:
        from faster_whisper.utils import download_model
    except ImportError as exc:
        raise SystemExit("Install runtime dependencies with: pip install .") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved = Path(download_model(args.model, output_dir=str(output)))
    manifest = write_manifest(resolved, args.model)
    print(f"Whisper checkpoint saved to {resolved}")
    print(f"Artifact manifest saved to {manifest}")


if __name__ == "__main__":
    main()
