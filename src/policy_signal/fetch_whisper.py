from __future__ import annotations

import argparse
from pathlib import Path


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
    resolved = download_model(args.model, output_dir=str(output))
    print(f"Whisper checkpoint saved to {resolved}")


if __name__ == "__main__":
    main()
