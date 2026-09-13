"""Market-return labeling utilities for the training machine.

Stance labels (hawkish/dovish) are weak supervision. Impact labels must come
from observed forward returns *conditioned on the pre-event consensus*:

    return_h = log(mid[t + horizon] / mid[t])

    surprise_bps = realized_bp_change - expected_bp_change (from futures/OIS)

A statement is only a positive example when the return exceeds a neutral band
calibrated from volatility, spread, and costs AND no confounding event is
flagged. Everything else is neutral (excluded from binary training) or
quarantined.

This module has no network access. It validates canonical research rows and
produces a trainer-ready CSV plus a manifest for the GPU machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ImpactLabel = Literal["bullish", "bearish", "neutral"]


@dataclass(frozen=True, slots=True)
class CanonicalRow:
    event_id: str
    text: str
    asset_context: str
    price_before: float
    price_after: float
    horizon_seconds: int
    spread_bps: float = 0.0
    expected_move_bps: float | None = None
    realized_vol_bps: float | None = None
    transaction_cost_bps: float = 0.0
    confounded: bool = False
    stale_quote: bool = False
    illiquid: bool = False
    uncertain_timing: bool = False
    surprise_bps: float | None = None


def forward_return_bps(price_before: float, price_after: float) -> float:
    if price_before <= 0 or price_after <= 0:
        raise ValueError("prices must be positive")
    return math.log(price_after / price_before) * 10000.0


def neutral_band_bps(
    spread_bps: float = 0.0,
    expected_move_bps: float | None = None,
    realized_vol_bps: float | None = None,
    transaction_cost_bps: float = 0.0,
    multiplier: float = 1.0,
    floor_bps: float = 5.0,
) -> float:
    """Calibrate the neutral band from microstructure noise, not a fixed 10 bps."""
    vol_component = 0.0
    candidates = [
        v for v in (expected_move_bps, realized_vol_bps) if v is not None and v > 0
    ]
    if candidates:
        vol_component = min(candidates) * 0.5
    band = (spread_bps + transaction_cost_bps + vol_component) * multiplier
    return max(band, floor_bps)


def label_return(return_bps: float, band_bps: float) -> ImpactLabel:
    if return_bps >= band_bps:
        return "bullish"
    if return_bps <= -band_bps:
        return "bearish"
    return "neutral"


def label_row(
    row: CanonicalRow,
    *,
    multiplier: float = 1.0,
    floor_bps: float = 5.0,
    min_abs_surprise_bps: float = 0.0,
) -> tuple[ImpactLabel, float, float, str | None]:
    """Return (label, return_bps, band_bps, exclusion_reason)."""
    for flag, name in (
        (row.confounded, "confounded_event"),
        (row.stale_quote, "stale_quote"),
        (row.illiquid, "illiquid"),
        (row.uncertain_timing, "uncertain_timing"),
    ):
        if flag:
            return "neutral", 0.0, 0.0, name
    ret = forward_return_bps(row.price_before, row.price_after)
    band = neutral_band_bps(
        spread_bps=row.spread_bps,
        expected_move_bps=row.expected_move_bps,
        realized_vol_bps=row.realized_vol_bps,
        transaction_cost_bps=row.transaction_cost_bps,
        multiplier=multiplier,
        floor_bps=floor_bps,
    )
    if row.surprise_bps is not None and abs(row.surprise_bps) < min_abs_surprise_bps:
        return "neutral", ret, band, "surprise_too_small"
    return label_return(ret, band), ret, band, None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    return float(str(value).strip())


def load_canonical(path: Path) -> list[CanonicalRow]:
    required = {
        "event_id",
        "text",
        "asset_context",
        "price_before",
        "price_after",
        "horizon_seconds",
    }
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(f"Missing columns: {', '.join(sorted(missing))}")
        rows: list[CanonicalRow] = []
        for line, record in enumerate(reader, start=2):
            try:
                rows.append(
                    CanonicalRow(
                        event_id=record["event_id"].strip(),
                        text=record["text"].strip(),
                        asset_context=(record.get("asset_context") or "US equities").strip(),
                        price_before=float(record["price_before"]),
                        price_after=float(record["price_after"]),
                        horizon_seconds=int(float(record["horizon_seconds"])),
                        spread_bps=float(record.get("spread_bps") or 0.0),
                        expected_move_bps=_parse_float(record.get("expected_move_bps")),
                        realized_vol_bps=_parse_float(record.get("realized_vol_bps")),
                        transaction_cost_bps=float(record.get("transaction_cost_bps") or 0.0),
                        confounded=_parse_bool(record.get("confounded") or ""),
                        stale_quote=_parse_bool(record.get("stale_quote") or ""),
                        illiquid=_parse_bool(record.get("illiquid") or ""),
                        uncertain_timing=_parse_bool(record.get("uncertain_timing") or ""),
                        surprise_bps=_parse_float(record.get("surprise_bps")),
                    )
                )
            except (ValueError, KeyError, AttributeError) as exc:
                raise SystemExit(f"Invalid row on line {line}: {exc}") from exc
    return rows


def build_training_rows(
    rows: list[CanonicalRow],
    *,
    multiplier: float = 1.0,
    floor_bps: float = 5.0,
    min_abs_surprise_bps: float = 0.0,
    include_neutral: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    labeled: list[dict[str, object]] = []
    counts = {"bullish": 0, "bearish": 0, "neutral": 0, "excluded": 0}
    for row in rows:
        label, ret_bps, band_bps, excluded = label_row(
            row,
            multiplier=multiplier,
            floor_bps=floor_bps,
            min_abs_surprise_bps=min_abs_surprise_bps,
        )
        if excluded:
            counts["excluded"] += 1
            continue
        if label == "neutral" and not include_neutral:
            counts["neutral"] += 1
            continue
        counts[label] += 1
        # Weight by distance outside the band so marginal 1-bps moves matter less.
        weight = 1.0 if band_bps <= 0 else min(2.0, max(0.5, abs(ret_bps) / band_bps / 2.0))
        labeled.append(
            {
                "text": row.text,
                "label": label,
                "asset_context": row.asset_context,
                "event_id": row.event_id,
                "horizon_seconds": row.horizon_seconds,
                "return_bps": round(ret_bps, 4),
                "neutral_band_bps": round(band_bps, 4),
                "weight": round(weight, 4),
            }
        )
    stats = {"input_rows": len(rows), **counts, "training_rows": len(labeled)}
    return labeled, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label canonical market rows into trainer-ready impact CSVs"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--band-multiplier", type=float, default=1.0)
    parser.add_argument("--band-floor-bps", type=float, default=5.0)
    parser.add_argument("--min-surprise-bps", type=float, default=0.0)
    parser.add_argument("--include-neutral", action="store_true")
    args = parser.parse_args()

    rows = load_canonical(args.input)
    labeled, stats = build_training_rows(
        rows,
        multiplier=args.band_multiplier,
        floor_bps=args.band_floor_bps,
        min_abs_surprise_bps=args.min_surprise_bps,
        include_neutral=args.include_neutral,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "text",
                "label",
                "asset_context",
                "event_id",
                "horizon_seconds",
                "return_bps",
                "neutral_band_bps",
                "weight",
            ],
        )
        writer.writeheader()
        writer.writerows(labeled)
    manifest = {
        "input_sha256": file_sha256(args.input),
        "output_sha256": file_sha256(args.output),
        "parameters": {
            "band_multiplier": args.band_multiplier,
            "band_floor_bps": args.band_floor_bps,
            "min_surprise_bps": args.min_surprise_bps,
            "include_neutral": args.include_neutral,
        },
        "stats": stats,
    }
    if args.manifest:
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
