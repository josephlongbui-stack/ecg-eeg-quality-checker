"""Clipping / saturation (DETECTORS.md §2)."""

from __future__ import annotations

import numpy as np

from ..config import BaseConfig
from ..features import longest_run, runs_of_true
from ..models import Finding
from ..preprocessing import iter_windows
from .common import DetectorOutput, fmt, safe_ratio, severity, to_display, unit_key

NAME = "clipping@1.0"


def detect(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    """Flag windows where many samples sit at the channel's extreme value in flat runs.

    Uses the declared physical min/max from the file header when available
    (meta["physical_range_V"]), otherwise the observed global max/min.
    """
    out = DetectorOutput()
    x = np.array(x, dtype=float, copy=True)
    meta = meta or {}
    declared = (meta.get("physical_range_V") or {}).get(channel)
    lo_obs, hi_obs = float(np.min(x)), float(np.max(x))
    if declared is not None:
        lo, hi = float(declared[0]), float(declared[1])
    else:
        lo, hi = lo_obs, hi_obs
    full_range = hi - lo
    if full_range <= 0:
        out.skipped.append(
            {"check": "clipping", "channel": channel, "reason": "channel is constant (range 0); see flatline"}
        )
        return out
    tol = config.clip_tolerance_frac * full_range
    min_run = max(config.clip_min_run_samples_floor, int(round(config.clip_min_run_s * sfreq)))
    sides = {"max": (x >= hi - tol, hi), "min": (x <= lo + tol, lo)}
    uk = unit_key(config)

    for start, stop in iter_windows(len(x), sfreq, config.window_s, config.overlap):
        for side, (mask, level) in sides.items():
            win_mask = mask[start:stop]
            frac = float(np.mean(win_mask))
            run = longest_run(win_mask)
            n_runs = sum(1 for a, b in runs_of_true(win_mask) if b - a >= min_run)
            if frac < config.clip_min_fraction or run < min_run or n_runs < config.clip_min_runs:
                continue
            at_declared = declared is not None
            if run < 2 * min_run:
                certainty = "uncertain"
            elif at_declared:
                certainty = "definite"
            else:
                certainty = "likely"
            ratio = safe_ratio(frac, config.clip_min_fraction)
            dur = (stop - start) / sfreq
            out.findings.append(
                Finding(
                    signal_type=config.signal_type,
                    channel=channel,
                    start_s=start / sfreq,
                    end_s=stop / sfreq,
                    category="clipping",
                    severity=severity(ratio, dur),
                    certainty=certainty,
                    rule=(
                        f"≥ {config.clip_min_fraction * 100:g}% of window within "
                        f"{config.clip_tolerance_frac * 100:g}% of range of the "
                        f"{'declared' if at_declared else 'observed'} {side}, "
                        f"in ≥ {config.clip_min_runs} flat runs of ≥ {min_run} samples"
                    ),
                    measurement={
                        "side": side,
                        "fraction_at_extreme_pct": frac * 100,
                        "longest_run_samples": run,
                        "runs_at_extreme": n_runs,
                        f"extreme_value_{uk}": to_display(level, config),
                        "threshold_fraction_pct": config.clip_min_fraction * 100,
                        "threshold_run_samples": min_run,
                        "ratio_to_threshold": ratio,
                    },
                    explanation=(
                        f"{frac * 100:.1f}% of samples in this window sit at the channel's {side} value "
                        f"in flat runs up to {run} samples: likely amplifier saturation (clipping)."
                    ),
                    detector=NAME,
                )
            )
    return out
