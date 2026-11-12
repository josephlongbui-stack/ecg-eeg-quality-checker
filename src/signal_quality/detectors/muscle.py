"""EEG high-frequency / possible muscle contamination (Phase 2)."""

from __future__ import annotations

import numpy as np

from ..config import BaseConfig
from ..features import integrated_power, welch_psd
from ..models import Finding
from ..preprocessing import iter_windows
from .common import DetectorOutput, safe_ratio, severity

NAME = "muscle@1.0"


def detect(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    """Flag windows where power in 20–45 Hz is more than half of the 1–45 Hz power."""
    out = DetectorOutput()
    lo = config.muscle_band_lo_hz
    hi = min(config.muscle_band_hi_hz, sfreq / 2 - 5)
    if hi <= lo + 5:
        out.skipped.append(
            {"check": "muscle", "channel": channel, "reason": f"sampling rate too low for a {lo:g}+ Hz band (upper edge {hi:g} Hz)"}
        )
        return out
    x = np.array(x, dtype=float, copy=True)
    for start, stop in iter_windows(len(x), sfreq, config.window_s, config.overlap):
        f, p = welch_psd(x[start:stop], sfreq, config.psd_segment_s)
        total = integrated_power(f, p, config.muscle_total_lo_hz, hi)
        if total <= 1e-24:  # flat window: nothing to measure
            continue
        ratio = integrated_power(f, p, lo, hi) / total
        if ratio <= config.muscle_ratio:
            continue
        rel = safe_ratio(ratio, config.muscle_ratio)
        out.findings.append(
            Finding(
                signal_type=config.signal_type,
                channel=channel,
                start_s=start / sfreq,
                end_s=stop / sfreq,
                category="muscle",
                severity=severity(rel, (stop - start) / sfreq),
                certainty="likely",
                rule=f"power {lo:g}–{hi:g} Hz ÷ power {config.muscle_total_lo_hz:g}–{hi:g} Hz > {config.muscle_ratio:g}",
                measurement={
                    "hf_power_fraction": ratio,
                    "threshold_fraction": config.muscle_ratio,
                    "band_hz": [lo, hi],
                    "ratio_to_threshold": rel,
                },
                explanation=(
                    f"{ratio * 100:.0f}% of the {config.muscle_total_lo_hz:g}–{hi:g} Hz power is above {lo:g} Hz: "
                    "possible muscle (high-frequency) contamination."
                ),
                detector=NAME,
            )
        )
    return out
