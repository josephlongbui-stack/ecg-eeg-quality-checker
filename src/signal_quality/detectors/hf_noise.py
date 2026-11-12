"""ECG high-frequency (muscle/electrical) noise (Phase 2)."""

from __future__ import annotations

import numpy as np

from ..config import BaseConfig
from ..features import integrated_power, welch_psd
from ..models import Finding
from ..preprocessing import iter_windows
from .common import DetectorOutput, safe_ratio, severity

NAME = "hf_noise@1.0"


def mains_exclusions(nyquist: float, halfwidth: float) -> list[tuple[float, float]]:
    """Bands around 50/60 Hz and their harmonics (power-line is measured by its own detector)."""
    bands = []
    for f0 in (50.0, 60.0):
        k = 1
        while k * f0 < nyquist:
            bands.append((k * f0 - halfwidth, k * f0 + halfwidth))
            k += 1
    return bands


def detect(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    """Flag windows where broadband power above 40 Hz exceeds 30% of the 1–40 Hz power."""
    out = DetectorOutput()
    nyquist = sfreq / 2
    split = config.hf_split_hz
    if nyquist <= split + 5:
        out.skipped.append(
            {"check": "hf_noise", "channel": channel, "reason": f"Nyquist {nyquist:g} Hz too low to measure power above {split:g} Hz"}
        )
        return out
    exclude = mains_exclusions(nyquist, config.hf_mains_exclude_hz)
    x = np.array(x, dtype=float, copy=True)
    for start, stop in iter_windows(len(x), sfreq, config.window_s, config.overlap):
        f, p = welch_psd(x[start:stop], sfreq, config.psd_segment_s)
        ref = integrated_power(f, p, config.hf_lo_hz, split, exclude)
        if ref <= 1e-24:
            continue
        ratio = integrated_power(f, p, split, nyquist, exclude) / ref
        if ratio <= config.hf_ratio:
            continue
        rel = safe_ratio(ratio, config.hf_ratio)
        out.findings.append(
            Finding(
                signal_type=config.signal_type,
                channel=channel,
                start_s=start / sfreq,
                end_s=stop / sfreq,
                category="hf_noise",
                severity=severity(rel, (stop - start) / sfreq),
                certainty="likely",
                rule=(
                    f"power > {split:g} Hz ÷ power {config.hf_lo_hz:g}–{split:g} Hz > {config.hf_ratio:g} "
                    f"(50/60 Hz ± {config.hf_mains_exclude_hz:g} Hz excluded)"
                ),
                measurement={"hf_ratio": ratio, "threshold_ratio": config.hf_ratio, "ratio_to_threshold": rel},
                explanation=(
                    f"Power above {split:g} Hz is {ratio * 100:.0f}% of the {config.hf_lo_hz:g}–{split:g} Hz power: "
                    "possible muscle or electrical noise."
                ),
                detector=NAME,
            )
        )
    return out
