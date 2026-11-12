"""Power-line interference at 50/60 Hz (DETECTORS.md §3).

Measured on the RAW window, before any notch filter: never filter out the
thing you are measuring.
"""

from __future__ import annotations

import numpy as np

from ..config import BaseConfig
from ..features import line_ratio, welch_psd
from ..models import Finding
from ..preprocessing import iter_windows
from .common import DetectorOutput, safe_ratio, severity

NAME = "powerline@1.0"


def detect(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    out = DetectorOutput()
    x = np.array(x, dtype=float, copy=True)
    nyquist = sfreq / 2
    margin = config.line_neighbor_outer_hz
    for f0 in config.mains_frequencies:
        if f0 + margin >= nyquist:
            out.skipped.append(
                {
                    "check": f"powerline_{f0:g}Hz",
                    "channel": channel,
                    "reason": f"Nyquist {nyquist:g} Hz ≤ {f0 + margin:g} Hz (sampling rate too low)",
                }
            )
            continue
        ratio_args = (f0, config.line_halfwidth_hz, config.line_neighbor_inner_hz, margin)

        # Whole-recording check for an applied notch filter (information only).
        freqs, psd = welch_psd(x, sfreq, config.psd_segment_s)
        overall = line_ratio(freqs, psd, *ratio_args)
        if np.isfinite(overall) and overall < config.notch_ratio_below:
            out.notes.append(
                {
                    "channel": channel,
                    "note": f"A notch filter at {f0:g} Hz appears to have been applied (ratio {overall:.3f}).",
                }
            )

        for start, stop in iter_windows(len(x), sfreq, config.window_s, config.overlap):
            freqs, psd = welch_psd(x[start:stop], sfreq, config.psd_segment_s)
            ratio = line_ratio(freqs, psd, *ratio_args)
            if not np.isfinite(ratio) or ratio < config.line_ratio_threshold:
                continue
            ratio = float(min(ratio, 1e6))
            rel = safe_ratio(ratio, config.line_ratio_threshold)
            dur = (stop - start) / sfreq
            out.findings.append(
                Finding(
                    signal_type=config.signal_type,
                    channel=channel,
                    start_s=start / sfreq,
                    end_s=stop / sfreq,
                    category="powerline",
                    severity=severity(rel, dur),
                    certainty="definite" if ratio >= config.line_ratio_definite else "likely",
                    rule=(
                        f"mean PSD in {f0:g}±{config.line_halfwidth_hz:g} Hz ÷ median PSD of neighbours "
                        f"≥ {config.line_ratio_threshold:g}"
                    ),
                    measurement={
                        "frequency_hz": f0,
                        "line_ratio": ratio,
                        "line_ratio_dB": float(10 * np.log10(ratio)),
                        "threshold_ratio": config.line_ratio_threshold,
                        "ratio_to_threshold": rel,
                    },
                    explanation=(
                        f"Power at {f0:g} Hz is {ratio:.0f}× higher than nearby frequencies: "
                        "likely electrical power-line interference."
                    ),
                    detector=NAME,
                )
            )
    return out
