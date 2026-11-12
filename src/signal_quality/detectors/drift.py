"""Baseline drift / baseline wander (DETECTORS.md §5)."""

from __future__ import annotations

import numpy as np

from ..config import BaseConfig
from ..models import Finding
from ..preprocessing import iter_windows, lowpass, remove_mean
from .common import DetectorOutput, fmt, safe_ratio, severity, to_display, unit_key

NAME = "drift@1.0"


def detect(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    """Low-pass the WHOLE channel (zero-phase), then flag windows where its ptp ≥ drift_ptp."""
    out = DetectorOutput()
    n = len(x)
    duration = n / sfreq
    cutoff = config.drift_cutoff_hz
    min_duration = config.drift_min_cycles / cutoff
    if duration < min_duration:
        out.skipped.append(
            {
                "check": "baseline_drift",
                "channel": channel,
                "reason": f"recording is {duration:.1f} s; at least {min_duration:.0f} s needed to estimate drift below {cutoff:g} Hz",
            }
        )
        return out
    if cutoff >= sfreq / 2:
        out.skipped.append({"check": "baseline_drift", "channel": channel, "reason": "cutoff above Nyquist"})
        return out

    low = lowpass(remove_mean(x), sfreq, cutoff, config.drift_filter_order)
    thr = config.volts("drift_ptp")
    uk = unit_key(config)
    thr_disp = getattr(config, "drift_ptp_" + uk)
    for start, stop in iter_windows(n, sfreq, config.window_s, config.overlap):
        p = float(np.ptp(low[start:stop]))
        if p < thr:
            continue
        start_s, end_s = start / sfreq, stop / sfreq
        near_edge = start_s < config.drift_edge_s or end_s > duration - config.drift_edge_s
        ratio = safe_ratio(p, thr)
        p_disp = to_display(p, config)
        out.findings.append(
            Finding(
                signal_type=config.signal_type,
                channel=channel,
                start_s=start_s,
                end_s=end_s,
                category="baseline_drift",
                severity=severity(ratio, end_s - start_s),
                certainty="uncertain" if near_edge else "likely",
                rule=f"ptp of < {cutoff:g} Hz component ≥ {thr_disp:g} {config.display_unit}",
                measurement={
                    f"lowfreq_ptp_{uk}": p_disp,
                    f"threshold_{uk}": thr_disp,
                    "cutoff_hz": cutoff,
                    "ratio_to_threshold": ratio,
                },
                explanation=(
                    f"The baseline wanders by {fmt(p_disp)} {config.display_unit} over this window "
                    f"(below {cutoff:g} Hz): likely baseline drift from movement, breathing or electrode contact."
                ),
                detector=NAME,
            )
        )
    return out
