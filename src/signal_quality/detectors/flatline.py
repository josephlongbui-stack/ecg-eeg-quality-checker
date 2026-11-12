"""Flatline / disconnected channel / lead dropout (DETECTORS.md §1)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import BaseConfig
from ..features import longest_run, runs_of_true
from .common import DetectorOutput, fmt, safe_ratio, severity, to_display, unit_key
from ..models import Finding

NAME = "flatline@1.0"


def _sliding_std_ptp(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """σ and ptp of every length-w window x[i:i+w] (memory-light, O(n))."""
    y = x - np.median(x)
    csum = np.concatenate(([0.0], np.cumsum(y)))
    csum2 = np.concatenate(([0.0], np.cumsum(y * y)))
    mean = (csum[w:] - csum[:-w]) / w
    var = (csum2[w:] - csum2[:-w]) / w - mean**2
    sd = np.sqrt(np.clip(var, 0.0, None))
    half = w // 2
    # max/min filters centered so that index i + half covers x[i : i + w]
    mx = ndimage.maximum_filter1d(x, size=w, mode="nearest")[half : half + len(sd)]
    mn = ndimage.minimum_filter1d(x, size=w, mode="nearest")[half : half + len(sd)]
    return sd, mx - mn


def flat_mask(x: np.ndarray, sfreq: float, config: BaseConfig) -> np.ndarray:
    """Boolean mask of samples inside flat stretches (σ and ptp below threshold for ≥ min duration).

    A 0.25 s sliding window locates the flat stretch to the sample level. Also used by the
    spike detector so dead stretches do not distort its estimate of normal variation.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = max(2, int(round(config.flat_refine_window_s * sfreq)))
    mask = np.zeros(n, dtype=bool)
    if n < w:
        return mask
    sd, pp = _sliding_std_ptp(x, w)
    flat_windows = (sd < config.volts("flat_std")) & (pp < config.volts("flat_ptp"))
    covered = np.convolve(flat_windows.astype(float), np.ones(w), mode="full")[:n] > 0
    min_len = int(round(config.flat_min_duration_s * sfreq))
    for start, stop in runs_of_true(covered):
        if stop - start >= min_len:
            mask[start:stop] = True
    return mask


def detect(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    """Flag stretches where σ < flat_std AND ptp < flat_ptp for at least flat_min_duration_s."""
    out = DetectorOutput()
    x = np.array(x, dtype=float, copy=True)
    if len(x) < max(2, int(round(config.flat_refine_window_s * sfreq))):
        out.skipped.append({"check": "flatline", "channel": channel, "reason": "recording shorter than 0.25 s"})
        return out
    std_thr = config.volts("flat_std")
    min_len = int(round(config.flat_min_duration_s * sfreq))
    uk = unit_key(config)
    for start, stop in runs_of_true(flat_mask(x, sfreq, config)):
        seg = x[start:stop]
        seg_std = float(np.std(seg))
        seg_ptp = float(np.ptp(seg))
        # 'definite' when the stretch contains an exactly constant run of at least the minimum duration
        constant_run = longest_run(np.diff(seg) == 0) + 1
        exactly_constant = constant_run >= min_len
        dur = (stop - start) / sfreq
        ratio = safe_ratio(std_thr, seg_std)
        std_disp = to_display(seg_std, config)
        out.findings.append(
            Finding(
                signal_type=config.signal_type,
                channel=channel,
                start_s=start / sfreq,
                end_s=stop / sfreq,
                category="flatline",
                severity=severity(ratio, dur),
                certainty="definite" if exactly_constant else "likely",
                rule=(
                    f"σ < {fmt(getattr(config, 'flat_std_' + uk))} {config.display_unit} and "
                    f"ptp < {fmt(getattr(config, 'flat_ptp_' + uk))} {config.display_unit} "
                    f"for ≥ {config.flat_min_duration_s:g} s"
                ),
                measurement={
                    f"std_{uk}": std_disp,
                    f"ptp_{uk}": to_display(seg_ptp, config),
                    f"threshold_std_{uk}": getattr(config, "flat_std_" + uk),
                    f"threshold_ptp_{uk}": getattr(config, "flat_ptp_" + uk),
                    "duration_s": dur,
                    "exactly_constant_s": constant_run / sfreq,
                    "ratio_to_threshold": ratio,
                },
                explanation=(
                    f"Channel {channel} is nearly flat for {dur:.1f} s (σ = {fmt(std_disp)} "
                    f"{config.display_unit}), consistent with a disconnected electrode or dropout."
                ),
                detector=NAME,
            )
        )
    return out
