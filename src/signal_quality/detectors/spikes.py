"""Sudden spikes and discontinuities (DETECTORS.md §4).

EEG and ECG use different rules on purpose: ECG R-peaks are naturally very
steep, so a robust z-score would flag every heartbeat.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..config import BaseConfig
from ..features import first_difference, robust_sigma
from ..models import Finding
from .flatline import flat_mask
from .common import DetectorOutput, fmt, safe_ratio, severity, to_display, unit_key

NAME = "spikes@1.0"


def _find_r_peaks(x: np.ndarray, sfreq: float) -> np.ndarray | None:
    """R-peak sample indices from NeuroKit2, or None if detection fails.

    Failure is NOT treated as an artifact or an arrhythmia; the cross-check is skipped.
    """
    try:
        import neurokit2 as nk

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate = int(round(sfreq))
            cleaned = nk.ecg_clean(x, sampling_rate=rate)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=rate)
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        return peaks if peaks.size > 0 else None
    except Exception:
        return None


def _group(indices: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """Split sorted indices into groups whose neighbours are at most max_gap apart."""
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > max_gap) + 1
    return np.split(indices, breaks)


def detect(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    out = DetectorOutput()
    x = np.array(x, dtype=float, copy=True)
    d = first_difference(x)
    if d.size < 2:
        return out
    uk = unit_key(config)
    # Normal variation is measured only outside flat stretches: a dead stretch has zero
    # differences, which would shrink the scale and make ordinary activity look extreme.
    active = ~flat_mask(x, sfreq, config)[1:]
    d_active = d[active] if active.sum() >= 10 else d

    if config.signal_type == "EEG":
        center = float(np.median(d_active))
        scale = robust_sigma(d_active)
        threshold = config.spike_z
        if scale <= 0:
            out.skipped.append(
                {"check": "spikes", "channel": channel, "reason": "first difference has zero spread (channel mostly constant)"}
            )
            return out
        score = np.abs(d - center) / scale
        rule = f"|Δx − median(Δx)| / robust σ(Δx) > {threshold:g}"
    else:
        scale = float(np.percentile(np.abs(d_active), config.spike_percentile))
        threshold = config.spike_factor
        if scale <= 0:
            out.skipped.append(
                {"check": "spikes", "channel": channel, "reason": "first difference is almost always zero"}
            )
            return out
        score = np.abs(d) / scale
        rule = f"|Δx| > {threshold:g} × P{config.spike_percentile:g}(|Δx|)"

    flagged = np.flatnonzero(score > threshold)
    if flagged.size == 0:
        return out

    local = max(1, int(round(sfreq)))  # 1 s each side to estimate local signal spread
    ctx = max(1, int(round(config.spike_context_s * sfreq)))
    groups = _group(flagged, max(1, int(round(config.spike_group_s * sfreq))))

    r_peaks = None
    if config.signal_type == "ECG":
        r_peaks = _find_r_peaks(x, sfreq)
        if r_peaks is None:
            out.skipped.append(
                {
                    "check": "spikes_r_peak_crosscheck",
                    "channel": channel,
                    "reason": "R-peak detection failed; spike flags were not cross-checked against beats",
                }
            )
    proximity = int(round(getattr(config, "rpeak_proximity_s", 0.05) * sfreq))

    for idx in groups:
        a, b = int(idx[0]), int(idx[-1])
        pre = x[max(0, a - ctx + 1) : a + 1]
        post = x[b + 1 : b + 1 + ctx]
        # Local robust σ of the signal (1 s before and after, each side centred on its own
        # median) so the step itself does not inflate the spread it is compared against.
        before = x[max(0, a - local + 1) : a + 1]
        after = x[b + 1 : b + 1 + local]
        signal_sigma = robust_sigma(
            np.concatenate([before - np.median(before), after - np.median(after) if after.size else after])
        )
        pre_level = float(np.median(pre))
        peak_dev = float(np.max(np.abs(x[a : b + 2] - pre_level)))
        if post.size:
            step = float(np.median(post)) - pre_level
            if abs(step) <= config.spike_return_frac * peak_dev:
                category = "spike"
            elif abs(step) > config.step_robust_sigma_factor * signal_sigma:
                category = "discontinuity"
            else:
                category = "large_transient"
        else:
            step = float("nan")
            category = "large_transient"

        k = float(np.max(score[idx]))
        # single-sample event: a one-sample step (1 flagged difference) or a one-sample spike (2)
        single_sample = (b - a) <= 1
        near_r_peak = False
        if r_peaks is not None:
            near_r_peak = bool(np.any(np.abs(r_peaks[:, None] - idx[None, :]) <= proximity))
        if near_r_peak:
            certainty = "uncertain"
        elif k >= 2 * threshold and single_sample:
            certainty = "definite"
        else:
            certainty = "likely"

        start_s, end_s = a / sfreq, (b + 1) / sfreq
        ratio = safe_ratio(k, threshold)
        out.findings.append(
            Finding(
                signal_type=config.signal_type,
                channel=channel,
                start_s=start_s,
                end_s=end_s,
                category=category,
                severity=severity(ratio, end_s - start_s),
                certainty=certainty,
                rule=rule,
                measurement={
                    "steepness_ratio": k,
                    "threshold": threshold,
                    f"max_jump_{uk}": to_display(float(np.max(np.abs(d[idx]))), config),
                    f"level_change_{uk}": to_display(step, config) if np.isfinite(step) else None,
                    "single_sample": single_sample,
                    "near_r_peak": near_r_peak,
                    "ratio_to_threshold": ratio,
                },
                explanation=(
                    f"An abrupt jump {k:.0f}× steeper than this channel's normal variation at "
                    f"{start_s:.2f} s: possible electrode pop or motion artifact."
                    + (" It is close to a detected heartbeat, so it may be an unusual but genuine beat." if near_r_peak else "")
                ),
                detector=NAME,
            )
        )
    return out
