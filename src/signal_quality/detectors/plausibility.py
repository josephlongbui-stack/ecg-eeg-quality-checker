"""Physiological plausibility HINTS (Phase 2). Always 'uncertain'; severity at most 'medium'.

These check measurements against known physiology (heart-rate range, beat timing, EEG
amplitude). They are hints, not verdicts: an implausible value usually means a recording
or beat-detection problem, but unusual genuine physiology is possible. Never diagnostic.
"""

from __future__ import annotations

import numpy as np

from ..beats import find_r_peaks
from ..config import BaseConfig
from ..features import robust_sigma, runs_of_true
from ..models import SEVERITY_ORDER, Finding
from ..preprocessing import iter_windows
from .common import DetectorOutput, fmt, safe_ratio, severity, to_display
from .flatline import flat_mask

NAME_ECG = "plausibility_rhythm@1.0"
NAME_EEG = "plausibility_amplitude@1.0"
REVIEW = "Review this interval."


def _capped(sev: str, config: BaseConfig) -> str:
    cap = config.plaus_max_severity
    return sev if SEVERITY_ORDER[sev] <= SEVERITY_ORDER[cap] else cap


def _hint(config, channel, start_s, end_s, category, ratio, rule, measurement, explanation, name) -> Finding:
    measurement = dict(measurement, ratio_to_threshold=ratio)
    return Finding(
        signal_type=config.signal_type,
        channel=channel,
        start_s=start_s,
        end_s=end_s,
        category=category,
        severity=_capped(severity(ratio, end_s - start_s), config),
        certainty="uncertain",
        rule=rule,
        measurement=measurement,
        explanation=explanation,
        detector=name,
    )


def detect_ecg_rhythm(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    out = DetectorOutput()
    meta = meta or {}
    x = np.array(x, dtype=float, copy=True)
    peaks = meta["r_peaks"] if "r_peaks" in meta else find_r_peaks(x, sfreq)
    if peaks is None or len(peaks) < 2:
        out.skipped.append(
            {"check": "plausibility_rhythm", "channel": channel,
             "reason": "beat detection did not find enough beats; rhythm plausibility not checked"}
        )
        return out
    unusable = flat_mask(x, sfreq, config)
    if meta.get("missing_mask") is not None:
        unusable = unusable | meta["missing_mask"]
    cat = "implausible_rhythm_measurement"
    lo_bpm, hi_bpm = config.plaus_hr_min_bpm, config.plaus_hr_max_bpm

    # 1. Beats closer than the refractory period
    rr = np.diff(peaks) / sfreq
    for i in np.flatnonzero(rr < config.plaus_rr_min_s):
        a, b = int(peaks[i]), int(peaks[i + 1])
        out.findings.append(_hint(
            config, channel, a / sfreq, b / sfreq, cat, safe_ratio(config.plaus_rr_min_s, rr[i]),
            f"RR interval < {config.plaus_rr_min_s * 1000:.0f} ms",
            {"rr_ms": rr[i] * 1000, "threshold_ms": config.plaus_rr_min_s * 1000},
            f"Two detected beats are only {rr[i] * 1000:.0f} ms apart, shorter than the heart's refractory period "
            f"(~{config.plaus_rr_min_s * 1000:.0f} ms). This usually means noise was mistaken for a beat, but unusual "
            f"rhythms are possible. {REVIEW}",
            NAME_ECG,
        ))

    # 2. Long stretches with no detected beat while the signal is present (edges included)
    bounds = np.concatenate(([0], peaks, [len(x)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        gap = (b - a) / sfreq
        if gap <= config.plaus_gap_s or float(np.mean(unusable[a:b])) >= 0.5:
            continue
        out.findings.append(_hint(
            config, channel, a / sfreq, b / sfreq, cat, safe_ratio(gap, config.plaus_gap_s),
            f"no detected beat for > {config.plaus_gap_s:g} s while the signal is not flat",
            {"gap_s": gap, "threshold_s": config.plaus_gap_s},
            f"No heartbeat was detected for {gap:.1f} s although the signal is not flat. This usually means beat "
            f"detection was confused by noise or a weak signal, but unusual rhythms are possible. {REVIEW}",
            NAME_ECG,
        ))

    # 3. Heart rate per window outside the plausible range
    for a, b in iter_windows(len(x), sfreq, config.window_s, config.overlap):
        inside = peaks[(peaks >= a) & (peaks < b)]
        if len(inside) < 2 or float(np.mean(unusable[a:b])) >= 0.5:
            continue
        hr = 60.0 / float(np.median(np.diff(inside) / sfreq))
        if lo_bpm <= hr <= hi_bpm:
            continue
        ratio = safe_ratio(hr, hi_bpm) if hr > hi_bpm else safe_ratio(lo_bpm, hr)
        out.findings.append(_hint(
            config, channel, a / sfreq, b / sfreq, cat, ratio,
            f"heart rate from detected beats outside {lo_bpm:g}–{hi_bpm:g} bpm",
            {"heart_rate_bpm": hr, "plausible_range_bpm": [lo_bpm, hi_bpm]},
            f"Detected beat timing implies {hr:.0f} bpm, which is outside the plausible range ({lo_bpm:g}–{hi_bpm:g} bpm). "
            f"This usually means beat detection was confused by noise, but unusual rhythms are possible. {REVIEW}",
            NAME_ECG,
        ))
    return out


def detect_eeg_amplitude(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    out = DetectorOutput()
    x = np.array(x, dtype=float, copy=True)
    thr = config.volts("plaus_amp")
    windows = iter_windows(len(x), sfreq, config.window_s, config.overlap)
    hot = np.zeros(len(x), dtype=bool)
    amps = []
    for a, b in windows:
        amp = robust_sigma(x[a:b])
        if amp > thr:
            hot[a:b] = True
            amps.append((a, b, amp))
    for a, b in runs_of_true(hot):
        dur = (b - a) / sfreq
        if dur < config.plaus_amp_min_duration_s:
            continue
        amp = max(v for s, e, v in amps if s < b and e > a)
        amp_disp = to_display(amp, config)
        out.findings.append(_hint(
            config, channel, a / sfreq, b / sfreq, "implausible_amplitude", safe_ratio(amp, thr),
            f"robust amplitude (1.4826·MAD) > {config.plaus_amp_uV:g} µV for ≥ {config.plaus_amp_min_duration_s:g} s",
            {"robust_amplitude_uV": amp_disp, "threshold_uV": config.plaus_amp_uV, "duration_s": dur},
            f"The signal's typical amplitude here is {fmt(amp_disp)} µV, well above usual scalp EEG "
            f"(threshold {config.plaus_amp_uV:g} µV). This usually means a recording problem, but unusually large "
            f"genuine activity is possible. {REVIEW}",
            NAME_EEG,
        ))
    return out
