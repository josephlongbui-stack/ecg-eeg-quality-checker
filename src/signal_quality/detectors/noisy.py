"""EEG noisy channel (Phase 2): one channel varies far more than the others."""

from __future__ import annotations

import numpy as np

from ..config import BaseConfig
from ..features import robust_sigma
from ..models import Finding
from ..preprocessing import highpass, iter_windows
from .common import DetectorOutput, fmt, safe_ratio, severity, to_display

NAME = "noisy_channel@1.0"


def detect(data: np.ndarray, sfreq: float, channel_names: list[str], config: BaseConfig) -> DetectorOutput:
    """Flag a channel whose robust σ is > noisy_ratio × the median robust σ of the other channels."""
    out = DetectorOutput()
    n_ch = len(channel_names)
    if n_ch < config.noisy_min_channels:
        out.skipped.append(
            {"check": "noisy_channel", "channel": "all",
             "reason": f"needs ≥ {config.noisy_min_channels} channels to compare against (found {n_ch})"}
        )
        return out
    # Noise is measured above noisy_highpass_hz so slow drift and steps (reported by their
    # own detectors) do not make a channel look "noisy".
    hp = np.vstack([highpass(data[i], sfreq, config.noisy_highpass_hz) for i in range(n_ch)])
    for start, stop in iter_windows(data.shape[1], sfreq, config.window_s, config.overlap):
        sig = np.array([robust_sigma(hp[i, start:stop]) for i in range(n_ch)])
        for i, ch in enumerate(channel_names):
            others = float(np.median(np.delete(sig, i)))
            if others <= 0:
                continue
            ratio = float(sig[i] / others)
            if ratio <= config.noisy_ratio:
                continue
            rel = safe_ratio(ratio, config.noisy_ratio)
            mine, typical = to_display(float(sig[i]), config), to_display(others, config)
            out.findings.append(
                Finding(
                    signal_type=config.signal_type,
                    channel=ch,
                    start_s=start / sfreq,
                    end_s=stop / sfreq,
                    category="noisy_channel",
                    severity=severity(rel, (stop - start) / sfreq),
                    certainty="likely",
                    rule=f"robust σ (> {config.noisy_highpass_hz:g} Hz) > {config.noisy_ratio:g} × median of the other channels",
                    measurement={
                        "robust_sigma_uV": mine,
                        "others_median_robust_sigma_uV": typical,
                        "ratio": ratio,
                        "ratio_to_threshold": rel,
                    },
                    explanation=(
                        f"Channel {ch} varies {ratio:.1f}× more than the typical channel (robust σ {fmt(mine)} vs "
                        f"{fmt(typical)} {config.display_unit}): possible loose electrode or poor contact."
                    ),
                    detector=NAME,
                )
            )
    return out
