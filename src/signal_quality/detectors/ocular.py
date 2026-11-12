"""EEG possible ocular artifact (Phase 2). Runs only when frontal (Fp/AF) channels exist. Never 'definite'."""

from __future__ import annotations

import re

import numpy as np

from ..config import BaseConfig
from ..models import Finding
from ..preprocessing import bandpass, iter_windows
from .common import DetectorOutput, fmt, safe_ratio, severity, to_display

NAME = "ocular@1.0"
_FRONTAL = re.compile(r"^(FP|AF)")
_POSTERIOR = re.compile(r"^(O|PO|P)(\d|Z)")


def _norm(name: str) -> str:
    """'EEG Fp1-REF' -> 'FP1'."""
    name = re.sub(r"^EEG\s*", "", str(name).strip().upper())
    return re.split(r"[-\s]", name)[0]


def detect(data: np.ndarray, sfreq: float, channel_names: list[str], config: BaseConfig) -> DetectorOutput:
    """Flag large 0.5–4 Hz deflections on frontal channels that are ≥ 3× those on posterior channels."""
    out = DetectorOutput()
    norm = [_norm(c) for c in channel_names]
    frontal = [i for i, c in enumerate(norm) if _FRONTAL.match(c)]
    if not frontal:
        out.skipped.append({"check": "ocular", "channel": "all", "reason": "no frontal channels (Fp/AF) to check for eye artifacts"})
        return out
    reference = [i for i, c in enumerate(norm) if _POSTERIOR.match(c)]
    ref_label = "posterior channels"
    if not reference:
        reference = [i for i in range(len(norm)) if i not in frontal]
        ref_label = "other channels"
    if not reference:
        out.skipped.append({"check": "ocular", "channel": "all", "reason": "no non-frontal channels to compare with"})
        return out
    if config.ocular_band_hi_hz >= sfreq / 2:
        out.skipped.append({"check": "ocular", "channel": "all", "reason": "sampling rate too low"})
        return out

    band = {i: bandpass(data[i], sfreq, config.ocular_band_lo_hz, config.ocular_band_hi_hz) for i in frontal + reference}
    thr = config.volts("ocular_ptp")
    for start, stop in iter_windows(data.shape[1], sfreq, config.window_s, config.overlap):
        ref_ptp = float(np.median([np.ptp(band[i][start:stop]) for i in reference]))
        hits = []
        for i in frontal:
            p = float(np.ptp(band[i][start:stop]))
            ratio = p / ref_ptp if ref_ptp > 0 else float("inf")
            if p >= thr and ratio >= config.ocular_frontal_ratio:
                hits.append((i, p, ratio))
        for i, p, ratio in hits:
            # Blinks are usually seen on both sides: 'likely' when ≥ 2 frontal channels agree.
            certainty = "likely" if len(hits) >= 2 else "uncertain"
            rel = safe_ratio(p, thr)
            p_disp = to_display(p, config)
            shown_ratio = min(ratio, 1e6)
            out.findings.append(
                Finding(
                    signal_type=config.signal_type,
                    channel=channel_names[i],
                    start_s=start / sfreq,
                    end_s=stop / sfreq,
                    category="ocular",
                    severity=severity(rel, (stop - start) / sfreq),
                    certainty=certainty,
                    rule=(
                        f"{config.ocular_band_lo_hz:g}–{config.ocular_band_hi_hz:g} Hz ptp ≥ "
                        f"{config.ocular_ptp_uV:g} µV and ≥ {config.ocular_frontal_ratio:g}× the {ref_label}"
                    ),
                    measurement={
                        "frontal_ptp_uV": p_disp,
                        "reference_median_ptp_uV": to_display(ref_ptp, config),
                        "frontal_to_reference": shown_ratio,
                        "frontal_channels_agreeing": len(hits),
                        "ratio_to_threshold": rel,
                    },
                    explanation=(
                        f"Large slow deflection on {channel_names[i]} ({fmt(p_disp)} µV, {shown_ratio:.1f}× the "
                        f"{ref_label}): possible eye blink or eye movement (ocular artifact)."
                    ),
                    detector=NAME,
                )
            )
    return out
