"""ECG electrode-motion artifact (Phase 2).

Combines evidence that other checks already produced in the same window:
  1. baseline drift, 2. a spike / discontinuity / large transient,
  3. R-peak detection failing in this window while both neighbouring windows succeed.
'likely' only when ≥ 2 indicators agree. Beat-detection failure alone gives an 'uncertain' finding.
(Drift alone or a spike alone is already reported by its own detector.)
"""

from __future__ import annotations

import numpy as np

from ..config import BaseConfig
from ..models import Finding
from ..preprocessing import iter_windows
from .common import DetectorOutput, severity

NAME = "electrode_motion@1.0"
JUMPS = {"spike", "discontinuity", "large_transient"}


def detect(
    x: np.ndarray,
    sfreq: float,
    channel: str,
    config: BaseConfig,
    findings: list[Finding],
    r_peaks: np.ndarray | None,
    unusable: np.ndarray | None = None,
) -> DetectorOutput:
    """`findings` are this channel's findings from the other detectors; `unusable` marks flat/missing samples."""
    out = DetectorOutput()
    windows = iter_windows(len(x), sfreq, config.window_s, config.overlap)
    counts = None
    if r_peaks is not None:
        counts = [int(np.sum((r_peaks >= a) & (r_peaks < b))) for a, b in windows]
    for k, (a, b) in enumerate(windows):
        ws, we = a / sfreq, b / sfreq
        overlapping = [f for f in findings if f.channel == channel and f.start_s < we and f.end_s > ws]
        indicators = []
        if any(f.category == "baseline_drift" for f in overlapping):
            indicators.append("baseline drift")
        if any(f.category in JUMPS for f in overlapping):
            indicators.append("abrupt jump")
        usable = unusable is None or float(np.mean(unusable[a:b])) < 0.5
        if counts is not None and usable and counts[k] == 0:
            # neighbours = nearest windows that do NOT overlap this one (overlapping ones share its gap)
            before = [j for j, (_, e) in enumerate(windows) if e <= a]
            after = [j for j, (s0, _) in enumerate(windows) if s0 >= b]
            if before and after and counts[before[-1]] > 0 and counts[after[0]] > 0:
                indicators.append("no beats detected (neighbouring windows have beats)")
        if len(indicators) >= config.motion_min_indicators:
            certainty = "likely"
        elif indicators == ["no beats detected (neighbouring windows have beats)"]:
            certainty = "uncertain"
        else:
            continue
        out.findings.append(
            Finding(
                signal_type=config.signal_type,
                channel=channel,
                start_s=ws,
                end_s=we,
                category="electrode_motion",
                severity=severity(len(indicators) / config.motion_min_indicators, we - ws),
                certainty=certainty,
                rule=f"≥ {config.motion_min_indicators} of: baseline drift, abrupt jump, beat detection failing locally",
                measurement={
                    "indicators": indicators,
                    "n_indicators": len(indicators),
                    "ratio_to_threshold": len(indicators) / config.motion_min_indicators,
                },
                explanation=(
                    f"{' and '.join(indicators).capitalize()} in the same window: "
                    + ("likely" if certainty == "likely" else "possible")
                    + " electrode-motion artifact (electrode or cable moving)."
                ),
                detector=NAME,
            )
        )
    return out
