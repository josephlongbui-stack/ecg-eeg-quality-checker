"""Non-finite / missing data (Phase 2): NaN/Inf runs become 'missing data' findings."""

from __future__ import annotations

import numpy as np

from ..config import BaseConfig
from ..features import runs_of_true
from ..models import Finding
from .common import DetectorOutput, severity

NAME = "missing@1.0"


def missing_mask(x: np.ndarray) -> np.ndarray:
    """True where the sample is NaN or ±Inf."""
    return ~np.isfinite(np.asarray(x, dtype=float))


def fill_missing(x: np.ndarray) -> np.ndarray:
    """Copy of x with NaN/Inf replaced by straight lines between the neighbouring good samples.

    Only so the other checks can run; the gaps themselves are reported as findings.
    """
    y = np.array(x, dtype=float, copy=True)
    bad = ~np.isfinite(y)
    if not bad.any():
        return y
    good = np.flatnonzero(~bad)
    if good.size == 0:
        return np.zeros_like(y)
    y[bad] = np.interp(np.flatnonzero(bad), good, y[good])
    return y


def detect(x: np.ndarray, sfreq: float, channel: str, config: BaseConfig, meta: dict | None = None) -> DetectorOutput:
    out = DetectorOutput()
    mask = missing_mask(x)
    n = len(x)
    for start, stop in runs_of_true(mask):
        dur = (stop - start) / sfreq
        out.findings.append(
            Finding(
                signal_type=config.signal_type,
                channel=channel,
                start_s=start / sfreq,
                end_s=stop / sfreq,
                category="missing_data",
                severity=severity(1.0, dur),
                certainty="definite",
                rule="sample value is NaN or Inf",
                measurement={
                    "missing_samples": stop - start,
                    "duration_s": dur,
                    "percent_of_channel": 100 * (stop - start) / n,
                    "ratio_to_threshold": 1.0,
                },
                explanation=(
                    f"{dur:.2f} s of data is missing (NaN/Inf) on channel {channel}. Other checks bridged "
                    "this gap with a straight line, so treat results inside it with care."
                ),
                detector=NAME,
            )
        )
    return out
