"""Simple, interpretable signal measurements used by the detectors."""

from __future__ import annotations

import numpy as np
from scipy import signal

MAD_TO_SIGMA = 1.4826  # robust sigma = 1.4826 * MAD for Gaussian data


def std(x: np.ndarray) -> float:
    return float(np.std(x))


def ptp(x: np.ndarray) -> float:
    return float(np.ptp(x)) if len(x) else 0.0


def mad(x: np.ndarray) -> float:
    """Median absolute deviation: median(|x - median(x)|)."""
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


def robust_sigma(x: np.ndarray) -> float:
    """1.4826 * MAD: a spread estimate that a few extreme samples cannot inflate."""
    return MAD_TO_SIGMA * mad(x)


def first_difference(x: np.ndarray) -> np.ndarray:
    """d[n] = x[n+1] - x[n] (volts per sample)."""
    return np.diff(np.asarray(x, dtype=float))


def welch_psd(x: np.ndarray, sfreq: float, segment_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with a Hann window and nperseg = min(len, segment_s * sfreq).

    segment_s = 2 gives ~0.5 Hz resolution (DETECTORS.md); 1 gives 1 Hz with more averaging.
    """
    x = np.asarray(x, dtype=float)
    nperseg = int(min(len(x), round(segment_s * sfreq)))
    return signal.welch(x - np.mean(x), fs=sfreq, window="hann", nperseg=nperseg)


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Mean PSD between lo and hi Hz (inclusive)."""
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.mean(psd[mask])) if mask.any() else float("nan")


def line_ratio(
    freqs: np.ndarray,
    psd: np.ndarray,
    f0: float,
    halfwidth_hz: float = 1.0,
    inner_hz: float = 2.0,
    outer_hz: float = 6.0,
) -> float:
    """mean PSD in [f0-1, f0+1] / median PSD in [f0-6, f0-2] ∪ [f0+2, f0+6]."""
    peak = (freqs >= f0 - halfwidth_hz) & (freqs <= f0 + halfwidth_hz)
    neighbors = ((freqs >= f0 - outer_hz) & (freqs <= f0 - inner_hz)) | (
        (freqs >= f0 + inner_hz) & (freqs <= f0 + outer_hz)
    )
    if not peak.any() or not neighbors.any():
        return float("nan")
    denom = float(np.median(psd[neighbors]))
    num = float(np.mean(psd[peak]))
    if denom <= 0:
        return float("inf") if num > 0 else float("nan")
    return num / denom


def runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, stop) index pairs (stop exclusive) of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    runs = runs_of_true(mask)
    return max((stop - start for start, stop in runs), default=0)


def integrated_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float, exclude: list[tuple[float, float]] | None = None) -> float:
    """Total power between lo and hi Hz (sum of PSD × bin width), optionally skipping excluded bands."""
    mask = (freqs >= lo) & (freqs <= hi)
    for a, b in exclude or []:
        mask &= ~((freqs >= a) & (freqs <= b))
    if not mask.any():
        return 0.0
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    return float(np.sum(psd[mask]) * df)
