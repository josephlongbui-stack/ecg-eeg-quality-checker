"""Windowing and filter helpers. Every function returns a NEW array (never in-place)."""

from __future__ import annotations

import numpy as np
from scipy import signal


def iter_windows(n_samples: int, sfreq: float, window_s: float, overlap: float) -> list[tuple[int, int]]:
    """Return (start, stop) sample indices of overlapping analysis windows.

    If the recording is shorter than one window, it is analyzed as one window.
    A final window aligned to the end is added so the tail is always covered.
    """
    if n_samples <= 0:
        return []
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    win = max(1, int(round(window_s * sfreq)))
    if n_samples <= win:
        return [(0, n_samples)]
    step = max(1, int(round(win * (1 - overlap))))
    windows = [(start, start + win) for start in range(0, n_samples - win + 1, step)]
    if windows[-1][1] < n_samples:
        windows.append((n_samples - win, n_samples))
    return windows


def remove_mean(x: np.ndarray) -> np.ndarray:
    """Return a mean-removed copy."""
    x = np.asarray(x, dtype=float)
    return x - np.mean(x)


def lowpass(x: np.ndarray, sfreq: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass (sosfiltfilt). Returns a new array.

    The edges are padded with a mirror image ("even") one cutoff-period long. SciPy's
    default (a ~10-sample point-reflection) creates large fake swings at the edges of a
    slow filter when the recording starts mid-heartbeat.
    """
    x = np.asarray(x, dtype=float)
    sos = signal.butter(order, cutoff_hz, btype="low", fs=sfreq, output="sos")
    padlen = int(min(len(x) - 1, round(sfreq / cutoff_hz)))
    return signal.sosfiltfilt(sos, x, padtype="even", padlen=padlen)


def bandpass(x: np.ndarray, sfreq: float, lo_hz: float, hi_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass with mirror padding. Returns a new array."""
    x = np.asarray(x, dtype=float)
    sos = signal.butter(order, [lo_hz, hi_hz], btype="band", fs=sfreq, output="sos")
    padlen = int(min(len(x) - 1, round(sfreq / lo_hz)))
    return signal.sosfiltfilt(sos, x, padtype="even", padlen=padlen)


def highpass(x: np.ndarray, sfreq: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth high-pass with mirror padding. Returns a new array."""
    x = np.asarray(x, dtype=float)
    sos = signal.butter(order, cutoff_hz, btype="high", fs=sfreq, output="sos")
    padlen = int(min(len(x) - 1, round(sfreq / cutoff_hz)))
    return signal.sosfiltfilt(sos, x, padtype="even", padlen=padlen)
