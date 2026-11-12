"""ECG beat (R-peak) detection with NeuroKit2, used as an AUXILIARY quality signal only.

A failure to find beats is never treated as an artifact or an arrhythmia by itself.
"""

from __future__ import annotations

import warnings

import numpy as np


def find_r_peaks(x: np.ndarray, sfreq: float) -> np.ndarray | None:
    """R-peak sample indices from NeuroKit2, or None if detection fails or finds nothing."""
    try:
        import neurokit2 as nk

        x = np.asarray(x, dtype=float)
        if x.size < 2 * sfreq or not np.all(np.isfinite(x)):
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rate = int(round(sfreq))
            cleaned = nk.ecg_clean(x, sampling_rate=rate)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=rate)
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        return peaks if peaks.size > 0 else None
    except Exception:
        return None
