"""Shared test fixtures. All signals are generated in code (no downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from signal_quality.demo import _ecg_base, _eeg_base

EEG_FS = 256.0
ECG_FS = 360.0


@pytest.fixture(scope="session")
def eeg_clean_channel() -> np.ndarray:
    """30 s of one realistic simulated EEG channel (volts)."""
    return _eeg_base(seed=1)[6].copy()


@pytest.fixture(scope="session")
def ecg_clean_signal() -> np.ndarray:
    """30 s of clean simulated ECG at 360 Hz (volts)."""
    return _ecg_base(seed=42)[0].copy()


def time_axis(n: int, fs: float) -> np.ndarray:
    return np.arange(n) / fs
