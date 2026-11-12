"""Detector registries: which detectors run for each modality.

* Per-channel detectors:  detect(x_volts, sfreq, channel_name, config, meta) -> DetectorOutput
* Multi-channel detectors: detect(data_volts, sfreq, channel_names, config) -> DetectorOutput
* Electrode motion (ECG) combines other detectors' findings; analyze.py calls it last.
All are pure functions that never modify their input.
"""

from __future__ import annotations

from . import clipping, drift, electrode_motion, flatline, hf_noise, missing, muscle, noisy, ocular, plausibility, powerline, spikes
from .common import DetectorOutput

PHASE1_DETECTORS = [flatline.detect, clipping.detect, powerline.detect, spikes.detect, drift.detect]

REGISTRY = {
    "EEG": PHASE1_DETECTORS + [muscle.detect, plausibility.detect_eeg_amplitude],
    "ECG": PHASE1_DETECTORS + [hf_noise.detect, plausibility.detect_ecg_rhythm],
}

MULTICHANNEL_REGISTRY = {
    "EEG": [noisy.detect, ocular.detect],
    "ECG": [],
}

__all__ = ["REGISTRY", "MULTICHANNEL_REGISTRY", "DetectorOutput", "electrode_motion", "missing"]
