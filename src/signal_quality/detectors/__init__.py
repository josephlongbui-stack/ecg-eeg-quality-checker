"""Detector registry: which detectors run for each modality.

Every detector is a pure function
    detect(x_volts, sfreq, channel_name, config, meta) -> DetectorOutput
that never modifies its input.
"""

from __future__ import annotations

from . import clipping, drift, flatline, powerline, spikes
from .common import DetectorOutput

PHASE1_DETECTORS = [flatline.detect, clipping.detect, powerline.detect, spikes.detect, drift.detect]

REGISTRY = {
    "EEG": PHASE1_DETECTORS,
    "ECG": PHASE1_DETECTORS,
}

__all__ = ["REGISTRY", "DetectorOutput"]
