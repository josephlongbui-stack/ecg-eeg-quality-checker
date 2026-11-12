"""Data models shared by every part of the package.

Signals are always stored in VOLTS inside a Recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SignalType = Literal["ECG", "EEG"]
DataKind = Literal["simulated", "real", "real_with_injected_artifacts"]
Severity = Literal["low", "medium", "high"]
Certainty = Literal["definite", "likely", "uncertain"]

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
CERTAINTY_ORDER = {"uncertain": 0, "likely": 1, "definite": 2}


@dataclass(frozen=True)
class Recording:
    """One multi-channel recording.

    `data` has shape (n_channels, n_samples), is in volts and is stored as a
    read-only copy so nothing downstream can modify the original signal.
    """

    signal_type: SignalType
    data: np.ndarray
    sfreq: float
    channel_names: list[str]
    source: str
    data_kind: DataKind
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.array(self.data, dtype=float, copy=True)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        arr.setflags(write=False)
        object.__setattr__(self, "data", arr)
        object.__setattr__(self, "channel_names", list(self.channel_names))
        object.__setattr__(self, "sfreq", float(self.sfreq))

    @property
    def is_simulated(self) -> bool:
        """True only for simulated (demo) signals."""
        return self.data_kind == "simulated"

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sfreq if self.sfreq > 0 else 0.0


@dataclass(frozen=True)
class Finding:
    """One flagged quality problem, with the exact rule and measurement behind it."""

    signal_type: str
    channel: str
    start_s: float
    end_s: float
    category: str
    severity: Severity
    certainty: Certainty
    rule: str
    measurement: dict
    explanation: str
    detector: str


@dataclass(frozen=True)
class InjectedArtifact:
    """Ground truth for an artifact that was added on purpose (demo / experiments)."""

    channel: str
    start_s: float
    end_s: float
    category: str
    parameters: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Everything one call to analyze_recording() produces."""

    recording_summary: dict
    findings: list[Finding]
    config: dict
    windows_analyzed: int
    windows_flagged: int
    skipped_checks: list[dict]
    package_version: str
    created_at: str
    notes: list[dict] = field(default_factory=list)
