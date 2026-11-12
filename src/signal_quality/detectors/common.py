"""Helpers shared by all detectors: output container, severity rule, formatting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import BaseConfig
from ..models import Finding


@dataclass
class DetectorOutput:
    """What one detector returns for one channel."""

    findings: list[Finding] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)  # checks that could not run, with reason
    notes: list[dict] = field(default_factory=list)  # information that is not a finding


def severity(ratio: float, duration_s: float) -> str:
    """DETECTORS.md rule. `ratio` = measurement / threshold (or inverse for 'below' rules).

    high if ratio >= 5 or duration >= 10 s; low if ratio < 2 and duration < 2 s; else medium.
    """
    if ratio >= 5 or duration_s >= 10:
        return "high"
    if ratio < 2 and duration_s < 2:
        return "low"
    return "medium"


def to_display(value_volts: float, config: BaseConfig) -> float:
    """Convert volts to the display unit of this modality (µV for EEG, mV for ECG)."""
    return value_volts * config.display_scale


def unit_key(config: BaseConfig) -> str:
    """Suffix for measurement keys, e.g. 'uV' or 'mV'."""
    return "uV" if config.display_unit == "µV" else "mV"


def fmt(value: float, digits: int = 3) -> str:
    """Compact number formatting for explanations."""
    if value == 0 or not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}g}"


def safe_ratio(numerator: float, denominator: float, cap: float = 1e6) -> float:
    """numerator / denominator, capped so it stays a finite, JSON-friendly number."""
    if denominator <= 0:
        return cap
    return float(min(numerator / denominator, cap))
