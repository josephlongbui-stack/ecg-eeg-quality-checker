"""All detection thresholds, with units in the field names and a rationale for each.

These defaults are reasonable starting points, NOT validated values (see DETECTORS.md).
EEG amplitude thresholds are in microvolts (_uV); ECG ones in millivolts (_mV).
Use `config.volts("flat_std")` to get any amplitude threshold in volts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

UNIT_SUFFIX_TO_VOLTS = {"_uV": 1e-6, "_mV": 1e-3, "_V": 1.0}


@dataclass(frozen=True)
class BaseConfig:
    """Settings shared by both modalities (no amplitude units here)."""

    signal_type: str = ""
    display_unit: str = ""  # unit used for display and for measurements in reports
    window_s: float = 2.0  # analysis window length (s)
    overlap: float = 0.5  # fraction of window overlap
    mains_hz: str = "60"  # "60" (US), "50" (Europe) or "both"

    # --- Flatline ---
    flat_min_duration_s: float = 1.0  # shorter flat stretches can occur naturally
    flat_refine_window_s: float = 0.25  # sliding window used to locate flat edges

    # --- Clipping ---
    clip_tolerance_frac: float = 0.001  # 0.1% of channel range: allows for quantization
    clip_min_fraction: float = 0.01  # 1% of window: real signals touch extremes only briefly
    clip_min_run_s: float = 0.01  # run length 0.01*sfreq samples...
    clip_min_run_samples_floor: int = 3  # ...but at least 3 samples (a true peak lasts ~1 sample)

    # --- Power line ---
    psd_segment_s: float = 1.0  # Welch segment length; 1 s averages 3 segments per 2 s EEG window (see DECISIONS.md)
    line_ratio_threshold: float = 10.0  # 10 dB narrow peak above neighbours: not broadband physiology
    line_ratio_definite: float = 100.0  # 20 dB: overwhelming peak
    line_halfwidth_hz: float = 1.0  # peak band f0 ± 1 Hz
    line_neighbor_inner_hz: float = 2.0  # neighbour bands start 2 Hz away...
    line_neighbor_outer_hz: float = 6.0  # ...and end 6 Hz away (also the Nyquist margin)
    notch_ratio_below: float = 0.1  # whole-recording ratio below this suggests a notch filter

    # --- Spikes / discontinuities ---
    spike_group_s: float = 0.2  # flagged samples within 200 ms form one event
    spike_context_s: float = 0.1  # look 100 ms before/after to classify the event
    spike_return_frac: float = 0.25  # "returns to within 25% of prior level" -> spike
    step_robust_sigma_factor: float = 5.0  # step > 5x robust sigma of signal -> discontinuity

    # --- Baseline drift ---
    drift_filter_order: int = 2  # gentle zero-phase Butterworth low-pass
    drift_min_cycles: float = 3.0  # need >= 3/cutoff seconds to estimate slow drift
    drift_edge_s: float = 1.0  # findings this close to start/end are 'uncertain' (filter edges)

    # --- Merging ---
    merge_gap_s: float = 0.5  # same-category findings closer than this are merged

    def volts(self, name: str) -> float:
        """Return amplitude threshold `name` (without unit suffix) in volts."""
        for suffix, scale in UNIT_SUFFIX_TO_VOLTS.items():
            if hasattr(self, name + suffix):
                return float(getattr(self, name + suffix)) * scale
        raise KeyError(f"No amplitude setting called '{name}' in {type(self).__name__}")

    @property
    def display_scale(self) -> float:
        """Multiply volts by this to get display units (µV for EEG, mV for ECG)."""
        return 1e6 if self.display_unit == "µV" else 1e3

    @property
    def mains_frequencies(self) -> list[float]:
        if str(self.mains_hz) == "both":
            return [50.0, 60.0]
        return [float(self.mains_hz)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EEGConfig(BaseConfig):
    signal_type: str = "EEG"
    display_unit: str = "µV"
    window_s: float = 2.0  # stable spectrum at 0.5 Hz resolution, still localizes artifacts
    flat_std_uV: float = 0.5  # well below the background activity of a connected EEG electrode
    flat_ptp_uV: float = 2.0  # guards against a single tiny wiggle passing
    spike_z: float = 10.0  # robust z-score of first difference; physiology rarely exceeds this
    drift_cutoff_hz: float = 0.3  # below most EEG delta activity
    drift_ptp_uV: float = 100.0  # larger than typical EEG amplitude (tens of µV)


@dataclass(frozen=True)
class ECGConfig(BaseConfig):
    signal_type: str = "ECG"
    display_unit: str = "mV"
    window_s: float = 4.0  # contains several heartbeats at normal rates
    flat_std_mV: float = 0.01  # 10 µV: well below a connected ECG lead's activity
    flat_ptp_mV: float = 0.05  # guards against a single tiny wiggle passing
    spike_factor: float = 3.0  # |d| > 3x P99(|d|): steeper than normal QRS slopes
    spike_percentile: float = 99.0  # real QRS slopes sit at or below P99
    rpeak_proximity_s: float = 0.05  # flags within 50 ms of an R-peak -> 'uncertain'
    drift_cutoff_hz: float = 0.5  # standard ECG baseline-wander band
    drift_ptp_mV: float = 0.5  # comparable to QRS amplitude (~1 mV): distorts the trace


def default_config(signal_type: str) -> BaseConfig:
    """Return the default config for 'ECG' or 'EEG'."""
    if signal_type == "EEG":
        return EEGConfig()
    if signal_type == "ECG":
        return ECGConfig()
    raise ValueError(f"Unknown signal type '{signal_type}'. Use 'ECG' or 'EEG'.")


def get_config(signal_type: str, overrides: dict[str, Any] | None = None) -> BaseConfig:
    """Default config for a modality, with optional overrides (e.g. {"mains_hz": "50"})."""
    cfg = default_config(signal_type)
    if not overrides:
        return cfg
    valid = {f.name for f in fields(cfg)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown setting(s) for {signal_type}: {sorted(unknown)}")
    if "signal_type" in overrides and overrides["signal_type"] != signal_type:
        raise ValueError("Cannot change signal_type through overrides.")
    return replace(cfg, **overrides)
