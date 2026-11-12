"""Each detector: detects its artifact (correct channel, time ±1 window) and stays quiet on clean data."""

from __future__ import annotations

import numpy as np
import pytest

from signal_quality.config import ECGConfig, EEGConfig, get_config
from signal_quality.detectors import clipping, drift, flatline, powerline, spikes

from conftest import ECG_FS, EEG_FS, time_axis

EEG = EEGConfig()
ECG = ECGConfig()


def _overlaps(findings, start, end, tol):
    return [f for f in findings if f.start_s <= end + tol and f.end_s >= start - tol]


# ---------------------------------------------------------------- input safety
@pytest.mark.parametrize("module", [flatline, clipping, powerline, spikes, drift])
def test_detectors_do_not_modify_input(module, eeg_clean_channel):
    x = eeg_clean_channel.copy()
    x[1000:1400] = x[1000]
    before = x.copy()
    module.detect(x, EEG_FS, "O1", EEG)
    assert np.array_equal(x, before)


# ---------------------------------------------------------------- flatline
def test_flatline_detects_eeg(eeg_clean_channel):
    x = eeg_clean_channel.copy()
    a, b = int(8 * EEG_FS), int(20 * EEG_FS)
    x[a:b] = x[a]
    out = flatline.detect(x, EEG_FS, "C3", EEG)
    assert len(out.findings) == 1
    f = out.findings[0]
    assert f.channel == "C3" and f.category == "flatline" and f.certainty == "definite"
    assert abs(f.start_s - 8) < EEG.window_s and abs(f.end_s - 20) < EEG.window_s
    assert f.measurement["std_uV"] < EEG.flat_std_uV


def test_flatline_detects_ecg_and_ignores_clean(ecg_clean_signal):
    assert flatline.detect(ecg_clean_signal, ECG_FS, "ECG", ECG).findings == []
    x = ecg_clean_signal.copy()
    a, b = int(12 * ECG_FS), int(15 * ECG_FS)
    x[a:b] = x[a]
    found = flatline.detect(x, ECG_FS, "ECG", ECG).findings
    assert len(_overlaps(found, 12, 15, ECG.window_s)) == 1


def test_flatline_clean_eeg(eeg_clean_channel):
    assert flatline.detect(eeg_clean_channel, EEG_FS, "O1", EEG).findings == []


def test_flatline_too_short_not_flagged(eeg_clean_channel):
    x = eeg_clean_channel.copy()
    a = int(5 * EEG_FS)
    x[a : a + int(0.5 * EEG_FS)] = x[a]  # 0.5 s < 1.0 s minimum
    assert flatline.detect(x, EEG_FS, "O1", EEG).findings == []


# ---------------------------------------------------------------- clipping
def _saturate(x, fs, a_s, b_s, gain):
    limit = 1.05 * np.max(np.abs(x))
    y = x.copy()
    a, b = int(a_s * fs), int(b_s * fs)
    y[a:b] *= gain
    return np.clip(y, -limit, limit)


def test_clipping_detects_eeg(eeg_clean_channel):
    x = _saturate(eeg_clean_channel, EEG_FS, 18, 24, 3.0)
    found = clipping.detect(x, EEG_FS, "O2", EEG).findings
    assert found and all(f.category == "clipping" for f in found)
    assert _overlaps(found, 18, 24, EEG.window_s)
    assert all(f.start_s >= 18 - EEG.window_s and f.end_s <= 24 + EEG.window_s for f in found)


def test_clipping_detects_ecg_and_clean_quiet(ecg_clean_signal):
    assert clipping.detect(ecg_clean_signal, ECG_FS, "ECG", ECG).findings == []
    x = _saturate(ecg_clean_signal, ECG_FS, 5, 9, 2.5)
    found = clipping.detect(x, ECG_FS, "ECG", ECG).findings
    assert _overlaps(found, 5, 9, ECG.window_s)


def test_clipping_clean_eeg(eeg_clean_channel):
    assert clipping.detect(eeg_clean_channel, EEG_FS, "O1", EEG).findings == []


def test_clipping_declared_physical_range_is_definite(eeg_clean_channel):
    x = _saturate(eeg_clean_channel, EEG_FS, 18, 24, 3.0)
    limit = float(np.max(x))
    meta = {"physical_range_V": {"O2": [-limit, limit]}}
    found = clipping.detect(x, EEG_FS, "O2", EEG, meta).findings
    assert any(f.certainty == "definite" for f in found)


# ---------------------------------------------------------------- power line
def test_powerline_detects_eeg(eeg_clean_channel):
    x = eeg_clean_channel.copy()
    t = time_axis(len(x), EEG_FS)
    m = (t >= 5) & (t < 15)
    x[m] += 15e-6 * np.sin(2 * np.pi * 60 * t[m])
    found = powerline.detect(x, EEG_FS, "Cz", EEG).findings
    assert found
    assert all(f.start_s >= 5 - EEG.window_s and f.end_s <= 15 + EEG.window_s for f in found)
    assert found[0].measurement["frequency_hz"] == 60


def test_powerline_clean_quiet(eeg_clean_channel, ecg_clean_signal):
    assert powerline.detect(eeg_clean_channel, EEG_FS, "O1", EEG).findings == []
    assert powerline.detect(ecg_clean_signal, ECG_FS, "ECG", ECG).findings == []


def test_powerline_50hz_setting(eeg_clean_channel):
    x = eeg_clean_channel.copy()
    t = time_axis(len(x), EEG_FS)
    x += 15e-6 * np.sin(2 * np.pi * 50 * t)
    assert powerline.detect(x, EEG_FS, "O1", EEG).findings == []  # default 60 Hz: not flagged
    cfg50 = get_config("EEG", {"mains_hz": "50"})
    assert powerline.detect(x, EEG_FS, "O1", cfg50).findings
    cfg_both = get_config("EEG", {"mains_hz": "both"})
    assert {f.measurement["frequency_hz"] for f in powerline.detect(x, EEG_FS, "O1", cfg_both).findings} == {50.0}


def test_powerline_skipped_when_nyquist_too_low():
    x = np.random.default_rng(0).normal(0, 20e-6, 100 * 10)
    out = powerline.detect(x, 100.0, "C3", EEG)  # Nyquist 50 Hz < 66 Hz
    assert out.findings == []
    assert out.skipped and "Nyquist" in out.skipped[0]["reason"]


def test_powerline_notch_note():
    from scipy import signal as sps

    fs = 256
    x = np.random.default_rng(3).normal(0, 20e-6, fs * 30)
    b, a = sps.iirnotch(60, 5, fs)
    out = powerline.detect(sps.filtfilt(b, a, x), fs, "C3", EEG)
    assert out.notes and "notch" in out.notes[0]["note"]


# ---------------------------------------------------------------- spikes
def test_eeg_step_and_pop(eeg_clean_channel):
    x = eeg_clean_channel.copy()
    t = time_axis(len(x), EEG_FS)
    x[int(12 * EEG_FS) :] += 200e-6  # step
    pop_at = 21.6
    m = t >= pop_at
    x[m] += 150e-6 * np.exp(-(t[m] - pop_at) / 0.015)  # electrode pop
    found = spikes.detect(x, EEG_FS, "F4", EEG).findings
    cats = {(f.category, round(f.start_s)) for f in found}
    assert ("discontinuity", 12) in cats
    assert any(abs(f.start_s - pop_at) < 0.1 for f in found)


def test_eeg_spikes_clean_quiet(eeg_clean_channel):
    assert spikes.detect(eeg_clean_channel, EEG_FS, "O1", EEG).findings == []


def test_ecg_clean_r_peaks_do_not_trigger_spikes(ecg_clean_signal):
    """Clean simulated ECG at 360 Hz must give zero spike findings (R-peaks are steep but normal)."""
    out = spikes.detect(ecg_clean_signal, ECG_FS, "ECG", ECG)
    assert out.findings == []


def test_ecg_single_sample_spikes_detected(ecg_clean_signal):
    x = ecg_clean_signal.copy()
    times = [6.2, 11.7, 18.4, 25.1]
    for t in times:
        x[int(round(t * ECG_FS))] += 3e-3
    found = spikes.detect(x, ECG_FS, "ECG", ECG).findings
    assert len(found) == 4
    for t in times:
        assert any(abs(f.start_s - t) < 0.05 and f.category == "spike" for f in found)
    for f in found:  # certainty follows the rule, never a probability
        assert f.certainty in {"definite", "likely", "uncertain"}
        if f.measurement["near_r_peak"]:
            assert f.certainty == "uncertain"


def test_flat_stretch_does_not_create_false_spikes(eeg_clean_channel):
    x = eeg_clean_channel.copy()
    a, b = int(8 * EEG_FS), int(20 * EEG_FS)
    x[a:b] = x[a]
    assert spikes.detect(x, EEG_FS, "C3", EEG).findings == []


# ---------------------------------------------------------------- drift
def test_drift_detects_eeg_and_ecg(eeg_clean_channel, ecg_clean_signal):
    x = eeg_clean_channel + 250e-6 * np.sin(2 * np.pi * 0.1 * time_axis(len(eeg_clean_channel), EEG_FS))
    found = drift.detect(x, EEG_FS, "Fp1", EEG).findings
    assert found and all(f.category == "baseline_drift" for f in found)
    y = ecg_clean_signal + 1e-3 * np.sin(2 * np.pi * 0.25 * time_axis(len(ecg_clean_signal), ECG_FS))
    assert drift.detect(y, ECG_FS, "ECG", ECG).findings


def test_drift_clean_quiet(eeg_clean_channel, ecg_clean_signal):
    assert drift.detect(eeg_clean_channel, EEG_FS, "O1", EEG).findings == []
    assert drift.detect(ecg_clean_signal, ECG_FS, "ECG", ECG).findings == []


def test_drift_skipped_on_short_recording(eeg_clean_channel):
    out = drift.detect(eeg_clean_channel[: int(5 * EEG_FS)], EEG_FS, "O1", EEG)  # needs 10 s at 0.3 Hz
    assert out.findings == [] and out.skipped


def test_drift_edge_findings_are_uncertain(eeg_clean_channel):
    x = eeg_clean_channel + 250e-6 * np.sin(2 * np.pi * 0.1 * time_axis(len(eeg_clean_channel), EEG_FS))
    for f in drift.detect(x, EEG_FS, "Fp1", EEG).findings:
        if f.start_s < 1.0:
            assert f.certainty == "uncertain"


# ---------------------------------------------------------------- config
def test_config_units_and_overrides():
    assert np.isclose(EEG.volts("flat_std"), 0.5e-6)
    assert np.isclose(ECG.volts("drift_ptp"), 0.5e-3)
    cfg = get_config("ECG", {"spike_factor": 4.0})
    assert cfg.spike_factor == 4.0
    with pytest.raises(ValueError):
        get_config("ECG", {"not_a_setting": 1})
