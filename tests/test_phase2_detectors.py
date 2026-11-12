"""Phase 2 detectors: each detects its artifact on a fixture and stays quiet on clean data."""

from __future__ import annotations

import numpy as np
import pytest

from signal_quality.analyze import analyze_recording
from signal_quality.config import ECGConfig, EEGConfig, get_config
from signal_quality.demo import EEG_CHANNELS, _band_noise, _blink, _ecg_base, _eeg_base, make_scenario
from signal_quality.detectors import electrode_motion, hf_noise, muscle, noisy, ocular, plausibility
from signal_quality.io import ValidationError
from signal_quality.models import Finding, Recording

from conftest import ECG_FS, EEG_FS, time_axis

EEG = EEGConfig()
ECG = ECGConfig()


@pytest.fixture(scope="module")
def eeg8() -> np.ndarray:
    return _eeg_base(seed=7)


# ---------------------------------------------------------------- noisy channel
def test_noisy_channel_detected_and_clean_quiet(eeg8):
    assert noisy.detect(eeg8, EEG_FS, EEG_CHANNELS, EEG).findings == []
    data = eeg8.copy()
    data[5] += _band_noise(data.shape[1], EEG_FS, 1, 40, 150e-6, np.random.default_rng(0))
    found = noisy.detect(data, EEG_FS, EEG_CHANNELS, EEG).findings
    assert found and {f.channel for f in found} == {"C4"}


def test_noisy_channel_ignores_slow_drift(eeg8):
    data = eeg8.copy()
    data[0] += 250e-6 * np.sin(2 * np.pi * 0.1 * time_axis(data.shape[1], EEG_FS))
    assert noisy.detect(data, EEG_FS, EEG_CHANNELS, EEG).findings == []


def test_noisy_channel_needs_four_channels(eeg8):
    out = noisy.detect(eeg8[:3], EEG_FS, EEG_CHANNELS[:3], EEG)
    assert out.findings == [] and out.skipped


# ---------------------------------------------------------------- EEG muscle
def test_muscle_detected_and_clean_quiet(eeg_clean_channel):
    assert muscle.detect(eeg_clean_channel, EEG_FS, "F3", EEG).findings == []
    x = eeg_clean_channel.copy()
    a, b = int(10 * EEG_FS), int(20 * EEG_FS)
    x[a:b] += _band_noise(b - a, EEG_FS, 20, 100, 45e-6, np.random.default_rng(1))
    found = muscle.detect(x, EEG_FS, "F3", EEG).findings
    assert found and all(9 - EEG.window_s <= f.start_s and f.end_s <= 21 + EEG.window_s for f in found)


def test_muscle_skipped_at_low_sampling_rate():
    x = np.random.default_rng(0).normal(0, 20e-6, 50 * 20)
    assert muscle.detect(x, 50.0, "C3", EEG).skipped


# ---------------------------------------------------------------- ECG high-frequency noise
def test_hf_noise_detected_and_clean_quiet(ecg_clean_signal):
    assert hf_noise.detect(ecg_clean_signal, ECG_FS, "ECG", ECG).findings == []
    x = ecg_clean_signal.copy()
    a, b = int(8 * ECG_FS), int(16 * ECG_FS)
    x[a:b] += _band_noise(b - a, ECG_FS, 20, 150, 0.2e-3, np.random.default_rng(2))
    found = hf_noise.detect(x, ECG_FS, "ECG", ECG).findings
    assert found and all(f.category == "hf_noise" for f in found)


def test_hf_noise_ignores_pure_powerline(ecg_clean_signal):
    """60 Hz hum is the power-line detector's job, not broadband high-frequency noise."""
    x = ecg_clean_signal + 0.5e-3 * np.sin(2 * np.pi * 60 * time_axis(len(ecg_clean_signal), ECG_FS))
    assert hf_noise.detect(x, ECG_FS, "ECG", ECG).findings == []


# ---------------------------------------------------------------- ocular
def test_ocular_blinks_detected_likely(eeg8):
    data = eeg8.copy()
    for t0 in (5.0, 15.0):
        for i, scale in ((0, 1.0), (1, 1.0), (2, 0.3), (3, 0.3)):
            _blink(data[i], EEG_FS, t0, 200e-6 * scale)
    found = ocular.detect(data, EEG_FS, EEG_CHANNELS, EEG).findings
    assert {f.channel for f in found} == {"Fp1", "Fp2"}
    assert all(f.certainty in ("likely", "uncertain") for f in found)  # never 'definite'
    assert any(f.certainty == "likely" for f in found)


def test_ocular_quiet_on_clean_and_on_slow_frontal_drift(eeg8):
    assert ocular.detect(eeg8, EEG_FS, EEG_CHANNELS, EEG).findings == []
    data = eeg8.copy()
    t = time_axis(data.shape[1], EEG_FS)
    data[0] += 250e-6 * np.sin(2 * np.pi * 0.1 * t)
    data[1] += 250e-6 * np.sin(2 * np.pi * 0.1 * t)
    assert ocular.detect(data, EEG_FS, EEG_CHANNELS, EEG).findings == []


def test_ocular_skipped_without_frontal_channels(eeg8):
    out = ocular.detect(eeg8[4:], EEG_FS, EEG_CHANNELS[4:], EEG)
    assert out.findings == [] and "frontal" in out.skipped[0]["reason"]


# ---------------------------------------------------------------- electrode motion (pure function)
def _f(cat, a, b):
    return Finding("ECG", "ECG", a, b, cat, "low", "likely", "r", {"ratio_to_threshold": 1}, "x", "t@1")


def test_electrode_motion_needs_two_indicators():
    x = np.zeros(int(30 * ECG_FS))
    beats = np.arange(0, len(x), int(0.85 * ECG_FS))
    drift_only = electrode_motion.detect(x, ECG_FS, "ECG", ECG, [_f("baseline_drift", 8, 12)], beats)
    assert drift_only.findings == []
    both = electrode_motion.detect(x, ECG_FS, "ECG", ECG, [_f("baseline_drift", 8, 12), _f("large_transient", 9, 9.01)], beats)
    assert both.findings and all(f.certainty == "likely" for f in both.findings)


def test_electrode_motion_beat_failure_alone_is_uncertain():
    x = np.zeros(int(30 * ECG_FS))
    beats = np.arange(0, len(x), int(0.85 * ECG_FS))
    beats = beats[(beats < 10 * ECG_FS) | (beats > 16 * ECG_FS)]  # no beats 10–16 s
    out = electrode_motion.detect(x, ECG_FS, "ECG", ECG, [], beats)
    assert out.findings and all(f.certainty == "uncertain" for f in out.findings)


def test_electrode_motion_demo():
    rec, truth = make_scenario("ecg_electrode_motion")
    res = analyze_recording(rec)
    assert any(f.category == "electrode_motion" and f.certainty == "likely" for f in res.findings)


# ---------------------------------------------------------------- missing data
def _with_nan(policy):
    x = _eeg_base(seed=3)
    x[2, int(10 * EEG_FS) : int(12 * EEG_FS)] = np.nan
    return Recording("EEG", x, EEG_FS, EEG_CHANNELS, "test", "real", meta={"missing_policy": policy})


def test_missing_data_flagged_when_allowed():
    res = analyze_recording(_with_nan("flag"))
    gaps = [f for f in res.findings if f.category == "missing_data"]
    assert len(gaps) == 1 and gaps[0].channel == "F3" and gaps[0].certainty == "definite"
    assert abs(gaps[0].start_s - 10) < 0.01 and abs(gaps[0].end_s - 12) < 0.01
    # nothing else is reported from inside the bridged gap
    assert not [f for f in res.findings if f.channel == "F3" and f.category != "missing_data" and 9.5 <= f.start_s and f.end_s <= 12.5]


def test_missing_data_rejected_by_default():
    with pytest.raises(ValidationError, match="NaN/Inf"):
        analyze_recording(_with_nan("reject"))


def test_all_missing_channel_is_reported_not_crashing():
    x = _eeg_base(seed=3)
    x[4] = np.nan
    rec = Recording("EEG", x, EEG_FS, EEG_CHANNELS, "test", "real", meta={"missing_policy": "flag"})
    res = analyze_recording(rec)
    assert any(f.channel == "C3" and f.category == "missing_data" and f.end_s - f.start_s > 29 for f in res.findings)


# ---------------------------------------------------------------- physiological plausibility
FORBIDDEN = ("arrhythmia", "tachycardia", "bradycardia", "diagnos", "abnormal", "fake")


@pytest.mark.parametrize("hr", [60, 80, 100])
def test_clean_ecg_60_to_100_bpm_gives_no_hint(hr):
    import neurokit2 as nk

    ecg = nk.ecg_simulate(duration=30, sampling_rate=360, heart_rate=hr, noise=0.01, random_state=11) * 1e-3
    rec = Recording("ECG", ecg, ECG_FS, ["ECG"], "test", "simulated")
    res = analyze_recording(rec)
    assert not [f for f in res.findings if f.category.startswith("implausible")]


def test_rhythm_hints_from_beat_timing():
    x = _ecg_base(seed=42)[0]
    fs = ECG_FS
    normal = np.arange(int(0.4 * fs), len(x), int(0.86 * fs))
    fast = np.concatenate([normal[normal < 10 * fs], np.arange(int(10 * fs), int(14 * fs), int(0.15 * fs)), normal[normal > 14 * fs]])
    out = plausibility.detect_ecg_rhythm(x, fs, "ECG", ECG, {"r_peaks": fast})
    assert out.findings
    for f in out.findings:
        assert f.category == "implausible_rhythm_measurement"
        assert f.certainty == "uncertain" and f.severity in ("low", "medium")
        assert not any(word in f.explanation.lower() for word in FORBIDDEN)
        assert "unusual rhythms are possible" in f.explanation


def test_rhythm_gap_hint_and_no_hint_for_flat_dropout():
    x = _ecg_base(seed=42)[0].copy()
    fs = ECG_FS
    beats = np.arange(int(0.4 * fs), len(x), int(0.86 * fs))
    missing_beats = beats[(beats < 10 * fs) | (beats > 15 * fs)]  # 5 s without detected beats
    out = plausibility.detect_ecg_rhythm(x, fs, "ECG", ECG, {"r_peaks": missing_beats})
    assert any("No heartbeat was detected" in f.explanation for f in out.findings)
    x[int(10 * fs) : int(15 * fs)] = x[int(10 * fs)]  # same gap, but the lead is flat (dropout)
    out = plausibility.detect_ecg_rhythm(x, fs, "ECG", ECG, {"r_peaks": missing_beats})
    assert not any("No heartbeat was detected" in f.explanation for f in out.findings)


def test_rhythm_check_skipped_when_beat_detection_fails():
    out = plausibility.detect_ecg_rhythm(np.zeros(3600), ECG_FS, "ECG", ECG, {"r_peaks": None})
    assert out.findings == [] and out.skipped


def test_eeg_amplitude_hint(eeg_clean_channel):
    assert plausibility.detect_eeg_amplitude(eeg_clean_channel, EEG_FS, "O1", EEG).findings == []
    x = eeg_clean_channel.copy()
    a, b = int(10 * EEG_FS), int(15 * EEG_FS)
    x[a:b] += _band_noise(b - a, EEG_FS, 1, 30, 400e-6, np.random.default_rng(4))
    found = plausibility.detect_eeg_amplitude(x, EEG_FS, "O1", EEG).findings
    assert found and all(f.certainty == "uncertain" and f.severity != "high" for f in found)
    assert not any(word in found[0].explanation.lower() for word in FORBIDDEN)


def test_no_diagnostic_words_in_any_demo_finding():
    from signal_quality.demo import list_scenarios

    for s in list_scenarios():
        rec, _ = make_scenario(s["id"])
        for f in analyze_recording(rec).findings:
            text = (f.explanation + " " + f.rule).lower()
            assert not any(word in text for word in FORBIDDEN), (s["id"], f.explanation)
            if f.category.startswith("implausible"):
                assert f.certainty == "uncertain" and f.severity != "high"


# ---------------------------------------------------------------- large recordings
def test_large_recording_note():
    rec, _ = make_scenario("ecg_clean")
    res = analyze_recording(rec, config={"large_warn_duration_min": 0.1})  # 30 s > 6 s at 360 Hz > 250 Hz
    assert any(n["check"] == "large_recording" for n in res.notes)
    assert not any(n["check"] == "large_recording" for n in analyze_recording(rec).notes)
