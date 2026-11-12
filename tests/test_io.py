"""CSV and EDF loading, unit conversion and friendly validation errors."""

from __future__ import annotations

import datetime as dt
import json

import mne
import numpy as np
import pytest

from signal_quality.analyze import analyze_recording
from signal_quality.io import ValidationError, load_csv, load_edf, load_file, suggest_signal_type, validate
from signal_quality.models import Recording


def _csv_bytes(header: str, rows: list[str]) -> bytes:
    return ("\n".join([header] + rows) + "\n").encode()


def _sine_rows(n: int, fs: float, amp: float, with_time: bool = False) -> list[str]:
    t = np.arange(n) / fs
    vals = amp * np.sin(2 * np.pi * 10 * t)
    if with_time:
        return [f"{ti:.6f},{v:.6f},{-v:.6f}" for ti, v in zip(t, vals)]
    return [f"{v:.6f},{-v:.6f}" for v in vals]


@pytest.mark.parametrize("unit,scale", [("V", 1.0), ("mV", 1e-3), ("uV", 1e-6), ("µV", 1e-6)])
def test_csv_units_converted_to_volts(unit, scale):
    content = _csv_bytes("C3,C4", _sine_rows(512, 256, 50.0))
    rec = load_csv(content, sfreq=256, unit=unit, signal_type="EEG")
    assert rec.channel_names == ["C3", "C4"]
    assert rec.data.shape == (2, 512)
    assert np.isclose(np.max(rec.data[0]), 50.0 * scale, rtol=1e-3)
    assert rec.source == "upload:csv" and rec.data_kind == "real"


def test_csv_time_column_ignored_and_checked():
    content = _csv_bytes("time,C3,C4", _sine_rows(512, 256, 50.0, with_time=True))
    rec = load_csv(content, sfreq=256, unit="uV", signal_type="EEG")
    assert rec.channel_names == ["C3", "C4"]
    assert rec.meta["warnings"] == []
    rec_bad = load_csv(content, sfreq=300, unit="uV", signal_type="EEG")
    assert rec_bad.meta["warnings"], "inconsistent time column should produce a warning"


def test_csv_missing_sfreq_is_friendly_error():
    content = _csv_bytes("C3", [str(v) for v in range(600)])
    with pytest.raises(ValidationError, match="Sampling rate is required"):
        load_csv(content, sfreq=None, unit="uV", signal_type="EEG")
    with pytest.raises(ValidationError, match="positive"):
        load_csv(content, sfreq=-5, unit="uV", signal_type="EEG")


def test_csv_nan_rejected_with_location():
    rows = [str(v) for v in range(600)]
    rows[300] = "nan"
    with pytest.raises(ValidationError, match="NaN/Inf"):
        load_csv(_csv_bytes("C3", rows), sfreq=256, unit="uV", signal_type="EEG")


def test_csv_unknown_or_missing_unit_rejected():
    content = _csv_bytes("C3", [str(v) for v in range(600)])
    with pytest.raises(ValidationError, match="Unknown unit"):
        load_csv(content, sfreq=256, unit="furlongs", signal_type="EEG")
    with pytest.raises(ValidationError, match="Unit is required"):
        load_csv(content, sfreq=256, unit=None, signal_type="EEG")


def test_csv_too_short_rejected():
    content = _csv_bytes("C3", [str(v) for v in range(100)])
    with pytest.raises(ValidationError, match="at least 1 s"):
        load_csv(content, sfreq=256, unit="uV", signal_type="EEG")


def test_csv_other_bad_inputs():
    with pytest.raises(ValidationError, match="empty"):
        load_csv(b"", sfreq=256, unit="uV", signal_type="EEG")
    with pytest.raises(ValidationError, match="Duplicate"):
        load_csv(_csv_bytes("C3,C3", ["1,2"] * 600), sfreq=256, unit="uV", signal_type="EEG")
    with pytest.raises(ValidationError, match="Non-numeric"):
        load_csv(_csv_bytes("C3", ["1"] * 300 + ["abc"] + ["1"] * 300), sfreq=256, unit="uV", signal_type="EEG")
    with pytest.raises(ValidationError, match="Signal type"):
        load_csv(_csv_bytes("C3", ["1"] * 600), sfreq=256, unit="uV", signal_type=None)
    with pytest.raises(ValidationError, match="Unsupported file type"):
        load_file("recording.txt", signal_type="EEG")


def test_validate_channel_name_mismatch():
    rec = Recording("EEG", np.zeros((2, 512)), 256, ["A"], "test", "real")
    with pytest.raises(ValidationError, match="channel names"):
        validate(rec)


def test_recording_is_read_only_copy():
    original = np.ones((1, 512))
    rec = Recording("EEG", original, 256, ["A"], "test", "real")
    original[0, 0] = 99
    assert rec.data[0, 0] == 1
    with pytest.raises(ValueError):
        rec.data[0, 0] = 5


@pytest.fixture()
def tiny_edf(tmp_path):
    """Write a 3 s, 2-channel EDF with fake patient info into a temp dir."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 20e-6, (2, 256 * 3))
    info = mne.create_info(["Fp1", "C3"], 256.0, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc))
    raw.info["subject_info"] = {
        "his_id": "PATIENT_SECRET_ID",
        "first_name": "Jane",
        "last_name": "Doe",
        "birthday": dt.date(1990, 1, 2),
    }
    path = tmp_path / "tiny.edf"
    mne.export.export_raw(str(path), raw, fmt="edf", verbose="error")
    return path, data


def test_edf_loads_in_volts_without_patient_info(tiny_edf):
    path, data = tiny_edf
    rec = load_edf(path, signal_type="EEG")
    assert rec.channel_names == ["Fp1", "C3"]
    assert rec.sfreq == 256.0
    assert rec.source == "upload:edf"
    assert np.allclose(rec.data, data, atol=1e-8)  # volts (EDF 16-bit quantization)
    blob = json.dumps(rec.meta, default=str)
    for secret in ("PATIENT_SECRET_ID", "Jane", "Doe", "1990"):
        assert secret not in blob
    result = analyze_recording(rec)
    result_blob = json.dumps(result.recording_summary, default=str)
    for secret in ("PATIENT_SECRET_ID", "Jane", "Doe", "1990"):
        assert secret not in result_blob


def test_edf_garbage_is_friendly_error(tmp_path):
    bad = tmp_path / "bad.edf"
    bad.write_bytes(b"this is not an EDF file at all" * 10)
    with pytest.raises(ValidationError):
        load_edf(bad, signal_type="EEG")


def test_suggest_signal_type():
    assert suggest_signal_type(["ECG II"]) == "ECG"
    assert suggest_signal_type(["EKG"]) == "ECG"
    assert suggest_signal_type(["Fp1", "Cz"]) == "EEG"
