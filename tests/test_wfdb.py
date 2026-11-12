"""WFDB (PhysioNet) loading."""

from __future__ import annotations

import json

import numpy as np
import pytest
import wfdb

from signal_quality.analyze import analyze_recording
from signal_quality.demo import _ecg_base
from signal_quality.io import ValidationError, load_file, load_wfdb


@pytest.fixture()
def wfdb_record(tmp_path):
    ecg_mV = _ecg_base(seed=42)[0] * 1e3
    sig = np.column_stack([ecg_mV, ecg_mV * 0.5])
    wfdb.wrsamp(
        "rec1", fs=360, units=["mV", "mV"], sig_name=["MLII", "V1"], p_signal=sig, fmt=["16", "16"],
        comments=["age: 45 sex: F", "patient secret note"], write_dir=str(tmp_path),
    )
    return tmp_path, ecg_mV


def test_wfdb_loads_from_path_in_volts(wfdb_record):
    folder, ecg_mV = wfdb_record
    rec = load_wfdb(folder / "rec1.hea", signal_type="ECG")
    assert rec.channel_names == ["MLII", "V1"]
    assert rec.sfreq == 360.0 and rec.source == "upload:wfdb"
    assert np.allclose(rec.data[0], ecg_mV * 1e-3, atol=2e-6)
    assert "secret" not in json.dumps(rec.meta) and "age" not in json.dumps(rec.meta)
    res = analyze_recording(rec)
    assert res.recording_summary["n_channels"] == 2


def test_wfdb_loads_from_uploaded_pairs(wfdb_record):
    folder, _ = wfdb_record
    files = [(p.name, p.read_bytes()) for p in folder.iterdir()]
    rec = load_file(files, signal_type="ECG")
    assert rec.channel_names == ["MLII", "V1"]


def test_wfdb_missing_dat_is_friendly_error(wfdb_record):
    folder, _ = wfdb_record
    with pytest.raises(ValidationError, match="missing|Upload"):
        load_wfdb([("rec1.hea", (folder / "rec1.hea").read_bytes())], signal_type="ECG")


def test_wfdb_needs_exactly_one_header(wfdb_record):
    folder, _ = wfdb_record
    with pytest.raises(ValidationError, match="exactly one"):
        load_wfdb([("rec1.dat", (folder / "rec1.dat").read_bytes())], signal_type="ECG")


def test_wfdb_unknown_unit_rejected(tmp_path):
    wfdb.wrsamp("rec2", fs=100, units=["mmHg"], sig_name=["BP"], p_signal=np.ones((300, 1)), fmt=["16"], write_dir=str(tmp_path))
    with pytest.raises(ValidationError, match="unit"):
        load_wfdb(tmp_path / "rec2.hea", signal_type="ECG")
