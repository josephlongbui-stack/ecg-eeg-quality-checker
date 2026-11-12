"""analyze_recording() output structure and CSV/JSON exports."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from signal_quality import __version__
from signal_quality.analyze import analyze_recording
from signal_quality.demo import make_scenario
from signal_quality.io import ValidationError
from signal_quality.models import Recording
from signal_quality.reporting import TABLE_COLUMNS, to_csv_bytes, to_dataframe, to_json_bytes


@pytest.fixture(scope="module")
def mixed_result():
    rec, _ = make_scenario("mixed_ecg")
    return rec, analyze_recording(rec)


def test_result_has_config_version_and_skipped(mixed_result):
    rec, res = mixed_result
    assert res.package_version == __version__
    assert res.config["signal_type"] == "ECG"
    assert res.config["flat_std_mV"] == 0.01
    assert isinstance(res.skipped_checks, list)
    assert res.windows_analyzed > 0 and 0 < res.windows_flagged <= res.windows_analyzed
    assert res.created_at
    assert res.recording_summary["is_simulated"] is True
    assert res.recording_summary["data_kind"] == "simulated"


def test_analysis_does_not_modify_recording(mixed_result):
    rec, _ = mixed_result
    before = rec.data.copy()
    analyze_recording(rec)
    assert np.array_equal(rec.data, before)


def test_skipped_checks_reported():
    rng = np.random.default_rng(0)
    rec = Recording("EEG", rng.normal(0, 20e-6, (1, 100 * 5)), 100.0, ["C3"], "test", "real")
    res = analyze_recording(rec)
    checks = {s["check"] for s in res.skipped_checks}
    assert "powerline_60Hz" in checks  # Nyquist 50 Hz
    assert "baseline_drift" in checks  # 5 s < 10 s


def test_config_overrides_accepted_and_embedded(mixed_result):
    rec, _ = mixed_result
    res = analyze_recording(rec, config={"mains_hz": "50"})
    assert res.config["mains_hz"] == "50"
    assert not any(f.category == "powerline" for f in res.findings)  # the injected hum is 60 Hz


def test_wrong_modality_config_rejected(mixed_result):
    from signal_quality.config import EEGConfig

    rec, _ = mixed_result
    with pytest.raises(ValidationError):
        analyze_recording(rec, config=EEGConfig())


def test_json_export_structure(mixed_result):
    _, res = mixed_result
    data = json.loads(to_json_bytes(res).decode())
    for key in ("package_version", "created_at", "recording", "config", "findings", "skipped_checks"):
        assert key in data
    assert data["package_version"] == __version__
    assert data["recording"]["data_kind"] == "simulated"
    assert data["findings"]
    f0 = data["findings"][0]
    for key in ("channel", "start_s", "end_s", "category", "severity", "certainty", "rule", "measurement", "explanation"):
        assert key in f0
    assert f0["certainty"] in {"definite", "likely", "uncertain"}


def test_csv_export_structure(mixed_result):
    _, res = mixed_result
    text = to_csv_bytes(res).decode()
    assert f"# package_version: {__version__}" in text
    assert "# config: " in text
    assert "SIMULATED" in text
    df = pd.read_csv(io.StringIO(text), comment="#")
    assert list(df.columns) == TABLE_COLUMNS
    assert len(df) == len(res.findings)


def test_dataframe_empty_has_columns():
    rec, _ = make_scenario("ecg_clean")
    df = to_dataframe(analyze_recording(rec))
    assert df.empty and list(df.columns) == TABLE_COLUMNS
