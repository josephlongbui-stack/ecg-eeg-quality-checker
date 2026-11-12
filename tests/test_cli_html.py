"""CLI (`sqc`) and HTML report. The CLI must produce exactly what analyze_recording() produces."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from signal_quality.analyze import analyze_recording
from signal_quality.cli import main
from signal_quality.demo import make_scenario
from signal_quality.io import load_csv
from signal_quality.reporting import to_dict, to_html


def _comparable(d: dict) -> dict:
    d = dict(d)
    d.pop("created_at", None)
    return d


@pytest.fixture()
def ecg_csv(tmp_path) -> Path:
    rec, _ = make_scenario("mixed_ecg")
    t = np.arange(rec.n_samples) / rec.sfreq
    path = tmp_path / "patient_like.csv"
    lines = ["time,ECG"] + [f"{a:.6f},{v * 1e3:.6f}" for a, v in zip(t, rec.data[0])]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_cli_output_equals_engine_output(ecg_csv, tmp_path):
    out = tmp_path / "out"
    code = main(["analyze", str(ecg_csv), "--type", "ECG", "--sfreq", "360", "--unit", "mV", "--out", str(out), "--quiet"])
    assert code == 0
    cli_json = json.loads((out / "patient_like_ECG_results.json").read_text())
    engine = to_dict(analyze_recording(load_csv(ecg_csv, sfreq=360, unit="mV", signal_type="ECG")))
    assert _comparable(cli_json) == _comparable(engine)
    assert (out / "patient_like_ECG_findings.csv").exists()
    assert (out / "patient_like_ECG_report.html").exists()


def test_cli_overrides_match_engine_overrides(ecg_csv, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("spike_factor: 4.0\n")
    out = tmp_path / "out"
    main(["analyze", str(ecg_csv), "--type", "ECG", "--sfreq", "360", "--unit", "mV", "--config", str(cfg),
          "--set", "line_ratio_threshold=20", "--mains", "both", "--out", str(out), "--quiet", "--no-html"])
    cli_json = json.loads((out / "patient_like_ECG_results.json").read_text())
    overrides = {"spike_factor": 4.0, "line_ratio_threshold": 20, "mains_hz": "both"}
    engine = to_dict(analyze_recording(load_csv(ecg_csv, sfreq=360, unit="mV", signal_type="ECG"), config=overrides))
    assert _comparable(cli_json) == _comparable(engine)
    assert cli_json["config"]["spike_factor"] == 4.0 and cli_json["config"]["mains_hz"] == "both"
    assert not (out / "patient_like_ECG_report.html").exists()


def test_cli_friendly_errors(ecg_csv, capsys):
    assert main(["analyze", str(ecg_csv), "--type", "ECG", "--unit", "mV"]) == 2
    assert "Sampling rate is required" in capsys.readouterr().err
    assert main(["analyze", "does_not_exist.csv", "--type", "ECG"]) == 2
    assert main(["analyze", str(ecg_csv), "--type", "ECG", "--sfreq", "360", "--unit", "mV", "--set", "bogus=1"]) == 2
    assert "Unknown setting" in capsys.readouterr().err
    assert main(["demo", "no_such_scenario"]) == 2


def test_cli_demo_list(capsys):
    assert main(["demo", "--list"]) == 0
    text = capsys.readouterr().out
    assert "ecg_clean" in text and "simulated" in text


def test_cli_entry_point_installed():
    """`sqc` is installed as a console script ([project.scripts])."""
    exe = Path(sys.executable).parent / "sqc"
    proc = subprocess.run([str(exe), "demo", "--list"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "eeg_clean" in proc.stdout


def test_html_report_contents():
    rec, truth = make_scenario("mixed_ecg")
    res = analyze_recording(rec)
    html = to_html(res, rec, truth)
    assert html.startswith("<!doctype html>")
    assert "Simulated signal — not a real recording" in html
    assert "not probabilities" in html
    assert res.package_version in html
    assert "spike_factor" in html  # settings included
    assert html.count("<tr>") >= len(res.findings) + 1
    assert "plotly" in html.lower()


def test_html_escapes_text_and_handles_no_findings():
    rec, _ = make_scenario("ecg_clean")
    res = analyze_recording(rec)
    res.recording_summary["meta"]["scenario_label"] = "<script>alert(1)</script>"
    html = to_html(res, include_plot=False)
    assert "<script>alert(1)</script>" not in html
    assert "No quality problems found" in html


def test_cli_simulated_flag_labels_result(ecg_csv, tmp_path):
    out = tmp_path / "o"
    main(["analyze", str(ecg_csv), "--type", "ECG", "--sfreq", "360", "--unit", "mV", "--simulated", "--out", str(out), "--quiet", "--no-html"])
    data = json.loads((out / "patient_like_ECG_results.json").read_text())
    assert data["recording"]["data_kind"] == "simulated" and data["recording"]["is_simulated"] is True
    main(["analyze", str(ecg_csv), "--type", "ECG", "--sfreq", "360", "--unit", "mV", "--out", str(out), "--quiet", "--no-html"])
    data = json.loads((out / "patient_like_ECG_results.json").read_text())
    assert data["recording"]["data_kind"] == "real"


def test_example_files_load():
    root = Path(__file__).resolve().parents[1] / "examples"
    from signal_quality.io import load_file

    assert load_file(root / "simulated_ecg_mixed_360Hz_mV.csv", "ECG", 360, "mV").n_samples == 10800
    assert load_file(root / "simulated_eeg_flat_channel.edf", "EEG").n_channels == 8
    assert load_file(root / "simulated_ecg_motion.hea", "ECG").channel_names == ["MLII", "V1"]
