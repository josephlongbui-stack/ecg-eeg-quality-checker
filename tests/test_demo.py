"""Phase 1 Definition of Done: demo scenarios behave as specified (with default settings)."""

from __future__ import annotations

import pytest

from signal_quality.analyze import analyze_recording
from signal_quality.demo import artifact_detected, list_scenarios, make_scenario

SCENARIO_IDS = [s["id"] for s in list_scenarios()]


def test_all_design_scenarios_exist():
    expected = {
        "ecg_clean", "ecg_powerline", "ecg_baseline_wander", "ecg_lead_dropout", "ecg_clipping",
        "ecg_spikes", "eeg_clean", "eeg_flat_channel", "eeg_powerline", "eeg_spikes", "eeg_drift",
        "eeg_clipping", "mixed_ecg",
    }
    phase2 = {"ecg_muscle_noise", "ecg_electrode_motion", "ecg_missing_data", "eeg_noisy_channel", "eeg_muscle", "eeg_ocular"}
    assert set(SCENARIO_IDS) == expected | phase2


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scenario(scenario_id):
    rec, truth = make_scenario(scenario_id, seed=42)
    assert rec.is_simulated and rec.data_kind == "simulated"
    result = analyze_recording(rec)
    if not truth:
        assert result.findings == [], f"clean scenario produced findings: {result.findings}"
    for artifact in truth:
        assert artifact_detected(result.findings, artifact), (
            f"{scenario_id}: missed {artifact.category} on {artifact.channel} "
            f"at {artifact.start_s:.2f}-{artifact.end_s:.2f} s"
        )


def test_scenarios_are_deterministic():
    a, _ = make_scenario("eeg_clean", seed=42)
    b, _ = make_scenario("eeg_clean", seed=42)
    assert (a.data == b.data).all()


def test_unknown_scenario():
    with pytest.raises(KeyError):
        make_scenario("does_not_exist")


def test_every_scenario_is_labelled_simulated():
    for sid in SCENARIO_IDS:
        rec, _ = make_scenario(sid)
        assert rec.meta["label"] == "Simulated signal — not a real recording"
