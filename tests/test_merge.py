"""Merging and suppression rules."""

from __future__ import annotations

from signal_quality.merge import merge_and_suppress, merge_findings, suppress_findings
from signal_quality.models import Finding


def _f(cat, start, end, ch="C3", sev="low", cert="likely", ratio=1.0):
    return Finding("EEG", ch, start, end, cat, sev, cert, "rule", {"ratio_to_threshold": ratio}, "x", "test@1")


def test_adjacent_same_category_merged_keeps_max():
    merged = merge_findings([_f("powerline", 0, 2, sev="low", ratio=1.5), _f("powerline", 2.3, 4, sev="high", ratio=8)])
    assert len(merged) == 1
    m = merged[0]
    assert (m.start_s, m.end_s) == (0, 4)
    assert m.severity == "high"
    assert m.measurement["ratio_to_threshold"] == 8
    assert m.measurement["merged_count"] == 2


def test_far_apart_or_other_channel_not_merged():
    assert len(merge_findings([_f("powerline", 0, 2), _f("powerline", 3, 4)])) == 2
    assert len(merge_findings([_f("powerline", 0, 2, ch="C3"), _f("powerline", 1, 3, ch="C4")])) == 2


def test_merged_duration_raises_severity():
    merged = merge_findings([_f("baseline_drift", s, s + 2, ratio=1.2) for s in range(0, 12)])
    assert merged[0].severity == "high"  # ≥ 10 s


def test_flatline_suppresses_inside_drift_spike_clipping():
    findings = [
        _f("flatline", 5, 15),
        _f("baseline_drift", 6, 8),
        _f("spike", 10, 10.01),
        _f("clipping", 11, 13),
        _f("powerline", 6, 8),  # different evidence: kept
        _f("spike", 4, 4.01),  # outside: kept
        _f("spike", 10, 10.01, ch="C4"),  # other channel: kept
    ]
    kept = {(f.category, f.channel, f.start_s) for f in suppress_findings(findings)}
    assert kept == {("flatline", "C3", 5), ("powerline", "C3", 6), ("spike", "C3", 4), ("spike", "C4", 10)}


def test_clipping_suppresses_spikes_inside():
    kept = suppress_findings([_f("clipping", 0, 4), _f("discontinuity", 1, 1.01), _f("baseline_drift", 1, 3)])
    assert {f.category for f in kept} == {"clipping", "baseline_drift"}


def test_merge_and_suppress_sorted():
    out = merge_and_suppress([_f("spike", 9, 9.01), _f("powerline", 1, 2)])
    assert [f.start_s for f in out] == [1, 9]
