"""Merging and de-duplication of findings (DETECTORS.md, 'De-duplication')."""

from __future__ import annotations

from dataclasses import replace

from .detectors.common import severity
from .models import CERTAINTY_ORDER, SEVERITY_ORDER, Finding

SPIKE_FAMILY = {"spike", "discontinuity", "large_transient"}
_EPS = 1e-9


def _strength(f: Finding) -> float:
    return float(f.measurement.get("ratio_to_threshold", 0.0) or 0.0)


def _combine(group: list[Finding]) -> Finding:
    """Combine overlapping same-category findings into one, keeping the strongest evidence."""
    if len(group) == 1:
        return group[0]
    strongest = max(group, key=_strength)
    start = min(f.start_s for f in group)
    end = max(f.end_s for f in group)
    sev = max((f.severity for f in group), key=SEVERITY_ORDER.get)
    sev_from_duration = severity(_strength(strongest), end - start)
    sev = max(sev, sev_from_duration, key=SEVERITY_ORDER.get)
    cert = max((f.certainty for f in group), key=CERTAINTY_ORDER.get)
    measurement = dict(strongest.measurement)
    measurement["merged_count"] = len(group)
    measurement["merged_duration_s"] = end - start
    return replace(strongest, start_s=start, end_s=end, severity=sev, certainty=cert, measurement=measurement)


def merge_findings(findings: list[Finding], gap_s: float = 0.5) -> list[Finding]:
    """Rule 1: merge same-category findings on the same channel that overlap or are < gap_s apart."""
    groups: dict[tuple[str, str], list[Finding]] = {}
    for f in findings:
        groups.setdefault((f.channel, f.category), []).append(f)
    merged: list[Finding] = []
    for items in groups.values():
        items = sorted(items, key=lambda f: f.start_s)
        cluster = [items[0]]
        cluster_end = items[0].end_s
        for f in items[1:]:
            if f.start_s <= cluster_end + gap_s:
                cluster.append(f)
                cluster_end = max(cluster_end, f.end_s)
            else:
                merged.append(_combine(cluster))
                cluster, cluster_end = [f], f.end_s
        merged.append(_combine(cluster))
    return merged


def _inside(inner: Finding, outer: Finding) -> bool:
    return (
        inner.channel == outer.channel
        and inner.start_s >= outer.start_s - _EPS
        and inner.end_s <= outer.end_s + _EPS
    )


def suppress_findings(findings: list[Finding]) -> list[Finding]:
    """Rules 2–3: flatline hides drift/spike/clipping inside it; clipping hides spikes inside it.

    Other overlapping categories stay as separate findings (they are different evidence).
    """
    flats = [f for f in findings if f.category == "flatline"]
    clips = [f for f in findings if f.category == "clipping"]
    kept = []
    for f in findings:
        if f.category in SPIKE_FAMILY | {"baseline_drift", "clipping"} and any(_inside(f, fl) for fl in flats):
            continue
        if f.category in SPIKE_FAMILY and any(_inside(f, c) for c in clips):
            continue
        kept.append(f)
    return kept


def merge_and_suppress(findings: list[Finding], gap_s: float = 0.5) -> list[Finding]:
    """Merge first (so window-level flags become intervals), then apply suppression; sort by time."""
    result = suppress_findings(merge_findings(findings, gap_s))
    return sorted(result, key=lambda f: (f.start_s, f.channel, f.category))
