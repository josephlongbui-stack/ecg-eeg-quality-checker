"""THE single analysis entry point. The app, CLI, tests and notebooks all call this."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

import numpy as np

from . import __version__
from .config import BaseConfig, get_config
from .detectors import REGISTRY
from .io import ValidationError, validate
from .merge import merge_and_suppress
from .models import AnalysisResult, Recording
from .preprocessing import iter_windows

# Metadata keys that are safe to show in reports (never patient-identifying).
SAFE_META_KEYS = ("file_name", "original_unit", "duration_s", "scenario_id", "scenario_label", "label", "seed", "warnings")


def _resolve_config(recording: Recording, config: BaseConfig | dict[str, Any] | None) -> BaseConfig:
    if config is None:
        return get_config(recording.signal_type)
    if isinstance(config, dict):
        return get_config(recording.signal_type, config)
    if config.signal_type != recording.signal_type:
        raise ValidationError(
            f"Config is for {config.signal_type} but the recording is {recording.signal_type}."
        )
    return config


def _dedupe(items: list[dict]) -> list[dict]:
    """Collapse identical skipped-check reasons across channels into one entry with a channel list."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for item in items:
        key = (item.get("check", ""), item.get("reason", item.get("note", "")))
        grouped.setdefault(key, []).append(item.get("channel", ""))
    return [{"check": c, "reason": r, "channels": chans} for (c, r), chans in grouped.items()]


def analyze_recording(
    recording: Recording,
    config: BaseConfig | dict[str, Any] | None = None,
    ml_model: Any = None,
) -> AnalysisResult:
    """Run every detector for this modality on every channel and return one AnalysisResult.

    `config` may be a config object, a dict of overrides, or None for defaults.
    The recording itself is never modified (each channel is copied before analysis).
    """
    validate(recording)
    cfg = _resolve_config(recording, config)

    raw_findings = []
    skipped: list[dict] = []
    notes: list[dict] = []
    detector_meta = {"physical_range_V": recording.meta.get("physical_range_V", {})}
    for i, channel in enumerate(recording.channel_names):
        x = np.array(recording.data[i], dtype=float, copy=True)
        for detect in REGISTRY[recording.signal_type]:
            out = detect(x, recording.sfreq, channel, cfg, detector_meta)
            raw_findings.extend(out.findings)
            skipped.extend(out.skipped)
            notes.extend(out.notes)

    if recording.duration_s < cfg.window_s:
        skipped.append(
            {
                "check": "windowing",
                "channel": "all",
                "reason": f"recording ({recording.duration_s:.1f} s) shorter than one {cfg.window_s:g} s window; analyzed as one window",
            }
        )
    if ml_model is not None:
        skipped.append({"check": "ml_anomaly", "channel": "all", "reason": "ML layer is not part of Phase 1; rule-based results only"})

    findings = merge_and_suppress(raw_findings, cfg.merge_gap_s)

    windows = iter_windows(recording.n_samples, recording.sfreq, cfg.window_s, cfg.overlap)
    n_flagged = 0
    for channel in recording.channel_names:
        ch_findings = [f for f in findings if f.channel == channel]
        for start, stop in windows:
            ws, we = start / recording.sfreq, stop / recording.sfreq
            if any(f.start_s < we and f.end_s > ws for f in ch_findings):
                n_flagged += 1

    summary = {
        "signal_type": recording.signal_type,
        "sfreq_hz": recording.sfreq,
        "duration_s": recording.duration_s,
        "n_channels": recording.n_channels,
        "channel_names": list(recording.channel_names),
        "source": recording.source,
        "data_kind": recording.data_kind,
        "is_simulated": recording.is_simulated,
        "display_unit": cfg.display_unit,
        "meta": {k: recording.meta[k] for k in SAFE_META_KEYS if k in recording.meta},
    }
    return AnalysisResult(
        recording_summary=summary,
        findings=findings,
        config=cfg.to_dict(),
        windows_analyzed=len(windows) * recording.n_channels,
        windows_flagged=n_flagged,
        skipped_checks=_dedupe(skipped),
        package_version=__version__,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=_dedupe(notes),
    )
