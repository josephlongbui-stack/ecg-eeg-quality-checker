"""Turn an AnalysisResult into a table, CSV bytes or JSON bytes."""

from __future__ import annotations

import io
import json
import math
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .models import AnalysisResult

TABLE_COLUMNS = [
    "Channel",
    "Start (s)",
    "End (s)",
    "Category",
    "Severity",
    "Certainty",
    "Rule",
    "Measurement",
    "Explanation",
    "Detector",
]


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and non-finite floats into JSON-friendly values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def _format_measurement(m: dict) -> str:
    parts = []
    for key, value in m.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def to_dataframe(result: AnalysisResult) -> pd.DataFrame:
    """Findings as a table (one row per finding). Measurements are in display units."""
    rows = [
        {
            "Channel": f.channel,
            "Start (s)": round(f.start_s, 3),
            "End (s)": round(f.end_s, 3),
            "Category": f.category,
            "Severity": f.severity,
            "Certainty": f.certainty,
            "Rule": f.rule,
            "Measurement": _format_measurement(f.measurement),
            "Explanation": f.explanation,
            "Detector": f.detector,
        }
        for f in result.findings
    ]
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def to_dict(result: AnalysisResult) -> dict:
    """Full result as a plain, JSON-safe dictionary."""
    return _json_safe(
        {
            "package": "signal_quality",
            "package_version": result.package_version,
            "created_at": result.created_at,
            "recording": result.recording_summary,
            "config": result.config,
            "windows_analyzed": result.windows_analyzed,
            "windows_flagged": result.windows_flagged,
            "skipped_checks": result.skipped_checks,
            "notes": result.notes,
            "findings": [asdict(f) for f in result.findings],
            "disclaimer": (
                "Recording-quality (artifact) flags only. Not a diagnosis. Certainty labels are "
                "rule-based, not probabilities. Thresholds are not clinically validated."
            ),
        }
    )


def to_json_bytes(result: AnalysisResult) -> bytes:
    """Full results (metadata, config, version, findings) as UTF-8 JSON."""
    return json.dumps(to_dict(result), indent=2, ensure_ascii=False).encode("utf-8")


def to_csv_bytes(result: AnalysisResult) -> bytes:
    """Findings CSV, preceded by '#' comment lines with metadata, version and config.

    Read it back with pandas.read_csv(path, comment="#").
    """
    summary = result.recording_summary
    header_lines = [
        "# ECG/EEG Signal Quality Checker - findings export",
        f"# package_version: {result.package_version}",
        f"# created_at: {result.created_at}",
        f"# signal_type: {summary.get('signal_type')}",
        f"# data_kind: {summary.get('data_kind')}",
        f"# source: {summary.get('source')}",
        f"# sfreq_hz: {summary.get('sfreq_hz')}",
        f"# duration_s: {summary.get('duration_s')}",
        f"# channels: {','.join(summary.get('channel_names', []))}",
        f"# measurement_unit: {summary.get('display_unit')}",
        f"# config: {json.dumps(_json_safe(result.config), ensure_ascii=False)}",
        "# note: certainty labels are rule-based, not probabilities; not a diagnosis",
    ]
    if summary.get("is_simulated"):
        header_lines.append("# SIMULATED SIGNAL - not a real recording")
    buffer = io.StringIO()
    buffer.write("\n".join(header_lines) + "\n")
    to_dataframe(result).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
