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


# ---------------------------------------------------------------- HTML report (Phase 2)
_SEV_COLORS = {"low": "#6B7B8C", "medium": "#C77700", "high": "#B3261E"}


def _cell(value: Any) -> str:
    import html as _html

    return _html.escape("" if value is None else str(value))


def to_html(result: AnalysisResult, recording=None, injected=None, include_plot: bool = True) -> str:
    """A single readable HTML report: metadata, settings, skipped checks and findings.

    If `recording` is given, an interactive waveform is embedded (Plotly loaded from its CDN).
    """
    summary = result.recording_summary
    unit = summary.get("display_unit", "")
    rows = []
    for f in result.findings:
        color = _SEV_COLORS.get(f.severity, "#1A1A1A")
        rows.append(
            "<tr>"
            f"<td>{_cell(f.channel)}</td><td>{f.start_s:.3f}</td><td>{f.end_s:.3f}</td>"
            f"<td>{_cell(f.category)}</td><td style='color:{color};font-weight:600'>{_cell(f.severity)}</td>"
            f"<td>{_cell(f.certainty)}</td><td>{_cell(f.rule)}</td>"
            f"<td class='m'>{_cell(_format_measurement(f.measurement))}</td><td>{_cell(f.explanation)}</td>"
            "</tr>"
        )
    findings_html = (
        "<table><thead><tr><th>Channel</th><th>Start (s)</th><th>End (s)</th><th>Category</th><th>Severity</th>"
        "<th>Certainty</th><th>Rule</th><th>Measurement</th><th>Explanation</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        if rows
        else "<p>No quality problems found with the current settings. This does not prove the recording is clean.</p>"
    )
    skipped = "".join(
        f"<li><code>{_cell(s.get('check'))}</code> ({_cell(', '.join(c for c in s.get('channels', []) if c))}): {_cell(s.get('reason'))}</li>"
        for s in result.skipped_checks
    ) or "<li>None: all checks ran.</li>"
    notes = "".join(f"<li>{_cell(n.get('reason'))}</li>" for n in result.notes)
    meta_rows = "".join(
        f"<tr><th>{_cell(k)}</th><td>{_cell(v)}</td></tr>"
        for k, v in [
            ("Signal type", summary.get("signal_type")),
            ("Data kind", summary.get("data_kind")),
            ("Source", summary.get("source")),
            ("File / scenario", summary.get("meta", {}).get("file_name") or summary.get("meta", {}).get("scenario_label")),
            ("Sampling rate", f"{summary.get('sfreq_hz'):g} Hz"),
            ("Duration", f"{summary.get('duration_s', 0):.1f} s"),
            ("Channels", ", ".join(summary.get("channel_names", []))),
            ("Measurement unit", unit),
            ("Windows flagged", f"{result.windows_flagged} of {result.windows_analyzed} channel-windows"),
            ("Package version", result.package_version),
            ("Created", result.created_at),
        ]
    )
    banner = (
        "<div class='banner'><b>Simulated signal — not a real recording.</b> Artifacts were injected on purpose.</div>"
        if summary.get("is_simulated")
        else ""
    )
    plot = ""
    if include_plot and recording is not None:
        from .plotting import waveform_figure

        channels = recording.channel_names[: (1 if recording.signal_type == "ECG" else 8)]
        fig, ds = waveform_figure(recording, result, channels, injected, bool(injected))
        plot = fig.to_html(full_html=False, include_plotlyjs="cdn") + (
            "<p class='note'>Display downsampled; analysis used full resolution.</p>" if ds else ""
        )
    config_json = json.dumps(_json_safe(result.config), indent=2, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signal quality report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; color:#1A1A1A; background:#FFFFFF; max-width:1200px; margin:24px auto; padding:0 16px; }}
h1 {{ color:#1F4E79; font-size:1.6rem; }} h2 {{ color:#1F4E79; font-size:1.2rem; margin-top:28px; }}
table {{ border-collapse:collapse; width:100%; font-size:0.85rem; }} th, td {{ border:1px solid #DDE2E7; padding:6px 8px; text-align:left; vertical-align:top; }}
th {{ background:#F4F6F8; }} td.m {{ font-family: Menlo, monospace; font-size:0.75rem; }}
.banner {{ background:#E8F0FA; border-left:4px solid #1F4E79; padding:10px 14px; margin:12px 0; }}
.note {{ color:#6B7B8C; font-size:0.85rem; }} pre {{ background:#F4F6F8; padding:12px; overflow:auto; font-size:0.75rem; }}
</style></head><body>
<h1>ECG/EEG Signal Quality Checker: report</h1>
{banner}
<p class="note">Flags recording-quality problems (artifacts) only. Not a diagnosis. Certainty labels are rule-based,
not probabilities. Thresholds are starting points, not clinically validated.</p>
<h2>Recording</h2><table>{meta_rows}</table>
{('<h2>Waveform</h2>' + plot) if plot else ''}
<h2>Findings ({len(result.findings)})</h2>
<p class="note">Measurements are in {_cell(unit)} unless the name says otherwise.</p>
{findings_html}
<h2>Checks skipped</h2><ul>{skipped}</ul>
{('<h2>Notes</h2><ul>' + notes + '</ul>') if notes else ''}
<h2>Settings used</h2><pre>{_cell(config_json)}</pre>
</body></html>
"""


def to_html_bytes(result: AnalysisResult, recording=None, injected=None, include_plot: bool = True) -> bytes:
    return to_html(result, recording, injected, include_plot).encode("utf-8")
