"""Plotly figure builders (used by the Streamlit app and, later, notebooks).

Downsampling here is for DISPLAY ONLY; analysis always uses the full-resolution signal.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import plotly.graph_objects as go

from .models import AnalysisResult, Finding, InjectedArtifact, Recording

TRACE_COLOR = "#2B3A4A"
FLAG_FILL = "rgba(200, 40, 40, 0.18)"
FLAG_LINE = "rgba(200, 40, 40, 0.85)"
TRUTH_LINE = "rgba(30, 140, 60, 0.95)"
PROCESSED_COLOR = "rgba(31, 78, 121, 0.75)"
CERTAINTY_DASH = {"definite": "solid", "likely": "dash", "uncertain": "dot"}
SEVERITY_COLORS = {"low": "#6B7B8C", "medium": "#C77700", "high": "#B3261E"}
DISPLAY = {"EEG": ("µV", 1e6), "ECG": ("mV", 1e3)}
MIN_FLAG_WIDTH_S = 0.08  # point-like events are drawn at least this wide so they stay visible


def display_unit(signal_type: str) -> tuple[str, float]:
    """(unit label, scale from volts) for a modality: EEG in µV, ECG in mV."""
    return DISPLAY[signal_type]


def minmax_decimate(t: np.ndarray, y: np.ndarray, max_points: int = 20000) -> tuple[np.ndarray, np.ndarray, bool]:
    """Keep the min and max of each bucket so spikes stay visible. Returns (t, y, was_downsampled)."""
    n = len(y)
    if n <= max_points:
        return t, y, False
    buckets = max_points // 2
    edges = np.linspace(0, n, buckets + 1).astype(int)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        if np.all(np.isnan(seg)):  # keep a gap visible where data is missing
            idx.append(a)
            continue
        i_min, i_max = a + int(np.nanargmin(seg)), a + int(np.nanargmax(seg))
        idx.extend(sorted((i_min, i_max)))
    idx = np.array(idx)
    return t[idx], y[idx], True


def processed_view(y: np.ndarray, sfreq: float) -> np.ndarray | None:
    """0.5–40 Hz band-pass copy for display only (None if the segment is too short or has gaps)."""
    from .preprocessing import bandpass

    hi = min(40.0, 0.45 * sfreq)
    if len(y) < 4 * sfreq or not np.all(np.isfinite(y)) or hi <= 1.0:
        return None
    return bandpass(y, sfreq, 0.5, hi)


def _rect(x0: float, x1: float, y0: float, y1: float) -> tuple[list[float], list[float]]:
    return [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]


def _widen(start: float, end: float) -> tuple[float, float]:
    if end - start >= MIN_FLAG_WIDTH_S:
        return start, end
    mid = (start + end) / 2
    return mid - MIN_FLAG_WIDTH_S / 2, mid + MIN_FLAG_WIDTH_S / 2


def waveform_figure(
    recording: Recording,
    result: AnalysisResult | None = None,
    channels: list[str] | None = None,
    injected: list[InjectedArtifact] | None = None,
    show_truth: bool = False,
    time_range: tuple[float, float] | None = None,
    max_points: int = 20000,
    height: int | None = None,
    show_processed: bool = False,
) -> tuple[go.Figure, bool]:
    """Waveform with shaded findings. Several channels are stacked with vertical offsets.

    show_processed overlays a 0.5–40 Hz band-pass filtered copy FOR VIEWING ONLY;
    analysis always uses the raw signal. Returns (figure, was_downsampled).
    """
    unit, scale = display_unit(recording.signal_type)
    channels = channels or recording.channel_names[:1]
    idx = [recording.channel_names.index(c) for c in channels]
    n = recording.n_samples
    a, b = 0, n
    if time_range is not None:
        a = max(0, int(np.floor(time_range[0] * recording.sfreq)))
        b = min(n, int(np.ceil(time_range[1] * recording.sfreq)) + 1)
    t = np.arange(a, b) / recording.sfreq
    traces = [recording.data[i, a:b] * scale for i in idx]

    stacked = len(channels) > 1
    if stacked:
        spreads = [np.subtract(*np.nanpercentile(y, [99.5, 0.5])) for y in traces if np.isfinite(y).any()]
        spacing = float(max(spreads)) if spreads else 1.0
        spacing = spacing if np.isfinite(spacing) and spacing > 0 else 1.0
        offsets = [(len(channels) - 1 - k) * spacing for k in range(len(channels))]
        centers = [float(np.nanmedian(y)) if np.isfinite(y).any() else 0.0 for y in traces]
    else:
        spacing = 0.0
        offsets, centers = [0.0], [0.0]

    fig = go.Figure()
    downsampled = False
    bands: dict[str, tuple[float, float]] = {}
    per_trace_points = max(2000, max_points // max(1, len(channels)))
    for ch, y, off, c in zip(channels, traces, offsets, centers):
        yy = y - c + off
        td, yd, ds = minmax_decimate(t, yy, per_trace_points)
        downsampled |= ds
        fig.add_trace(
            go.Scattergl(
                x=td, y=yd, mode="lines", line=dict(color=TRACE_COLOR, width=1), name=ch,
                showlegend=False, hovertemplate=f"{ch}<br>%{{x:.3f}} s<extra></extra>",
            )
        )
        if show_processed:
            proc = processed_view(y, recording.sfreq)
            if proc is not None:
                tp, yp, _ = minmax_decimate(t, proc - np.nanmedian(proc) + off, per_trace_points)
                fig.add_trace(
                    go.Scattergl(
                        x=tp, y=yp, mode="lines", line=dict(color=PROCESSED_COLOR, width=1),
                        name="Processed (0.5–40 Hz, display only)", legendgroup="processed",
                        showlegend=ch == channels[0], hoverinfo="skip",
                    )
                )
        if stacked:
            bands[ch] = (off - spacing / 2, off + spacing / 2)
        else:
            lo, hi = (float(np.nanmin(yy)), float(np.nanmax(yy))) if np.isfinite(yy).any() else (0.0, 1.0)
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            bands[ch] = (lo - pad, hi + pad)

    t0, t1 = (t[0], t[-1]) if len(t) else (0.0, recording.duration_s)
    if result is not None:
        legend_shown = False
        for f in result.findings:
            if f.channel not in bands or f.end_s < t0 or f.start_s > t1:
                continue
            x0, x1 = _widen(f.start_s, f.end_s)
            y0, y1 = bands[f.channel]
            xs, ys = _rect(x0, x1, y0, y1)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, fill="toself", fillcolor=FLAG_FILL, mode="lines",
                    line=dict(color=FLAG_LINE, width=1.2, dash=CERTAINTY_DASH[f.certainty]),
                    name="Flagged interval", legendgroup="flag", showlegend=not legend_shown,
                    hoveron="fills+points",
                    text=_hover(f), hoverinfo="text",
                )
            )
            legend_shown = True
    if show_truth and injected:
        legend_shown = False
        for art in injected:
            if art.channel not in bands or art.end_s < t0 or art.start_s > t1:
                continue
            x0, x1 = _widen(art.start_s, art.end_s)
            y0, y1 = bands[art.channel]
            xs, ys = _rect(x0, x1, y0, y1)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines", line=dict(color=TRUTH_LINE, width=2),
                    name="Injected artifact (ground truth)", legendgroup="truth",
                    showlegend=not legend_shown, text=f"Injected: {art.category} on {art.channel}",
                    hoverinfo="text",
                )
            )
            legend_shown = True

    fig.update_layout(
        height=height or (260 if not stacked else max(320, 90 * len(channels))),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Time (s)", showgrid=True, gridcolor="#E6E9ED", rangeslider=dict(visible=time_range is None))
    if stacked:
        fig.update_yaxes(
            tickmode="array", tickvals=offsets, ticktext=channels, showgrid=False,
            title_text=f"Channel (traces offset; spacing {spacing:.3g} {unit})",
        )
    else:
        fig.update_yaxes(title_text=f"Amplitude ({unit})", showgrid=True, gridcolor="#E6E9ED")
    if time_range is not None:
        fig.update_xaxes(range=list(time_range))
    return fig, downsampled


def _hover(f: Finding) -> str:
    return (
        f"<b>{f.category}</b> on {f.channel}<br>{f.start_s:.2f}–{f.end_s:.2f} s<br>"
        f"Severity: {f.severity} · Certainty: {f.certainty} (rule-based label)<br>Rule: {f.rule}"
    )


def findings_by_category_figure(result: AnalysisResult) -> go.Figure:
    counts = Counter(f.category for f in result.findings)
    cats = sorted(counts)
    fig = go.Figure(go.Bar(x=cats, y=[counts[c] for c in cats], marker_color="#1F4E79"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#FFFFFF")
    fig.update_xaxes(title_text="Category")
    fig.update_yaxes(title_text="Number of findings", dtick=1, gridcolor="#E6E9ED")
    return fig


def timeline_figure(result: AnalysisResult) -> go.Figure:
    """Flagged intervals over time, one row per channel, coloured by severity."""
    channels = list(result.recording_summary.get("channel_names", []))
    fig = go.Figure()
    shown = set()
    for f in result.findings:
        x0, x1 = _widen(f.start_s, f.end_s)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[f.channel, f.channel], mode="lines",
                line=dict(color=SEVERITY_COLORS[f.severity], width=14),
                name=f"{f.severity} severity", legendgroup=f.severity, showlegend=f.severity not in shown,
                text=_hover(f), hoverinfo="text",
            )
        )
        shown.add(f.severity)
    fig.update_layout(
        height=max(200, 40 * len(channels) + 80), margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#FFFFFF", legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_xaxes(title_text="Time (s)", range=[0, result.recording_summary.get("duration_s", 1)], gridcolor="#E6E9ED")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(channels)), type="category")
    return fig
