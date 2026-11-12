"""Streamlit dashboard for the ECG/EEG Signal Quality Checker.

UI only: every finding comes from signal_quality.analyze.analyze_recording().
Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from signal_quality import __version__
from signal_quality.analyze import analyze_recording
from signal_quality.config import default_config
from signal_quality.demo import SIMULATED_LABEL, list_scenarios, make_scenario
from signal_quality.io import ValidationError, load_file, mark_simulated, peek_channel_names, suggest_signal_type
from signal_quality.plotting import (
    SEVERITY_COLORS,
    findings_by_category_figure,
    timeline_figure,
    waveform_figure,
)
from signal_quality.reporting import to_csv_bytes, to_dataframe, to_html_bytes, to_json_bytes

st.set_page_config(layout="wide", page_title="ECG/EEG Signal Quality Checker")

DEMO = "Demo signal (simulated)"
UPLOAD = "Upload a file"
SCENARIOS = {s["id"]: s for s in list_scenarios()}
DATA_KIND_LABEL = {
    "simulated": "Simulated",
    "real": "Real recording",
    "real_with_injected_artifacts": "Real recording + injected artifacts",
}
MAINS_LABELS = {"60": "60 Hz (US)", "50": "50 Hz (Europe)", "both": "Both"}
REPORTS_DIR = Path(__file__).parent / "reports"


# ------------------------------------------------------------------ cached work
@st.cache_data(show_spinner=False)
def cached_scenario(scenario_id: str, seed: int = 42):
    return make_scenario(scenario_id, seed)


@st.cache_data(show_spinner="Analyzing…")
def cached_demo_analysis(scenario_id: str, overrides: tuple):
    recording, injected = cached_scenario(scenario_id)
    return recording, injected, analyze_recording(recording, config=dict(overrides))


def _named_buffer(content: bytes, name: str) -> io.BytesIO:
    buffer = io.BytesIO(content)
    buffer.name = name
    return buffer


def _as_loader_input(files: tuple):
    """files = ((name, bytes), ...): one CSV/EDF, or a WFDB .hea + .dat set."""
    if any(name.lower().endswith(".hea") for name, _ in files):
        return list(files)
    if len(files) != 1:
        raise ValidationError("Upload one .csv or .edf file, or a WFDB .hea + .dat pair.")
    name, content = files[0]
    return _named_buffer(content, name)


@st.cache_data(show_spinner="Reading channel names…")
def cached_peek(files: tuple) -> list[str]:
    target = _as_loader_input(files)
    return peek_channel_names(target)


@st.cache_data(show_spinner="Loading and analyzing file…")
def cached_upload_analysis(files: tuple, signal_type: str, sfreq, unit, missing: str, simulated: bool, overrides: tuple):
    recording = load_file(_as_loader_input(files), signal_type=signal_type, sfreq=sfreq, unit=unit, missing=missing)
    if simulated:
        recording = mark_simulated(recording)
    return recording, [], analyze_recording(recording, config=dict(overrides))


# ------------------------------------------------------------------ helpers
def try_demo() -> None:
    """Overview button: switch to the power-line demo and analyze it."""
    st.session_state["source"] = DEMO
    st.session_state["scenario"] = "ecg_powerline"
    st.session_state["run_now"] = True


def advanced_settings(signal_type: str) -> dict:
    """Advanced settings expander; returns only values that differ from the defaults."""
    cfg = default_config(signal_type)
    unit = cfg.display_unit
    uk = "uV" if unit == "µV" else "mV"
    overrides: dict = {}
    with st.sidebar.expander("Advanced settings"):
        st.caption(f"Defaults for {signal_type}. Starting points, not clinically validated values.")
        fields = [
            ("window_s", "Window length (s)", 0.5, 0.5),
            ("overlap", "Window overlap (fraction)", 0.0, 0.05),
            (f"flat_std_{uk}", f"Flatline σ threshold ({unit})", 0.0, None),
            (f"drift_ptp_{uk}", f"Drift ptp threshold ({unit})", 0.0, None),
            ("line_ratio_threshold", "Power-line ratio threshold (×)", 1.0, 1.0),
        ]
        fields.append(("spike_z", "Spike robust z threshold", 1.0, 1.0) if signal_type == "EEG"
                      else ("spike_factor", "Spike factor (× P99 of |Δx|)", 1.0, 0.5))
        for name, label, min_value, step in fields:
            default = float(getattr(cfg, name))
            kwargs = {"step": step} if step else {}
            value = st.number_input(
                label, min_value=min_value, value=default, key=f"adv_{signal_type}_{name}",
                max_value=0.9 if name == "overlap" else None, format="%g", **kwargs,
            )
            if value != default:
                overrides[name] = float(value)
    return overrides


def _request_key(request: tuple) -> tuple:
    """Small hashable identity of what was analyzed (file names + sizes, not the bytes)."""
    if request[0] == "demo":
        return request
    _, files, *rest = request
    return ("upload", tuple((n, len(b)) for n, b in files), *rest)


def file_stub(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "recording"


def fmt_duration(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Data and settings")
source = st.sidebar.radio("Data source", [DEMO, UPLOAD], key="source")
request = None  # what the Analyze button would run
signal_type = None

if source == DEMO:
    scenario_id = st.sidebar.selectbox(
        "Demo scenario", list(SCENARIOS), key="scenario", format_func=lambda s: SCENARIOS[s]["label"]
    )
    st.sidebar.caption(f"{SIMULATED_LABEL}. Artifacts were injected on purpose.")
    signal_type = SCENARIOS[scenario_id]["signal_type"]
    request = ("demo", scenario_id)
else:
    uploads = st.sidebar.file_uploader(
        "CSV, EDF, or WFDB .hea + .dat (max 50 MB)", type=["csv", "edf", "hea", "dat"], accept_multiple_files=True,
        help="For PhysioNet WFDB records, select the .hea and .dat files together.",
    )
    if uploads:
        files = tuple((u.name, u.getvalue()) for u in uploads)
        first_name = files[0][0] if len(files) == 1 else next((n for n, _ in files if n.lower().endswith(".hea")), files[0][0])
        suggestion = "EEG"
        try:
            suggestion = suggest_signal_type(cached_peek(files))
        except ValidationError as exc:
            st.sidebar.error(str(exc))
        signal_type = st.sidebar.radio(
            "Signal type", ["ECG", "EEG"], index=["ECG", "EEG"].index(suggestion), horizontal=True,
            key=f"stype_{first_name}", help=f"Suggested from channel names: {suggestion}. Please confirm.",
        )
        missing = "flag" if st.sidebar.checkbox(
            "Report NaN/Inf gaps as missing-data findings", value=False,
            help="Off: files with NaN/Inf values are rejected with an explanation. On: each gap becomes a "
                 "'missing data' finding and the other checks bridge it.",
        ) else "reject"
        simulated = st.sidebar.checkbox(
            "This file is simulated (not a real recording)", value=first_name.lower().startswith("simulated"),
            help="Labels the results 'Simulated' everywhere. The example files in examples/ are simulated.",
        )
        sfreq = unit = None
        if len(files) == 1 and first_name.lower().endswith(".csv"):
            st.sidebar.caption("CSV: one column per channel, header row = channel names; optional first column `time`.")
            sfreq = st.sidebar.number_input("Sampling rate (Hz)", min_value=0.0, value=None, placeholder="e.g. 256")
            unit = st.sidebar.selectbox("Unit", ["V", "mV", "µV"], index=None, placeholder="Choose the unit")
        request = ("upload", files, signal_type, sfreq, unit, missing, simulated)
    else:
        st.sidebar.info("Upload a .csv or .edf file, or a WFDB .hea + .dat pair, to analyze it.")

mains = st.sidebar.radio("Mains frequency", list(MAINS_LABELS), format_func=MAINS_LABELS.get, key="mains", horizontal=True)
overrides = {"mains_hz": mains} if mains != "60" else {}
if signal_type:
    overrides.update(advanced_settings(signal_type))
overrides_key = tuple(sorted(overrides.items()))

analyze_clicked = st.sidebar.button("Analyze", type="primary", width="stretch", disabled=request is None)
run_now = st.session_state.pop("run_now", False)

if (analyze_clicked or run_now) and request is not None:
    st.session_state.pop("error", None)
    try:
        if request[0] == "demo":
            recording, injected, result = cached_demo_analysis(request[1], overrides_key)
        else:
            _, files, stype, sfreq, unit, missing, simulated = request
            recording, injected, result = cached_upload_analysis(files, stype, sfreq, unit, missing, simulated, overrides_key)
        st.session_state.update(recording=recording, injected=injected, result=result,
                                analysis_key=(_request_key(request), overrides_key))
    except ValidationError as exc:
        st.session_state["error"] = str(exc)
        for key in ("recording", "injected", "result"):
            st.session_state.pop(key, None)

st.sidebar.divider()
st.sidebar.caption(
    f"v{__version__} · Detects recording-quality problems only. Not a medical device; not a diagnosis."
)

recording = st.session_state.get("recording")
injected = st.session_state.get("injected") or []
result = st.session_state.get("result")

# ------------------------------------------------------------------ header
st.title("ECG/EEG Signal Quality Checker")
if st.session_state.get("error"):
    st.error(st.session_state["error"])
current_key = None
if request is not None:
    current_key = (_request_key(request), overrides_key)
if result is not None and current_key is not None and current_key != st.session_state.get("analysis_key"):
    st.info("The data source or settings changed. Press **Analyze** to update the results.")

tabs = st.tabs(["Overview", "Analyze", "Findings", "Summary", "Export", "About/Evaluation"])


def need_result() -> bool:
    if result is None:
        st.info("Choose a data source in the sidebar and press **Analyze**. Or try the demo from the Overview tab.")
        return False
    return True


def simulated_banner() -> None:
    if recording is not None and recording.is_simulated:
        st.info(
            "**Simulated signal — not a real recording.** Artifacts were injected on purpose so you can check the detector."
        )


# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.markdown("Find recording-quality problems in ECG and EEG signals and see exactly why each was flagged.")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("**What it detects**")
        st.markdown(
            "- Flatline / disconnected channel\n- Clipping (amplifier saturation)\n"
            "- Power-line interference (50/60 Hz)\n- Sudden spikes and discontinuities\n- Baseline drift / wander\n"
            "- Noisy channel, muscle / high-frequency noise\n- Possible eye blinks (EEG), electrode motion (ECG)\n"
            "- Missing data (NaN gaps)\n- Physiological plausibility **hints** (always *uncertain*)"
        )
    with c2, st.container(border=True):
        st.markdown("**How it works**")
        st.markdown(
            "The signal is cut into short overlapping windows. Simple measurements (spread, peak-to-peak, "
            "spectrum, steepness, slow drift) are compared with **documented rules**. Every flag shows the rule, "
            "the measurement and a plain-English explanation."
        )
    with c3, st.container(border=True):
        st.markdown("**Limitations**")
        st.markdown(
            "- Flags recording problems; it is **not** a diagnosis\n"
            "- Does **not** decide whether a signal is AI-generated\n"
            "- Thresholds are starting points, **not clinically validated**\n"
            "- Certainty labels are rule-based, **not probabilities**"
        )
    st.button("Try a demo signal", type="primary", on_click=try_demo)
    st.caption("Loads “ECG — power-line interference” (simulated) and analyzes it.")

# ------------------------------------------------------------------ Analyze
with tabs[1]:
    if need_result():
        summary = result.recording_summary
        unit = summary["display_unit"]
        name = summary["meta"].get("scenario_label") or summary["meta"].get("file_name") or summary["source"]
        st.markdown(f"#### {name} &nbsp; :gray-background[{DATA_KIND_LABEL[summary['data_kind']]}]")
        simulated_banner()
        cols = st.columns(6)
        cols[0].metric("Signal type", summary["signal_type"])
        cols[1].metric("Sampling rate", f"{summary['sfreq_hz']:g} Hz")
        cols[2].metric("Duration", fmt_duration(summary["duration_s"]))
        cols[3].metric("Channels", summary["n_channels"])
        cols[4].metric("Display unit", unit)
        cols[5].metric("Source", summary["source"])
        for warning in summary["meta"].get("warnings", []):
            st.warning(warning)

        names = recording.channel_names
        if recording.signal_type == "ECG" and len(names) == 1:
            shown = names
        elif recording.signal_type == "ECG":
            shown = [st.selectbox("Lead", names)]
        else:
            shown = st.multiselect("Channels (stacked)", names, default=names[:4]) or names[:1]
        show_processed = st.toggle(
            "Show processed signal", value=False,
            help="Overlays a 0.5–40 Hz band-pass filtered copy for viewing only. Analysis always uses the raw signal.",
        )
        show_truth = False
        if injected:
            show_truth = st.checkbox("Show injected-artifact ground truth", value=False,
                                     help="Green outlines show where artifacts were injected, to compare injected vs detected.")
        fig, downsampled = waveform_figure(recording, result, shown, injected, show_truth, show_processed=show_processed)
        st.plotly_chart(fig, width="stretch")
        notes = ["Red shading = flagged interval (border: solid = definite, dashed = likely, dotted = uncertain)."]
        if downsampled:
            notes.append("Display downsampled; analysis uses full resolution.")
        st.caption(" ".join(notes))
        st.markdown(f"**{len(result.findings)} finding(s).** See the Findings tab for details.")

# ------------------------------------------------------------------ Findings
with tabs[2]:
    if need_result():
        simulated_banner()
        df = to_dataframe(result)
        if df.empty:
            st.success("No quality problems found with the current settings.")
            st.caption("This does not prove the recording is clean: only the checks listed in About were run.")
        else:
            f1, f2, f3 = st.columns(3)
            cats = f1.multiselect("Category", sorted(df["Category"].unique()))
            chans = f2.multiselect("Channel", list(dict.fromkeys(df["Channel"])))
            sevs = f3.multiselect("Severity", ["low", "medium", "high"])
            view = df
            if cats:
                view = view[view["Category"].isin(cats)]
            if chans:
                view = view[view["Channel"].isin(chans)]
            if sevs:
                view = view[view["Severity"].isin(sevs)]
            view = view.reset_index(drop=True)
            styled = view.drop(columns=["Detector"]).style.map(
                lambda v: f"color: {SEVERITY_COLORS.get(v, '#1A1A1A')}; font-weight: 600", subset=["Severity"]
            )
            event = st.dataframe(
                styled, width="stretch", hide_index=True, on_select="rerun",
                selection_mode="single-row", key="findings_table",
            )
            st.caption(f"Measurements are in {result.recording_summary['display_unit']}. "
                       "Certainty labels are rule-based, not probabilities. Select a row to zoom the waveform.")
            rows = event.selection.rows if event is not None else []
            if rows:
                row = view.iloc[rows[0]]
                t0 = max(0.0, float(row["Start (s)"]) - 2)
                t1 = min(recording.duration_s, float(row["End (s)"]) + 2)
                st.markdown(f"**{row['Category']}** on **{row['Channel']}**, {row['Start (s)']}–{row['End (s)']} s")
                st.markdown(f"> {row['Explanation']}")
                zfig, _ = waveform_figure(recording, result, [row["Channel"]], injected, bool(injected), time_range=(t0, t1))
                st.plotly_chart(zfig, width="stretch")

# ------------------------------------------------------------------ Summary
with tabs[3]:
    if need_result():
        simulated_banner()
        n_find = len(result.findings)
        pct = 100 * result.windows_flagged / result.windows_analyzed if result.windows_analyzed else 0.0
        by_channel = pd.Series([f.channel for f in result.findings]).value_counts()
        m = st.columns(4)
        m[0].metric("Windows flagged", f"{pct:.1f}%", help=f"{result.windows_flagged} of {result.windows_analyzed} channel-windows")
        m[1].metric("Findings", n_find)
        m[2].metric("Categories detected", len({f.category for f in result.findings}))
        m[3].metric("Most affected channel", by_channel.index[0] if n_find else "—")
        if n_find:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Findings by category**")
                st.plotly_chart(findings_by_category_figure(result), width="stretch")
            with c2:
                st.markdown("**Flagged intervals over time**")
                st.plotly_chart(timeline_figure(result), width="stretch")
        else:
            st.success("No quality problems found with the current settings.")
        st.markdown("**Checks that were skipped**")
        if result.skipped_checks:
            for item in result.skipped_checks:
                chans = ", ".join(c for c in item.get("channels", []) if c)
                st.markdown(f"- `{item['check']}` ({chans}): {item['reason']}")
        else:
            st.caption("All checks ran on every channel.")
        if result.notes:
            st.markdown("**Notes**")
            for item in result.notes:
                st.markdown(f"- {item['reason']} ({', '.join(item.get('channels', []))})")

# ------------------------------------------------------------------ Export
with tabs[4]:
    if need_result():
        simulated_banner()
        summary = result.recording_summary
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        stub = file_stub(summary["meta"].get("scenario_id") or summary["source"])
        base = f"{stub}_{summary['signal_type']}"
        st.markdown("Both files include the recording metadata (no patient-identifying fields), the exact settings used and the package version.")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Findings (CSV)", to_csv_bytes(result), file_name=f"{base}_findings_{stamp}.csv",
                           mime="text/csv", width="stretch")
        c2.download_button("Full results (JSON)", to_json_bytes(result), file_name=f"{base}_results_{stamp}.json",
                           mime="application/json", width="stretch")
        c3.download_button("HTML report", to_html_bytes(result, recording, injected), file_name=f"{base}_report_{stamp}.html",
                           mime="text/html", width="stretch")
        st.caption("The HTML report includes an interactive waveform (it loads the Plotly library from the internet when opened).")

# ------------------------------------------------------------------ About / Evaluation
with tabs[5]:
    st.markdown("### Method")
    st.markdown(
        "Each channel is analyzed separately on the **raw** signal (a copy; the original is never changed). "
        "Signals are stored in volts and shown in µV (EEG) or mV (ECG). Findings from overlapping windows are "
        "merged, and side effects are removed (for example, a flat stretch hides drift or spike flags inside it)."
    )
    eeg, ecg = default_config("EEG"), default_config("ECG")
    rules = pd.DataFrame(
        [
            ("Flatline", f"σ < {eeg.flat_std_uV} µV and ptp < {eeg.flat_ptp_uV} µV for ≥ {eeg.flat_min_duration_s:g} s",
             f"σ < {ecg.flat_std_mV} mV and ptp < {ecg.flat_ptp_mV} mV for ≥ {ecg.flat_min_duration_s:g} s"),
            ("Clipping", "≥ 1% of window at the channel max/min (±0.1% of range) in flat runs ≥ max(3, 0.01·fs) samples", "same"),
            ("Power line", f"PSD at f0±1 Hz ≥ {eeg.line_ratio_threshold:g}× the median of neighbours (f0 = 60/50 Hz; skipped if Nyquist too low)", "same"),
            ("Spikes", f"robust z of first difference > {eeg.spike_z:g}",
             f"|Δx| > {ecg.spike_factor:g} × P99(|Δx|); near an R-peak → uncertain"),
            ("Baseline drift", f"ptp of < {eeg.drift_cutoff_hz} Hz component ≥ {eeg.drift_ptp_uV:g} µV",
             f"ptp of < {ecg.drift_cutoff_hz} Hz component ≥ {ecg.drift_ptp_mV} mV"),
            ("Noisy channel", f"robust σ (> 1 Hz) > {eeg.noisy_ratio:g}× the median of the other channels (needs ≥ 4 channels)", "—"),
            ("Muscle / high-frequency", f"power 20–45 Hz > {eeg.muscle_ratio:g} of 1–45 Hz power",
             f"power > 40 Hz > {ecg.hf_ratio:g}× power 1–40 Hz (50/60 Hz excluded)"),
            ("Ocular (EEG) / electrode motion (ECG)",
             f"0.5–4 Hz deflection ≥ {eeg.ocular_ptp_uV:g} µV on Fp/AF and ≥ {eeg.ocular_frontal_ratio:g}× posterior; never definite",
             "≥ 2 of: drift, abrupt jump, beat detection failing locally"),
            ("Missing data", "NaN/Inf runs (when the option is on)", "same"),
            ("Plausibility hints (uncertain only)", f"robust amplitude > {eeg.plaus_amp_uV:g} µV for ≥ 2 s",
             f"heart rate outside {ecg.plaus_hr_min_bpm:g}–{ecg.plaus_hr_max_bpm:g} bpm, RR < 200 ms, or > 3 s without a beat"),
        ],
        columns=["Detector", "EEG rule", "ECG rule"],
    )
    st.dataframe(rules, hide_index=True, width="stretch")
    st.caption("Thresholds are reasonable starting points with a stated rationale, not validated values. "
               "Certainty (definite / likely / uncertain) is a rule-based label, never a probability.")

    st.markdown("### Evaluation")
    reports = sorted(REPORTS_DIR.glob("*.json")) if REPORTS_DIR.exists() else []
    st.markdown("**Controlled injected artifacts (PhysioNet recordings)**")
    if reports:
        for path in reports:
            with st.expander(path.name):
                st.json(json.loads(path.read_text()))
    else:
        st.warning("Not yet evaluated.")
    st.markdown("**Independently labelled real artifacts**")
    st.warning("Not yet evaluated.")
    st.caption("The demo scenarios above are simulated checks of the code, not an evaluation of real-world accuracy.")
