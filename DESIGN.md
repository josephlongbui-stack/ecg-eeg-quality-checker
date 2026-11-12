# DESIGN — Dashboard (Streamlit)

## 1. Design goals
- Looks like a **professional scientific tool**, not a class demo: restrained colors, clear hierarchy, no emoji clutter.
- A first-time visitor gets to a result in **one click** (Load demo → Analyze).
- Advanced settings are hidden in an expander. Defaults just work.
- Honest by design: the simulated-data label, units and limitations are always visible.

## 2. Visual style
| Element | Spec |
|---|---|
| Theme | Light. `.streamlit/config.toml`: `primaryColor="#1F4E79"`, `backgroundColor="#FFFFFF"`, `secondaryBackgroundColor="#F4F6F8"`, `textColor="#1A1A1A"`, `font="sans serif"` |
| Signal traces | Dark slate `#2B3A4A`, line width 1 |
| Flag highlight | Translucent red `rgba(200, 40, 40, 0.18)` shaded rectangles; certainty shown by border: definite = solid, likely = dashed, uncertain = dotted |
| Severity chips | low `#6B7B8C` · medium `#C77700` · high `#B3261E` |
| Typography | Streamlit defaults. Titles in sentence case. Units always in axis labels, e.g. "Amplitude (µV)" |
| Layout | `st.set_page_config(layout="wide", page_title="ECG/EEG Signal Quality Checker")` |

## 3. Navigation
Single page `app.py` with `st.tabs`: **Overview · Analyze · Findings · Summary · Export · About/Evaluation**. The sidebar holds the data source and settings.

### Sidebar
1. **Data source** (radio): `Demo signal (simulated)` | `Upload a file`
2. If **Demo:** select box of scenarios from `demo.list_scenarios()`, e.g. "ECG — clean", "ECG — power-line interference", "ECG — lead dropout", "EEG — flat channel", ... (see §5).
3. If **Upload:** `st.file_uploader` (`.csv`, `.edf`); then **Signal type** (ECG/EEG; suggest from channel names). For CSV, also **Sampling rate (Hz)** (required number input, no default) and **Unit** (V / mV / µV, required).
4. **Mains frequency:** 60 Hz (default) / 50 Hz / both.
5. `▸ Advanced settings` expander: window length, overlap, key thresholds (prefilled from config, with units).
6. **Analyze** primary button. Analysis runs **only** on click; store the result in `st.session_state`.

### Tab: Overview
- Title + one-line purpose: "Find recording-quality problems in ECG and EEG signals and see exactly why each was flagged."
- Three short cards: **What it detects** (list) · **How it works** (rules → measurements → flags) · **Limitations** (not diagnosis; not AI-detection; thresholds not clinically validated).
- Button: "Try a demo signal" → sets the sidebar to Demo with "ECG — power-line interference" and runs the analysis.

### Tab: Analyze (main view)
- **Metadata strip** (`st.columns` metrics): Signal type · Sampling rate · Duration · Channels · Unit · Source.
- If simulated: a persistent `st.info` banner: "Simulated signal — not a real recording. Artifacts were injected on purpose so you can check the detector."
- **Channel selector:** single channel for ECG; for EEG, a multiselect (default first 4 channels) shown as a **stacked** plot (vertical offsets, channel name labels on the y-axis).
- **Waveform (Plotly):** zoom/pan, range slider, shaded flagged intervals, hover on a shaded region shows category, severity, certainty and rule. Display downsampled to about 20k points per trace (see ARCHITECTURE §8) with a small caption "Display downsampled; analysis uses full resolution."
- Toggle **"Show processed signal"** (Phase 2; off by default): overlays a band-pass-filtered copy for viewing only. The raw signal is always the default view, and analysis always uses the raw signal.
- A data-kind badge next to the title: **Simulated** / **Real recording** / **Real recording + injected artifacts**.
- Demo only: checkbox "Show injected-artifact ground truth" adds green outlines where artifacts were injected, so visitors can compare **injected vs detected**.

### Tab: Findings
- `st.dataframe` with columns: Channel · Start (s) · End (s) · Category · Severity · Certainty · Rule · Measurement · Explanation.
- Filter by category, channel, severity.
- Selecting a row (`on_select="rerun"`) zooms the waveform to that interval ± 2 s, shown in a plot under the table.
- Empty state: "No quality problems found with the current settings." plus a note that this does not prove the recording is clean.

### Tab: Summary
- Metrics: % windows flagged · number of findings · categories detected · most affected channel.
- Bar chart: findings by category. Timeline strip: flagged intervals over time per channel.
- **No** single "quality score" in Phase 1. If one is added later, show its formula next to it and never call it a probability.
- List `skipped_checks` with reasons (e.g., "60 Hz check skipped: sampling rate too low").

### Tab: Export
- `st.download_button` × 2 (Phase 1): **Findings (CSV)** and **Full results (JSON)**, which includes metadata, config, package version and timestamp. Phase 2 adds **HTML report**.
- File names: `{source}_{signal_type}_findings_{YYYYMMDD-HHMM}.csv`.

### Tab: About / Evaluation
- Short method description and links to `DETECTORS.md` content (render the key rules).
- **Evaluation:** until Phase 3 outputs exist in `reports/`, show **"Not yet evaluated"**. After that, load real metrics from `reports/*.json`. Keep "controlled injected artifacts" and "independently labelled real artifacts" in separate sections.

## 4. Copy and tone
Plain, precise, non-alarming. Say "possible" / "likely". Never "diagnosis", "abnormal heart" or "fake signal". Errors say what's wrong and what to do: "Sampling rate is required for CSV files. Enter it in the sidebar (e.g., 256)."

## 5. Demo scenarios (generated by `demo.py`, fixed seeds)
| Scenario id | Content | Injected artifact (ground truth) |
|---|---|---|
| `ecg_clean` | 30 s, 360 Hz, 1 lead, HR 70 (NeuroKit2 `ecg_simulate`, small noise) | none |
| `ecg_powerline` | ecg_clean base | 60 Hz sine, 0.1 mV, 10–20 s |
| `ecg_baseline_wander` | base | 0.25 Hz sine, 1.0 mV, whole recording |
| `ecg_lead_dropout` | base | flat (constant) 12–15 s |
| `ecg_clipping` | base | clip at ±0.6 mV, 5–9 s |
| `ecg_spikes` | base | 4 single-sample spikes of 3 mV at 6.2, 11.7, 18.4, 25.1 s |
| `eeg_clean` | 30 s, 256 Hz, 8 ch (Fp1, Fp2, F3, F4, C3, C4, O1, O2): pink (1/f) noise about 20 µV RMS + 10 Hz alpha strongest at O1/O2 | none |
| `eeg_flat_channel` | eeg_clean base | C3 constant 8–20 s |
| `eeg_powerline` | base | 60 Hz, 15 µV on all channels, 5–15 s |
| `eeg_spikes` | base | step of 200 µV on F4 at 12 s (discontinuity) + 2 pops on O1 |
| `eeg_drift` | base | 0.1 Hz drift 250 µV on Fp1/Fp2 |
| `eeg_clipping` | base | O2 clipped at ±40 µV, 18–24 s |
| `mixed_ecg` | base | power-line + 2 spikes + dropout |

Each scenario returns `(Recording, list[InjectedArtifact])`. `InjectedArtifact` has channel, start_s, end_s, category and parameters. Clean scenarios should produce **zero** findings with default settings. If they don't, first check whether the simulated signal is unrealistic (fix the simulation) or the rule is genuinely too sensitive (fix the rule, with a rationale). Never add special cases that only make the demo look clean. Record the decision in `docs/DECISIONS.md`.
