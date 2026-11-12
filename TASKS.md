# TASKS — Phase-by-phase build checklist

Rough calendar (requested timeframe, not a guarantee): Phase 1 = first session (a few hours) · Phase 2 = week 1–2 · Phase 3 = week 3 · Phase 4 = week 3–4 · Phase 5 = week 4. Joseph chose to keep tests, CSV/EDF upload, export and the summary tab in Phase 1 (an earlier schedule had pushed them to week 2).

Claude Code: build **one phase per session** unless told otherwise. Tick boxes (`- [x]`) as you go. A phase is done only when every item in its **Definition of Done** has been verified by actually running it.

---

## Phase 1 — Demo (target: one Claude Code session, a few hours)

### 1.1 Setup
- [x] Check `python3 --version` (3.12/3.13) and OS; create `.venv`; activate
- [x] `pyproject.toml` (package `signal_quality` in `src/`), `requirements.txt` (pinned compatible ranges from ARCHITECTURE §1 + `-e .`), `.gitignore` (`.venv`, `data/*` except README, `__pycache__`, `.pytest_cache`, `reports/*.tmp`)
- [x] `.streamlit/config.toml` with theme (DESIGN §2) and `[server] maxUploadSize = 50`
- [x] `docs/DECISIONS.md` created

### 1.2 Core engine
- [x] `models.py`: `Recording`, `Finding`, `AnalysisResult`, `InjectedArtifact`
- [x] `config.py`: `EEGConfig`, `ECGConfig` with all Phase 1 thresholds from DETECTORS.md (units in names, rationale comments)
- [x] `io.py`: `load_csv(file, sfreq, unit, signal_type)`, `load_edf(file, signal_type)`, `validate(recording)`; friendly `ValidationError`; strip identifying EDF header info
- [x] `preprocessing.py`: `iter_windows(n_samples, sfreq, window_s, overlap)`; filter helpers that return copies
- [x] `features.py`: std, ptp, MAD/robust σ, Welch PSD, band power, line_ratio, first-difference stats
- [x] Detectors: `flatline.py`, `clipping.py`, `powerline.py`, `spikes.py`, `drift.py` exactly per DETECTORS.md (incl. Nyquist skip, ECG P99 spike rule, NeuroKit2 R-peak cross-check)
- [x] `merge.py`: merging + suppression rules
- [x] `analyze.py`: `analyze_recording()` returning `AnalysisResult` with config, skipped_checks, window counts, version, timestamp
- [x] `reporting.py`: `to_dataframe`, `to_csv_bytes`, `to_json_bytes`

### 1.3 Demo data
- [x] `demo.py`: `list_scenarios()`, `make_scenario(id, seed=42)` for every scenario in DESIGN §5, returning ground-truth `InjectedArtifact`s; all marked `is_simulated=True`

### 1.4 Tests (see §Testing below)
- [x] Write Phase 1 tests; run `pytest -q`; fix until all pass

### 1.5 Dashboard
- [x] `plotting.py`: waveform figure with shaded findings, optional ground-truth outlines, stacked EEG, display downsampling
- [x] `app.py`: sidebar + tabs (Overview, Analyze, Findings, Summary, Export, About/Evaluation) per DESIGN.md; analysis only on button click; session_state; caching
- [x] Launch `streamlit run app.py` headless once (`--server.headless true`), confirm no errors, then stop it

### 1.6 Docs and CI
- [x] `README.md`: overview, what it detects, **limitations**, install/run (Mac), demo link placeholder, how it works, roadmap, citations for NeuroKit2/MNE/WFDB/PhysioNet
- [x] `.github/workflows/tests.yml`: Ubuntu, Python 3.12, `pip install -r requirements.txt`, `pytest -q`
- [x] `data/README.md`: datasets used later + download commands (no data committed)

### Phase 1 Definition of Done
- [x] `pip install -r requirements.txt` works in a fresh venv
- [x] All tests pass (real output shown in the report)
- [x] Every non-clean demo scenario: each injected artifact is detected (overlapping interval, correct channel, correct category)
- [x] Every clean scenario: zero findings
- [x] CSV upload (with sfreq + unit) and EDF upload both analyze successfully; bad input shows a friendly error
- [x] CSV and JSON exports download and contain metadata + config + version
- [x] Streamlit app launches without errors
- [x] Simulated-data banner visible; About/Evaluation shows "Not yet evaluated"
- [x] Report delivered in the CLAUDE.md format

---

## Phase 2 — Full detectors, formats, CLI (week 1–2)
- [x] WFDB loading (`load_wfdb`; multi-file upload of `.hea` + `.dat`)
- [x] Phase 2 detectors from DETECTORS.md: noisy channel, EEG muscle, ECG high-frequency noise, electrode motion, ocular, non-finite data
- [x] **Physiological plausibility hints** (ECG heart rate / RR timing; EEG amplitude) exactly per DETECTORS.md: always `uncertain`, non-diagnostic wording, tests showing clean simulated ECG at 60–100 bpm produces no hint
- [x] HTML report export (`reporting.to_html`) + config/version included
- [x] `cli.py` + `[project.scripts] sqc = "signal_quality.cli:main"`:
      `sqc analyze rec.edf --type EEG --out results/`; `sqc analyze rec.csv --type ECG --sfreq 360 --unit mV --out results/`; `sqc demo --list`
- [x] Test: CLI output == `analyze_recording()` output for the same input
- [x] Large-file handling: per-channel processing; warn above 30 min of data at high sfreq
- [x] **DoD:** all formats load; all detectors tested with fixtures; CLI documented in README with working examples; tests pass

## Phase 3 — Evaluation & notebooks (week 3)
- [ ] `scripts/download_data.py` (NSTDB + a small EEGMMIDB subset, e.g. 5 subjects, baseline runs R01/R02) with `wfdb.dl_database` / MNE helpers
- [ ] `evaluation.py`: interval matching (a detection matches a reference event if they overlap ≥ 1 sample on the same channel and category family), **event-level** and **window-level** precision / recall / F1 / FPR / confusion matrix; undefined metrics reported as "undefined (single class)" — never invented
- [ ] ECG experiment on NSTDB. Structure (checked against PhysioNet docs on 2026-11-12; re-verify before coding): 12 half-hour records built from clean MIT-BIH records **118 and 119** with **electrode-motion ('em') noise** added at SNRs 24, 18, 12, 6, 0, −6 dB (`118e24` … `119e_6`). Noise starts after the first 5 minutes and runs in 2-minute noisy segments alternating with 2-minute clean segments, which gives reference intervals for "noisy vs clean". The separate noise-only records `bw` (baseline wander), `ma` (muscle) and `em` can be added to clean ECG with the same schedule for baseline-wander and muscle experiments. The `.atr` files are **beat** annotations copied from the clean originals, **not** artifact labels. Report performance per SNR; there are only 2 independent source records, so say so in the limitations and never treat the 12 records as 12 subjects
- [ ] EEG experiment: inject configurable artifacts (deterministic seeds) into EEGMMIDB baseline segments; inspect for pre-existing artifacts first; report per severity
- [ ] Investigate **one smaller, expert-labelled public EEG artifact dataset**. Add it only if its annotations, licence, availability and documentation support a defensible benchmark; otherwise document why it was skipped
- [ ] Strict split: threshold-tuning subset vs final evaluation subset (by subject/record)
- [ ] Results saved to `reports/eval_ecg.json`, `reports/eval_eeg.json` with dataset sizes, channels, configs, seeds
- [ ] Notebooks: `01_understanding_signals.ipynb`, `02_artifact_detection.ipynb`, `03_evaluation.ipynb`. They import the package and never reimplement detectors
- [ ] App's About/Evaluation tab reads `reports/*.json`
- [ ] `docs/METHODOLOGY.md`: research question, artifact definitions, data, methods, baselines, metrics, limitations, reproducibility
- [ ] **DoD:** scripts run end-to-end when data is available; real metrics saved; limitations written; if downloads were impossible in the environment, the commands are documented and the step is marked not run

## Phase 4 — Optional ML (week 3–4)
- [ ] `ml.py`: feature matrix per window; Isolation Forest (`random_state=42`); train only on the tuning split's designated reference windows; save model + `model_card.json` (features, data, params, date)
- [ ] Subject/record-level split; no overlapping windows across train and test
- [ ] Compare **rules alone**, **ML alone** (where meaningful) and **rules + ML combined** on the same evaluation set; save to `reports/eval_ml_comparison.json`
- [ ] Handle a missing or untrained model explicitly (app and CLI say "No trained model; rule-based results only")
- [ ] App: "ML anomaly score (uncalibrated)" shown only if a model exists; rules remain default
- [ ] **DoD:** honest comparison written in the README, including "ML did not improve results" if that's the outcome

## Phase 5 — Polish & verification (week 4)
- [ ] UX pass against DESIGN.md; error messages; empty states
- [ ] README final: architecture Mermaid diagram, screenshots/GIF, real results, citations, limitations, future work
- [ ] Re-run all tests + evaluation scripts; check README commands work verbatim on a fresh clone
- [ ] Deploy final version (DEPLOYMENT.md); verify the live link
- [ ] **v1 DoD:** everything in PRD §8; no claim in README/app that isn't backed by code or saved results

---

## Testing (all phases)
Fixtures are generated in code (no big downloads). Minimum Phase 1 tests:
- CSV load: units converted to volts correctly (µV/mV/V); missing sfreq → error; NaN → error; wrong unit → error; < 1 s → error
- EDF load: write a tiny EDF in a temp dir (e.g., with `mne.export.export_raw` or `pyedflib` if needed) → loads, volts, channel names correct, no patient info in the result
- Windowing: counts, overlap, short-recording case
- Each detector: **detects** its artifact on a synthetic fixture with the correct channel and time (±1 window) **and** gives **no** finding on a clean fixture
- Power-line: skipped (with a reason) when Nyquist is too low
- ECG spikes: clean simulated ECG at 360 Hz → zero spike findings (R-peaks must not trigger)
- Merge/suppression rules
- `analyze_recording` returns config, version, skipped_checks; JSON/CSV export structure
- Demo: every scenario's injected artifacts detected; clean scenarios → zero findings
Run with `pytest -q`. CI runs the same command.
