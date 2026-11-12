# PRD — ECG/EEG Signal Quality Checker

**Owner:** Joseph Bui · **Status:** Phase 1 (demo) next · **Last updated:** 2026-11-12
**Former name:** ECG/EEG Signal Authenticity Checker

## 0. Background, history and decisions
**Origin.** Joseph first described the project in August 2026 (JHU lab outreach) as a Python system that checks ECG/EEG signals against physiological rules (heart-rate ranges, wave timing, expected frequency patterns) and **explains which rule was violated**. He called it early-stage. Later bios described it as flagging *synthetic or corrupted* waveforms. A separate AI-image detector (lighting/shadow consistency) shares the same "check consistency with physical constraints" idea but is a **different project** and is not part of this repo.

**Why it exists.** Trust in biomedical datasets: corrupted recordings undermine downstream analysis and the ML models trained on them. Bringing physiological and signal-processing knowledge into quality checking, and explaining *why* something was flagged, is the defining idea.

**Rebuild decision (2026-11-12, Joseph's choices).**
| Decision | Choice |
|---|---|
| Overall goal | Strong portfolio project + useful research tool + foundation for rigorous experiments or a possible publication |
| Starting point | **Build from scratch** |
| Signals | EEG **and** ECG |
| Detection target | **Corrupted recordings only**: noise, interference, equipment artifacts. Not synthetic/AI-origin detection |
| Product | Interactive web app + research Python package, experiments, visualizations, notebooks |
| Approach | Hybrid: physiological/signal rules + statistics + optional ML, explainable and rigorously validated, **"but not too complicated"** |
| Main job (next 6 months) | Résumé / internship portfolio |
| Deployment | Laptop + free public demo |
| Spec vs old résumé stack | **Spec wins; fix the résumé** (no FastAPI/Docker in v1) |
| Timing | Prototype in a few hours; full version in about 1 month (requested timeframe, not a guarantee) |

**Naming.** Original: *ECG/EEG Signal Authenticity Checker*. Current: **ECG/EEG Signal Quality Checker**. "Signal QA" appeared once as informal shorthand; it is not the project name.

**Evidence status.** As of 2026-11-12, **nothing in this repo is implemented yet**. Earlier résumé text describing a "built" checker refers to an earlier iteration whose working state was never audited. All status claims must come from code that exists and has been run, not from résumé or chat descriptions. Full background: `context/EEG_ECG_Project_Context.pdf`.

## 1. Problem
Clinical and research AI models are increasingly trained on ECG (heart) and EEG (brain) recordings. Those recordings often contain **artifacts**: disconnected electrodes, clipped amplifiers, 60 Hz electrical interference, motion spikes and slow baseline drift. Bad data that slips into analysis or training quietly degrades results. Researchers usually catch this by eye, which is slow, inconsistent and doesn't scale.

## 2. Solution
A Python tool that analyzes ECG and EEG recordings and, for every problem it finds, reports:
- signal type, channel/lead
- start and end time
- artifact category
- severity (`low` / `medium` / `high`)
- the **exact rule and measurement** that triggered it
- a one-sentence plain-English explanation
- certainty (`definite` / `likely` / `uncertain`), a rule-based label and not a probability

Key principle: **explainable over black-box.** Every flag traces back to a measurable rule a researcher can check.

## 3. Users
| User | Need |
|---|---|
| Undergraduate / lab researcher | Quickly screen recordings before analysis |
| Recruiter / professor viewing Joseph's portfolio | Open a link, click once, see it work in under a minute |
| Joseph (developer) | A credible, honest, extendable research-software project |

## 4. Goals (v1, about 4 weeks)
1. Detect core EEG and ECG artifacts with interpretable rules.
2. Interactive Streamlit dashboard: load, analyze, view highlighted waveform, inspect findings, export.
3. Command-line interface using the same engine.
4. Reproducible evaluation on PhysioNet data with controlled injected artifacts.
5. Optional lightweight anomaly model (Isolation Forest), honestly compared with rules.
6. Public GitHub repo with tests in CI and a live demo link.

## 5. Phase 1 demo scope (build first)
- Simulated ECG and EEG demo signals (clean plus several with artifacts injected on purpose, with known labels)
- **Upload your own file:** CSV (user supplies sampling rate and unit) and EDF/EDF+
- Five detectors: **flatline, clipping, power-line interference, sudden spikes/discontinuities, baseline drift**
- Waveform viewer with highlighted flagged intervals and a findings table
- **Export** findings as CSV and full results as JSON
- pytest tests, GitHub Actions CI, deployable to Streamlit Community Cloud

## 6. Full v1 detector scope
**EEG:** flatline/disconnected channel · spikes & discontinuities · excessively noisy channel · power-line (50/60 Hz) · baseline drift · clipping/saturation · possible muscle (high-frequency) contamination · possible ocular artifact (only when frontal channels exist)
**ECG:** baseline wander · power-line · high-frequency muscle/electrical noise · electrode-motion artifact · flatline/lead dropout · clipping · abrupt non-physiological discontinuities
Ambiguous events are reported as **"large transient artifact"** rather than a guessed cause.
**Physiological plausibility hints (Phase 2):** ECG heart rate / beat-timing outside plausible ranges and EEG amplitude far above typical ranges, always labelled `uncertain` and never diagnostic. This brings back Joseph's original "check against known physiology" idea in a safe form (see `DETECTORS.md`).

## 7. Non-goals (v1)
Medical diagnosis · arrhythmia or seizure detection · AI-generated vs authentic classification · real-time monitoring · live hardware · deep learning / GPUs / transformers · mobile apps · user accounts · Docker / FastAPI / cloud infrastructure · paid services · editing or deleting original recordings.

## 8. Success criteria
- **Phase 1:** a public demo link where a visitor picks a demo signal, clicks Analyze, and sees correct highlights for every injected artifact and no flags on the clean signals. CSV/EDF upload and export work. All tests pass in CI.
- **v1:** the definition of done in `TASKS.md` Phase 5, with real measured evaluation metrics (precision, recall, F1, false-positive rate) for ECG and EEG reported separately, plus honest limitations.

## 9. Datasets (Phase 3+)
- **ECG:** MIT-BIH Noise Stress Test Database — https://physionet.org/content/nstdb/1.0.0/ (noise versions of the same record are **not** independent subjects; beat annotations are **not** artifact labels).
- **EEG:** EEG Motor Movement/Imagery Dataset — https://physionet.org/content/eegmmidb/1.0.0/ (no artifact ground truth; use only for loading tests and controlled injection experiments).

## 10. Risks
| Risk | Mitigation |
|---|---|
| Thresholds flag genuine physiology (e.g., ECG R-peaks as spikes) | Modality-specific rules; ECG spike rule is relative to typical QRS slope; R-peak proximity lowers certainty |
| Free hosting has about 1 GB RAM | 50 MB upload cap; analyze per channel; downsample only for plotting |
| Overclaiming on résumé/README | `DEPLOYMENT.md` + README limitations section; "Not yet evaluated" until Phase 3 |
| Joseph is new to code | Plain-English reports at each phase; simple architecture |

## 11. Future directions (after v1)
More signal types (PPG, EMG) · population-specific thresholds · expert-labelled EEG artifact benchmark · cross-channel / spatial consistency checks, entropy and autocorrelation features (discussed earlier, **not** in v1 unless they clearly help) · FastAPI service if others need programmatic access.
An earlier chat sketched a weighted "violation score". That was conceptual only and is **not** part of v1; findings stay as interpretable flags.
