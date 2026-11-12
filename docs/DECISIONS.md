# Decision log

Minor decisions made while building, with the reason for each. Scientific-threshold changes are listed separately so they can be reviewed.

## Phase 1 (2026-11-12)

### Environment
- **Python 3.12.16 from Homebrew** (`/opt/homebrew/bin/python3.12`). The system `python3` is 3.9.6, which is too old.
- **Library versions:** every minimum version listed in ARCHITECTURE.md was confirmed on PyPI and installed: numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, mne 1.13.2, wfdb 4.3.1, neurokit2 0.3.0, streamlit 1.67.0, plotly 7.1.0, pytest 9.2.0. `requirements.txt` uses `>=minimum,<next major` ranges.
- **`edfio` added** to the requirements. MNE needs it to *write* EDF files, which the EDF upload test does.

### Changes to detector behaviour (deviations from DETECTORS.md, with reasons)
1. **Power-line PSD segment length is 1 s, not 2 s** (`psd_segment_s = 1.0`, configurable). With 2 s EEG windows, `nperseg = 2·sfreq` puts one Welch segment in the whole window, so nothing is averaged. That raw spectrum is noisy enough that the 10× rule fired on about 1 in 1,000 clean EEG windows: 10 of 9,280 windows across 40 simulated seeds × 8 channels. With 1 s segments (3 averaged segments, 1 Hz resolution) the same 9,280 windows gave **0** false flags (maximum ratio 6.4). Sensitivity barely changed: a 15 µV 60 Hz sine still gave a minimum ratio of 20.5, against 24.8 before. The peak band (f0 ± 1 Hz) and neighbour bands (2–6 Hz away) are unchanged. Setting `psd_segment_s = 2.0` restores the original behaviour.
2. **The drift low-pass uses mirror ("even") edge padding one cutoff-period long.** SciPy's default `sosfiltfilt` padding is a ~10-sample point reflection. When a recording starts on an R-peak, that padding produced a fake 1 mV "drift" in the first window of clean ECG.
3. **The spike scale is measured only outside flat stretches.** A flat stretch has zero first differences, which shrinks the robust σ (EEG) and lowers the P99 (ECG). Ordinary activity elsewhere then looks extreme, and in testing a channel with a 12 s dropout got 8 false spike flags. The flatline mask is shared from `flatline.flat_mask`.
4. **The "discontinuity" test uses the local robust σ of the signal** (1 s before and after the event, each side centred on its own median), not the robust σ of the whole channel. A 200 µV step shifts half the channel, which inflates the whole-channel σ so much that the step could never count as a discontinuity.
5. **The "single-sample" event for `definite` spike certainty** is a one-sample step (1 flagged difference) or a one-sample spike (2 adjacent flagged differences: up and back down).
6. **Flatline detection runs on a 0.25 s sliding window over the whole channel, not on the 2 s / 4 s analysis windows.** The demo ECG dropout lasts 3 s, so no 4 s window is ever fully flat. The rule (σ and ptp thresholds, ≥ 1 s) is unchanged; only the time resolution is finer.
7. **Flatline `definite` certainty** means the flat stretch contains an exactly constant run at least `flat_min_duration_s` long. The sliding window can extend a stretch a few samples into a quiet stretch of real signal, so "ptp == 0 over the whole stretch" would otherwise never hold.
8. **Merged findings take the highest certainty and severity of their parts.** Severity is also recomputed from the merged duration (≥ 10 s → high).
9. **`AnalysisResult.notes` was added** (not in ARCHITECTURE.md §3). It carries information that is not a finding, e.g. "a notch filter appears to have been applied".

### Demo simulation (DESIGN.md §5)
- **Clipping scenarios, Joseph's decision "fix the demo":** the clipping detector compares samples with the channel's overall max/min, as in DETECTORS.md. Clipping only 5–9 s at ±0.6 mV, as the original table said, would leave normal R-peaks elsewhere above the "clip" level, so the clipped samples would never be the channel max. The demo now simulates real amplifier saturation. The signal surges (ECG ×2.5 during 5–9 s; EEG O2 ×3 during 18–24 s, with 0.1 s ramps inside the interval), and the whole trace is clipped at one fixed limit 5% above the clean signal's largest excursion. The ground-truth intervals are unchanged.
- **Simulated EEG spectrum:** power ∝ 1/f from 0.5 to 30 Hz, then falling off faster (∝ 1/f³), plus a 0.5 µV RMS white amplifier noise floor. Plain 1/f noise up to 100 Hz put about 9 µV RMS above 30 Hz, far more than real scalp EEG. That made sample-to-sample jumps unrealistically large, and a 150 µV electrode pop barely passed the spike rule. The noise starts at 0.5 Hz, like hardware high-pass filtering.
- **Lead dropout / flat channel** freezes the trace at its last value, so the start of the dropout has no artificial jump.
- **EEG pops** are a 150 µV instant jump that decays back with a 15 ms time constant. **The EEG step** is +200 µV on F4 from 12 s to the end.
- **Mixed ECG timings:** power line 3–8 s, spikes at 10.5 and 22.3 s, dropout 15–18 s.

### Known side effects (kept on purpose; not special-cased)
- The ECG dropout scenario also gets a `baseline_drift` finding around the dropout, because freezing the trace shifts its slow baseline. The EEG step scenario gets `baseline_drift` around the step for the same reason. Both are real evidence of a slow level change, and the flatline suppression rule only removes findings *fully inside* the flat interval.
- **Robustness across random seeds** (20 seeds × 13 scenarios, checked during the build): 1 clean ECG recording (seed 5) got a `baseline_drift` flag from a 0.64 mV slow swing that NeuroKit2's simulator produced. 2 of 40 EEG pops were found at the right time but labelled `large_transient` instead of `spike`, because the background moved within 100 ms; that is the spec's "don't guess a cause" rule. The demo uses seed 42.
- Some injected ECG spikes land within 50 ms of a detected R-peak and are therefore labelled `uncertain`, as DETECTORS.md requires.

### App
- The CSV export starts with `#` comment lines holding the metadata, config and version. Read it with `pandas.read_csv(path, comment="#")`.
- The JSON/CSV exports never include EDF patient fields. Only file name, unit, duration, scenario info and warnings are kept (`analyze.SAFE_META_KEYS`).

## Phase 2 (2026-11-12)

### New detectors: choices DETECTORS.md left open
- **Detector types.** Per-channel detectors keep the Phase 1 signature. Noisy-channel and ocular checks compare channels, so they are "multi-channel" detectors (`detect(data, sfreq, channel_names, config)`). Electrode motion combines other detectors' findings, so `analyze.py` runs it last for each ECG channel.
- **Noisy channel** compares robust σ **above 1 Hz** (`noisy_highpass_hz`). On the raw signal, a slow 0.1 Hz drift (eeg_drift) and a 200 µV step (eeg_spikes) each made a channel look "5× noisier", which duplicated the drift and discontinuity findings. It runs per analysis window, so it can localize a contact that comes and goes.
- **EEG muscle** uses integrated (summed) power, not mean PSD. The upper edge is min(45 Hz, Nyquist − 5 Hz). It is skipped when that leaves less than a 5 Hz band. Certainty is `likely`; the wording says "possible muscle".
- **ECG high-frequency noise** excludes ±2 Hz around 50/60 Hz and their harmonics. Pure power-line hum is the power-line detector's job and must not also count as broadband noise (tested). The high band runs up to Nyquist.
- **Ocular**
  - **Band:** it uses a 0.5–4 Hz band-pass (the spec says "< 4 Hz"), so slow drift below 0.5 Hz, which has its own detector, isn't called an eye artifact (tested with a 0.1 Hz frontal drift).
  - **Rule:** frontal (Fp/AF) ptp ≥ 100 µV **and** ≥ `ocular_frontal_ratio` × the median of the posterior channels (O/PO/P). If no posterior channels exist, the median of all non-frontal channels is used.
  - **Certainty:** `likely` when ≥ 2 frontal channels agree in the window (blinks are bilateral); otherwise `uncertain`. Never `definite`.
  - **Threshold 2× (was 3× in my first draft):** chosen on **simulated** data. At 3×, 116/240 simulated 150 µV blinks were detected; at 2×, 239/240. Clean EEG gave 0 false ocular findings in 30 recordings at either value. Phase 3 must re-check this on a separate tuning split of real data.
- **Electrode motion.** Indicators per ECG window: baseline drift, a spike/discontinuity/large transient, and "no beats detected while the nearest **non-overlapping** windows before and after have beats". Overlapping neighbours share the gap, so they can't serve as the comparison. A finding needs ≥ 2 indicators (`likely`). Beat-detection failure alone gives an `uncertain` finding. Drift alone or a jump alone is **not** reported again as electrode motion, since its own detector already reports it. The indicators come from findings **after** suppression, so a jump hidden inside a dropout doesn't count.
- **Missing data** stays off by default ("reject", as in Phase 1). With `missing="flag"` (loader argument, CLI `--missing flag`, app checkbox), each NaN/Inf run becomes a `definite` `missing_data` finding. The gap is bridged by a straight line so the other checks can run, and any other finding within the gap ± 0.5 s is suppressed. A channel with no valid samples is reported and skipped.
- **Physiological plausibility (ECG)**
  - **Beats:** NeuroKit2 R-peaks are computed once per channel and shared with the spike and electrode-motion checks.
  - **Rules:** each RR < 200 ms, each beat-free gap > 3 s (including from the recording edges), and each window whose median-RR heart rate falls outside 25–250 bpm.
  - **Exclusions:** gaps or windows that are ≥ 50% flat or missing are skipped, because the dropout is already reported.
  - **Labels and wording:** all hints are `uncertain`, with severity capped at `medium`. The wording is tested against a list of forbidden words.
- **Physiological plausibility (EEG):** windows with robust amplitude > 200 µV, merged into stretches; reported only if ≥ 2 s.

### Changes to Phase 1 behaviour found while building Phase 2
- **Clipping now also needs ≥ 2 separate flat runs at the extreme in the window** (`clip_min_runs = 2`). The electrode-motion demo exposed a false "clipping" finding. Every channel's single global minimum sat at the bottom of a slow, smooth trough, where 18 consecutive samples were within 0.1% of the extreme. Real clipping repeats peak after peak. A single long stuck stretch is constant, so the flatline detector reports it. With this rule, 20 seeds × 19 scenarios gave 0 clipping findings outside the clipping scenarios, and all clipping scenarios were still detected.
- **Muscle (EEG) and high-frequency noise (ECG) now suppress spike-type findings fully inside them**, as clipping does. An EMG burst is made of steep jumps, and without this rule it produced several "spike" findings for the same event. Trade-off: a genuine electrode pop inside a muscle burst is hidden.

### Demo scenarios added (Phase 2)
`ecg_muscle_noise` (20–150 Hz noise, 0.2 mV RMS, 8–16 s) · `ecg_electrode_motion` (0.05–0.5 Hz wander, 0.5 mV RMS, 18–24 s, plus two 0.8 mV tugs relaxing over 0.5 s) · `ecg_missing_data` (NaN 20–22 s; this demo uses the "flag" option) · `eeg_noisy_channel` (C4 + 120 µV RMS pink noise) · `eeg_muscle` (F3, 20–100 Hz, 45 µV RMS, 10–20 s) · `eeg_ocular` (four 150 µV blinks on Fp1/Fp2, weaker on F3/F4/C3/C4).
- **Robustness over 20 seeds × 19 scenarios:** 1 extra flag on clean signals (the same NeuroKit2 seed-5 slow ECG swing as in Phase 1). Electrode motion was missed in 1 of 20 seeds (the wander stayed under the drift threshold). 2 of 40 EEG pops were labelled `large_transient` (as in Phase 1). The app uses seed 42, where every injected artifact is found.

### Other
- **Uploaded files are labelled "Real recording" unless the user says they are simulated:** app checkbox (pre-ticked for file names starting with "simulated"), CLI `--simulated`. A loader can't know where the data came from.
- **`examples/`** holds three small simulated files (CSV, EDF, WFDB), made by `scripts/make_examples.py`, so the README commands run verbatim.
- **HTML report:** one page with metadata, findings, skipped checks, settings and an interactive waveform. Plotly is loaded from its CDN, which keeps the file around 0.3 MB instead of about 4 MB, so the waveform needs internet; the tables don't.
- **"Show processed signal"** overlays a 0.5–40 Hz band-pass copy **for display only**.
- **Large recordings** get a note above 30 min of data at more than 250 Hz. Measured on this Mac: 30 min of 2-lead ECG at 360 Hz took 1.3 s; 30 min of 19-channel EEG at 256 Hz took 15 s, with a peak memory of about 630 MB for the whole Python process.
- **CI:** `actions/checkout@v7` and `actions/setup-python@v7`; v4/v5 used the deprecated Node 20 runtime.
