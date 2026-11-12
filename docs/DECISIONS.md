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
