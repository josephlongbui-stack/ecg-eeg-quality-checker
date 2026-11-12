# DETECTORS — Detection rules (scientific core)

Every detector is a pure function: `detect(channel_data_volts, sfreq, channel_name, config) -> list[Finding]`.
It never modifies its input. Each one reports the measurement behind every flag.

> **Threshold policy.** The defaults below are reasonable starting points with a stated rationale, **not validated values**. They are configurable. Phase 3 tunes them on a *separate* tuning split, never on the final evaluation data. Record any change, with its reason, in `docs/DECISIONS.md`.

## Shared definitions
- **Window:** see `ARCHITECTURE.md` §6. Window features are computed on the **raw** signal (a copy), mean-removed unless stated otherwise.
- **MAD (median absolute deviation):** `median(|x − median(x)|)`. **Robust σ** = `1.4826 × MAD`. Robust statistics are used so a few extreme samples don't hide themselves by inflating the spread.
- **PSD:** Welch's method (`scipy.signal.welch`, Hann window, `nperseg = min(len(window), 2·sfreq)` → ~0.5 Hz resolution).
- **Nyquist rule:** a frequency check at `f` Hz runs only if `f + margin < sfreq / 2`. Otherwise it is **skipped** and recorded in `skipped_checks` with the reason. Never silently skipped.
- **Severity:** from `ratio = measurement / threshold` (or its inverse for "below" rules) and duration:
  `low` if ratio < 2 and duration < 2 s · `high` if ratio ≥ 5 or duration ≥ 10 s · otherwise `medium`.
- **Certainty:** defined per detector below. These are rule-based labels, **never probabilities**.

---

## 1. Flatline / disconnected channel / lead dropout
**Idea:** a real EEG/ECG electrode always shows some activity. A near-constant trace means disconnection, dropout or a dead channel.
**Feature:** window standard deviation (σ) and peak-to-peak (ptp) of the raw signal.
**Rule:** `σ < flat_std` **and** `ptp < flat_ptp`, sustained for at least `flat_min_duration_s` (merge consecutive windows; then refine edges to the sample level using a 0.25 s sliding σ).
| Param | EEG | ECG | Rationale |
|---|---|---|---|
| `flat_std` | 0.5 µV | 0.01 mV (10 µV) | Well below the background activity of a connected electrode in each modality |
| `flat_ptp` | 2 µV | 0.05 mV | Guards against a single tiny wiggle passing |
| `flat_min_duration_s` | 1.0 | 1.0 | Shorter flat stretches can occur naturally between ECG beats at low resolution |
**Certainty:** `definite` if ptp == 0 (exactly constant) · `likely` otherwise.
**Explanation template:** "Channel {ch} is nearly flat for {dur:.1f} s (σ = {std} {unit}), consistent with a disconnected electrode or dropout."
**Suppresses:** drift and spike findings fully inside the same interval (they are side effects).

## 2. Clipping / saturation
**Idea:** when the amplifier hits its limit, samples pile up at one exact max/min value, and the waveform tops look cut flat.
**Feature:** for each window, the fraction of samples within `clip_tolerance` of the channel's **global** max or min, plus the longest run of consecutive samples at that extreme.
**Rule:** `fraction_at_extreme ≥ clip_min_fraction` **and** `longest_run ≥ clip_min_run_samples`.
If the file declares a physical min/max (EDF header), use that range instead of the observed global extremes.
| Param | EEG | ECG | Rationale |
|---|---|---|---|
| `clip_tolerance` | 0.1% of channel range | same | Accounts for quantization |
| `clip_min_fraction` | 1% of window | 1% | Real signals touch their extreme only briefly |
| `clip_min_run_samples` | max(3, 0.01·sfreq) | same | A true peak lasts about 1 sample; clipping produces flat runs |
**Certainty:** `definite` if the samples sit at the declared physical limit from the file header · `likely` if they sit at the observed extreme with long runs · `uncertain` if the run is under 2× the minimum.
**Explanation:** "{pct}% of samples in this window sit at the channel's {max|min} value in flat runs up to {run} samples: likely amplifier saturation (clipping)."

## 3. Power-line interference (50/60 Hz)
**Idea:** mains electricity leaks into recordings as a narrow peak at 60 Hz (US) or 50 Hz (Europe).
**Feature:** Welch PSD of the raw window. `line_ratio = mean PSD in [f0 − 1, f0 + 1] Hz ÷ median PSD in [f0 − 6, f0 − 2] ∪ [f0 + 2, f0 + 6] Hz`.
**Important:** measure this **before** any notch filter. Never filter out the thing you are measuring.
**Rule:** `line_ratio ≥ line_ratio_threshold` in a window. Check `f0 = mains_hz` (default 60, configurable to 50, or `"both"`).
**Nyquist:** skip if `f0 + 6 ≥ sfreq/2`, e.g. EEGMMIDB at 160 Hz has Nyquist 80 Hz, so 60 Hz is OK; a 100 Hz recording → skipped and reported.
| Param | EEG | ECG | Rationale |
|---|---|---|---|
| `line_ratio_threshold` | 10 (10 dB) | 10 (10 dB) | A 10× narrow peak above neighbors is clearly not broadband physiology |
| `line_ratio_definite` | 100 (20 dB) | 100 | Overwhelming peak |
**Certainty:** `definite` if ratio ≥ 100 · `likely` if ≥ 10.
**Note:** if the whole recording shows a deep notch at f0 (ratio < 0.1), record an info note "a notch filter appears to have been applied". This is not a finding.
**Explanation:** "Power at {f0} Hz is {ratio:.0f}× higher than nearby frequencies: likely electrical power-line interference."

## 4. Sudden spikes and discontinuities
**Idea:** electrode pops, cable movement and data glitches cause jumps far steeper than physiology allows.
**Feature:** first difference `d[n] = x[n+1] − x[n]` (in volts per sample), across the whole channel.
**EEG rule:** `|d − median(d)| / robust_σ(d) > spike_z_eeg` (default **10**).
**ECG rule (different on purpose):** ECG R-peaks are naturally very steep, so a MAD-based z-score would flag every heartbeat. Instead:
`|d| > spike_factor_ecg × P99(|d|)` (default factor **3**, where P99 is the 99th percentile of |d| over the channel). Real QRS slopes sit at or below P99; artifacts that are 3× steeper are not normal beats.
- Then use NeuroKit2 (`nk.ecg_peaks`) to find R-peaks. If a flagged sample lies within 50 ms of a detected R-peak, set certainty to `uncertain` (it may be an unusual but genuine beat).
- If R-peak detection fails, do **not** treat that failure as an artifact or arrhythmia. Just skip the R-peak cross-check and note it.
**Classifying the event** (look 100 ms before and after):
- **spike:** the signal returns to within 25% of its prior level within 100 ms
- **discontinuity:** step change. The median level 100 ms after differs from 100 ms before by more than 5× robust_σ of the signal
- otherwise **large_transient**: don't guess a cause
**Merging:** flagged samples within 200 ms of each other form one event.
**Certainty:** `likely` by default · `definite` if the jump exceeds 2× the threshold **and** is a single-sample step (typical of data glitches) · `uncertain` per the R-peak rule.
**Explanation:** "An abrupt jump {k:.0f}× steeper than this channel's normal variation at {t:.2f} s: possible electrode pop or motion artifact."

## 5. Baseline drift / baseline wander
**Idea:** slow movement of the whole trace (sweat, breathing, loose electrode) under the real signal.
**Feature:** low-frequency component = zero-phase Butterworth low-pass (order 2, `sosfiltfilt`) applied to the **whole channel** (not per window, to avoid edge effects), with cutoff `drift_cutoff_hz`. Then per window: ptp of that component.
| Param | EEG | ECG | Rationale |
|---|---|---|---|
| `drift_cutoff_hz` | 0.3 Hz | 0.5 Hz | Below most EEG delta activity; standard ECG baseline-wander band |
| `drift_ptp` | 100 µV | 0.5 mV | Comparable to or larger than typical signal amplitude (EEG tens of µV; ECG QRS about 1 mV), so it meaningfully distorts the trace |
**Rule:** low-freq ptp ≥ `drift_ptp`. Skip, with a note, if the recording is under 3/`drift_cutoff_hz` seconds (too short to estimate slow drift).
**Certainty:** `likely` · `uncertain` if within 1 s of the recording start or end (filter edge effects).
**Explanation:** "The baseline wanders by {ptp} {unit} over this window (below {cutoff} Hz): likely baseline drift from movement, breathing or electrode contact."

---

## Phase 2 detectors (spec summary)
| Detector | Modality | Feature | Default rule |
|---|---|---|---|
| Noisy channel | EEG | channel robust σ vs median robust σ across channels | > 5× the median of other channels (needs ≥ 4 channels; otherwise skipped) |
| High-frequency / muscle noise | EEG | band power 20–(min(45, Nyquist−5)) Hz ÷ total 1–45 Hz power | ratio > 0.5 → "possible muscle contamination" |
| High-frequency noise | ECG | power > 40 Hz ÷ power 1–40 Hz | ratio > 0.3 → "possible muscle/electrical noise" |
| Electrode motion | ECG | windows with drift **and** spikes/large transients together, or R-peak detection failing where neighbors succeed | `likely` only when ≥ 2 indicators agree; otherwise `uncertain` |
| Ocular artifact | EEG | low-frequency (<4 Hz) large deflections that are larger in frontal channels (Fp1/Fp2/AF*) than posterior ones | only if frontal channels exist; label "possible ocular artifact"; never `definite` |
| Non-finite / missing data | both | NaN/Inf runs | `definite` "missing data" finding (instead of rejecting the file) |
| **Physiological plausibility (ECG)** | ECG | NeuroKit2 R-peaks per window → heart rate and RR intervals | HR < 25 or > 250 bpm, **or** any RR < 200 ms, **or** a gap > 3 s with no detected beat while the signal is not flat → category `implausible_rhythm_measurement`. Always `uncertain` |
| **Physiological plausibility (EEG)** | EEG | robust amplitude (1.4826·MAD) per window | > 200 µV sustained ≥ 2 s → category `implausible_amplitude`. Always `uncertain` |

### About the physiological plausibility checks (Joseph's original idea, added back 2026-11-12)
These bring back the original concept of checking signals against known physiology (heart-rate ranges, beat timing, amplitude ranges). They are **hints, not verdicts**:
- Always certainty `uncertain` and severity at most `medium`.
- Explanation must say that both a recording problem **and** genuine but unusual physiology are possible, e.g. "Detected beat timing implies 310 bpm, which is outside the plausible range. This usually means beat detection was confused by noise, but unusual rhythms are possible. Review this interval."
- Never use words like arrhythmia, tachycardia, diagnosis, abnormal or fake.
- If NeuroKit2 fails to find beats, record that as context (`skipped_checks`), not as a finding by itself.
- Ranges are configurable with rationale comments (25–250 bpm covers extreme bradycardia to extreme exercise/pathological tachycardia; RR < 200 ms is shorter than the heart's refractory period; EEG > 200 µV sustained is well above typical scalp EEG background).
- Phase 3 should report how often these hints fire on clean vs contaminated NSTDB segments, so their usefulness is measured rather than assumed.

## De-duplication (`merge.py`)
1. Merge same-category findings on the same channel whose intervals overlap or sit within 0.5 s of each other. Keep the max severity and the strongest measurement.
2. Flatline suppresses drift/spike/clipping inside its interval.
3. Clipping suppresses spike findings inside its interval.
4. Different categories that overlap otherwise stay as separate findings (they are different evidence).

## Optional ML layer (Phase 4)
Isolation Forest on per-window feature vectors (σ, ptp, MAD, line_ratio, band powers, max |d| z, drift ptp). The output is an **anomaly score**, labelled "ML anomaly score (uncalibrated)". It is never a probability. Shown only when a trained model with documented training data exists. Rules remain the default.
