"""Simulated demo signals with artifacts injected on purpose (known ground truth).

Everything here is SIMULATED: "Simulated signal — not a real recording".
All findings shown for these signals still come from the real analysis engine.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from .models import InjectedArtifact, Recording

SIMULATED_LABEL = "Simulated signal — not a real recording"

ECG_SFREQ = 360.0
ECG_DURATION_S = 30.0
ECG_CHANNEL = "ECG"
EEG_SFREQ = 256.0
EEG_DURATION_S = 30.0
EEG_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]
# Alpha (10 Hz) amplitude per channel in µV: strongest over the occipital channels.
EEG_ALPHA_UV = {"Fp1": 2, "Fp2": 2, "F3": 3, "F4": 3, "C3": 5, "C4": 5, "O1": 12, "O2": 12}


# ---------------------------------------------------------------- base signals
def _ecg_base(seed: int) -> np.ndarray:
    """30 s, 360 Hz, HR 70 bpm ECG from NeuroKit2 (output in mV) -> volts, shape (1, n)."""
    import neurokit2 as nk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ecg_mV = nk.ecg_simulate(
            duration=int(ECG_DURATION_S),
            sampling_rate=int(ECG_SFREQ),
            heart_rate=70,
            noise=0.01,
            random_state=seed,
        )
    return (np.asarray(ecg_mV, dtype=float) * 1e-3)[np.newaxis, :]


def _pink_noise(
    n: int, sfreq: float, rng: np.random.Generator, f_lo: float = 0.5, f_hi: float = 100.0, knee_hz: float = 30.0
) -> np.ndarray:
    """Unit-RMS EEG-like background noise.

    Power ∝ 1/f ("pink") between f_lo and knee_hz, falling off faster (∝ 1/f³) above the
    knee as real scalp EEG does, and zero outside [f_lo, f_hi] (like hardware filtering).
    """
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    shape = np.zeros_like(freqs)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    f = freqs[band]
    shape[band] = 1 / np.sqrt(f) / np.sqrt(1 + (f / knee_hz) ** 2)
    noise = np.fft.irfft(spectrum * shape, n)
    return noise / np.sqrt(np.mean(noise**2))


def _eeg_base(seed: int) -> np.ndarray:
    """8-channel EEG-like signal: pink noise (~20 µV RMS) + amplifier noise + 10 Hz alpha, in volts."""
    rng = np.random.default_rng(seed)
    n = int(EEG_DURATION_S * EEG_SFREQ)
    t = np.arange(n) / EEG_SFREQ
    data = np.empty((len(EEG_CHANNELS), n))
    for i, ch in enumerate(EEG_CHANNELS):
        # ~20 µV RMS brain background + 0.5 µV RMS white amplifier noise floor
        background = 20.0 * _pink_noise(n, EEG_SFREQ, rng) + 0.5 * rng.standard_normal(n)
        # alpha waxes and wanes slowly (amplitude modulation keeps power near 10 Hz)
        envelope = 1 + 0.4 * np.sin(2 * np.pi * 0.15 * t + rng.uniform(0, 2 * np.pi))
        alpha = EEG_ALPHA_UV[ch] * envelope * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
        data[i] = (background + alpha) * 1e-6
    return data


def _time(n: int, sfreq: float) -> np.ndarray:
    return np.arange(n) / sfreq


def _idx(t_s: float, sfreq: float) -> int:
    return int(round(t_s * sfreq))


# ---------------------------------------------------------------- injections
def _add_sine(x: np.ndarray, sfreq: float, freq: float, amp_V: float, start_s: float, end_s: float) -> None:
    t = _time(len(x), sfreq)
    mask = (t >= start_s) & (t < end_s)
    x[mask] += amp_V * np.sin(2 * np.pi * freq * t[mask])


def _hold_flat(x: np.ndarray, sfreq: float, start_s: float, end_s: float) -> None:
    """Dropout: the trace freezes at its last value (no artificial jump at the start)."""
    a, b = _idx(start_s, sfreq), _idx(end_s, sfreq)
    x[a:b] = x[a]


def _saturate(x: np.ndarray, sfreq: float, start_s: float, end_s: float, gain: float) -> float:
    """Simulated amplifier saturation.

    During [start, end) the signal surges by `gain` (smooth 0.1 s ramps inside the
    interval), then the WHOLE trace is clipped at one fixed amplifier limit placed
    5% above the clean signal's largest excursion. Real amplifiers clip at a fixed
    limit, so only the surge reaches it. Returns the limit in volts.
    """
    limit = 1.05 * float(np.max(np.abs(x)))
    t = _time(len(x), sfreq)
    ramp = 0.1
    env = np.clip(np.minimum((t - start_s) / ramp, (end_s - t) / ramp), 0, 1)
    x *= 1 + (gain - 1) * env
    np.clip(x, -limit, limit, out=x)
    return limit


def _single_spike(x: np.ndarray, sfreq: float, t_s: float, amp_V: float) -> None:
    x[_idx(t_s, sfreq)] += amp_V


def _pop(x: np.ndarray, sfreq: float, t_s: float, amp_V: float, tau_s: float = 0.015) -> None:
    """Electrode pop: instant jump that decays back exponentially."""
    t = _time(len(x), sfreq)
    mask = t >= t_s
    x[mask] += amp_V * np.exp(-(t[mask] - t_s) / tau_s)


def _step(x: np.ndarray, sfreq: float, t_s: float, amp_V: float) -> None:
    x[_idx(t_s, sfreq) :] += amp_V


def _band_noise(n: int, sfreq: float, lo: float, hi: float, rms: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise limited to [lo, hi] Hz with the given RMS (e.g. muscle/EMG-like noise)."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    spectrum[(freqs < lo) | (freqs > hi)] = 0
    y = np.fft.irfft(spectrum, n)
    return rms * y / np.sqrt(np.mean(y**2))


def _add_burst(x: np.ndarray, sfreq: float, noise: np.ndarray, start_s: float, end_s: float) -> None:
    """Add `noise` only during [start, end) with 0.1 s fade-in/out (no artificial steps)."""
    t = _time(len(x), sfreq)
    env = np.clip(np.minimum((t - start_s) / 0.1, (end_s - t) / 0.1), 0, 1)
    x += env * noise


def _blink(x: np.ndarray, sfreq: float, t_s: float, amp_V: float, width_s: float = 0.07) -> None:
    """Eye blink: a smooth positive bump (~0.3 s wide) strongest at the frontal-polar electrodes."""
    t = _time(len(x), sfreq)
    x += amp_V * np.exp(-0.5 * ((t - t_s) / width_s) ** 2)


# ---------------------------------------------------------------- scenarios
def _ecg(seed: int, kind: str) -> tuple[np.ndarray, list[InjectedArtifact]]:
    data = _ecg_base(seed)
    x = data[0]
    fs = ECG_SFREQ
    truth: list[InjectedArtifact] = []
    one = 1 / fs
    if kind in ("powerline", "mixed"):
        a, b = (10.0, 20.0) if kind == "powerline" else (3.0, 8.0)
        _add_sine(x, fs, 60.0, 0.1e-3, a, b)
        truth.append(InjectedArtifact(ECG_CHANNEL, a, b, "powerline", {"freq_hz": 60, "amplitude_mV": 0.1}))
    if kind == "baseline_wander":
        _add_sine(x, fs, 0.25, 1.0e-3, 0.0, ECG_DURATION_S)
        truth.append(InjectedArtifact(ECG_CHANNEL, 0.0, ECG_DURATION_S, "baseline_drift", {"freq_hz": 0.25, "amplitude_mV": 1.0}))
    if kind in ("lead_dropout", "mixed"):
        a, b = (12.0, 15.0) if kind == "lead_dropout" else (15.0, 18.0)
        _hold_flat(x, fs, a, b)
        truth.append(InjectedArtifact(ECG_CHANNEL, a, b, "flatline", {"mode": "hold last value"}))
    if kind == "clipping":
        limit = _saturate(x, fs, 5.0, 9.0, gain=2.5)
        truth.append(InjectedArtifact(ECG_CHANNEL, 5.0, 9.0, "clipping", {"gain": 2.5, "limit_mV": limit * 1e3}))
    if kind == "muscle_noise":
        rng = np.random.default_rng(seed + 1)
        _add_burst(x, fs, _band_noise(len(x), fs, 20.0, 150.0, 0.2e-3, rng), 8.0, 16.0)
        truth.append(InjectedArtifact(ECG_CHANNEL, 8.0, 16.0, "hf_noise", {"band_hz": [20, 150], "rms_mV": 0.2}))
    if kind == "electrode_motion":
        rng = np.random.default_rng(seed + 2)
        wander = _band_noise(len(x), fs, 0.05, 0.5, 0.5e-3, rng)
        _add_burst(x, fs, wander, 18.0, 24.0)
        for t0 in (19.5, 22.0):  # cable tugs: sudden shift that relaxes over ~0.5 s
            t = _time(len(x), fs)
            m = t >= t0
            x[m] += 0.8e-3 * np.exp(-(t[m] - t0) / 0.5)
        truth.append(InjectedArtifact(ECG_CHANNEL, 18.0, 24.0, "electrode_motion", {"wander_rms_mV": 0.5, "jumps_mV": 0.8}))
    if kind == "missing":
        a, b = _idx(20.0, fs), _idx(22.0, fs)
        x[a:b] = np.nan
        truth.append(InjectedArtifact(ECG_CHANNEL, 20.0, 22.0, "missing_data", {"value": "NaN"}))
    if kind in ("spikes", "mixed"):
        times = [6.2, 11.7, 18.4, 25.1] if kind == "spikes" else [10.5, 22.3]
        for t in times:
            _single_spike(x, fs, t, 3.0e-3)
            truth.append(InjectedArtifact(ECG_CHANNEL, t, t + one, "spike", {"amplitude_mV": 3.0, "width_samples": 1}))
    return data, truth


def _eeg(seed: int, kind: str) -> tuple[np.ndarray, list[InjectedArtifact]]:
    data = _eeg_base(seed)
    fs = EEG_SFREQ
    ch = {name: i for i, name in enumerate(EEG_CHANNELS)}
    truth: list[InjectedArtifact] = []
    one = 1 / fs
    if kind == "flat_channel":
        _hold_flat(data[ch["C3"]], fs, 8.0, 20.0)
        truth.append(InjectedArtifact("C3", 8.0, 20.0, "flatline", {"mode": "hold last value"}))
    if kind == "powerline":
        for name in EEG_CHANNELS:
            _add_sine(data[ch[name]], fs, 60.0, 15e-6, 5.0, 15.0)
            truth.append(InjectedArtifact(name, 5.0, 15.0, "powerline", {"freq_hz": 60, "amplitude_uV": 15}))
    if kind == "spikes":
        _step(data[ch["F4"]], fs, 12.0, 200e-6)
        truth.append(InjectedArtifact("F4", 12.0, 12.0 + one, "discontinuity", {"step_uV": 200}))
        for t in (7.3, 21.6):
            _pop(data[ch["O1"]], fs, t, 150e-6)
            truth.append(InjectedArtifact("O1", t, t + one, "spike", {"amplitude_uV": 150, "decay_ms": 15}))
    if kind == "drift":
        for name in ("Fp1", "Fp2"):
            _add_sine(data[ch[name]], fs, 0.1, 250e-6, 0.0, EEG_DURATION_S)
            truth.append(InjectedArtifact(name, 0.0, EEG_DURATION_S, "baseline_drift", {"freq_hz": 0.1, "amplitude_uV": 250}))
    if kind == "clipping":
        limit = _saturate(data[ch["O2"]], fs, 18.0, 24.0, gain=3.0)
        truth.append(InjectedArtifact("O2", 18.0, 24.0, "clipping", {"gain": 3.0, "limit_uV": limit * 1e6}))
    if kind == "noisy_channel":
        rng = np.random.default_rng(seed + 3)
        data[ch["C4"]] += 120e-6 * _pink_noise(data.shape[1], fs, rng)  # loose electrode: broadband noise
        truth.append(InjectedArtifact("C4", 0.0, EEG_DURATION_S, "noisy_channel", {"extra_noise_rms_uV": 120}))
    if kind == "muscle":
        rng = np.random.default_rng(seed + 4)
        _add_burst(data[ch["F3"]], fs, _band_noise(data.shape[1], fs, 20.0, 100.0, 45e-6, rng), 10.0, 20.0)
        truth.append(InjectedArtifact("F3", 10.0, 20.0, "muscle", {"band_hz": [20, 100], "rms_uV": 45}))
    if kind == "ocular":
        for t0 in (4.0, 9.5, 16.2, 23.8):
            for name, scale in (("Fp1", 1.0), ("Fp2", 1.0), ("F3", 0.35), ("F4", 0.35), ("C3", 0.1), ("C4", 0.1)):
                _blink(data[ch[name]], fs, t0, 150e-6 * scale)
            for name in ("Fp1", "Fp2"):
                truth.append(InjectedArtifact(name, t0 - 0.25, t0 + 0.25, "ocular", {"blink_amplitude_uV": 150}))
    return data, truth


SCENARIOS: dict[str, tuple[str, str, Callable[[int], tuple[np.ndarray, list[InjectedArtifact]]]]] = {
    "ecg_clean": ("ECG — clean", "ECG", lambda s: _ecg(s, "clean")),
    "ecg_powerline": ("ECG — power-line interference", "ECG", lambda s: _ecg(s, "powerline")),
    "ecg_baseline_wander": ("ECG — baseline wander", "ECG", lambda s: _ecg(s, "baseline_wander")),
    "ecg_lead_dropout": ("ECG — lead dropout", "ECG", lambda s: _ecg(s, "lead_dropout")),
    "ecg_clipping": ("ECG — clipping", "ECG", lambda s: _ecg(s, "clipping")),
    "ecg_spikes": ("ECG — spikes", "ECG", lambda s: _ecg(s, "spikes")),
    "eeg_clean": ("EEG — clean", "EEG", lambda s: _eeg(s, "clean")),
    "eeg_flat_channel": ("EEG — flat channel", "EEG", lambda s: _eeg(s, "flat_channel")),
    "eeg_powerline": ("EEG — power-line interference", "EEG", lambda s: _eeg(s, "powerline")),
    "eeg_spikes": ("EEG — spikes and a discontinuity", "EEG", lambda s: _eeg(s, "spikes")),
    "eeg_drift": ("EEG — baseline drift", "EEG", lambda s: _eeg(s, "drift")),
    "eeg_clipping": ("EEG — clipping", "EEG", lambda s: _eeg(s, "clipping")),
    "mixed_ecg": ("ECG — mixed (power line + spikes + dropout)", "ECG", lambda s: _ecg(s, "mixed")),
    # Phase 2 scenarios
    "ecg_muscle_noise": ("ECG — muscle / high-frequency noise", "ECG", lambda s: _ecg(s, "muscle_noise")),
    "ecg_electrode_motion": ("ECG — electrode motion", "ECG", lambda s: _ecg(s, "electrode_motion")),
    "ecg_missing_data": ("ECG — missing data (NaN gap)", "ECG", lambda s: _ecg(s, "missing")),
    "eeg_noisy_channel": ("EEG — noisy channel", "EEG", lambda s: _eeg(s, "noisy_channel")),
    "eeg_muscle": ("EEG — muscle contamination", "EEG", lambda s: _eeg(s, "muscle")),
    "eeg_ocular": ("EEG — eye blinks (ocular)", "EEG", lambda s: _eeg(s, "ocular")),
}


def list_scenarios() -> list[dict[str, str]]:
    """All demo scenarios as [{"id", "label", "signal_type"}]."""
    return [{"id": sid, "label": label, "signal_type": st} for sid, (label, st, _) in SCENARIOS.items()]


def make_scenario(scenario_id: str, seed: int = 42) -> tuple[Recording, list[InjectedArtifact]]:
    """Build one simulated scenario. Returns (Recording, ground-truth injected artifacts)."""
    if scenario_id not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario_id}'. Choose one of: {', '.join(SCENARIOS)}")
    label, signal_type, builder = SCENARIOS[scenario_id]
    data, truth = builder(seed)
    sfreq = ECG_SFREQ if signal_type == "ECG" else EEG_SFREQ
    channels = [ECG_CHANNEL] if signal_type == "ECG" else list(EEG_CHANNELS)
    recording = Recording(
        signal_type=signal_type,
        data=data,
        sfreq=sfreq,
        channel_names=channels,
        source="simulated",
        data_kind="simulated",
        meta={
            "scenario_id": scenario_id,
            "scenario_label": label,
            "duration_s": data.shape[1] / sfreq,
            "original_unit": "V (simulated)",
            "seed": seed,
            "label": SIMULATED_LABEL,
            # the NaN-gap demo shows the "mark missing data" option; others never contain NaN
            "missing_policy": "flag" if scenario_id == "ecg_missing_data" else "reject",
        },
    )
    return recording, truth


def artifact_detected(findings, artifact: InjectedArtifact, tolerance_s: float = 0.0) -> bool:
    """True if some finding on the same channel and category overlaps the injected interval.

    Used to check the demo (and later evaluation); it does not detect anything itself.
    """
    for f in findings:
        if f.channel != artifact.channel or f.category != artifact.category:
            continue
        if f.start_s <= artifact.end_s + tolerance_s and f.end_s >= artifact.start_s - tolerance_s:
            return True
    return False
