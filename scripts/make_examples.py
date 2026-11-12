"""Write small SIMULATED example files (CSV, EDF, WFDB) into examples/ for trying the app and CLI.

Run:  python scripts/make_examples.py
All files are simulated signals, not real recordings. File names start with "simulated_".
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import wfdb

from signal_quality.demo import make_scenario

OUT = Path(__file__).resolve().parents[1] / "examples"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # CSV: mixed ECG scenario in millivolts, with a time column (enter 360 Hz and mV when uploading)
    rec, _ = make_scenario("mixed_ecg")
    t = np.arange(rec.n_samples) / rec.sfreq
    lines = ["time_s,ECG"] + [f"{a:.5f},{v * 1e3:.5f}" for a, v in zip(t, rec.data[0])]
    (OUT / "simulated_ecg_mixed_360Hz_mV.csv").write_text("\n".join(lines) + "\n")

    # EDF: EEG with a flat C3 channel (8–20 s)
    rec, _ = make_scenario("eeg_flat_channel")
    raw = mne.io.RawArray(rec.data, mne.create_info(rec.channel_names, rec.sfreq, "eeg"), verbose="error")
    mne.export.export_raw(str(OUT / "simulated_eeg_flat_channel.edf"), raw, fmt="edf", overwrite=True, verbose="error")

    # WFDB: two-lead ECG with electrode motion (millivolts)
    rec, _ = make_scenario("ecg_electrode_motion")
    sig = np.column_stack([rec.data[0] * 1e3, rec.data[0] * 0.6e3])
    wfdb.wrsamp(
        "simulated_ecg_motion", fs=int(rec.sfreq), units=["mV", "mV"], sig_name=["MLII", "V1"],
        p_signal=sig, fmt=["16", "16"], comments=["Simulated signal - not a real recording"], write_dir=str(OUT),
    )
    for path in sorted(OUT.iterdir()):
        print(f"wrote {path.relative_to(OUT.parent)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
