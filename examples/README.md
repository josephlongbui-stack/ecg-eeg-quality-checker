# Example files (all SIMULATED)

These are **simulated signals, not real recordings**, made by `python scripts/make_examples.py` from the package's demo generator. Use them to try uploads in the app or the `sqc` command line.

| File | Format | What's inside | Upload settings |
|---|---|---|---|
| `simulated_ecg_mixed_360Hz_mV.csv` | CSV (time column + 1 lead) | 60 Hz hum 3–8 s, spikes at 10.5 s and 22.3 s, dropout 15–18 s | ECG, **360 Hz**, **mV** |
| `simulated_eeg_flat_channel.edf` | EDF | 8-channel EEG; C3 flat from 8 to 20 s | EEG |
| `simulated_ecg_motion.hea` + `.dat` | WFDB | 2-lead ECG; electrode motion 18–24 s | ECG (upload both files together) |
