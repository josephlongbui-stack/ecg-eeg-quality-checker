# Data

No data is committed to this repository (`data/*` is git-ignored). Phase 1 uses only **simulated** signals generated in code (`signal_quality.demo`).

Later phases (Phase 3) will use these public PhysioNet datasets:

| Dataset | Use | Notes |
|---|---|---|
| [MIT-BIH Noise Stress Test Database (nstdb)](https://physionet.org/content/nstdb/1.0.0/) | ECG experiments with controlled noise | Records are built from only two clean records (118, 119). The noise versions are **not** independent subjects. The `.atr` files are **beat** annotations, **not** artifact labels. |
| [EEG Motor Movement/Imagery Dataset (eegmmidb)](https://physionet.org/content/eegmmidb/1.0.0/) | Real EEG loading + controlled artifact injection | Has no artifact ground truth. |

Planned download commands (not run yet; Phase 3 adds `scripts/download_data.py`):

```bash
source .venv/bin/activate
python -c "import wfdb; wfdb.dl_database('nstdb', dl_dir='data/nstdb')"
python -c "import mne; mne.datasets.eegbci.load_data(subjects=[1,2,3,4,5], runs=[1,2], path='data/eegmmidb')"
```
