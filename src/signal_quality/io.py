"""Loading and validating recordings (CSV and EDF/EDF+).

Every loader returns a Recording in VOLTS. Problems raise ValidationError with
a friendly message that says what is wrong and what to do.
"""

from __future__ import annotations

import csv
import io as _stdio
import os
import tempfile
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from .models import Recording

MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB, matches .streamlit/config.toml
MIN_DURATION_S = 1.0
UNIT_TO_VOLTS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6}
TIME_COLUMNS = {"time", "time_s"}
SIGNAL_TYPES = ("ECG", "EEG")


class ValidationError(ValueError):
    """Raised when an input file or recording cannot be analyzed safely."""


# ---------------------------------------------------------------- helpers
def _read_bytes(file: Any) -> tuple[bytes, str]:
    """Accept a path, raw bytes or a file-like object (e.g. Streamlit upload)."""
    if isinstance(file, (str, Path)):
        path = Path(file)
        if not path.exists():
            raise ValidationError(f"File not found: {path.name}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValidationError("File is larger than 50 MB. Please use a shorter excerpt.")
        return path.read_bytes(), path.name
    if isinstance(file, (bytes, bytearray)):
        return bytes(file), "uploaded file"
    name = getattr(file, "name", "uploaded file")
    if hasattr(file, "getvalue"):
        content = file.getvalue()
    else:
        content = file.read()
    if isinstance(content, str):
        content = content.encode("utf-8")
    return content, os.path.basename(str(name))


def _check_size(content: bytes) -> None:
    if len(content) == 0:
        raise ValidationError("The file is empty. Please choose a file that contains data.")
    if len(content) > MAX_FILE_BYTES:
        raise ValidationError("File is larger than 50 MB. Please use a shorter excerpt.")


def _check_signal_type(signal_type: str | None) -> str:
    if signal_type not in SIGNAL_TYPES:
        raise ValidationError("Signal type is required: choose ECG or EEG.")
    return signal_type


def unit_to_volts(unit: str | None) -> float:
    """Scale factor from a unit string (V, mV, uV/µV) to volts."""
    if unit is None or str(unit).strip() == "":
        raise ValidationError("Unit is required for CSV files. Choose V, mV or µV in the sidebar.")
    key = str(unit).strip()
    if key not in UNIT_TO_VOLTS:
        raise ValidationError(f"Unknown unit '{unit}'. Accepted units are V, mV and µV (or uV).")
    return UNIT_TO_VOLTS[key]


def suggest_signal_type(channel_names: list[str]) -> str:
    """Suggest 'ECG' if any channel name mentions ECG/EKG, else 'EEG'. Only a suggestion."""
    for name in channel_names:
        upper = str(name).upper()
        if "ECG" in upper or "EKG" in upper:
            return "ECG"
    return "EEG"


def peek_channel_names(file: Any) -> list[str]:
    """Read only the channel names of a CSV or EDF file (used for suggestions in the app)."""
    content, name = _read_bytes(file)
    _check_size(content)
    if name.lower().endswith(".edf"):
        raw = _read_edf_raw(content, preload=False)
        return list(raw.ch_names)
    header = next(csv.reader(_stdio.StringIO(content.decode("utf-8-sig", errors="replace"))), [])
    return [h.strip() for h in header if h.strip().lower() not in TIME_COLUMNS]


# ---------------------------------------------------------------- validation
def validate(recording: Recording) -> Recording:
    """Check a Recording is safe to analyze. Returns it unchanged or raises ValidationError."""
    if recording.signal_type not in SIGNAL_TYPES:
        raise ValidationError("Signal type must be ECG or EEG.")
    if not np.isfinite(recording.sfreq) or recording.sfreq <= 0:
        raise ValidationError("Sampling rate must be a positive number of samples per second (Hz).")
    data = recording.data
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValidationError("The recording contains no samples.")
    if len(recording.channel_names) != data.shape[0]:
        raise ValidationError(
            f"There are {data.shape[0]} data channels but {len(recording.channel_names)} channel names."
        )
    if len(set(recording.channel_names)) != len(recording.channel_names):
        dupes = sorted({c for c in recording.channel_names if recording.channel_names.count(c) > 1})
        raise ValidationError(f"Duplicate channel names: {dupes}. Each channel needs a unique name.")
    bad = ~np.isfinite(data)
    if bad.any():
        ch_idx, sample_idx = np.argwhere(bad)[0]
        raise ValidationError(
            f"Found {int(bad.sum())} missing or infinite values (NaN/Inf). The first is in channel "
            f"'{recording.channel_names[ch_idx]}' at sample {int(sample_idx)} "
            f"({sample_idx / recording.sfreq:.3f} s). Please clean or remove them and try again."
        )
    if recording.duration_s < MIN_DURATION_S:
        raise ValidationError(
            f"The recording is {recording.duration_s:.2f} s long; at least {MIN_DURATION_S:.0f} s is needed."
        )
    return recording


# ---------------------------------------------------------------- CSV
def load_csv(
    file: Any,
    sfreq: float | None,
    unit: str | None,
    signal_type: str | None,
) -> Recording:
    """Load a CSV: one column per channel, header row = channel names.

    An optional first column named `time` / `time_s` is not analyzed; it is only
    used to sanity-check the sampling rate (a warning if off by more than 1%).
    The sampling rate and unit must be given by the user; they are never guessed.
    """
    signal_type = _check_signal_type(signal_type)
    if sfreq is None:
        raise ValidationError("Sampling rate is required for CSV files. Enter it in the sidebar (e.g., 256).")
    try:
        sfreq = float(sfreq)
    except (TypeError, ValueError):
        raise ValidationError("Sampling rate must be a number (e.g., 256).") from None
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValidationError("Sampling rate must be a positive number (e.g., 256).")
    scale = unit_to_volts(unit)

    content, name = _read_bytes(file)
    _check_size(content)
    text = content.decode("utf-8-sig", errors="replace")
    header = next(csv.reader(_stdio.StringIO(text)), [])
    header = [h.strip() for h in header]
    if not header or all(h == "" for h in header):
        raise ValidationError("The CSV has no header row. The first row must contain channel names.")
    if any(h == "" for h in header):
        raise ValidationError("Every column in the CSV header needs a channel name.")
    if len(set(header)) != len(header):
        dupes = sorted({h for h in header if header.count(h) > 1})
        raise ValidationError(f"Duplicate channel names in the CSV header: {dupes}.")

    try:
        df = pd.read_csv(_stdio.StringIO(text))
    except Exception as exc:  # pandas raises several error types for malformed CSV
        raise ValidationError(f"Could not read the CSV file: {exc}") from None
    if df.shape[0] == 0:
        raise ValidationError("The CSV has a header but no data rows.")
    df.columns = header

    warnings: list[str] = []
    time_col = header[0] if header[0].lower() in TIME_COLUMNS else None
    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(t)
        if len(dt) > 0 and np.all(np.isfinite(dt)) and np.median(dt) > 0:
            implied = 1.0 / float(np.median(dt))
            if abs(implied - sfreq) / sfreq > 0.01:
                warnings.append(
                    f"The time column implies about {implied:.2f} Hz, but you entered {sfreq:g} Hz. "
                    "The entered sampling rate was used."
                )
        df = df.drop(columns=[time_col])
    if df.shape[1] == 0:
        raise ValidationError("The CSV has no signal columns (only a time column).")

    numeric = df.apply(pd.to_numeric, errors="coerce")
    non_numeric = numeric.isna() & df.notna()
    if non_numeric.to_numpy().any():
        row, col = np.argwhere(non_numeric.to_numpy())[0]
        raise ValidationError(
            f"Non-numeric value '{df.iat[row, col]}' in column '{df.columns[col]}' (data row {row + 1})."
        )
    values = numeric.to_numpy(dtype=float).T * scale  # (n_channels, n_samples), volts

    recording = Recording(
        signal_type=signal_type,
        data=values,
        sfreq=sfreq,
        channel_names=list(df.columns),
        source="upload:csv",
        data_kind="real",
        meta={
            "file_name": name,
            "original_unit": str(unit),
            "duration_s": values.shape[1] / sfreq,
            "warnings": warnings,
        },
    )
    return validate(recording)


# ---------------------------------------------------------------- EDF
def _read_edf_raw(content: bytes, preload: bool):
    import mne

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload.edf")
        with open(path, "wb") as fh:
            fh.write(content)
        try:
            raw = mne.io.read_raw_edf(path, preload=preload, verbose="error")
        except Exception as exc:
            raise ValidationError(f"Could not read this EDF file: {exc}") from None
        if preload:
            raw.load_data()
        return raw


def load_edf(file: Any, signal_type: str | None) -> Recording:
    """Load an EDF/EDF+ file with MNE (which returns volts).

    Patient-identifying header fields (name, birthdate, IDs, recording date) are
    never copied into the Recording. Only signal data, channel names, the sampling
    rate and the declared physical range (for clipping checks) are kept.
    """
    signal_type = _check_signal_type(signal_type)
    content, name = _read_bytes(file)
    _check_size(content)
    raw = _read_edf_raw(content, preload=True)

    types = raw.get_channel_types()
    keep = [i for i, t in enumerate(types) if t in ("eeg", "ecg")]
    if not keep:
        raise ValidationError("This EDF file contains no EEG or ECG channels.")
    data = raw.get_data(picks=keep)  # volts
    ch_names = [raw.ch_names[i] for i in keep]

    physical_range: dict[str, list[float]] = {}
    try:  # declared physical min/max per channel, converted to volts
        extras = raw._raw_extras[0]
        for i in keep:
            unit = float(extras["units"][i])
            physical_range[raw.ch_names[i]] = [
                float(extras["physical_min"][i]) * unit,
                float(extras["physical_max"][i]) * unit,
            ]
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        physical_range = {}

    sfreq = float(raw.info["sfreq"])
    recording = Recording(
        signal_type=signal_type,
        data=data,
        sfreq=sfreq,
        channel_names=ch_names,
        source="upload:edf",
        data_kind="real",
        meta={
            "file_name": name,
            "original_unit": "V (EDF physical units, converted by MNE)",
            "duration_s": data.shape[1] / sfreq,
            "physical_range_V": physical_range,
            "warnings": [],
        },
    )
    return validate(recording)


def load_file(file: Any, signal_type: str | None, sfreq: float | None = None, unit: str | None = None) -> Recording:
    """Dispatch on file extension (.csv or .edf)."""
    name = str(getattr(file, "name", file if isinstance(file, (str, Path)) else ""))
    lower = name.lower()
    if lower.endswith(".csv"):
        return load_csv(file, sfreq=sfreq, unit=unit, signal_type=signal_type)
    if lower.endswith(".edf"):
        return load_edf(file, signal_type=signal_type)
    raise ValidationError("Unsupported file type. Please upload a .csv or .edf file.")
