"""Windowing and feature helpers."""

from __future__ import annotations

import numpy as np

from signal_quality.features import line_ratio, longest_run, mad, robust_sigma, runs_of_true, welch_psd
from signal_quality.preprocessing import iter_windows, lowpass


def test_window_counts_and_overlap():
    windows = iter_windows(n_samples=256 * 10, sfreq=256, window_s=2.0, overlap=0.5)
    assert len(windows) == 9  # starts 0,1,...,8 s
    assert windows[0] == (0, 512)
    assert windows[1][0] == 256  # 50% overlap
    assert windows[-1][1] == 256 * 10


def test_window_tail_is_covered():
    windows = iter_windows(n_samples=1000, sfreq=100, window_s=3.0, overlap=0.5)
    assert windows[-1] == (700, 1000)


def test_short_recording_is_one_window():
    assert iter_windows(n_samples=300, sfreq=256, window_s=2.0, overlap=0.5) == [(0, 300)]


def test_robust_stats():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert mad(x) == 1.0
    assert np.isclose(robust_sigma(x), 1.4826)


def test_runs():
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert runs_of_true(mask) == [(1, 3), (4, 7)]
    assert longest_run(mask) == 3


def test_line_ratio_detects_60hz():
    fs = 256
    t = np.arange(fs * 4) / fs
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, len(t))
    f, p = welch_psd(noise, fs)
    assert line_ratio(f, p, 60) < 5
    f, p = welch_psd(noise + 2 * np.sin(2 * np.pi * 60 * t), fs)
    assert line_ratio(f, p, 60) > 100


def test_lowpass_returns_copy():
    x = np.random.default_rng(0).normal(size=2000)
    before = x.copy()
    y = lowpass(x, 256, 1.0)
    assert np.array_equal(x, before)
    assert y is not x
