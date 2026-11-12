"""Command-line interface: `sqc`. Uses the same engine as the app (analyze_recording).

Examples:
    sqc analyze recording.edf --type EEG --out results/
    sqc analyze recording.csv --type ECG --sfreq 360 --unit mV --out results/
    sqc analyze examples/simulated_ecg_mixed_360Hz_mV.csv --type ECG --sfreq 360 --unit mV --simulated
    sqc analyze 100.hea --type ECG --out results/          # WFDB (the .dat must sit next to it)
    sqc demo --list
    sqc demo ecg_powerline --out results/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .analyze import analyze_recording
from .demo import list_scenarios, make_scenario
from .io import ValidationError, load_file, mark_simulated
from .models import AnalysisResult
from .reporting import to_csv_bytes, to_dataframe, to_html_bytes, to_json_bytes


def _parse_overrides(config_file: str | None, sets: list[str] | None, mains: str | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if config_file:
        path = Path(config_file)
        if not path.exists():
            raise ValidationError(f"Config file not found: {config_file}")
        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValidationError("The config file must contain key: value pairs (YAML or JSON).")
        overrides.update(loaded)
    for item in sets or []:
        if "=" not in item:
            raise ValidationError(f"--set expects key=value, got '{item}'.")
        key, value = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(value)
    if mains:
        overrides["mains_hz"] = str(mains)
    if "mains_hz" in overrides:
        overrides["mains_hz"] = str(overrides["mains_hz"])
    return overrides


def _stub(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-") or "recording"


def _write_outputs(result: AnalysisResult, out_dir: Path, stem: str, recording=None, injected=None, html: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{_stub(stem)}_{result.recording_summary['signal_type']}"
    paths = [out_dir / f"{base}_findings.csv", out_dir / f"{base}_results.json"]
    paths[0].write_bytes(to_csv_bytes(result))
    paths[1].write_bytes(to_json_bytes(result))
    if html:
        paths.append(out_dir / f"{base}_report.html")
        paths[2].write_bytes(to_html_bytes(result, recording, injected))
    return paths


def _print_summary(result: AnalysisResult) -> None:
    s = result.recording_summary
    print(f"{s['signal_type']} · {s['n_channels']} channel(s) · {s['sfreq_hz']:g} Hz · {s['duration_s']:.1f} s · {s['data_kind']}")
    if s.get("is_simulated"):
        print("Simulated signal — not a real recording.")
    print(f"{len(result.findings)} finding(s); {result.windows_flagged}/{result.windows_analyzed} channel-windows flagged.")
    if result.findings:
        df = to_dataframe(result)[["Channel", "Start (s)", "End (s)", "Category", "Severity", "Certainty"]]
        print(df.to_string(index=False))
    for item in result.skipped_checks:
        print(f"Skipped {item['check']}: {item['reason']}")
    for item in result.notes:
        print(f"Note: {item['reason']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sqc", description="ECG/EEG Signal Quality Checker (recording-quality flags, not diagnosis).")
    parser.add_argument("--version", action="version", version=f"sqc {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mains", choices=["60", "50", "both"], help="mains frequency to check (default 60)")
    common.add_argument("--config", help="YAML/JSON file with setting overrides (e.g. flat_std_uV: 0.3)")
    common.add_argument("--set", action="append", metavar="KEY=VALUE", help="override one setting; repeatable")
    common.add_argument("--out", help="folder for findings CSV, results JSON and HTML report")
    common.add_argument("--no-html", action="store_true", help="skip the HTML report")
    common.add_argument("--quiet", action="store_true", help="do not print the summary")

    a = sub.add_parser("analyze", parents=[common], help="analyze a CSV, EDF or WFDB (.hea) file")
    a.add_argument("path", help="file to analyze (.csv, .edf, or WFDB .hea with its .dat alongside)")
    a.add_argument("--type", required=True, choices=["ECG", "EEG"], help="signal type")
    a.add_argument("--sfreq", type=float, help="sampling rate in Hz (required for CSV)")
    a.add_argument("--unit", choices=["V", "mV", "uV", "µV"], help="amplitude unit (required for CSV)")
    a.add_argument("--missing", choices=["reject", "flag"], default="reject",
                   help="NaN/Inf values: reject the file (default) or report them as missing-data findings")
    a.add_argument("--simulated", action="store_true", help="label the file as simulated (not a real recording)")

    d = sub.add_parser("demo", parents=[common], help="list or analyze a simulated demo scenario")
    d.add_argument("scenario", nargs="?", help="scenario id (see --list)")
    d.add_argument("--list", action="store_true", help="list demo scenarios")
    d.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo" and (args.list or not args.scenario):
            for s in list_scenarios():
                print(f"{s['id']:24s} {s['signal_type']}  {s['label']}  (simulated)")
            return 0
        overrides = _parse_overrides(args.config, args.set, args.mains)
        injected = None
        if args.command == "demo":
            try:
                recording, injected = make_scenario(args.scenario, seed=args.seed)
            except KeyError as exc:
                raise ValidationError(str(exc).strip("'\"")) from None
            stem = args.scenario
        else:
            path = Path(args.path)
            if not path.exists() and not path.with_suffix(".hea").exists():
                raise ValidationError(f"File not found: {args.path}")
            recording = load_file(path, signal_type=args.type, sfreq=args.sfreq, unit=args.unit, missing=args.missing)
            if args.simulated:
                recording = mark_simulated(recording)
            stem = path.stem
        try:
            result = analyze_recording(recording, config=overrides or None)
        except ValueError as exc:  # unknown setting names etc.
            raise ValidationError(str(exc)) from None
        if not args.quiet:
            _print_summary(result)
        if args.out:
            for p in _write_outputs(result, Path(args.out), stem, recording, injected, html=not args.no_html):
                print(f"Wrote {p}")
        return 0
    except ValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
