# CLAUDE.md — Rules for building the ECG/EEG Signal Quality Checker

You are building this project for Joseph, an undergraduate (Applied Math & Statistics, JHU) who is **new to coding** and reads code at a beginner level. Build real, working software and explain what you did in plain English.

## Project documents — read all of them before writing code
| File | What it is |
|---|---|
| `PRD.md` | Product requirements: what we build, for whom, scope, non-goals, success criteria |
| `ARCHITECTURE.md` | Stack, folder layout, modules, data models, data flow, validation, config |
| `DETECTORS.md` | The scientific core: every detection rule, threshold, unit and rationale |
| `DESIGN.md` | Streamlit dashboard: pages, layout, components, visual style, copy |
| `TASKS.md` | Phase-by-phase task checklist with a Definition of Done per phase |
| `DEPLOYMENT.md` | GitHub + Streamlit Community Cloud steps (Mac) |
| `START_HERE.md` | Human instructions for Joseph (you can skim it) |
| `context/EEG_ECG_Project_Context.pdf` | Background only: project history, Joseph's decisions, evidence limits. Read it once; the `.md` files above take priority |

**Evidence rule:** older résumé lines and chat descriptions say a checker was "built". Treat those as history, not as code. Nothing exists until you find it in this folder and run it. Before claiming any capability, inspect the actual files and run them.

If documents disagree, priority is: `CLAUDE.md` > `DETECTORS.md` > `PRD.md` > `ARCHITECTURE.md` > `DESIGN.md` > `TASKS.md`. If something is genuinely ambiguous and affects scientific correctness, stop and ask; for minor choices, decide, and note the decision in `docs/DECISIONS.md`.

## How to work
- **Build one phase at a time** from `TASKS.md`. Default is Phase 1. Do not start the next phase unless asked.
- Tick off checkboxes in `TASKS.md` (`- [x]`) as you complete tasks.
- Start by checking the environment (`python3 --version`, `uname -a`) and creating a virtual environment in `.venv`.
- After each module, **run the tests**. Never say something works unless you ran it. Never invent test or benchmark output.
- Keep code simple: small functions, type hints, docstrings, no clever tricks.
- **One analysis engine.** `signal_quality.analyze.analyze_recording()` is the only entry point. `app.py`, the CLI, tests and notebooks call it. Never copy detection logic into the app, notebooks or tests.
- Every threshold lives in `src/signal_quality/config.py` with a unit and a one-line rationale comment, and can be overridden.
- **Units:** store signals internally in **volts**. Display EEG in **µV** and ECG in **mV**. Label units on every plot, table and export.
- Never silently assume a sampling rate or unit. Missing → clear, friendly error.
- Never modify the original loaded signal array. Copy before processing.
- Out of scope, do not add: Docker, FastAPI, databases, user accounts, cloud services, paid APIs, deep learning, GPUs.

## Scientific honesty (non-negotiable)
- This tool detects **recording-quality problems (artifacts)**. It does **not** diagnose disease, detect arrhythmias or seizures, or decide whether a signal was AI-generated.
- Unusual physiology is not automatically corruption. Use "possible" / "likely" language.
- Certainty labels (`definite` / `likely` / `uncertain`) are rule-based, **not probabilities**. Never show them as percentages.
- Demo signals are **simulated**. Label them "Simulated signal — not a real recording" everywhere they appear.
- Never fabricate accuracy, benchmarks or results. Not evaluated → show "Not yet evaluated".
- Do not store patient-identifying header fields (EDF patient name, birthdate, etc.) in reports or exports.

## When you finish a phase, report in this format
1. **What was built** (plain English, 3–6 bullets)
2. **How to run it** (exact Mac Terminal commands)
3. **Tests:** the command you ran and real pass/fail counts
4. **Not done yet / known limitations**
5. **One concept Joseph should understand from this phase** (3–5 sentences, beginner level)
