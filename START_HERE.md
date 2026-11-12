# START HERE (for Joseph)

This folder is the full blueprint for the **ECG/EEG Signal Quality Checker**. Claude Code reads it and builds the project. You don't need to read every file; this page tells you what to do.

| File | What it's for |
|---|---|
| `CLAUDE.md` | Rules Claude Code follows (it reads this automatically) |
| `PRD.md` | What the product is, scope, non-goals, success criteria |
| `ARCHITECTURE.md` | Tech stack, folders, data models, data flow |
| `DETECTORS.md` | The science: every detection rule and threshold |
| `DESIGN.md` | How the dashboard looks and behaves; demo scenarios |
| `TASKS.md` | Phase-by-phase checklist (Phase 1 = demo) |
| `DEPLOYMENT.md` | Putting it on GitHub and a free public link |
| `context/EEG_ECG_Project_Context.pdf` | Your project history and decisions (background for Claude Code) |

## 1. One-time Mac setup (skip what you already have)
```bash
python3 --version   # want 3.12 or 3.13
git --version
```
- Need Python? https://www.python.org/downloads/macos/ (or `brew install python@3.12`).
- Need git? `git --version` offers to install Apple's command-line tools; accept.
- Free accounts: **GitHub** and **Streamlit Community Cloud** (https://share.streamlit.io, sign in with GitHub).

## 2. Put the folder in place and start Claude Code
```bash
mkdir -p ~/Projects
# move the unzipped ecg-eeg-quality-checker folder into ~/Projects, then:
cd ~/Projects/ecg-eeg-quality-checker
git init
claude
```

## 3. Paste this to build the demo (Phase 1)
```
Read CLAUDE.md, PRD.md, ARCHITECTURE.md, DETECTORS.md, DESIGN.md, TASKS.md and
DEPLOYMENT.md. Then build Phase 1 exactly as defined in TASKS.md. Create the
virtual environment, install dependencies, write the code and tests, run the
tests and fix failures until they pass, check every demo scenario meets the
Phase 1 Definition of Done, and launch the Streamlit app once headless to
confirm it starts without errors. Tick the boxes in TASKS.md as you go. Do not
start Phase 2. Finish with the report format in CLAUDE.md.
```

## 4. Try it yourself
```bash
source .venv/bin/activate
streamlit run app.py
```
Pick a demo signal → **Analyze** → check that the red highlights line up with the injected problems (turn on "Show injected-artifact ground truth").

## 5. Put it online
Ask Claude Code: `Walk me through DEPLOYMENT.md step by step and run the git commands for me.`

## 6. Later phases (one per session)
```
Read all the project .md files. Phase N-1 is complete. Build Phase N from
TASKS.md, run all tests, tick the boxes, and report in the CLAUDE.md format.
```

## 7. What you can honestly put on your résumé
Only claim what is built and live.

Dates: **Jul 2026 – Present** (one continuous project: the earlier version plus the from-scratch rebuild). Describe it as in progress until Phase 5 is done, and don't call it "built" or "created" while it's being rebuilt.

**After Phase 1 is deployed:**
> **ECG/EEG Signal Quality Checker** | Python, SciPy, MNE, NeuroKit2, Streamlit | Jul 2026 – Present: Building an explainable artifact detector that flags flatlines, clipping, power-line noise, spikes and baseline drift in EEG/ECG recordings, reporting the channel, time window and rule behind each flag; live demo on Streamlit Cloud.

**After Phase 3 (real evaluation done):** add "evaluated on PhysioNet's MIT-BIH Noise Stress Test Database and EEG Motor Movement/Imagery data with controlled artifact injection" plus your *actual* measured numbers (e.g., recall at each noise level).

**After Phase 4:** add scikit-learn **only** if the Isolation Forest is in the code; report honestly whether it helped.

**Don't claim:** clinical validation, diagnosis, detecting AI-generated signals, FastAPI or Docker (not in this build), or "published", unless those become true.
