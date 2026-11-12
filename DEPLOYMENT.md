# DEPLOYMENT — GitHub + Streamlit Community Cloud (Mac)

Free. Takes about 25 minutes the first time. Claude Code can run the git commands for you; the website steps are yours.

## 1. Create the GitHub repo
1. Go to https://github.com/new
2. Name: `ecg-eeg-quality-checker` · **Public** · do **not** add a README (we already have one) → **Create repository**
3. In Terminal, inside the project folder:
```bash
git add .
git commit -m "Phase 1: demo version of ECG/EEG Signal Quality Checker"
git branch -M main
git remote add origin https://github.com/<your-username>/ecg-eeg-quality-checker.git
git push -u origin main
```
If git asks for a password, use a GitHub **personal access token** or install the GitHub CLI (`brew install gh`, then `gh auth login`).

4. On GitHub, open the **Actions** tab and confirm the tests workflow passes (green check).

## 2. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub
2. **Create app** → **Deploy a public app from GitHub**
3. Repository: `<you>/ecg-eeg-quality-checker` · Branch: `main` · Main file: `app.py`
4. **Advanced settings** → Python version **3.12**
5. Custom URL: e.g. `ecg-eeg-quality-checker` → `https://ecg-eeg-quality-checker.streamlit.app`
6. **Deploy**. The first build takes a few minutes (MNE and SciPy are large).

If the build fails on the `-e .` line in `requirements.txt`, ask Claude Code: "Streamlit Cloud failed to install the local package; read the build log I'm pasting and fix requirements.txt or app.py imports."

## 3. After deploying
- Open the link, run a demo scenario, and check that highlights appear.
- Put the link in `README.md` (top, as a badge or line) and push again.
- Free-tier facts (checked 2026-11-12, may change): about 1 GB memory per app; apps **sleep after about 12 hours without traffic** and wake on visit. Open the link yourself before sending it to a recruiter or professor.

## 4. Updating
Every `git push` to `main` redeploys automatically.
