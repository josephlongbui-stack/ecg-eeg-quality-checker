"""Smoke test: the Streamlit app runs, analyzes a demo and shows the honesty labels."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_demo_flow():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    demo_button = next(b for b in at.button if b.label == "Try a demo signal")
    demo_button.click().run()
    assert not at.exception and not at.error
    infos = " ".join(i.value for i in at.info)
    assert "Simulated signal — not a real recording" in infos
    findings = [m.value for m in at.metric if m.label == "Findings"]
    assert findings and int(findings[0]) >= 1
    warnings = " ".join(w.value for w in at.warning)
    assert "Not yet evaluated" in warnings


def test_app_clean_demo_has_no_findings():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.selectbox(key="scenario").set_value("eeg_clean").run()
    next(b for b in at.button if b.label == "Analyze").click().run()
    assert not at.exception and not at.error
    assert [m.value for m in at.metric if m.label == "Findings"] == ["0"]
