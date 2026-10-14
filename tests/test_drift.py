import numpy as np
import pytest

from src.drift import drift_report, psi_categorical, psi_numeric, status, to_markdown
from src.synth import generate


def test_identical_distributions_have_zero_psi():
    x = np.random.default_rng(0).normal(0, 1, 5000)
    assert psi_numeric(x, x) == pytest.approx(0, abs=1e-9)


def test_shifted_numeric_has_high_psi():
    rng = np.random.default_rng(1)
    assert psi_numeric(rng.normal(0, 1, 5000), rng.normal(1, 1, 5000)) > 0.25


def test_categorical_psi():
    same = psi_categorical(["a", "b"] * 500, ["a", "b"] * 500)
    moved = psi_categorical(["a"] * 900 + ["b"] * 100, ["a"] * 300 + ["b"] * 700)
    assert same == pytest.approx(0, abs=1e-9) and moved > 0.25


def test_new_category_does_not_crash():
    assert psi_categorical(["a"] * 100, ["a"] * 50 + ["c"] * 50) > 0


@pytest.mark.parametrize("value, expected", [(0.05, "ok"), (0.15, "warn"), (0.4, "alert")])
def test_status(value, expected):
    assert status(value) == expected


def test_no_drift_without_shift():
    report = drift_report(generate(5000, seed=0), generate(2000, seed=9))
    assert all(r["status"] == "ok" for r in report.values())


def test_strong_shift_triggers_alert():
    report = drift_report(generate(5000, seed=0), generate(2000, seed=9, shift=1.0))
    assert any(r["status"] == "alert" for r in report.values())


def test_markdown_lists_every_feature():
    report = drift_report(generate(2000, seed=0), generate(1000, seed=1))
    md = to_markdown(report, "batch.csv")
    assert all(col in md for col in report)
