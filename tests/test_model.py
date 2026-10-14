"""Train + evaluate end to end, and check the release gate logic."""
import json

import pytest

from src import config as C
from src import evaluate, train
from src.synth import generate


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("reports")
    data = d / "train.csv"
    generate(6000, seed=21).to_csv(data, index=False)
    train.main(data, d)
    return d


def test_artifacts_written(trained):
    for name in ["model.joblib", "test_set.csv", "top_features.json"]:
        assert (trained / name).exists()


def test_model_passes_gate(trained, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    overall, subgroups, failures = evaluate.main(trained, "test")
    assert failures == []
    assert overall["auc"] >= C.MIN_AUC
    assert set(subgroups["sex"]) == {"F", "M"}
    card = (trained / "model_card.md").read_text()
    assert "synthetic" in card.lower() and "Passed" in card
    assert json.loads((trained / "metrics.json").read_text())["gate_passed"] is True


def test_refuses_invalid_data(tmp_path):
    df = generate(500, seed=1)
    df.loc[0, "age"] = 500
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        train.main(path, tmp_path)


def _m(auc, recall=0.7, n=500):
    return {"auc": auc, "recall": recall, "n": n}


def test_gate_fails_on_low_auc():
    assert any("overall AUC" in r for r in evaluate.gate(_m(0.6), {}))


def test_gate_fails_on_low_recall():
    assert any("recall" in r for r in evaluate.gate(_m(0.8, recall=0.3), {}))


def test_gate_fails_when_a_group_is_left_behind():
    subgroups = {"sex": {"F": _m(0.82), "M": _m(0.64)}}
    assert any("sex=M" in r for r in evaluate.gate(_m(0.78), subgroups))


def test_gate_allows_a_group_doing_better_than_average():
    subgroups = {"age_band": {"18-49": _m(0.92), "80+": _m(0.77)}}
    assert evaluate.gate(_m(0.78), subgroups) == []


def test_gate_ignores_tiny_groups():
    subgroups = {"sex": {"F": _m(0.85), "M": _m(0.40, n=20)}}
    assert evaluate.gate(_m(0.8), subgroups) == []
