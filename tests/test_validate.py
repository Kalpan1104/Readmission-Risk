import numpy as np
import pytest

from src.synth import generate
from src.validate import validate


@pytest.fixture
def df():
    return generate(500, seed=11)


def test_clean_data_passes(df):
    assert validate(df) == []


def test_missing_column(df):
    assert "missing columns" in validate(df.drop(columns=["age"]))[0]


def test_empty_file(df):
    assert validate(df.iloc[0:0]) == ["file has no rows"]


@pytest.mark.parametrize("column, value, expected", [
    ("age", 140, "age: 1 values outside"),
    ("length_of_stay", 0, "length_of_stay: 1 values outside"),
    ("primary_dx", "Heart Failure", "unexpected categories"),
    ("sex", "X", "unexpected categories"),
    ("discharge_to", None, "discharge_to: 0.2% missing"),
])
def test_single_bad_value_is_caught(df, column, value, expected):
    df.loc[0, column] = value
    assert any(expected in p for p in validate(df))


def test_duplicate_ids(df):
    df.loc[1, "patient_id"] = df.loc[0, "patient_id"]
    assert any("duplicate patient_id" in p for p in validate(df))


def test_hba1c_can_be_partly_missing(df):
    df["hba1c"] = 6.0
    df.loc[:99, "hba1c"] = np.nan           # 20%, under the 30% limit
    assert validate(df) == []


def test_hba1c_too_much_missing(df):
    df.loc[:300, "hba1c"] = np.nan          # ~60%
    assert any(p.startswith("hba1c") for p in validate(df))


def test_bad_labels(df):
    df["readmitted_30d"] = 2
    assert any("labels must be 0/1" in p for p in validate(df))


def test_single_class(df):
    df["readmitted_30d"] = 0
    assert any("only one class" in p for p in validate(df))


def test_scoring_data_without_target(df):
    assert validate(df.drop(columns=["readmitted_30d"]), require_target=False) == []
