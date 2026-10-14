"""
Train a logistic regression readmission model.

Logistic regression on purpose: clinicians tend to ask "why did it flag this
patient?", and coefficients are easy to explain. Missing HbA1c is imputed with
the median and gets its own "was missing" indicator, because not being tested
is itself informative.

Usage: python -m src.train data/reference.csv --out reports/
"""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd

from src import config as C
from src.validate import validate


def build_pipeline():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([("num", numeric, C.NUMERIC), ("cat", categorical, C.CATEGORICAL)])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])


def split(df, seed=42):
    return train_test_split(df, test_size=0.25, stratify=df[C.TARGET], random_state=seed)


def top_coefficients(model, k=8):
    names = model.named_steps["pre"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    pairs = sorted(zip(names, coefs), key=lambda t: -abs(t[1]))[:k]
    return [(n.split("__", 1)[-1], round(float(c), 3)) for n, c in pairs]


def main(data_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    problems = validate(df)
    if problems:
        raise SystemExit("Refusing to train on invalid data:\n  " + "\n  ".join(problems))

    train_df, test_df = split(df)
    model = build_pipeline().fit(train_df[C.NUMERIC + C.CATEGORICAL], train_df[C.TARGET])

    joblib.dump(model, out_dir / "model.joblib")
    test_df.to_csv(out_dir / "test_set.csv", index=False)
    (out_dir / "top_features.json").write_text(json.dumps(top_coefficients(model), indent=2))
    print(f"Trained on {len(train_df)} discharges, held out {len(test_df)} for evaluation")
    return model, test_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--out", default="reports")
    a = p.parse_args()
    main(a.data, a.out)
