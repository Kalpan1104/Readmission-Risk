"""
Data contract checks. Run before training or scoring anything.

Returns a list of human-readable problems. An empty list means the file passed.
Usage: python -m src.validate data/reference.csv
"""
import sys

import pandas as pd

from src import config as C

REQUIRED = ["patient_id", *C.NUMERIC, *C.CATEGORICAL]


def validate(df, require_target=True):
    problems = []
    cols = REQUIRED + ([C.TARGET] if require_target else [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]

    if df.empty:
        return ["file has no rows"]

    dupes = df["patient_id"].duplicated().sum()
    if dupes:
        problems.append(f"{dupes} duplicate patient_id values")

    for col in C.NUMERIC + C.CATEGORICAL:
        rate = df[col].isna().mean()
        limit = C.MAX_NULL_RATE.get(col, C.DEFAULT_MAX_NULL_RATE)
        if rate > limit:
            problems.append(f"{col}: {rate:.1%} missing (limit {limit:.0%})")

    for col, (lo, hi) in C.RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().sum() > df[col].isna().sum():
            problems.append(f"{col}: contains non-numeric values")
        bad = ((values < lo) | (values > hi)).sum()
        if bad:
            problems.append(f"{col}: {bad} values outside [{lo}, {hi}]")

    for col, allowed in C.ALLOWED.items():
        unknown = set(df[col].dropna().unique()) - allowed
        if unknown:
            problems.append(f"{col}: unexpected categories {sorted(unknown)}")

    if require_target:
        labels = set(df[C.TARGET].dropna().unique())
        if not labels <= {0, 1}:
            problems.append(f"{C.TARGET}: labels must be 0/1, found {sorted(labels)}")
        elif labels != {0, 1}:
            problems.append(f"{C.TARGET}: only one class present")
    return problems


if __name__ == "__main__":
    path = sys.argv[1]
    has_target = "--no-target" not in sys.argv
    issues = validate(pd.read_csv(path), require_target=has_target)
    if issues:
        print(f"❌ {path} failed validation:")
        for i in issues:
            print("  -", i)
        sys.exit(1)
    print(f"✅ {path} passed all data checks")
