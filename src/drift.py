"""
Population Stability Index (PSI) between the training data and a new batch.

PSI < 0.10 stable | 0.10-0.25 worth watching | > 0.25 the population has moved
Usage: python -m src.drift data/reference.csv data/batch.csv --report reports/drift.md
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src import config as C

EPS = 1e-4


def _psi(ref_share, new_share):
    ref_share = np.clip(ref_share, EPS, None)
    new_share = np.clip(new_share, EPS, None)
    return float(np.sum((new_share - ref_share) * np.log(new_share / ref_share)))


def psi_numeric(ref, new, bins=10):
    ref, new = pd.Series(ref).dropna(), pd.Series(new).dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, edges)[0] / len(ref)
    new_share = np.histogram(new, edges)[0] / len(new)
    return _psi(ref_share, new_share)


def psi_categorical(ref, new):
    cats = sorted(set(pd.Series(ref).dropna()) | set(pd.Series(new).dropna()))
    ref_share = pd.Series(ref).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    new_share = pd.Series(new).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_share, new_share)


def status(psi):
    if psi >= C.PSI_ALERT:
        return "alert"
    return "warn" if psi >= C.PSI_WARN else "ok"


def drift_report(ref_df, new_df):
    rows = []
    for col in C.NUMERIC:
        rows.append((col, psi_numeric(ref_df[col], new_df[col])))
    for col in C.CATEGORICAL:
        rows.append((col, psi_categorical(ref_df[col], new_df[col])))
    return {col: {"psi": round(v, 4), "status": status(v)} for col, v in rows}


def to_markdown(report, batch_name):
    icon = {"ok": "🟢", "warn": "🟡", "alert": "🔴"}
    lines = [f"## Drift check: `{batch_name}` vs training data", "",
             "| Feature | PSI | Status |", "|---|---|---|"]
    for col, r in sorted(report.items(), key=lambda kv: -kv[1]["psi"]):
        lines.append(f"| {col} | {r['psi']} | {icon[r['status']]} {r['status']} |")
    alerts = [c for c, r in report.items() if r["status"] == "alert"]
    lines += ["", f"**{len(alerts)} feature(s) over the alert threshold ({C.PSI_ALERT})**"
              + (f": {', '.join(alerts)}" if alerts else "")]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("batch")
    p.add_argument("--report", default="reports/drift.md")
    p.add_argument("--json", default="reports/drift.json")
    a = p.parse_args()
    rep = drift_report(pd.read_csv(a.reference), pd.read_csv(a.batch))
    md = to_markdown(rep, a.batch)
    open(a.report, "w").write(md)
    json.dump(rep, open(a.json, "w"), indent=2)
    print(md)
    # exit code 2 = drift alert, so the workflow can react
    sys.exit(2 if any(r["status"] == "alert" for r in rep.values()) else 0)
