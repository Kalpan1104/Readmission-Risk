"""
Synthetic hospital discharge records. No real patients, no PHI.

Each row is one discharge. `readmitted_30d` is 1 if the patient came back
within 30 days. `shift` (0 to 1) makes the population older and sicker, which
is how we simulate data drift for the monitoring workflow.

Usage:
    python -m src.synth --rows 5000 --seed 0 --out data/reference.csv
    python -m src.synth --rows 2000 --seed 12 --shift 0.6 --out data/batch.csv
"""
import argparse

import numpy as np
import pandas as pd

DX = ["heart_failure", "copd", "diabetes", "pneumonia", "other"]
DX_RISK = {"heart_failure": 1.1, "copd": 0.8, "diabetes": 0.5, "pneumonia": 0.3, "other": 0.0}
DISCHARGE = ["home", "home_health", "skilled_nursing"]


def generate(n=5000, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)

    age = rng.normal(64 + 8 * shift, 15, n).clip(18, 100).round()
    sex = rng.choice(["F", "M"], n)
    dx_p = np.array([0.18, 0.14, 0.16, 0.17, 0.35])
    dx_p = dx_p + shift * np.array([0.08, 0.04, 0.0, 0.0, -0.12])
    dx = rng.choice(DX, n, p=dx_p / dx_p.sum())
    los = (rng.gamma(2.0, 2.4 + 1.2 * shift, n) + 1).clip(1, 60).round()
    prior = rng.poisson(0.8 + 0.02 * (age - 60).clip(0) + 0.6 * shift, n).clip(0, 30)
    meds = rng.poisson(6 + 0.08 * (age - 50).clip(0), n).clip(0, 40)
    ed = rng.poisson(0.5 + 0.4 * prior, n).clip(0, 30)

    hba1c = np.where(dx == "diabetes", rng.normal(8.4, 1.5, n), rng.normal(5.9, 0.7, n)).clip(4, 15).round(1)
    measured = rng.uniform(size=n) < np.where(dx == "diabetes", 0.97, 0.75)
    hba1c = np.where(measured, hba1c, np.nan)

    older = (age >= 75).astype(float)
    p_snf = 0.05 + 0.25 * older + 0.02 * (los - 4).clip(0)
    p_hh = np.full(n, 0.22)
    u = rng.uniform(size=n)
    discharge = np.where(u < p_snf.clip(0, 0.7), "skilled_nursing",
                         np.where(u < (p_snf + p_hh).clip(0, 0.95), "home_health", "home"))

    logit = (
        -4.3
        + 0.022 * (age - 60)
        + 0.10 * los
        + 0.50 * prior
        + 0.05 * meds
        + 0.35 * ed
        + np.vectorize(DX_RISK.get)(dx)
        + 0.12 * np.nan_to_num(hba1c - 7, nan=0).clip(0)
        + 0.35 * (discharge == "skilled_nursing")
        - 0.15 * (discharge == "home_health")
        + rng.normal(0, 0.2, n)
    )
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    return pd.DataFrame({
        "patient_id": [f"P{seed:03d}{i:06d}" for i in range(n)],
        "age": age.astype(int), "sex": sex, "primary_dx": dx,
        "length_of_stay": los.astype(int), "prior_admissions": prior,
        "num_medications": meds, "hba1c": hba1c, "ed_visits_6m": ed,
        "discharge_to": discharge, "readmitted_30d": y,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shift", type=float, default=0.0)
    p.add_argument("--out", default="data/reference.csv")
    a = p.parse_args()
    df = generate(a.rows, a.seed, a.shift)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows to {a.out} (readmission rate {df.readmitted_30d.mean():.1%})")
