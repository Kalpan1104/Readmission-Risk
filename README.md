# Readmission Risk: data checks, drift alerts and gated releases with GitHub Actions

This project predicts whether a patient will be readmitted to hospital within 30 days of being discharged. Hospitals care about this a lot: readmissions are expensive, often preventable, and in the US they're penalised. A model like this is usually used to decide who gets a follow-up phone call after going home.

**Everything here runs on synthetic data.** The generator in `src/synth.py` makes up patients from scratch. There is no real patient information anywhere in the repo, and the model is for learning only, not for real care decisions.

I wanted this project to focus on the parts of MLOps that matter most once health data is involved:

- **Checking data before trusting it.** Every file goes through a data contract (required columns, valid ranges, allowed categories, missing-value limits, duplicate IDs) before anything is trained on it.
- **Watching for drift.** Patient populations change (an older catchment area, a flu season, a new hospital joining), and a model trained on last year's patients can quietly get worse. A weekly job measures that shift and opens a GitHub issue when it's big.
- **Not shipping a model that's unfair or weak.** A release gate checks overall performance and also checks the model separately for women and men and for each age band. If any group is left far behind, the release is blocked.
- **Treating a model release like a software release.** You push a git tag, Actions rebuilds the model from scratch, and if the gate passes it publishes a GitHub Release with the model file and a model card attached.

## The three workflows

**`ci.yml`** runs on every push and every pull request. It runs the tests, validates the training data, trains the model and runs the release gate. If a change to the code makes the model worse, or worse for one group, the PR turns red. The model card from each run is saved as an artifact.

**`drift_monitor.yml`** runs every Monday (or whenever you click Run workflow). It simulates "this week's discharges", validates them, and compares each feature against the training data using the Population Stability Index (PSI). The PSI table goes on the run's summary page. If any feature is over 0.25, it opens an issue labelled `data-drift`. If one is already open, it adds a comment instead of piling up duplicates. On scheduled runs the amount of drift rotates, so roughly every fourth week you'll see an alert. When you run it by hand you can set the drift yourself.

**`release.yml`** runs when you push a tag like `v1.0.0`. It does a full clean build: tests, data check, train, gate. Only if all of those pass does it create a GitHub Release with three files attached: the model (`.joblib`), `metrics.json`, and `model_card.md`, which also becomes the release notes.

## The data

One row per discharge:

| Column | What it is |
|---|---|
| `patient_id` | Made-up ID |
| `age`, `sex` | Demographics |
| `primary_dx` | Main diagnosis: heart failure, COPD, diabetes, pneumonia or other |
| `length_of_stay` | Days in hospital |
| `prior_admissions` | Admissions in the past year |
| `num_medications` | Medications at discharge |
| `hba1c` | Blood sugar marker. Missing for about a quarter of patients, as in real records where not everyone gets tested |
| `ed_visits_6m` | Emergency department visits in the last 6 months |
| `discharge_to` | Home, home health care, or a skilled nursing facility |
| `readmitted_30d` | The label: 1 if they came back within 30 days |

About 16% of patients are readmitted, which is close to real-world rates. Heart failure patients and people with lots of prior admissions come back most often.

`data/reference.csv` is the training set (8,000 patients). `data/example_bad_batch.csv` is a small file with mistakes in it on purpose, so you can see the data contract catch them.

## The model

It's a logistic regression with scaling and one-hot encoding. Missing HbA1c values are filled with the median, and the model also gets a "was missing" flag, because whether a test was ordered can carry information too. I picked logistic regression on purpose: in healthcare people will ask why a patient was flagged, and coefficients are easy to explain. The model card lists the strongest ones.

Patients with a predicted risk of 16% or more get flagged. That threshold leans toward catching more readmissions at the cost of more false alarms, since missing someone who comes back is the more expensive mistake.

## The release gate

All thresholds live in `src/config.py`:

- overall AUC of at least 0.70
- recall of at least 0.60 at the flagging threshold
- no subgroup (sex, or age band with at least 150 patients) more than 0.10 AUC below the overall AUC

One thing I learned while building this: AUC inside a single age band is always somewhat lower than overall. Age is one of the predictors, so slicing by age removes some of the spread the model relies on. The youngest band is also small and noisy. That's why the rule is about falling *far* below overall, not about every group being equal. If you train on a different random seed, you'll occasionally see the gate block a model for exactly this reason, which is a good way to see it working.

## Project layout

```
readmission-risk-mlops/
├── .github/workflows/
│   ├── ci.yml               # tests + data check + gate on every push / PR
│   ├── drift_monitor.yml    # weekly PSI check, opens an issue on drift
│   └── release.yml          # tag v*.*.* -> gated GitHub Release
├── src/
│   ├── config.py            # every threshold in one place
│   ├── synth.py             # synthetic patients (with optional drift)
│   ├── validate.py          # data contract
│   ├── drift.py             # PSI for numeric and categorical features
│   ├── train.py             # logistic regression pipeline
│   └── evaluate.py          # overall + subgroup metrics, gate, model card
├── tests/                   # pytest
├── data/
│   ├── reference.csv
│   └── example_bad_batch.csv
├── reports/                 # filled in when you run things (git-ignored)
├── requirements.txt
└── pytest.ini
```

## Running it on your machine

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest

python -m src.validate data/reference.csv           # should pass
python -m src.validate data/example_bad_batch.csv   # should list five problems

python -m src.train data/reference.csv --out reports
python -m src.evaluate --dir reports --version local
cat reports/model_card.md

# make a drifted batch and compare it
python -m src.synth --rows 2000 --seed 5 --shift 1.0 --out data/batch.csv
python -m src.drift data/reference.csv data/batch.csv
```

---

## Putting it on GitHub from the terminal

### Option A: GitHub CLI

If you don't have it: `brew install gh` (Mac), `winget install GitHub.cli` (Windows), or see https://cli.github.com. Then:

```bash
gh auth login                    # GitHub.com -> HTTPS -> log in with browser

cd readmission-risk-mlops
git init
git branch -M main
git add .
git commit -m "Initial commit: readmission risk pipeline"
gh repo create readmission-risk-mlops --public --source=. --remote=origin --push
```

### Option B: plain git

Create an empty repo at https://github.com/new called `readmission-risk-mlops`. Don't tick any of the "add README / .gitignore / license" boxes. Then:

```bash
cd readmission-risk-mlops
git init
git branch -M main
git add .
git commit -m "Initial commit: readmission risk pipeline"
git remote add origin https://github.com/<your-username>/readmission-risk-mlops.git
git push -u origin main
```

If it asks for a password, use a personal access token (GitHub → Settings → Developer settings → Personal access tokens), not your account password.

### Then try each workflow

**CI** starts on its own after the first push. Open the **Actions** tab, click the run, and scroll down to read the model card on the summary page.

**Drift monitor.** Trigger it with a big shift so you see the alert straight away:

```bash
gh workflow run "Weekly drift monitor" -f shift=1.0
gh run watch
gh issue list --label data-drift
```

Run it again with `-f shift=0` and you'll get an all-green PSI table and no issue.

**Release.** Tag a version and push the tag:

```bash
git tag -a v1.0.0 -m "First readmission model"
git push origin v1.0.0
gh run watch
gh release view v1.0.0
```

The release page will have the model file, the metrics and the model card. To ship a new version later, change something, merge it, and tag `v1.1.0`.

**See the gate block a bad change.** On a branch, make a change that hurts patients. For example, in `src/config.py` raise `DECISION_THRESHOLD` from `0.16` to `0.40`. Fewer people get flagged, so the model now misses about two out of three readmissions. Then:

```bash
git checkout -b raise-threshold
git commit -am "Only flag high-risk patients"
git push -u origin raise-threshold
gh pr create --fill
```

CI fails on the pull request, and the model card in the run summary says recall dropped to about 0.31, below the 0.60 minimum.

### If something goes wrong

- **Release or issue step fails with a 403.** Go to Settings → Actions → General → Workflow permissions, pick **Read and write permissions**, save, and re-run the job.
- **The drift job's issue step fails because issues are turned off.** Enable Issues under Settings → General → Features.
- **A tag that failed the gate.** Fix the problem, then delete the tag and push it again:

  ```bash
  git tag -d v1.0.0
  git push origin :refs/tags/v1.0.0
  git tag -a v1.0.0 -m "..."
  git push origin v1.0.0
  ```

## What I'd add for a real hospital

- Real EHR data with proper de-identification and access controls. Synthetic data can live in a repo; real patient data never should.
- Fairness checks across more groups (race and ethnicity, insurance type, primary language), where that's lawful and appropriate.
- Calibration checks per subgroup, not just AUC.
- A human sign-off step before a release goes live, using GitHub Environments with required reviewers.

## License

MIT
