"""All thresholds in one place, so changing a rule is a one-line PR."""

NUMERIC = ["age", "length_of_stay", "prior_admissions", "num_medications", "hba1c", "ed_visits_6m"]
CATEGORICAL = ["sex", "primary_dx", "discharge_to"]
TARGET = "readmitted_30d"

ALLOWED = {
    "sex": {"F", "M"},
    "primary_dx": {"heart_failure", "copd", "diabetes", "pneumonia", "other"},
    "discharge_to": {"home", "home_health", "skilled_nursing"},
}
RANGES = {
    "age": (18, 105),
    "length_of_stay": (1, 60),
    "prior_admissions": (0, 30),
    "num_medications": (0, 40),
    "hba1c": (4.0, 15.0),
    "ed_visits_6m": (0, 30),
}
MAX_NULL_RATE = {"hba1c": 0.30}          # hba1c isn't measured for everyone
DEFAULT_MAX_NULL_RATE = 0.0              # everything else must be filled in

# Release gate
MIN_AUC = 0.70
MIN_RECALL = 0.60                        # missing a readmission is the costly error
MAX_SUBGROUP_AUC_DROP = 0.10             # no group's AUC may sit this far below the overall AUC
MIN_SUBGROUP_SIZE = 150                  # too few patients to judge fairly
DECISION_THRESHOLD = 0.16                # flag for follow-up call above this risk

# Drift
PSI_WARN = 0.10
PSI_ALERT = 0.25
