from src import config as C
from src.synth import generate
from src.validate import validate


def test_reproducible():
    assert generate(300, seed=1).equals(generate(300, seed=1))


def test_generated_data_passes_contract():
    assert validate(generate(3000, seed=2)) == []


def test_shifted_data_still_passes_contract():
    # Drift changes the population, not the schema
    assert validate(generate(3000, seed=2, shift=1.0)) == []


def test_readmission_rate_is_realistic():
    rate = generate(8000, seed=3)[C.TARGET].mean()
    assert 0.10 < rate < 0.25


def test_shift_makes_population_older_and_sicker():
    base, shifted = generate(5000, seed=4), generate(5000, seed=4, shift=1.0)
    assert shifted["age"].mean() > base["age"].mean() + 5
    assert shifted[C.TARGET].mean() > base[C.TARGET].mean()


def test_heart_failure_riskier_than_other():
    df = generate(10000, seed=5)
    rates = df.groupby("primary_dx")[C.TARGET].mean()
    assert rates["heart_failure"] > rates["other"]


def test_hba1c_missingness_within_limit():
    assert generate(5000, seed=6)["hba1c"].isna().mean() <= C.MAX_NULL_RATE["hba1c"]
