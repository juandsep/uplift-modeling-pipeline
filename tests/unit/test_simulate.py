"""Each drift scenario breaks exactly what it claims to."""

import numpy as np
import pandas as pd
import pytest

from uplift_pipeline.simulate import SCENARIOS, make_scenario


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    n = 4000
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "total_spend": rng.gamma(2.0, 4000.0, n),
            "n_transactions": rng.poisson(20, n).astype(float),
            "treatment": rng.integers(0, 2, n),
            "y": rng.integers(0, 2, n),
        }
    )


def test_baseline_only_resamples(df):
    out = make_scenario(df, "baseline", 1000)
    assert len(out) == 1000
    assert out["total_spend"].isin(df["total_spend"]).all()


def test_covariate_moves_inputs_not_labels(df):
    base = make_scenario(df, "baseline", 3000, seed=2)
    out = make_scenario(df, "covariate", 3000, seed=2)
    assert out["total_spend"].mean() == pytest.approx(1.5 * base["total_spend"].mean())
    assert out["age"].mean() == pytest.approx(base["age"].mean() + 10)
    assert out["y"].equals(base["y"])


def test_missing_nulls_about_30_percent_of_rows(df):
    out = make_scenario(df, "missing", 3000, seed=3)
    assert 0.25 < out["age"].isna().mean() < 0.35
    assert out["treatment"].notna().all()


def test_concept_keeps_inputs_and_rates_but_removes_the_effect(df):
    df["y"] = ((df["treatment"] == 1) & (df["age"] > 50)).astype(int)
    base = make_scenario(df, "baseline", 3000, seed=4)
    out = make_scenario(df, "concept", 3000, seed=4)
    assert out["age"].equals(base["age"])
    assert out["y"].equals(base["y"])
    assert out["treatment"].sum() == base["treatment"].sum()
    # Among older clients, treated and control now convert alike.
    old = out[out["age"] > 50]
    lift = old.groupby("treatment")["y"].mean()
    assert abs(lift[1] - lift[0]) < 0.2


def test_strength_zero_is_baseline(df):
    base = make_scenario(df, "baseline", 500, seed=5)
    for scenario in SCENARIOS:
        assert make_scenario(df, scenario, 500, seed=5, strength=0).equals(base)


def test_unknown_scenario(df):
    with pytest.raises(ValueError, match="unknown scenario"):
        make_scenario(df, "nope", 10)
