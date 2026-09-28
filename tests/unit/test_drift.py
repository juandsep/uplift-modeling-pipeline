"""PSI and top-k lift catch the drift they are meant to catch."""

import numpy as np
import pandas as pd
import pytest

from uplift_pipeline.drift import DRIFT, profile, psi, report, top_lift


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"spend": rng.gamma(2.0, 100.0, 5000), "age": rng.normal(45, 12, 5000)}
    )


def test_same_distribution_has_no_drift(df):
    ref = profile(df.iloc[:2500], ["spend", "age"])
    result = report(ref, df.iloc[2500:])
    assert all(v < 0.05 for v in result["feature_psi"].values())


def test_shift_is_drift(df):
    ref = profile(df, ["spend"])
    assert psi(ref["spend"], df["spend"] * 1.5) > DRIFT


def test_nulls_are_drift(df):
    ref = profile(df, ["age"])
    age = df["age"].mask(np.random.default_rng(1).random(len(df)) < 0.3)
    assert psi(ref["age"], age) > DRIFT


def test_shares_sum_to_one_with_nulls(df):
    df.loc[:99, "age"] = np.nan
    shares = profile(df, ["age"])["age"]["shares"]
    assert sum(shares) == pytest.approx(1.0)
    assert shares[-1] == pytest.approx(100 / len(df))


def test_top_lift_and_uplift_psi():
    n = 1000
    cur = pd.DataFrame(
        {
            "uplift": np.linspace(1, 0, n),
            "treatment": np.tile([0, 1], n // 2),
            "y": 0,
        }
    )
    cur.loc[: n // 5 - 1, "y"] = cur.loc[: n // 5 - 1, "treatment"]  # top: treated buy
    assert top_lift(cur) == pytest.approx(1.0)
    ref = profile(cur, ["uplift"])
    result = report(ref, cur)
    assert result["uplift_psi"] == pytest.approx(0.0, abs=1e-6)
    assert result["top20_lift"] == pytest.approx(1.0)
    assert "uplift" not in result["feature_psi"]


def test_push_sends_every_gauge(monkeypatch):
    import prometheus_client

    from uplift_pipeline.drift import push

    sent = {}

    def fake_push(gateway, job, grouping_key, registry):
        sent.update(gateway=gateway, job=job, key=grouping_key)
        sent["names"] = {m.name for m in registry.collect()}

    monkeypatch.setattr(prometheus_client, "push_to_gateway", fake_push)
    push(
        {
            "feature_psi": {"age": 0.3},
            "rows": 10,
            "uplift_psi": 0.1,
            "top20_lift": 0.05,
        },
        "localhost:9091",
        "covariate",
    )
    assert sent["key"] == {"window": "covariate"}
    assert sent["names"] == {
        "uplift_feature_psi",
        "uplift_drift_rows",
        "uplift_score_psi",
        "uplift_top20_lift",
    }


def test_reference_from_json_or_feature_table(tmp_path, df):
    import json

    from uplift_pipeline.drift import _load_reference

    ref = profile(df, ["age"])
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    assert _load_reference(str(tmp_path / "ref.json")) == ref

    table = df.assign(
        client_id=range(len(df)), treatment=np.tile([0, 1], len(df) // 2), y=0
    )
    table.loc[::7, "y"] = 1
    table.to_parquet(tmp_path / "features.parquet")
    assert set(_load_reference(str(tmp_path / "features.parquet"))) == {"spend", "age"}
