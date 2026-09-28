"""Meta-learner wrapper: learner choice and the X-learner's fixed propensity."""

import numpy as np
import pandas as pd
import pytest

from uplift_pipeline.models import UpliftModel


def data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "treatment": np.tile([0, 1, 1, 1], n // 4),
            "y": rng.integers(0, 2, n),
        }
    )


def test_unknown_learner_rejected():
    with pytest.raises(ValueError, match="unknown learner 'r_xgb'"):
        UpliftModel(["a"], learner="r_xgb")


@pytest.mark.parametrize("learner", ["t_xgb", "x_xgb", "s_xgb"])
def test_each_learner_predicts_one_uplift_per_row(learner):
    model = UpliftModel(["a", "b"], learner=learner).fit(data())
    # A null feature is treated as missing, not a crash.
    batch = pd.DataFrame({"a": [0.1, None], "b": [1.0, 2.0]})
    pred = model.predict(None, batch)
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_only_x_learner_fixes_trial_treatment_rate():
    assert UpliftModel(["a", "b"], learner="x_xgb").fit(data()).propensity == 0.75
    assert UpliftModel(["a", "b"], learner="t_xgb").fit(data()).propensity is None
