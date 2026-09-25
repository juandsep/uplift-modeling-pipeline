"""X5 feature table: schema is validated where it enters the pipeline."""

import numpy as np
import pandas as pd
import pytest

from uplift_pipeline.data import load_x5_features


def table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["a", "b", "c"],
            "treatment": [0, 1, 1],
            "y": [1, 0, 1],
            "age": pd.array([45, None, 30], dtype="Int64"),
            "spend": [10.5, np.nan, 3.0],
        }
    )


def test_loads_valid_table(tmp_path):
    path = tmp_path / "f.parquet"
    table().to_parquet(path)

    df, features = load_x5_features(path)

    assert features == ["age", "spend"]
    assert list(df.columns) == ["age", "spend", "treatment", "y"]
    assert (df[features].dtypes == "float64").all()
    assert df["age"].isna().sum() == 1


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda d: d.drop(columns="y"), "missing required columns \\['y'\\]"),
        (lambda d: d.assign(treatment=[0, 2, 1]), "'treatment' must be 0/1"),
        (lambda d: d.assign(y=[1.0, None, 0.0]), "'y' must be 0/1"),
        (lambda d: d.assign(gender=["F", "M", "U"]), "non-numeric.*gender"),
        (lambda d: d.assign(client_id=["a", "a", "c"]), "client_id is not unique"),
    ],
)
def test_rejects_bad_table(tmp_path, change, error):
    path = tmp_path / "f.parquet"
    change(table()).to_parquet(path)
    with pytest.raises(ValueError, match=error):
        load_x5_features(path)
