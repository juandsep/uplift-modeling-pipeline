"""Incrementality metrics: Qini and AUUC (normalized, from causalml).

`ate` (observed difference in y means) vs `mean_uplift` (model average) is a
sanity check: a model far off the observed lift is miscalibrated.
"""

import pandas as pd
from causalml.metrics import auuc_score, qini_score


def uplift_metrics(y, treatment, uplift) -> dict[str, float]:
    df = pd.DataFrame({"y": y, "w": treatment, "uplift": uplift})
    means = df.groupby("w")["y"].mean()
    return {
        "qini": float(qini_score(df)["uplift"]),
        "auuc": float(auuc_score(df)["uplift"]),
        "ate": float(means[1] - means[0]),
        "mean_uplift": float(df["uplift"].mean()),
    }
