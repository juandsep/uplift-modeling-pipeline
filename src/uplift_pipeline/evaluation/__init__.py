"""Incrementality metrics: Qini and AUUC (normalized, from causalml)."""

import pandas as pd
from causalml.metrics import auuc_score, qini_score


def uplift_metrics(y, treatment, uplift) -> dict[str, float]:
    df = pd.DataFrame({"y": y, "w": treatment, "uplift": uplift})
    return {
        "qini": float(qini_score(df)["uplift"]),
        "auuc": float(auuc_score(df)["uplift"]),
    }
