"""Incrementality metrics: Qini and AUUC (normalized, from causalml).

`ate` (observed difference in y means) vs `mean_uplift` (model average) is a
sanity check: a model far off the observed lift is miscalibrated.
"""

import numpy as np
import pandas as pd
from causalml.metrics import auuc_score, get_qini, qini_score
from matplotlib import style

# Figure without pyplot renders through Agg: no display or GUI backend needed.
from matplotlib.figure import Figure

# First three slots of the default categorical palette (validated all-pairs).
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def uplift_metrics(y, treatment, uplift) -> dict[str, float]:
    df = pd.DataFrame({"y": y, "w": treatment, "uplift": uplift})
    means = df.groupby("w")["y"].mean()
    return {
        "qini": float(qini_score(df)["uplift"]),
        "auuc": float(auuc_score(df)["uplift"]),
        "ate": float(means[1] - means[0]),
        "mean_uplift": float(df["uplift"].mean()),
    }


def qini_curve(y, treatment, uplifts: dict[str, np.ndarray], points: int = 101):
    """Normalized Qini curve per model plus the random line, at `points` fractions.

    Same curve `qini_score` integrates (causalml's get_qini, normalized), so
    the area between a column and `random` matches uplift_metrics' qini.
    """
    df = pd.DataFrame({"y": np.asarray(y), "w": np.asarray(treatment), **uplifts})
    curve = get_qini(df, normalize=True)
    n = len(curve) - 1
    # Every model ends at the same point (whole population), so one random line.
    curve["random"] = np.linspace(0.0, curve.iloc[-1, 0], n + 1)
    rows = np.unique(np.linspace(0, n, points).round().astype(int))
    out = curve.iloc[rows].reset_index(drop=True)
    out.insert(0, "fraction", rows / n)
    return out


def plot_qini(curve: pd.DataFrame, colors: dict[str, str]) -> Figure:
    """Qini curves (one per model column) against the random baseline."""
    # causalml switches the global style to fivethirtyeight on import.
    with style.context("default"):
        fig = Figure(figsize=(6.4, 4.8), layout="constrained")
        ax = fig.subplots()
        for name in curve.columns.drop(["fraction", "random"]):
            ax.plot(
                curve["fraction"], curve[name], lw=2, color=colors[name], label=name
            )
        ax.plot(
            curve["fraction"],
            curve["random"],
            "--",
            lw=1.5,
            color="#52514e",
            label="random",
        )
        ax.set_xlabel("Fraction targeted, by predicted uplift")
        ax.set_ylabel("Cumulative incremental outcome (normalized)")
        ax.set_title("Qini curve")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    return fig
