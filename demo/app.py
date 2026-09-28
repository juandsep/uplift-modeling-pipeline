"""Targeting demo: who to send the campaign to, based on precomputed uplift scores.

Reads scores_sample.parquet only. No API calls, no credentials.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "scores_sample.parquet"


@st.cache_data
def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    return df.sort_values("uplift", ascending=False, ignore_index=True)


def cumulative_uplift(df: pd.DataFrame) -> np.ndarray:
    """Incremental conversions when targeting the top n clients, for every n.

    (treated conversion rate - control conversion rate) in the top n, times n.
    Zero until the top n holds at least one treated and one control client.
    """
    t = df["treatment"].to_numpy()
    y = df["y"].to_numpy()
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = y_t / n_t - y_c / n_c
    n = np.arange(1, len(df) + 1)
    return np.nan_to_num(lift * n)


st.set_page_config(page_title="Uplift targeting demo")
st.title("Uplift targeting demo")
st.write(
    "Uplift is the change in a client's chance to convert caused by the campaign: "
    "the conversion probability if treated minus the probability if not. "
    "Targeting the clients with the highest predicted uplift spends the budget on "
    "people the campaign actually moves, not on those who would buy anyway."
)

df = load()
curve = cumulative_uplift(df)
total = len(df)

k = st.slider("Target top k% of clients by predicted uplift", 1, 100, 20)
n_k = max(1, round(total * k / 100))
model_inc = curve[n_k - 1]
random_inc = curve[-1] * n_k / total

c1, c2, c3 = st.columns(3)
c1.metric("Clients targeted", f"{n_k:,}")
c2.metric("Incremental conversions (model)", f"{model_inc:,.0f}")
c3.metric(
    "Random targeting, same size",
    f"{random_inc:,.0f}",
    delta=f"{model_inc - random_inc:+,.0f} from the model",
    delta_color="off",
)
st.caption(
    "Estimates come from the observed treatment vs control conversion rates "
    "inside the targeted group, times its size. "
    f"Sample of {total:,} clients."
)

x = np.arange(1, total + 1) / total * 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, curve, label="Model ranking")
ax.plot([0, 100], [0, curve[-1]], linestyle="--", color="gray", label="Random")
ax.scatter([k], [model_inc], color="black", zorder=3, label=f"Top {k}%")
ax.set_xlabel("Clients targeted (%)")
ax.set_ylabel("Incremental conversions")
ax.set_title("Cumulative uplift (Qini-style) curve")
ax.legend()
st.pyplot(fig)
