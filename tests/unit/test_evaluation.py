import numpy as np

from uplift_pipeline.evaluation import qini_curve, uplift_metrics


def test_true_ranking_beats_reversed():
    rng = np.random.default_rng(0)
    n = 4000
    tau = rng.uniform(0, 0.5, n)
    w = rng.integers(0, 2, n)
    y = (rng.uniform(size=n) < 0.2 + tau * w).astype(int)

    good = uplift_metrics(y, w, tau)
    bad = uplift_metrics(y, w, -tau)

    assert good["qini"] > bad["qini"]
    assert good["auuc"] > bad["auuc"]


def test_qini_curve_area_matches_qini_score():
    rng = np.random.default_rng(1)
    n = 500
    tau = rng.uniform(0, 0.5, n)
    w = rng.integers(0, 2, n)
    y = (rng.uniform(size=n) < 0.2 + tau * w).astype(int)

    curve = qini_curve(y, w, {"m": tau}, points=n + 1)

    assert curve["fraction"].iloc[[0, -1]].tolist() == [0.0, 1.0]
    area = (curve["m"] - curve["random"]).mean()
    assert np.isclose(area, uplift_metrics(y, w, tau)["qini"])
