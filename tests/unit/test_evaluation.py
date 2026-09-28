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


def test_qini_curve_never_repeats_rows_when_points_exceed_n():
    y, w = [1, 0, 1, 0], [1, 1, 0, 0]
    curve = qini_curve(y, w, {"m": np.array([0.4, 0.3, 0.2, 0.1])}, points=101)
    assert curve["fraction"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_metrics_report_observed_lift():
    # Treated all convert, control never: ATE is exactly 1 whatever the model says.
    y, w = [1, 1, 0, 0], [1, 1, 0, 0]
    m = uplift_metrics(y, w, [0.0, 0.0, 0.0, 0.0])
    assert m["ate"] == 1.0
    assert m["mean_uplift"] == 0.0
