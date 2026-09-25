import numpy as np

from uplift_pipeline.evaluation import uplift_metrics


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
