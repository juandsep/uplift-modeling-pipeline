"""X5 features: shards partition clients, merge is numeric, no post-cutoff leakage."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from uplift_pipeline.features.x5 import build_shard, merge_shards

CUTOFF = datetime(2019, 3, 18, 12, 0)
N_CLIENTS = 40
NUM_SHARDS = 4


def _write_inputs(root):
    ids = [f"c{i:03d}" for i in range(N_CLIENTS)]
    ages = [30] * N_CLIENTS
    ages[:3] = [-7491, 1901, 13]  # all outside 14..100
    redeem: list[datetime | None] = [CUTOFF - timedelta(days=10)] * N_CLIENTS
    redeem[0] = CUTOFF + timedelta(days=60)  # after the cutoff: post-treatment
    redeem[1] = None
    pd.DataFrame(
        {
            "client_id": ids,
            "first_issue_date": [CUTOFF - timedelta(days=100)] * N_CLIENTS,
            "first_redeem_date": redeem,
            "age": ages,
            "gender": ["F", "M", "U", "F"] * (N_CLIENTS // 4),
        }
    ).to_parquet(root / "clients.parquet")
    labeled = ids[: N_CLIENTS // 2]
    pd.DataFrame(
        {
            "client_id": labeled,
            "treatment_flg": [i % 2 for i in range(len(labeled))],
            "target": [1] * len(labeled),
        },
    ).astype({"treatment_flg": "int8", "target": "int8"}).to_parquet(
        root / "uplift_train.parquet"
    )
    rows = []
    for i, cid in enumerate(ids):
        for t in range(2):  # two transactions, two product lines each
            ts = CUTOFF - timedelta(days=5 + 40 * t) if i else CUTOFF
            for product in ("p1", "p2"):
                rows.append(
                    (
                        cid,
                        f"t{t}",
                        ts,
                        10.0,
                        0.0,
                        -5.0,
                        0.0,
                        100.0,
                        "s1",
                        product,
                        1.0,
                        45.0,
                        5.0 if product == "p1" else None,
                    )
                )
    cols = ["client_id", "transaction_id", "transaction_datetime",
            "regular_points_received", "express_points_received",
            "regular_points_spent", "express_points_spent", "purchase_sum",
            "store_id", "product_id", "product_quantity", "trn_sum_from_iss",
            "trn_sum_from_red"]  # fmt: skip
    purchases = pd.DataFrame(rows, columns=cols)
    for month, part in purchases.groupby(
        purchases.transaction_datetime.dt.strftime("%Y-%m")
    ):
        (root / "purchases" / f"month={month}").mkdir(parents=True)
        part.to_parquet(root / "purchases" / f"month={month}" / "data_0.parquet")
    return ids, labeled


@pytest.fixture
def built(tmp_path):
    processed, out = tmp_path / "processed", tmp_path / "features"
    processed.mkdir()
    ids, labeled = _write_inputs(processed)
    shards = [
        pd.read_parquet(build_shard(processed, out, s, NUM_SHARDS))
        for s in range(NUM_SHARDS)
    ]
    merged = pd.read_parquet(merge_shards(out, NUM_SHARDS, processed))
    return processed, out, ids, labeled, shards, merged


def test_shards_partition_clients(built):
    processed, out, ids, _, shards, _ = built
    seen = [cid for s in shards for cid in s.client_id]
    assert sorted(seen) == ids  # complete and disjoint
    assert all(len(s) for s in shards)
    again = pd.read_parquet(build_shard(processed, out, 0, NUM_SHARDS))
    pd.testing.assert_frame_equal(again, shards[0])  # idempotent


def test_merge_schema(built):
    _, _, _, labeled, _, merged = built
    assert sorted(merged.client_id) == labeled
    assert list(merged.columns[:3]) == ["client_id", "treatment", "y"]
    assert str(merged.treatment.dtype).startswith("int")
    assert str(merged.y.dtype).startswith("int")
    features = merged.drop(columns=["client_id", "treatment", "y"])
    assert all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes)
    assert {"gender_f", "gender_m", "gender_u"} <= set(features.columns)
    assert (merged.gender_f + merged.gender_m + merged.gender_u == 1).all()
    row = merged.set_index("client_id").loc["c005"]
    assert row.n_transactions == 2
    assert row.n_products == 2
    assert row.total_spend == 200.0  # purchase_sum counted once per transaction
    assert row.regular_points_spent == 10.0  # sign flipped to positive
    assert row.redeemed_amount == 10.0  # null trn_sum_from_red means 0


def test_leakage_censoring(built):
    merged = built[-1].set_index("client_id")
    assert merged.loc["c000", "redeemed_before_cutoff"] == 0
    assert pd.isna(merged.loc["c000", "days_issue_to_redeem"])
    assert merged.loc["c001", "redeemed_before_cutoff"] == 0
    assert merged.loc["c003", "redeemed_before_cutoff"] == 1
    assert merged.loc["c003", "days_issue_to_redeem"] == pytest.approx(90.0)
    # c000 has the latest purchase, so it defines the cutoff.
    assert merged.loc["c000", "recency_days"] == 0
    assert merged.loc["c003", "tenure_days"] == pytest.approx(100.0)


def test_age_cleaning(built):
    merged = built[-1].set_index("client_id")
    assert merged.loc[["c000", "c001", "c002"], "age"].isna().all()
    assert merged.loc["c003", "age"] == 30


def test_merge_requires_every_shard(built):
    _, out, *_ = built
    (out / "shards" / "shard=2.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="shard=2"):
        merge_shards(out, NUM_SHARDS)
