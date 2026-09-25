"""X5 ingestion: tiny gzipped CSVs in, typed and month-partitioned Parquet out."""

import gzip

import duckdb

from uplift_pipeline.data.x5 import convert_x5

UPLIFT_TRAIN = """client_id,treatment_flg,target
000012768d,0,1
000036f903,1,1
"""
CLIENTS = """client_id,first_issue_date,first_redeem_date,age,gender
000012768d,2017-08-05 15:40:48,2018-01-04 19:30:07,45,U
000036f903,2017-04-10 13:54:23,,72,F
"""
PURCHASES = """client_id,transaction_id,transaction_datetime,regular_points_received,\
express_points_received,regular_points_spent,express_points_spent,purchase_sum,\
store_id,product_id,product_quantity,trn_sum_from_iss,trn_sum_from_red
000012768d,7e3e2e3984,2018-12-01 07:12:45,10.0,0.0,0.0,0.0,1007.0,\
54a4a11a29,9a80204f78,2.0,80.0,
000012768d,7e3e2e3984,2018-12-01 07:12:45,10.0,0.0,0.0,0.0,1007.0,\
54a4a11a29,da89ebd374,1.0,65.0,
000036f903,0a1b2c3d4e,2019-01-15 20:01:02,0.0,0.0,-5.0,0.0,300.5,\
54a4a11a29,9a80204f78,1.0,,300.5
"""


def test_convert_x5(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    for name, text in [
        ("uplift_train", UPLIFT_TRAIN),
        ("clients", CLIENTS),
        ("purchases", PURCHASES),
    ]:
        with gzip.open(raw / f"{name}.csv.gz", "wt") as f:
            f.write(text)

    counts = convert_x5(raw, out)

    assert counts == {"uplift_train": 2, "clients": 2, "purchases": 3}
    assert sorted(p.name for p in (out / "purchases").iterdir()) == [
        "month=2018-12",
        "month=2019-01",
    ]
    con = duckdb.connect()

    def types(path: str) -> dict[str, str]:
        rows = con.execute(f"DESCRIBE SELECT * FROM {path}").fetchall()
        return {r[0]: r[1] for r in rows}

    train = types(f"'{out / 'uplift_train.parquet'}'")
    assert train == {
        "client_id": "VARCHAR",
        "treatment_flg": "TINYINT",
        "target": "TINYINT",
    }
    clients = types(f"'{out / 'clients.parquet'}'")
    assert clients["first_redeem_date"] == "TIMESTAMP"
    assert clients["age"] == "INTEGER"
    purchases = types(
        f"read_parquet('{out}/purchases/*/*.parquet', hive_partitioning=1)"
    )
    assert purchases["client_id"] == "VARCHAR"
    assert purchases["transaction_datetime"] == "TIMESTAMP"
    assert purchases["purchase_sum"] == "DOUBLE"
    assert purchases["month"] == "VARCHAR"
    clients_path = out / "clients.parquet"
    null_redeem = con.execute(
        f"SELECT count(*) FROM '{clients_path}' WHERE first_redeem_date IS NULL"
    ).fetchone()
    assert null_redeem == (1,)
