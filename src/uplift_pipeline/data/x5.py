"""X5 RetailHero uplift dataset: raw gzipped CSVs to Parquet, streamed by DuckDB."""

import argparse
import json
from pathlib import Path

import duckdb

UPLIFT_TRAIN = {"client_id": "VARCHAR", "treatment_flg": "TINYINT", "target": "TINYINT"}
CLIENTS = {
    "client_id": "VARCHAR",
    "first_issue_date": "TIMESTAMP",
    "first_redeem_date": "TIMESTAMP",
    "age": "INTEGER",
    "gender": "VARCHAR",
}
PURCHASES = {
    "client_id": "VARCHAR",
    "transaction_id": "VARCHAR",
    "transaction_datetime": "TIMESTAMP",
    "regular_points_received": "DOUBLE",
    "express_points_received": "DOUBLE",
    "regular_points_spent": "DOUBLE",
    "express_points_spent": "DOUBLE",
    "purchase_sum": "DOUBLE",
    "store_id": "VARCHAR",
    "product_id": "VARCHAR",
    "product_quantity": "DOUBLE",
    "trn_sum_from_iss": "DOUBLE",
    "trn_sum_from_red": "DOUBLE",
}
# Next to the purchase buckets: the shard count they were written for, and cutoff.
LAYOUT = "_layout.json"


def _read_csv(path: Path, columns: dict[str, str]) -> str:
    cols = ", ".join(f"'{name}': '{kind}'" for name, kind in columns.items())
    return f"read_csv('{path}', header = true, columns = {{{cols}}})"


def write_purchases(
    con: duckdb.DuckDBPyConnection, src: str, out_dir: Path, num_shards: int
) -> int:
    """Write purchases/shard=K/month=M/ and its layout; return the row count.

    Shard K holds the clients with ``hash(client_id) % num_shards == K``, so a
    feature shard reads only its own files. The cutoff (last purchase over all
    shards) is stored in the layout so every shard agrees on it.
    """
    dst = Path(out_dir) / "purchases"
    row = con.execute(
        f"COPY (SELECT *, hash(client_id) % {num_shards} AS shard,"
        f" strftime(transaction_datetime, '%Y-%m') AS month FROM {src})"
        f" TO '{dst}' (FORMAT parquet, PARTITION_BY (shard, month), OVERWRITE true)"
    ).fetchone()
    cutoff = con.execute(
        f"SELECT max(transaction_datetime) FROM '{dst}/*/*/*.parquet'"
    ).fetchone()
    if cutoff is None or cutoff[0] is None:
        raise ValueError(f"no purchases in {src}")
    layout = {"num_shards": num_shards, "cutoff": str(cutoff[0])}
    (dst / LAYOUT).write_text(json.dumps(layout))
    return row[0] if row else 0


def read_layout(processed_dir: Path, num_shards: int) -> dict:
    """Return the purchases layout; fail if it was written for another N."""
    path = Path(processed_dir) / "purchases" / LAYOUT
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; rerun ingestion")
    layout = json.loads(path.read_text())
    if layout["num_shards"] != num_shards:
        raise ValueError(
            f"purchases are bucketed for num_shards={layout['num_shards']}, not "
            f"{num_shards}; rerun ingestion with --num-shards {num_shards}"
        )
    return layout


def convert_x5(raw_dir: Path, out_dir: Path, num_shards: int) -> dict[str, int]:
    """Write clients, uplift_train and bucketed purchases; return row counts."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        # Lets DuckDB stream the COPY in parallel instead of buffering to keep order.
        con.execute("SET preserve_insertion_order = false")
        counts = {}
        for name, columns in [("uplift_train", UPLIFT_TRAIN), ("clients", CLIENTS)]:
            src = _read_csv(raw_dir / f"{name}.csv.gz", columns)
            dst = out_dir / f"{name}.parquet"
            row = con.execute(f"COPY (SELECT * FROM {src}) TO '{dst}'").fetchone()
            counts[name] = row[0] if row else 0
        src = _read_csv(raw_dir / "purchases.csv.gz", PURCHASES)
        counts["purchases"] = write_purchases(con, src, out_dir, num_shards)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/x5"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/x5"))
    # Must match the --num-shards of the feature build.
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    for name, n in convert_x5(args.raw, args.out, args.num_shards).items():
        print(f"{name}: {n:,} rows")


if __name__ == "__main__":
    main()
