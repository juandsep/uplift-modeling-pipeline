"""X5 RetailHero uplift dataset: raw gzipped CSVs to Parquet, streamed by DuckDB."""

import argparse
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


def _read_csv(path: Path, columns: dict[str, str]) -> str:
    cols = ", ".join(f"'{name}': '{kind}'" for name, kind in columns.items())
    return f"read_csv('{path}', header = true, columns = {{{cols}}})"


def convert_x5(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    """Write clients, uplift_train and monthly purchases; return row counts."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    # Lets DuckDB stream the COPY in parallel instead of buffering to keep order.
    con.execute("SET preserve_insertion_order = false")
    counts = {}
    for name, columns in [("uplift_train", UPLIFT_TRAIN), ("clients", CLIENTS)]:
        src = _read_csv(raw_dir / f"{name}.csv.gz", columns)
        dst = out_dir / f"{name}.parquet"
        row = con.execute(f"COPY (SELECT * FROM {src}) TO '{dst}'").fetchone()
        counts[name] = row[0] if row else 0
    src = _read_csv(raw_dir / "purchases.csv.gz", PURCHASES)
    row = con.execute(
        f"COPY (SELECT *, strftime(transaction_datetime, '%Y-%m') AS month FROM {src})"
        f" TO '{out_dir / 'purchases'}'"
        " (FORMAT parquet, PARTITION_BY (month), OVERWRITE true)"
    ).fetchone()
    counts["purchases"] = row[0] if row else 0
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/x5"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/x5"))
    args = parser.parse_args()
    for name, n in convert_x5(args.raw, args.out).items():
        print(f"{name}: {n:,} rows")


if __name__ == "__main__":
    main()
