"""X5 per-client features, built in hash shards so workers can run in parallel.

Each shard owns the clients with ``hash(client_id) % num_shards == shard`` and is
independent of every other shard, so shards map 1:1 onto Airflow mapped tasks or
Cloud Run Job tasks. ``merge_shards`` is the single fan-in step. DuckDB's
``hash`` is stable for a given DuckDB version (pinned in uv.lock); all shards of
one run must use the same version.

Cutoff and leakage: the uplift campaign (treatment and target) happens after the
purchase history ends, so the cutoff is the last observed purchase,
``max(transaction_datetime)`` over all purchases (2019-03-18 on the real data).
Every shard derives it from the same files, so all shards agree. All time
features are measured back from it, and ``first_redeem_date`` (which runs to
2019-11, well into the campaign) is censored at it: a redemption after the
cutoff counts as "not redeemed yet".
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb

DAY = 86400.0
RECENT_DAYS = 30


def _purchases(processed_dir: Path) -> str:
    return f"read_parquet('{processed_dir}/purchases/*/*.parquet', hive_partitioning=1)"


def build_shard(
    processed_dir: Path, out_dir: Path, shard: int, num_shards: int, threads: int = 2
) -> Path:
    """Write the features of one client shard; rerunning overwrites it."""
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    dst = out_dir / "shards" / f"shard={shard}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={"threads": threads})
    # Parallel float sums change in the last bit with thread scheduling; summing
    # as DECIMAL is exact, so a rerun (or another shard count) is bit-identical.
    for fn in ("sum", "avg"):
        con.execute(f"CREATE MACRO d{fn}(x) AS {fn}(x::DECIMAL(18, 4))::DOUBLE")
    row = con.execute(
        f"SELECT max(transaction_datetime) FROM {_purchases(processed_dir)}"
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"no purchases under {processed_dir}")
    cutoff = f"TIMESTAMP '{row[0]}'"
    recent = f"{cutoff} - INTERVAL {RECENT_DAYS} DAY"
    in_shard = f"hash(client_id) % {num_shards} = {shard}"
    # ponytail: every shard scans all purchase files and drops ~(N-1)/N of the
    # rows (about +45% total CPU at 8 shards); bucket purchases by shard at
    # ingestion if that read amplification starts to dominate.
    # NOT MATERIALIZED: scan the shard's lines twice instead of buffering them
    # in memory (peak RSS per shard ~1.4 GB -> ~0.4 GB at 8 shards).
    query = f"""
    WITH lines AS NOT MATERIALIZED (
        SELECT * FROM {_purchases(processed_dir)} WHERE {in_shard}
    ),
    clients AS (
        SELECT * FROM '{processed_dir}/clients.parquet' WHERE {in_shard}
    ),
    txn AS (  -- header fields repeat on every product line of a transaction
        SELECT
            client_id,
            transaction_id,
            any_value(transaction_datetime) AS ts,
            any_value(month) AS month,
            any_value(store_id) AS store_id,
            any_value(purchase_sum) AS purchase_sum,
            any_value(regular_points_received) AS reg_received,
            any_value(express_points_received) AS exp_received,
            -any_value(regular_points_spent) AS reg_spent,
            -any_value(express_points_spent) AS exp_spent,
            count(*) AS n_lines,
            dsum(product_quantity) AS quantity,
            dsum(coalesce(trn_sum_from_red, 0)) AS paid_redeemed,
            dsum(coalesce(trn_sum_from_iss, 0)) AS paid_issued
        FROM lines
        GROUP BY client_id, transaction_id
    ),
    per_client AS (
        SELECT
            client_id,
            count(*) AS n_transactions,
            count(DISTINCT store_id) AS n_stores,
            count(DISTINCT month) AS active_months,
            count(*) / count(DISTINCT month) AS txn_per_active_month,
            dsum(purchase_sum) AS total_spend,
            davg(purchase_sum) AS mean_spend,
            max(purchase_sum) AS max_spend,
            (epoch({cutoff}) - epoch(max(ts))) / {DAY} AS recency_days,
            (epoch({cutoff}) - epoch(min(ts))) / {DAY} AS first_purchase_days,
            (epoch(max(ts)) - epoch(min(ts))) / {DAY} / nullif(count(*) - 1, 0)
                AS mean_days_between_txn,
            count(*) FILTER (ts > {recent}) AS n_transactions_30d,
            dsum(CASE WHEN ts > {recent} THEN purchase_sum ELSE 0 END) AS spend_30d,
            avg(n_lines) AS mean_basket_lines,
            davg(quantity) AS mean_basket_quantity,
            dsum(reg_received) AS regular_points_received,
            dsum(exp_received) AS express_points_received,
            dsum(reg_spent) AS regular_points_spent,
            dsum(exp_spent) AS express_points_spent,
            avg((exp_received > 0)::INTEGER) AS express_received_txn_share,
            dsum(paid_redeemed) AS redeemed_amount,
            dsum(paid_redeemed) / nullif(dsum(paid_redeemed + paid_issued), 0)
                AS redeemed_spend_share,
            avg((paid_redeemed > 0)::INTEGER) AS redeem_txn_share
        FROM txn
        GROUP BY client_id
    ),
    products AS (
        SELECT client_id, count(DISTINCT product_id) AS n_products
        FROM lines
        GROUP BY client_id
    )
    SELECT
        c.client_id,
        CASE WHEN c.age BETWEEN 14 AND 100 THEN c.age END AS age,
        (c.gender = 'F')::INTEGER AS gender_f,
        (c.gender = 'M')::INTEGER AS gender_m,
        (c.gender = 'U')::INTEGER AS gender_u,
        (epoch({cutoff}) - epoch(c.first_issue_date)) / {DAY} AS tenure_days,
        coalesce(c.first_redeem_date <= {cutoff}, false)::INTEGER
            AS redeemed_before_cutoff,
        CASE WHEN c.first_redeem_date <= {cutoff}
            AND c.first_redeem_date >= c.first_issue_date
            THEN (epoch(c.first_redeem_date) - epoch(c.first_issue_date)) / {DAY}
        END AS days_issue_to_redeem,
        p.* EXCLUDE (client_id),
        pr.n_products
    FROM clients AS c
    LEFT JOIN per_client AS p USING (client_id)
    LEFT JOIN products AS pr USING (client_id)
    ORDER BY c.client_id
    """
    # Write then rename so a crashed worker never leaves a half-written shard.
    tmp = dst.with_suffix(".tmp")
    con.execute(f"COPY ({query}) TO '{tmp}' (FORMAT parquet)")
    tmp.replace(dst)
    return dst


def merge_shards(
    out_dir: Path, num_shards: int, processed_dir: Path = Path("data/processed/x5")
) -> Path:
    """Join all shards with the uplift labels into ``client_features.parquet``."""
    out_dir, processed_dir = Path(out_dir), Path(processed_dir)
    shards = [out_dir / "shards" / f"shard={i}.parquet" for i in range(num_shards)]
    missing = [p.name for p in shards if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing shards: {', '.join(missing)}")
    files = ", ".join(f"'{p}'" for p in shards)
    labels = processed_dir / "uplift_train.parquet"
    dst = out_dir / "client_features.parquet"
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE merged AS
        SELECT
            f.client_id,
            u.treatment_flg::INTEGER AS treatment,
            u.target::INTEGER AS y,
            f.* EXCLUDE (client_id)
        FROM read_parquet([{files}], hive_partitioning=0) AS f
        JOIN '{labels}' AS u USING (client_id)
        ORDER BY f.client_id
    """)
    # Stale shards from a run with another num_shards would duplicate or drop
    # clients; refuse to write rather than train on that.
    got = con.execute(
        "SELECT count(*), count(DISTINCT client_id) FROM merged"
    ).fetchone()
    want = con.execute(f"SELECT count(*) FROM '{labels}'").fetchone()
    if got is None or want is None or got != (want[0], want[0]):
        raise ValueError(f"merged rows/distinct {got} != labeled clients {want}")
    con.execute(f"COPY merged TO '{dst}' (FORMAT parquet)")
    return dst


def build_all(
    processed_dir: Path, out_dir: Path, num_shards: int, threads: int
) -> Path:
    """Run every shard in parallel local processes, then merge."""
    with ProcessPoolExecutor(max_workers=num_shards) as pool:
        futures = [
            pool.submit(build_shard, processed_dir, out_dir, i, num_shards, threads)
            for i in range(num_shards)
        ]
        for f in futures:
            f.result()
    return merge_shards(out_dir, num_shards, processed_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("shard", "merge", "all"):
        p = sub.add_parser(name)
        p.add_argument("--processed", type=Path, default=Path("data/processed/x5"))
        p.add_argument("--out", type=Path, default=Path("data/features/x5"))
        p.add_argument("--num-shards", type=int, required=True)
    sub.choices["shard"].add_argument("--shard", type=int, required=True)
    sub.choices["shard"].add_argument("--threads", type=int, default=2)
    # Default: split the machine's cores evenly across the parallel shards.
    sub.choices["all"].add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.cmd == "shard":
        path = build_shard(
            args.processed, args.out, args.shard, args.num_shards, args.threads
        )
    elif args.cmd == "merge":
        path = merge_shards(args.out, args.num_shards, args.processed)
    else:
        threads = args.threads or max(1, (os.cpu_count() or 1) // args.num_shards)
        path = build_all(args.processed, args.out, args.num_shards, threads)
    print(path)


if __name__ == "__main__":
    main()
