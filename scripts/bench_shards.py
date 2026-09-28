"""Time X5 feature shards for several shard counts on synthetic purchases.

Not a test: run by hand, e.g. ``uv run python scripts/bench_shards.py``. Per
shard count it reports the summed and slowest shard wall time (shards run one
after another), the Parquet input each shard scans, and a checksum of all
shard features, which must not change with the shard count.
"""

import argparse
import tempfile
import time
from pathlib import Path

import duckdb

from uplift_pipeline.data.x5 import write_purchases
from uplift_pipeline.features.x5 import build_shard

PURCHASES = """
    SELECT
        printf('c%08d', c) AS client_id,
        printf('t%010d', t) AS transaction_id,
        TIMESTAMP '2018-11-21' + to_seconds((hash(t) % 10195200)::BIGINT)
            AS transaction_datetime,
        round(amount * 0.02, 1)::DOUBLE AS regular_points_received,
        0.0::DOUBLE AS express_points_received,
        -(hash(t, 1) % 3 * 10)::DOUBLE AS regular_points_spent,
        0.0::DOUBLE AS express_points_spent,
        amount AS purchase_sum,
        printf('s%03d', hash(c) % 400) AS store_id,
        printf('p%05d', hash(t, l) % 40000) AS product_id,
        (1 + l % 3)::DOUBLE AS product_quantity,
        round(amount / 3, 2)::DOUBLE AS trn_sum_from_iss,
        CASE WHEN l = 0 AND t % 5 = 0 THEN 10.0 END::DOUBLE AS trn_sum_from_red
    FROM (
        SELECT i AS t, hash(i, 2) % {clients} AS c, (hash(i, 3) % 50000) / 100
            AS amount
        FROM range({txns}) AS r(i)
    ), range(3) AS lines(l)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clients", type=int, default=200_000)
    parser.add_argument("--txns", type=int, default=2_000_000, help="3 lines each")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--shards", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as con:
        processed, out = Path(tmp, "processed"), Path(tmp, "features")
        processed.mkdir()
        con.execute(
            "COPY (SELECT printf('c%08d', i) AS client_id,"
            " TIMESTAMP '2017-06-01' + to_days((i % 500)::INTEGER)"
            " AS first_issue_date, NULL::TIMESTAMP AS first_redeem_date,"
            " (20 + i % 60)::INTEGER AS age, 'F' AS gender"
            f" FROM range({args.clients}) AS r(i))"
            f" TO '{processed / 'clients.parquet'}'"
        )
        con.execute(
            "CREATE TABLE purchases AS "
            + PURCHASES.format(clients=args.clients, txns=args.txns)
        )
        print("shards  sum_s  max_s  rows/shard  MB/shard  checksum")
        for n in args.shards:
            write_purchases(con, "purchases", processed, n)
            times, rows, size = [], 0, 0
            for shard in range(n):
                files = list(processed.glob(f"purchases/shard={shard}/*/*.parquet"))
                size += sum(f.stat().st_size for f in files)
                (count,) = con.execute(
                    f"SELECT count(*) FROM read_parquet({[str(f) for f in files]})"
                ).fetchone() or (0,)
                rows += count
                start = time.perf_counter()
                build_shard(processed, out, shard, n, args.threads)
                times.append(time.perf_counter() - start)
            shards = [str(out / "shards" / f"shard={i}.parquet") for i in range(n)]
            (checksum,) = con.execute(
                "SELECT md5(string_agg(f::VARCHAR, '|' ORDER BY client_id))"
                f" FROM read_parquet({shards}) AS f"
            ).fetchone() or ("",)
            print(
                f"{n:>6}  {sum(times):5.2f}  {max(times):5.2f}  {rows // n:>10,}"
                f"  {size / n / 2**20:8.1f}  {checksum[:12]}"
            )


if __name__ == "__main__":
    main()
