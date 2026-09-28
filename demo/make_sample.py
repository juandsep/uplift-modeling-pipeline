"""Write demo/scores_sample.csv, the only data the demo page reads.

Usage: python demo/make_sample.py path/to/scores.parquet

Input columns: uplift (float), treatment (0/1), y (0/1), and optionally split;
when split is present only the "test" rows are kept.
"""

import argparse
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "scores_sample.csv"
COLUMNS = ["uplift", "treatment", "y"]


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Keep the treatment x outcome mix of the full file."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    return df.groupby(["treatment", "y"]).sample(frac=frac, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scores", help="full scores Parquet file")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.scores)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    if "split" in df.columns:
        # Training rows would overstate the gain: keep the held-out clients only.
        df = df[df["split"] == "test"]
    sample = stratified_sample(df[COLUMNS], args.rows, args.seed)
    # Sorted here so the page only has to accumulate.
    sample = sample.sort_values("uplift", ascending=False)
    sample.to_csv(OUT, index=False, float_format="%.5f")
    print(f"wrote {len(sample)} rows to {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
