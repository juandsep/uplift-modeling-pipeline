"""Write demo/scores_sample.parquet, the only data the demo app reads.

Real scores:  python demo/make_sample.py path/to/scores.parquet
Placeholder:  uv run python demo/make_sample.py --synthetic   (needs the repo package)

Input columns: client_id (str), uplift (float), treatment (0/1), y (0/1).
"""

import argparse
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "scores_sample.parquet"
COLUMNS = ["client_id", "uplift", "treatment", "y"]


def synthetic_scores() -> pd.DataFrame:
    # Train on one half of the synthetic data, score the other half.
    from uplift_pipeline.data import load_training_data
    from uplift_pipeline.models import UpliftModel

    df, features = load_training_data(n_samples=20_000, seed=42)
    train = df.sample(frac=0.5, random_state=0)
    test = df.drop(train.index)
    model = UpliftModel(features).fit(train)
    return pd.DataFrame(
        {
            "client_id": [f"synthetic-{i:06d}" for i in test.index],
            "uplift": model.predict(None, test),
            "treatment": test["treatment"].to_numpy(),
            "y": test["y"].to_numpy(),
        }
    )


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Keep the treatment x outcome mix of the full file."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    return df.groupby(["treatment", "y"]).sample(frac=frac, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scores", nargs="?", help="full scores Parquet file")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if bool(args.scores) == args.synthetic:
        parser.error("pass a scores file or --synthetic, not both")

    df = synthetic_scores() if args.synthetic else pd.read_parquet(args.scores)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    if "split" in df.columns:
        # Training rows would overstate the gain: keep the held-out clients only.
        df = df[df["split"] == "test"]
    df = df[COLUMNS].astype(
        {"client_id": str, "uplift": "float32", "treatment": "int8", "y": "int8"}
    )
    sample = stratified_sample(df, args.rows, args.seed).reset_index(drop=True)
    sample.to_parquet(OUT, index=False)
    print(f"wrote {len(sample)} rows to {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
