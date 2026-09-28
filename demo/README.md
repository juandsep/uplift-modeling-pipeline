---
title: Uplift Targeting Demo
sdk: docker
app_port: 8501
pinned: false
---

# Uplift targeting demo

Pick what share of clients to target, ranked by predicted uplift, and see how
many extra conversions that brings compared with targeting at random.

The app reads precomputed scores from `scores_sample.parquet` (columns
`client_id`, `uplift`, `treatment`, `y`). It calls no API and needs no
credentials. The sample holds only clients the model did not train on.

Incremental conversions for the top k% are estimated as the treated minus
control conversion rate inside that group, times the group size.

## Run locally

From the repository root:

```bash
uv run --with-requirements demo/requirements.txt streamlit run demo/app.py
```

Or with pip:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

The Space builds `Dockerfile` (Hugging Face no longer offers a Streamlit
SDK, only Docker). Publish or update it with:

```bash
hf upload sepulvedajd/uplift-targeting-demo demo . --type space --exclude make_sample.py
```

## Refresh the data

From a full scores Parquet file (any size, same columns):

```bash
uv run python demo/make_sample.py path/to/scores.parquet
```

This writes a stratified sample of 20,000 rows (fixed seed) to
`demo/scores_sample.parquet`. Use `--rows` to change the size.

To rebuild the synthetic placeholder instead (trains a quick model with the
repository package):

```bash
uv run python demo/make_sample.py --synthetic
```
