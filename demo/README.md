---
title: Uplift Targeting Demo
sdk: static
app_file: index.html
pinned: false
---

# Uplift targeting demo

Pick what share of clients to target, ranked by predicted uplift, and see how
many extra conversions that brings compared with targeting at random.

The page is plain HTML and JavaScript. It reads `scores_sample.csv` (columns
`uplift`, `treatment`, `y`, sorted by uplift), a 20,000-row sample of X5
RetailHero clients the model did not train on. It calls no API and needs no
credentials.

Incremental conversions for the top k% are estimated as the treated minus
control conversion rate inside that group, times the group size.

## Run locally

```bash
python -m http.server -d demo 8000   # then open http://localhost:8000
```

## Refresh the data and publish

From the scores the training DAG writes (`gs://<data bucket>/scores/x5/scores.parquet`):

```bash
uv run python demo/make_sample.py data/scores/x5/scores.parquet
hf upload sepulvedajd/uplift-targeting-demo demo . --type space --exclude make_sample.py
```

The Space is a static Space: Hugging Face only hosts Docker and Gradio Spaces
on a paid plan.
