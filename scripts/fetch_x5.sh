#!/usr/bin/env bash
# Download the X5 RetailHero uplift dataset, check it, and upload it to GCS.
# Usage: scripts/fetch_x5.sh [gs://bucket]   (no argument: download only)
# Files and checksums come from scikit-uplift's fetch_x5.
set -euo pipefail

base=https://sklift.s3.eu-west-2.amazonaws.com
dest=data/raw/x5
mkdir -p "$dest"

while read -r file md5; do
  path="$dest/$file"
  if [ ! -f "$path" ]; then
    curl -fsSL --retry 3 -o "$path.part" "$base/$file"
    mv "$path.part" "$path"
  fi
  actual=$(md5 -q "$path" 2>/dev/null || md5sum "$path" | cut -d' ' -f1)
  if [ "$actual" != "$md5" ]; then
    echo "checksum mismatch for $file; delete it and run again" >&2
    exit 1
  fi
  echo "ok $file"
done <<LIST
uplift_train.csv.gz 2720bbb659daa9e0989b2777b6a42d19
clients.csv.gz b9cdeb2806b732771de03e819b3354c5
purchases.csv.gz 48d2de13428e24e8b61d66fef02957a8
LIST

if [ $# -gt 0 ]; then
  gcloud storage cp "$dest"/*.csv.gz "$1/raw/x5/"
fi
