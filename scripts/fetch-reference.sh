#!/usr/bin/env sh
# Fetch the upstream LemonGraph source we're rewriting, pinned for reproducibility.
# Lands in reference/ (gitignored) so it never enters our history or PR diffs.
set -eu

REPO=https://github.com/NationalSecurityAgency/lemongraph.git
REV=f7363c6fb14046bb3f4f2ca6c832076fb0430245 # master @ 2022-09-07
DEST="$(dirname "$0")/../reference/lemongraph"

if [ ! -d "$DEST/.git" ]; then
  git clone --quiet "$REPO" "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$REV"
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$REV"
echo "lemongraph @ $(git -C "$DEST" rev-parse --short HEAD) -> $DEST"
