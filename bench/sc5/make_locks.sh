#!/bin/bash
# bench/sc5/make_locks.sh -- SC5's provenance locks, regenerated from nothing but PyPI (SC5-PREREG.md, Provenance).
#
#   bash bench/sc5/make_locks.sh [OUTDIR]        # default: bench/sc5/locks
#
# Writes, for the box's platform (x86_64, manylinux_2_34, CPython 3.12):
#   vllm.lock.txt / sglang.lock.txt  the competitor's whole dependency closure, every distribution's sha256
#                                    (uv pip compile --generate-hashes); the box installs it with
#                                    `pip install --require-hashes --no-deps -r <lock>` into the framework's own venv
#   e4b-wheels.lock                  grouped-nf4-gemm and experts4bit-qlora: one line per release wheel,
#                                    "<name> <version> <filename> <sha256> <url>", read from PyPI's JSON (never typed)
# Needs uv (pip install uv) and network. Re-running it on another day can only differ where PyPI changed a release's
# files, which it does not allow; a diff against the committed locks is therefore a finding, not noise.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/locks}
UV=${UV:-uv}
PLATFORM=x86_64-manylinux_2_34; PYVER=3.12
VLLM_VERSION=0.31.0; SGLANG_VERSION=0.5.21
GNF4_VERSION=0.45.0; E4B_VERSION=0.52.0
mkdir -p "$OUT"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
printf 'vllm==%s\nhuggingface_hub>=0.23\n' "$VLLM_VERSION" > "$TMP/vllm.in"
printf 'sglang==%s\nninja\n' "$SGLANG_VERSION" > "$TMP/sglang.in"
for f in vllm sglang; do
  "$UV" pip compile "$TMP/$f.in" --python-platform "$PLATFORM" --python-version "$PYVER" --generate-hashes \
      --no-header --quiet -o "$TMP/$f.lock.txt"
  { printf '# SC5 %s lock: uv pip compile --python-platform %s --python-version %s --generate-hashes\n' "$f" "$PLATFORM" "$PYVER"
    printf '# input: %s\n' "$(tr '\n' ' ' < "$TMP/$f.in")"
    cat "$TMP/$f.lock.txt"; } > "$OUT/$f.lock.txt"
done
python3 - "$OUT/e4b-wheels.lock" "$GNF4_VERSION" "$E4B_VERSION" <<'PY'
import json, sys, urllib.request
out, gnf4, e4b = sys.argv[1:4]
rows = ["# SC5 e4b wheels: <name> <version> <filename> <sha256> <url>, read from https://pypi.org/pypi/<name>/<version>/json"]
for name, ver in (("grouped-nf4-gemm", gnf4), ("experts4bit-qlora", e4b)):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=60) as r:
        meta = json.load(r)
    wheels = [u for u in meta["urls"] if u["packagetype"] == "bdist_wheel"]
    if len(wheels) != 1:
        sys.exit(f"{name} {ver}: expected one wheel on PyPI, found {len(wheels)}")
    w = wheels[0]
    rows.append(f"{name} {ver} {w['filename']} {w['digests']['sha256']} {w['url']}")
open(out, "w").write("\n".join(rows) + "\n")
PY
for f in vllm sglang; do
  printf '%s: %s packages\n' "$f" "$(grep -c '^[A-Za-z0-9]' "$OUT/$f.lock.txt")"
done
cat "$OUT/e4b-wheels.lock"
