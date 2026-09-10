#!/usr/bin/env python3
"""tp4_alpaca.py -- the fixed Alpaca text for lane tp4 (TP4-PREREG.md "Fixture"), built deterministically.

Source: the dataset the Unsloth notebooks load -- `unsloth/alpaca-cleaned`, file `alpaca_data_cleaned.json`
(51,760 rows of {instruction, input, output}), pinned by dataset revision AND by the file's sha256. The rows
are shuffled once with a fixed seed and split into a 1,200-row train set and a 48-row held-out set; the
output JSON is written with sorted keys so its sha256 is reproducible and is what every receipt records.

    tp4_alpaca.py --out ds_alpaca.json [--src alpaca_data_cleaned.json]   # builds (downloads if --src absent)
    tp4_alpaca.py --check ds_alpaca.json                                  # prints the sha256 only

Nothing here tokenises; tp4_arm.py --prepare does that per family with the Alpaca prompt template.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import urllib.request

REPO_ID = "unsloth/alpaca-cleaned"
REVISION = "0fe581eb78617869b6e17dd195a3fe4045b3723d"       # dataset revision (HF API, 2026-09-10)
FILE = "alpaca_data_cleaned.json"
FILE_SHA256 = "bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b"   # LFS sha256 of FILE at REVISION
SEED = 3407           # the Unsloth notebooks' random_state / seed
N_TRAIN, N_EVAL = 1200, 48


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dst):
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{FILE}"
    req = urllib.request.Request(url, headers={"User-Agent": "tp4_alpaca/1.0"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    return dst


def build(src, out):
    got = sha256_file(src)
    if got != FILE_SHA256:
        print(f"SOURCE MISMATCH {src}: sha256 {got} != registered {FILE_SHA256}")
        sys.exit(13)
    rows = json.load(open(src))
    assert isinstance(rows, list) and len(rows) == 51760, len(rows)
    idx = list(range(len(rows)))
    random.Random(SEED).shuffle(idx)
    pick = [rows[i] for i in idx[: N_TRAIN + N_EVAL]]
    clean = [{"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]} for r in pick]
    ds = {"source": {"repo_id": REPO_ID, "revision": REVISION, "file": FILE, "file_sha256": FILE_SHA256},
          "seed": SEED, "n_train": N_TRAIN, "n_eval": N_EVAL, "train": clean[:N_TRAIN], "eval": clean[N_TRAIN:]}
    body = json.dumps(ds, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with open(out, "wb") as f:
        f.write(body)
    print(f"DS alpaca n_train={N_TRAIN} n_eval={N_EVAL} seed={SEED} sha256={hashlib.sha256(body).hexdigest()} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ds_alpaca.json")
    ap.add_argument("--src", default=None, help="a local copy of alpaca_data_cleaned.json (downloaded from the pinned revision when absent)")
    ap.add_argument("--check", default=None, help="print the sha256 of an existing ds file and exit")
    a = ap.parse_args()
    if a.check:
        print(sha256_file(a.check))
        return
    src = a.src or fetch(os.path.join(os.path.dirname(os.path.abspath(a.out)) or ".", FILE))
    build(src, a.out)


if __name__ == "__main__":
    main()
