#!/usr/bin/env python
"""A layer slice of a released checkpoint: the first N decoder layers of the REAL weights, as a local snapshot directory.

Shards are symlinked (nothing is copied); the index lists only the kept layers' tensors plus every non-layer tensor; the
config's layer count and every per-layer list (``layer_types``, ``layers_block_type``, ...) are truncated to N. The loader
then reads only those tensors, so a 60 GB checkpoint's harness path can be exercised in minutes on a slow pool. A slice is
a real-weight harness smoke and a per-layer profile -- never a model-quality or parity receipt (its outputs are a truncated
network's).

    python slice_snapshot.py <model_id_or_dir> <n_layers> <out_dir> [--revision R]
"""
from __future__ import annotations

import json
import os
import re
import sys


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rev = None
    if "--revision" in sys.argv:
        rev = sys.argv[sys.argv.index("--revision") + 1]
        args = [a for a in args if a != rev]
    src, n, out = args[0], int(args[1]), args[2]
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, revision=rev, local_files_only=True)
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    full = tcfg["num_hidden_layers"]
    assert 0 < n <= full, (n, full)
    tcfg["num_hidden_layers"] = n
    for k, v in list(tcfg.items()):
        if isinstance(v, list) and len(v) == full and ("layer" in k or "block" in k):
            tcfg[k] = v[:n]
    for k in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
        if k in tcfg:
            tcfg[k] = 0
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    idx_path = os.path.join(src, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    keep = {}
    for k, f in idx["weight_map"].items():
        m = pat.search(k)
        if m and int(m.group(1)) >= n:
            continue
        if "mtp" in k:
            continue
        keep[k] = f
    json.dump({"metadata": idx.get("metadata", {}), "weight_map": keep}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    for f in os.listdir(src):
        p = os.path.join(out, f)
        if f in ("config.json", "model.safetensors.index.json") or os.path.exists(p):
            continue
        os.symlink(os.path.realpath(os.path.join(src, f)), p)
    print(f"{out}: {n}/{full} layers, {len(keep)}/{len(idx['weight_map'])} tensors")


if __name__ == "__main__":
    main()
