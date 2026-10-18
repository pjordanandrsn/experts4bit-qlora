"""HP1 shakedown fixture (HP1-PREREG.md, "Shakedown"): the first N decoder layers of the pinned checkpoint (real weights), laid out as
an offline HF cache repo `local/<name>-L<N>` at a derived 40-hex revision, so the TC1 harness and hp1_eval.py load it through their
normal --model/--revision path with HF_HUB_CACHE pointed at --cache. Correctness only: never a scored number.

    python hp1_trunc.py --src <snapshot dir of the pin> --cache <dir> [--layers 2]
"""
import argparse
import hashlib
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--layers", type=int, default=2)
a = ap.parse_args()
N, SRC = a.layers, a.src.rstrip("/")
pin = os.path.basename(SRC)
name = os.path.basename(os.path.dirname(os.path.dirname(SRC))).replace("models--", "").split("--")[-1]
REV = hashlib.sha1(f"{name}@{pin} first {N} layers".encode()).hexdigest()
ROOT = os.path.join(a.cache, f"models--local--{name}-L{N}")
DST = f"{ROOT}/snapshots/{REV}"
os.makedirs(DST, exist_ok=True)
os.makedirs(f"{ROOT}/refs", exist_ok=True)
open(f"{ROOT}/refs/main", "w").write(REV)
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]


def keep(k):
    return (not k.startswith("model.layers.")) or int(k.split(".")[2]) < N


by_file = {}
for k, f in idx.items():
    if keep(k):
        by_file.setdefault(f, []).append(k)
tensors = {}
for f, ks in sorted(by_file.items()):
    with safe_open(f"{SRC}/{f}", "pt") as fh:
        for k in ks:
            tensors[k] = fh.get_tensor(k)
save_file(tensors, f"{DST}/model.safetensors", metadata={"format": "pt"})
cfg = json.load(open(f"{SRC}/config.json"))
cfg.update(num_hidden_layers=N, max_window_layers=N)
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
for f in ("generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
    shutil.copy(f"{SRC}/{f}", f"{DST}/{f}")
print(json.dumps({"model": f"local/{name}-L{N}", "revision": REV, "hf_hub_cache": a.cache,
                  "tensors": len(tensors), "bytes": os.path.getsize(f"{DST}/model.safetensors")}))
