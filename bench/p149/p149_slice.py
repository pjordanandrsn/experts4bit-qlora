#!/usr/bin/env python3
"""p149_slice.py -- Amendment 5's dry model (P149_DRY=1): the first LAYERS decoder layers of Mixtral-8x7B (eba92302),
with the embedding, the final norm and the head, copied byte for byte from the snapshot into one safetensors file, and
the config cut to num_hidden_layers = LAYERS. The tokenizer and generation config are copied unchanged. At LAYERS = 1
only shards 1 and 19 are read. An instrument input, never a reading (p149-dry-1 cut it on the lab A2000).

    python p149_slice.py --snapshot DIR --out DIR [--layers 1]
"""
import argparse
import json
import os
import shutil


def keep(name, layers):
    if not name.startswith("model.layers."):
        return True                                       # embed_tokens, norm, lm_head
    return int(name.split(".")[2]) < layers


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=1)
    a = ap.parse_args(argv)
    from safetensors import safe_open
    from safetensors.torch import save_file
    idx = json.load(open(os.path.join(a.snapshot, "model.safetensors.index.json")))
    want = sorted(n for n in idx["weight_map"] if keep(n, a.layers))
    by_file = {}
    for n in want:
        by_file.setdefault(idx["weight_map"][n], []).append(n)
    os.makedirs(a.out, exist_ok=True)
    tensors = {}
    for f, names in sorted(by_file.items()):
        with safe_open(os.path.join(a.snapshot, f), framework="pt") as h:
            for n in names:
                tensors[n] = h.get_tensor(n).clone()
    save_file(tensors, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(a.snapshot, "config.json")))
    cfg["num_hidden_layers"] = a.layers
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in os.listdir(a.snapshot):                     # the tokenizer and generation config, unchanged
        if f.startswith(("tokenizer", "special_tokens", "generation_config")):
            shutil.copyfile(os.path.join(a.snapshot, f), os.path.join(a.out, f))
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"SLICE layers={a.layers} tensors={len(tensors)} bytes={size} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
