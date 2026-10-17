"""DeepSeek-V4.1-Flash-1017, gate A: parameter and byte counts per component, from metadata only.

Reads the safetensors index and each shard's header (the 8-byte length prefix and the JSON header, by HTTP range
requests), never a tensor byte. Prints parameters (FP4 experts count two per I8 byte; quantization scales are not
parameters) and stored bytes per component.

    python ds41_metadata.py [--revision SHA] [--headers DIR]

``--headers DIR`` reads ``<shard>.header.json`` files saved by an earlier run instead of fetching them.
"""
import argparse
import collections
import json
import math
import os
import struct
import urllib.request

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def fetch(url, first=None, last=None):
    req = urllib.request.Request(url)
    if first is not None:
        req.add_header("Range", f"bytes={first}-{last}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def header(base, shard, cache):
    path = os.path.join(cache, shard + ".header.json") if cache else None
    if path and os.path.exists(path):
        return json.load(open(path))
    n = struct.unpack("<Q", fetch(base + shard, 0, 7))[0]
    raw = fetch(base + shard, 8, 8 + n - 1)
    if len(raw) != n:
        raise SystemExit(f"{shard}: {len(raw)} of {n} header bytes (range not honoured?)")
    if path:
        open(path, "wb").write(raw)
    return json.loads(raw)


def component(n):
    if n.startswith(("vision.", "aligner.", "image_")):
        return "vision + aligner"
    if ".engram.embed." in n:
        return "Engram tables"
    if ".engram." in n:
        return "Engram projections"
    if n.startswith("mtp."):
        return "MTP / DSpark routed experts" if ".ffn.experts." in n else "MTP / DSpark other"
    if ".ffn.experts." in n:
        return "routed experts (40 layers)"
    if ".ffn.shared_experts." in n or ".ffn.gate." in n:
        return "shared experts + router"
    if ".attn." in n:
        return "attention (+ compressor, indexer)"
    if ".hc_" in n:
        return "hyper-connections"
    if n in ("embed.weight", "head.weight"):
        return "embedding + head"
    return "norms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--headers", default=None, help="a directory to read (or save) the shard headers")
    a = ap.parse_args()
    base = f"https://huggingface.co/{REPO}/resolve/{a.revision}/"
    if a.headers:
        os.makedirs(a.headers, exist_ok=True)
    index = json.loads(fetch(base + "model.safetensors.index.json"))
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        h = header(base, shard, a.headers)
        h.pop("__metadata__", None)
        tensors.update(h)
    params, stored = collections.Counter(), collections.Counter()
    for n, v in tensors.items():
        first, last = v["data_offsets"]
        c = component(n)
        stored[c] += last - first
        if n.endswith(".scale"):
            continue
        k = math.prod(v["shape"])
        params[c] += 2 * k if (v["dtype"] == "I8" and ".experts." in n) else k
    for c in sorted(params, key=lambda c: -params[c]):
        print(f"{c:36s} {params[c] / 1e9:8.2f} B params {stored[c] / 1e9:8.2f} GB")
    print(f"{'total':36s} {sum(params.values()) / 1e9:8.2f} B params {sum(stored.values()) / 1e9:8.2f} GB "
          f"(index total_size {index['metadata']['total_size'] / 1e9:.2f} GB)")
    text = sum(params.values()) - sum(params[c] for c in ("Engram tables", "vision + aligner", "MTP / DSpark routed experts",
                                                           "MTP / DSpark other"))
    print(f"text backbone without the Engram tables, vision and MTP: {text / 1e9:.2f} B params")


if __name__ == "__main__":
    main()
