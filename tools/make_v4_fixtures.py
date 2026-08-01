#!/usr/bin/env python3
"""Regenerate the DeepSeek-V4 test fixtures from a real checkpoint.

Several tests assert against the CHECKPOINT'S OWN BYTES rather than synthetic tensors --
that is the point of them, since the failures they catch (a value-cast on an e8m0 scale,
a nibble order, an inverted module nesting) are invisible on data you generated yourself.
But a 161 GB checkpoint cannot live in the repo, so those tests skip unless the fixtures
are present, and a permanently-skipped test is not coverage.

This produces all three, ~62 MB total, from a local DeepSeek-V4 snapshot::

    python tools/make_v4_fixtures.py --ckpt /path/to/DeepSeek-V4-Flash --out ./fixtures
    E4B_V4_TENSORS=./fixtures/v4_expert.safetensors \\
    E4B_V4_KEYS=./fixtures/v4_keys_l0_5.txt \\
    E4B_V4_CONFIG=./fixtures/v4cfg \\
        pytest tests/test_fp8_blocks.py tests/test_deepseek_v4_keys.py

Runs on plain python3 -- no torch, no safetensors package. safetensors is just
``[u64 header_len][json header][data]``, so slicing one is a seek and a copy, which also
means this works on a machine too old to *name* the dtypes involved.

Every scale tensor is emitted TWICE: once with the checkpoint's own ``F8_E8M0`` label and
once relabelled ``U8`` over identical bytes. torch < 2.7 cannot materialize ``F8_E8M0`` at
all, and torch >= 2.7 hands back a FLOAT whose ``.to(int32)`` is the value rather than the
exponent byte -- so the ``.as_u8`` copy is what a correct reader must end up with, and
having both in one file lets a test show the difference instead of asserting it.
"""
import argparse
import json
import os
import struct

# One layer's worth of MXFP4 experts + the FP8 dense tensors the tests exercise.
LAYER, EXPERTS = 18, (0,)
DENSE = ("attn.wq_a", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1")
KEY_LAYERS = 6          # test_deepseek_v4_keys builds a 6-layer skeleton to compare against


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def shards(ckpt):
    return sorted(f for f in os.listdir(ckpt) if f.endswith(".safetensors"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="a DeepSeek-V4 snapshot directory")
    ap.add_argument("--out", default="fixtures")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "v4cfg"), exist_ok=True)

    wanted = [f"layers.{LAYER}.ffn.experts.{e}.{p}.{k}"
              for e in EXPERTS for p in ("w1", "w2", "w3") for k in ("weight", "scale")]
    wanted += [f"layers.{LAYER}.{d}.{k}" for d in DENSE for k in ("weight", "scale")]

    loc, keys = {}, []
    for name in shards(args.ckpt):
        path = os.path.join(args.ckpt, name)
        hdr, base = read_header(path)
        for k, meta in hdr.items():
            if k in wanted:
                loc[k] = (path, base, meta)
            lay = k.split(".")[1] if k.startswith("layers.") else None
            if lay is None or int(lay) < KEY_LAYERS:
                e = k.split(".experts.")[1].split(".")[0] if ".experts." in k and "shared" not in k else None
                if e is None or int(e) <= 1:
                    keys.append(k)
    missing = [k for k in wanted if k not in loc]
    if missing:
        raise SystemExit(f"not found in {args.ckpt}: {missing[:3]} ...")

    # --- key list ---
    kp = os.path.join(args.out, "v4_keys_l0_5.txt")
    with open(kp, "w") as f:
        f.write("\n".join(sorted(set(keys))) + "\n")

    # --- config ---
    cp = os.path.join(args.out, "v4cfg", "config.json")
    with open(os.path.join(args.ckpt, "config.json")) as src, open(cp, "w") as dst:
        dst.write(src.read())

    # --- tensor sample, with the doubled scales ---
    emit = []
    for k in wanted:
        _p, _b, meta = loc[k]
        emit.append((k, meta["dtype"], meta["shape"]))
        if meta["dtype"] == "F8_E8M0":
            emit.append((k + ".as_u8", "U8", meta["shape"]))

    header, off = {}, 0
    for k, dt, shape in emit:
        _p, _b, meta = loc[k[:-len(".as_u8")] if k.endswith(".as_u8") else k]
        n = meta["data_offsets"][1] - meta["data_offsets"][0]
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)          # keep the data section 8-byte aligned

    tp = os.path.join(args.out, "v4_expert.safetensors")
    with open(tp, "wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for k, _dt, _shape in emit:
            src_key = k[:-len(".as_u8")] if k.endswith(".as_u8") else k
            path, base, meta = loc[src_key]
            a, b = meta["data_offsets"]
            with open(path, "rb") as fh:
                fh.seek(base + a)
                left = b - a
                while left:
                    chunk = fh.read(min(left, 1 << 24))
                    out.write(chunk)
                    left -= len(chunk)

    print(f"wrote {tp} ({os.path.getsize(tp)} bytes, {len(emit)} tensors)")
    print(f"wrote {kp} ({len(set(keys))} keys)")
    print(f"wrote {cp}")


if __name__ == "__main__":
    main()
