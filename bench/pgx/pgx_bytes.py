"""PGX Z3: bytes per expert for each arm's checkpoint, from the safetensors headers alone, and the K table they give.

    python bench/pgx/pgx_bytes.py --fetch --out bench/pgx/bytes_per_expert.json   # header-only range reads (~50 MB)
    python bench/pgx/pgx_bytes.py --table bench/pgx/bytes_per_expert.json         # print the registered K table

Each shard's header is read with two HTTP range requests (the 8-byte length, then the JSON header); no tensor data
is fetched. A routed expert's bytes are the sum over its ``mlp.experts.<e>.*`` tensors of dtype size times shape, so
the FP8 figure carries its block scales and the GPTQ figure its qzeros, scales and g_idx. The GPTQ checkpoint has
``desc_act: false``, so its g_idx is the trivial group map; the figure without it is recorded too, and both give the
same K at every registered budget. NF4 is not a checkpoint format here: E-hyb bakes its arena on the box, and its
figure (2,654,208 B) is P233's placement record. These are checkpoint bytes. The box measures each engine's loaded
slot bytes before the first timed cell, and a disagreement moves K by the registered rule.

Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.request

LAYERS, EXPERTS = 48, 128
BUDGETS_GB = (4, 8, 12)
NF4_BYTES_PER_EXPERT = 2_654_208
CHECKPOINTS = {
    "bf16": ("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"),
    "fp8": ("Qwen/Qwen3-30B-A3B-FP8", "d206ba732169f29bb77fbf80fc2c4b81d4d30782"),
    "int4": ("Qwen/Qwen3-30B-A3B-GPTQ-Int4", "9b534e4318b7ebc3c961a839f13eb18b1833f441"),
}
ARM_FORMAT = {"S8": "fp8", "V8": "fp8", "S4": "int4", "E-hyb": "nf4"}
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "F8_E4M3": 1, "U8": 1, "I8": 1}
_EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.(\w+)$")


def expert_bytes(headers: dict, skip=()) -> dict:
    """{(layer, expert): bytes} over every shard header; ``skip`` names tensor suffixes left out (e.g. 'g_idx')."""
    per: dict = {}
    for hd in headers.values():
        for name, t in hd.items():
            m = _EXPERT.match(name)
            if name == "__metadata__" or not m or m[4] in skip:
                continue
            n = DTYPE_BYTES[t["dtype"]]
            for s in t["shape"]:
                n *= s
            per[(int(m[1]), int(m[2]))] = per.get((int(m[1]), int(m[2])), 0) + n
    return per


def uniform(per: dict) -> int:
    """The single bytes-per-expert figure; refuses a partial checkpoint or experts of unequal size."""
    if len(per) != LAYERS * EXPERTS:
        raise SystemExit(f"{len(per)} experts in the headers, expected {LAYERS * EXPERTS}")
    vals = set(per.values())
    if len(vals) != 1:
        raise SystemExit(f"experts differ in size: {sorted(vals)[:4]}")
    return vals.pop()


def k_for(budget_gb: float, bytes_per_expert: int) -> int:
    """The registered rule: K = floor(budget / (48 x bytes_per_expert)), budget in decimal GB."""
    return int(budget_gb * 10**9 // (LAYERS * bytes_per_expert))


def k_table(record: dict) -> dict:
    fmt = {"fp8": record["fp8"]["bytes_per_expert"], "int4": record["int4"]["bytes_per_expert"],
           "nf4": NF4_BYTES_PER_EXPERT}
    return {arm: {str(b): k_for(b, fmt[f]) for b in BUDGETS_GB} for arm, f in ARM_FORMAT.items()}


def _get(url: str, rng: str) -> bytes:
    req = urllib.request.Request(url, headers={"Range": rng, "User-Agent": "pgx-bytes"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch(repo: str, rev: str) -> dict:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}", timeout=60) as r:
        api = json.load(r)
    out = {}
    for s in sorted(x["rfilename"] for x in api["siblings"] if x["rfilename"].endswith(".safetensors")):
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{s}"
        n = struct.unpack("<Q", _get(url, "bytes=0-7")[:8])[0]
        if n > 100_000_000:
            raise SystemExit(f"{s}: header length {n} is not a safetensors header")
        out[s] = json.loads(_get(url, f"bytes=8-{7 + n}"))
    return out


def record_from(headers_by_fmt: dict) -> dict:
    rec = {}
    for fmt, (repo, rev) in CHECKPOINTS.items():
        h = headers_by_fmt[fmt]
        per = expert_bytes(h)
        kinds = sorted({_EXPERT.match(k)[3] + "." + _EXPERT.match(k)[4] for hd in h.values() for k in hd
                        if _EXPERT.match(k)})
        rec[fmt] = {"repo": repo, "revision": rev, "shards": len(h), "bytes_per_expert": uniform(per),
                    "tensors_per_expert": kinds,
                    "header_sha256": hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()}
        if fmt == "int4":
            rec[fmt]["bytes_per_expert_without_g_idx"] = uniform(expert_bytes(h, skip=("g_idx",)))
    rec["nf4"] = {"bytes_per_expert": NF4_BYTES_PER_EXPERT, "source": "P233's placement record"}
    rec["k_table"] = k_table(rec)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--table")
    a = ap.parse_args(argv)
    if a.fetch:
        rec = record_from({fmt: fetch(*rv) for fmt, rv in CHECKPOINTS.items()})
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
            f.write("\n")
    else:
        with open(a.table, encoding="utf-8") as f:
            rec = json.load(f)
    for arm, ks in k_table(rec).items():
        print(f"{arm:6s} " + "  ".join(f"{b} GB: K={k}" for b, k in ks.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
