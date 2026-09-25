# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Run router_epilogue's licensing probe on REAL gpt-oss-20b and
granite-3.0-1b-a400m router weights, in the pinned transformers' own router
classes, at fp32 and bf16, over 50 probe seeds.

    python bench/router-probe-bf16/probe_real.py <repo-root> <dir-with-rt_*.{bin,json}>

The rt_* files come from extract_router.py (stdlib only, run where the
checkpoints are): `mlp.router.` for gpt-oss-20b, `router.layer` for granite.
Receipts: before.txt (origin/main 81f137d), after.txt (this change).
"""
import collections
import json
import sys

import torch

sys.path.insert(0, sys.argv[1])
from experts4bit_qlora.engines import router_epilogue as RE  # noqa: E402
from transformers import GptOssConfig, GraniteMoeConfig  # noqa: E402
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter  # noqa: E402
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeTopKRouter  # noqa: E402

D = sys.argv[2]
SEEDS = [4242] + list(range(1, 50))      # 4242 is the probe's own seed


def load(tag):
    idx = json.load(open(f"{D}/{tag}.json"))
    raw = open(f"{D}/{tag}.bin", "rb").read()
    return {k: torch.frombuffer(bytearray(raw[v["off"]:v["off"] + v["len"]]), dtype=torch.bfloat16)
            .reshape(v["shape"]).clone() for k, v in idx.items()}


def routers(dtype):
    g = load("rt_gptoss")
    cfg = GptOssConfig(hidden_size=2880, num_local_experts=32, num_experts_per_tok=4)
    for layer in range(24):
        m = GptOssTopKRouter(cfg)
        m.weight.data = g[f"model.layers.{layer}.mlp.router.weight"].clone()
        m.bias.data = g[f"model.layers.{layer}.mlp.router.bias"].clone()
        yield "gpt-oss-20b", m.to(dtype)
    r = load("rt_granite")
    cfg = GraniteMoeConfig(hidden_size=1024, num_local_experts=32, num_experts_per_tok=8)
    for layer in range(24):
        m = GraniteMoeTopKRouter(cfg)
        m.weight.data = r[f"model.layers.{layer}.block_sparse_moe.router.layer.weight"].clone()
        yield "granite-1b-a400m", m.to(dtype)


def probe(mod, kind, spec, seed):
    """RE._probe_matches's checks in its order, with the seed as a parameter
    and the failing check named. Uses the module under test's own
    _reference_for, so it measures whichever router_epilogue.py is on the path."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = mod.weight
    x = torch.randn(4, spec["hidden"], generator=g).to(w.device, w.dtype)
    with torch.no_grad():
        out = mod(x)
        ref_f, ref_w, ref_i = RE._reference_for(mod, kind, spec, x)
    pos = RE._out_positions(out)
    if pos is None:
        return "positions"
    got_f, got_w, got_i = out[pos[0]], out[pos[1]], out[pos[2]]
    if not torch.equal(got_i, ref_i):
        same_set = torch.equal(got_i.sort(-1).values, ref_i.sort(-1).values)
        return "index-order" if same_set else "index-SET"
    if not torch.allclose(got_w.float(), ref_w.float(), rtol=2 ** -8, atol=2 ** -12):
        return "weights"
    if not torch.allclose(got_f.float(), ref_f.float(), rtol=2 ** -6, atol=2 ** -8):
        return "first"
    return "PASS"


for dtype in (torch.float32, torch.bfloat16):
    res = collections.defaultdict(collections.Counter)
    licensed = collections.Counter()
    n = collections.Counter()
    for fam, m in routers(dtype):
        n[fam] += 1
        # the real licensing call, at its own seed, over every candidate in order
        if any(RE._probe_matches(m, k, dict(s)) for k, s in RE._structural(m)):
            licensed[fam] += 1
        kind, spec = [(k, s) for k, s in RE._structural(m) if k == "topk_softmax"][0]
        for sd in SEEDS:
            res[fam][probe(m, kind, dict(spec), sd)] += 1
    print(f"== {dtype}")
    for fam in n:
        print(f"  {fam}: licensed by _probe_matches (seed 4242): {licensed[fam]}/{n[fam]} layers;"
              f" across {len(SEEDS)} seeds x {n[fam]} layers: {dict(res[fam])}")
