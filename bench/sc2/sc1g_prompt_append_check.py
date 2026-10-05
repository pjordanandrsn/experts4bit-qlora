#!/usr/bin/env python3
"""sc1g_prompt_append_check.py -- lane SC1g (#846), e4b#1175: is Fp8PagedKV.append_prompt (the bulk path the served loop
writes a prompt with) bitwise the per-layer append at k_groups 4 / 8 / 16, on gpt-oss's geometry?

sc1g-diag-2's attention check ran the fp8 decode kernel over a cache filled by append() and read it agreeing at every group
count (kernel vs reference <= 1.9e-3), yet MXFP4 served at --kv-groups 16 reads +0.130 nats over the default 4. The served
loop writes its 512-token prompt through append_prompt (PagedModelRunner's bulk flush), and every decode token through
append_many (the shim's batched_append, on by default) -- neither of which the check ran. This fills two caches with the
same K/V for 24 layers -- the served way (append_prompt, then --decode tokens one at a time through append_many, crossing
block boundaries) and the plain way (append per layer, the same tokens one at a time through append) -- and compares
reference_kv per layer. It also reports each cache's reconstruction error against the bf16 originals. No kernel runs
(sm_86 ok).

  sc1g_prompt_append_check.py --out OUT.json [--tokens 512] [--decode 100] [--mutate]

--mutate is the positive control: the per-layer cache is filled with the layers' K in REVERSED layer order, which must
read DIFFERENT.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

L, HKV, D = 24, 8, 64


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def one(kg, T, g, mutate, n_dec=100):
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    ks, vs = [], []
    for _ in range(L):
        k = torch.randn(T + n_dec, HKV, D, generator=g)
        k[:, :, torch.randperm(D, generator=g)[:3]] *= 20.0          # a few outlier key channels, per layer
        ks.append(k.to("cuda", torch.bfloat16))
        vs.append(torch.randn(T + n_dec, HKV, D, generator=g).to("cuda", torch.bfloat16))
    bulk = Fp8PagedKV(L, HKV, D, batch=1, max_tokens_per_seq=T + n_dec + 64, k_groups=kg, device="cuda")
    per = Fp8PagedKV(L, HKV, D, batch=1, max_tokens_per_seq=T + n_dec + 64, k_groups=kg, device="cuda")
    took_bulk = bulk.append_prompt(0, list(range(L)), [k[:T] for k in ks], [v[:T] for v in vs])
    for layer in range(L):
        src = L - 1 - layer if mutate else layer
        per.append(layer, 0, ks[src][:T].contiguous(), vs[layer][:T].contiguous())
    for t in range(T, T + n_dec):                                      # decode tokens, one step at a time, every layer
        for layer in range(L):
            kt, vt = ks[layer][t:t + 1], vs[layer][t:t + 1]
            bulk.append_many(layer, [0], kt[None].contiguous(), vt[None].contiguous())   # the shim's batched_append
            per.append(layer, 0, kt.contiguous(), vt.contiguous())
    torch.cuda.synchronize()
    rows = []
    for layer in range(L):
        kb, vb = bulk.reference_kv(layer, 0, dtype=torch.float32)
        kp, vp = per.reference_kv(layer, 0, dtype=torch.float32)
        rows.append({"layer": layer, "k_equal": bool(torch.equal(kb, kp)), "v_equal": bool(torch.equal(vb, vp)),
                     "k_bulk_vs_per": rel(kb, kp), "k_recon_bulk": rel(kb, ks[layer].float()),
                     "k_recon_per": rel(kp, ks[layer].float()), "v_recon_bulk": rel(vb, vs[layer].float())})
    return {"k_groups": kg, "k_group_width": D // kg, "bulk_path_taken": bool(took_bulk), "decode_tokens": n_dec,
            "layers_k_equal": sum(r["k_equal"] for r in rows), "layers_v_equal": sum(r["v_equal"] for r in rows),
            "k_bulk_vs_per_worst": max(r["k_bulk_vs_per"] for r in rows),
            "k_recon_bulk_worst": max(r["k_recon_bulk"] for r in rows), "k_recon_per_worst": max(r["k_recon_per"] for r in rows),
            "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--decode", type=int, default=100)
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args(argv)
    g = torch.Generator().manual_seed(0)
    res = [one(kg, a.tokens, g, a.mutate, a.decode) for kg in (4, 8, 16)]
    same = all(r["layers_k_equal"] == L and r["layers_v_equal"] == L for r in res)
    bulk = all(r["bulk_path_taken"] for r in res)
    verdict = ("NOT_BULK" if not bulk else "BITWISE_EQUAL" if same else "DIFFERENT")
    rec = {"device": {"name": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability(0))},
           "verdict": verdict, "mutate": a.mutate, "tokens": a.tokens, "per_k_groups": res}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"SC1G_PROMPT_APPEND {verdict} mutate={a.mutate} " + " ".join(
        f"kg{r['k_groups']}:bulk={r['bulk_path_taken']},k_eq={r['layers_k_equal']}/{L},v_eq={r['layers_v_equal']}/{L},"
        f"recon_bulk={r['k_recon_bulk_worst']:.3e},recon_per={r['k_recon_per_worst']:.3e}" for r in res), flush=True)
    return 0 if verdict == "BITWISE_EQUAL" else 1


if __name__ == "__main__":
    sys.exit(main())
