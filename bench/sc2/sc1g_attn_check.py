#!/usr/bin/env python3
"""sc1g_attn_check.py -- lane SC1g amendment A3's tail step (e4b#1175): grouped-nf4-gemm's fp8 paged decode attention at
k_groups 4 / 8 / 16 on gpt-oss's geometry (64 query heads, 8 KV heads, head_dim 64, sinks), with the sliding window (128) and
full attention, through e4b's own Fp8PagedKV (the serving path's pack and kernel call).

For each (k_groups, window, K kind):
  kernel    Fp8PagedKV.attention -- the kernel reading the fp8 pool (bf16 out)
  deq       the SAME stored K/V dequantized by Fp8PagedKV.reference_kv, then fp32 attention with sinks and the window
  raw       the original bf16 K/V, fp32 attention with sinks and the window
All three take the same bf16-rounded q the kernel takes. kernel vs deq (rounded to bf16) is the kernel's error; deq vs raw
is the fp8 scheme's error at this group count, which finer groups should only shrink. The stored K and V are also compared
with the bf16 originals element-wise (k_recon / v_recon, ||a - b|| / ||b||): e4m3 keeps 3 mantissa bits, so a correct pack
reads a few percent and a broken one (wrong group layout, wrong scale slot) reads O(1) -- the kernel and reference_kv share
the pack, so only this read can see a pack bug. A mutation (the reference without sinks) must disagree. Each row records the
compute mode the kernel ran (fp8_paged_attn.compute_counts() before vs after). sm_89+ only: the A2000 cannot run the
kernel (no native e4m3).

Verdict, in order: INERT (the mutation agrees), PACK_BAD (any recon > --recon-tol), KERNEL_DISAGREES (kernel vs deq_bf16 >
--tol at any k_groups), else KERNEL_AGREES.

--pack-only skips the kernel (so it runs on sm_86, at $0 on the A2000): the recon and scheme reads only, verdict PACK_OK or
PACK_BAD. --mutate-pack is its positive control: recon is read against the originals shifted one token, which must read
PACK_BAD.

  sc1g_attn_check.py --out OUT.json [--tokens 2000] [--tol 1e-2] [--recon-tol 0.1] [--pack-only [--mutate-pack]]
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

HQ, HKV, D = 64, 8, 64


def attend(q, K, V, window, sinks, scale, use_sinks=True):
    """q [Hq, D] fp32; K, V [T, Hkv, D] fp32 -> [Hq, D]: the last `window` keys (0 = all), gpt-oss sinks as an extra logit."""
    if window:
        K, V = K[-window:], V[-window:]
    G = HQ // HKV
    Kr = K.repeat_interleave(G, dim=1).permute(1, 0, 2)          # [Hq, T, D]
    Vr = V.repeat_interleave(G, dim=1).permute(1, 0, 2)
    s = torch.einsum("hd,htd->ht", q, Kr) * scale
    if use_sinks:
        s = torch.cat([s, sinks[:, None]], dim=-1)
        p = torch.softmax(s, dim=-1)[:, :-1]
    else:
        p = torch.softmax(s, dim=-1)
    return torch.einsum("ht,htd->hd", p, Vr)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _counts():
    try:
        import fp8_paged_attn
        return dict(fp8_paged_attn.compute_counts() or {})
    except Exception:                                  # noqa: BLE001 -- the tally is a record, not the verdict
        return {}


def one(kg, window, kind, T, g, pack_only=False, mutate_pack=False):
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    K = torch.randn(T, HKV, D, generator=g)
    if kind == "outlier":                              # a few large key channels, as attention keys often carry
        K[:, :, torch.randperm(D, generator=g)[:3]] *= 20.0
    V = torch.randn(T, HKV, D, generator=g)
    q = torch.randn(HQ, D, generator=g)
    sinks = torch.randn(HQ, generator=g)
    kv = Fp8PagedKV(1, HKV, D, batch=1, max_tokens_per_seq=T + 64, k_groups=kg, device="cuda")
    Kb, Vb = K.to("cuda", torch.bfloat16), V.to("cuda", torch.bfloat16)
    kv.append(0, 0, Kb, Vb)
    scale = D ** -0.5
    qb = q.to("cuda", torch.bfloat16)
    if pack_only:
        Kd, Vd = kv.reference_kv(0, 0, dtype=torch.float32)
        Kr, Vr = (Kb.roll(1, 0), Vb.roll(1, 0)) if mutate_pack else (Kb, Vb)
        qf = qb.float()
        return {"k_groups": kg, "window": window, "kind": kind, "k_group_width": D // kg, "k_recon": rel(Kd, Kr.float()),
                "v_recon": rel(Vd, Vr.float()), "scheme_deq_vs_raw": rel(attend(qf, Kd, Vd, window, sinks.cuda(), scale),
                                                                     attend(qf, Kb.float(), Vb.float(), window, sinks.cuda(), scale))}
    before = _counts()
    out = kv.attention(0, qb[None], window=window, sinks=sinks.to("cuda", torch.float32), sm_scale=scale)[0].float()
    torch.cuda.synchronize()
    after = _counts()
    ran = sorted(m for m in after if after[m] > before.get(m, 0))
    Kd, Vd = kv.reference_kv(0, 0, dtype=torch.float32)
    qf = qb.float()
    deq = attend(qf, Kd, Vd, window, sinks.cuda(), scale)
    raw = attend(qf, Kb.float(), Vb.float(), window, sinks.cuda(), scale)
    nosink = attend(qf, Kd, Vd, window, sinks.cuda(), scale, use_sinks=False)
    return {"k_groups": kg, "window": window, "kind": kind, "k_group_width": D // kg, "compute_ran": ran,
            "k_recon": rel(Kd, Kb.float()), "v_recon": rel(Vd, Vb.float()),
            "kernel_vs_deq_bf16": rel(out, deq.to(torch.bfloat16).float()), "kernel_vs_deq": rel(out, deq),
            "scheme_deq_vs_raw": rel(deq, raw), "kernel_vs_raw": rel(out, raw),
            "mutation_kernel_vs_nosink": rel(out, nosink.to(torch.bfloat16).float())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-2, help="kernel vs the bf16-rounded reference above this = KERNEL_DISAGREES")
    ap.add_argument("--recon-tol", type=float, default=0.1, help="stored K or V vs its bf16 original above this = PACK_BAD")
    ap.add_argument("--pack-only", action="store_true", help="no kernel (sm_86 ok): recon and scheme only")
    ap.add_argument("--mutate-pack", action="store_true", help="--pack-only's positive control: must read PACK_BAD")
    a = ap.parse_args(argv)
    cc = list(torch.cuda.get_device_capability(0))
    g = torch.Generator().manual_seed(0)
    rows = [one(kg, w, kind, a.tokens, g, a.pack_only, a.mutate_pack)
            for kg in (4, 8, 16) for w in (128, 0) for kind in ("normal", "outlier")]
    if a.pack_only:
        per_kg = {}
        for r in rows:
            o = per_kg.setdefault(str(r["k_groups"]), {"recon_worst": 0.0, "k_recon_worst": 0.0, "scheme_worst": 0.0})
            o["recon_worst"] = max(o["recon_worst"], r["k_recon"], r["v_recon"])
            o["k_recon_worst"] = max(o["k_recon_worst"], r["k_recon"])
            o["scheme_worst"] = max(o["scheme_worst"], r["scheme_deq_vs_raw"])
        packbad = [kg for kg, v in per_kg.items() if v["recon_worst"] > a.recon_tol]
        verdict = "PACK_BAD" if packbad else "PACK_OK"
        json.dump({"device": {"name": torch.cuda.get_device_name(0), "cc": cc}, "verdict": verdict, "pack_only": True,
                   "mutate_pack": a.mutate_pack, "pack_bad_k_groups": packbad, "per_k_groups": per_kg, "recon_tol": a.recon_tol,
                   "rows": rows}, open(a.out, "w"), indent=1)
        print(f"SC1G_ATTN_PACK {verdict} cc={cc} mutate={a.mutate_pack} " + " ".join(
            f"kg{k}:recon={v['recon_worst']:.3e},k_recon={v['k_recon_worst']:.3e},scheme={v['scheme_worst']:.3e}"
            for k, v in per_kg.items()), flush=True)
        return 0 if verdict == "PACK_OK" else 1
    try:
        import fp8_paged_attn
        compute = {k: v for k, v in (fp8_paged_attn.compute_counts() or {}).items() if v}
    except Exception as e:                             # noqa: BLE001 -- the tally is a record, not the verdict
        compute = f"unavailable: {e!r}"[:120]
    per_kg = {}
    for r in rows:
        o = per_kg.setdefault(str(r["k_groups"]), {"kernel_worst": 0.0, "scheme_worst": 0.0, "mutation_min": 1e9,
                                                   "recon_worst": 0.0, "compute_ran": []})
        o["kernel_worst"] = max(o["kernel_worst"], r["kernel_vs_deq_bf16"])
        o["recon_worst"] = max(o["recon_worst"], r["k_recon"], r["v_recon"])
        o["compute_ran"] = sorted(set(o["compute_ran"]) | set(r["compute_ran"]))
        o["scheme_worst"] = max(o["scheme_worst"], r["scheme_deq_vs_raw"])
        o["mutation_min"] = min(o["mutation_min"], r["mutation_kernel_vs_nosink"])
    inert = any(v["mutation_min"] <= a.tol for v in per_kg.values())
    packbad = [kg for kg, v in per_kg.items() if v["recon_worst"] > a.recon_tol]
    bad = [kg for kg, v in per_kg.items() if v["kernel_worst"] > a.tol]
    verdict = "INERT" if inert else "PACK_BAD" if packbad else "KERNEL_DISAGREES" if bad else "KERNEL_AGREES"
    rec = {"device": {"name": torch.cuda.get_device_name(0), "cc": cc}, "verdict": verdict, "disagreeing_k_groups": bad,
           "pack_bad_k_groups": packbad, "per_k_groups": per_kg, "tol": a.tol, "recon_tol": a.recon_tol,
           "compute_counts": compute, "rows": rows}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"SC1G_ATTN_CHECK {verdict} cc={cc} " + " ".join(
        f"kg{k}:kernel={v['kernel_worst']:.2e},scheme={v['scheme_worst']:.2e},recon={v['recon_worst']:.2e},"
        f"compute={'+'.join(v['compute_ran']) or '?'}" for k, v in per_kg.items()), flush=True)
    return 0 if verdict == "KERNEL_AGREES" else 1


if __name__ == "__main__":
    sys.exit(main())
