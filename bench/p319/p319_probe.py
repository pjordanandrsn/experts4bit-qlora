"""gnf4#319 evidence run: what the f32 compute modes cost in accuracy and
throughput at each dot precision, on whatever card this is.

Drives the SHIPPED knob (``GNF4_ATTN_F32_PRECISION``), not a rewritten
source copy, so the numbers describe the code that would ship. Prints
one table per section and a machine-readable JSON blob at the end.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import triton  # noqa: E402

PRECS = ("tf32", "tf32x3", "ieee")
TOL = 2e-2
SHAPES = [
    ("permuted_tables",  dict(B=3, hq=16, hkv=4, d=64,  seq=[64, 128, 96])),
    ("partial_tails",    dict(B=4, hq=16, hkv=4, d=64,  seq=[17, 33, 1, 47])),
    ("grouped_k_scales", dict(B=2, hq=16, hkv=4, d=128, seq=[96, 160], kg=4)),
    ("small_gqa_group",  dict(B=2, hq=32, hkv=8, d=64,  seq=[64, 80])),
    ("single_token",     dict(B=2, hq=16, hkv=4, d=64,  seq=[1, 2])),
    ("head_dim_256",     dict(B=1, hq=16, hkv=8, d=256, seq=[64], kg=4)),
]


def fresh(prec):
    """Re-import the kernel under a given precision. The constexpr is read
    at launch, but Triton caches a compiled kernel per constexpr value, so
    a plain env flip is enough -- the import reset only keeps the module
    state (fallback memo, counters) from leaking between arms."""
    os.environ["GNF4_ATTN_F32_PRECISION"] = prec
    for m in ("fp8_paged_attn", "test_fp8_paged_attn"):
        sys.modules.pop(m, None)
    import test_fp8_paged_attn as T
    return T


def accuracy():
    rows = []
    print("\n== accuracy: max abs error vs the fp32 oracle "
          "(assert_close rtol=atol=%g) ==" % TOL)
    print("%-17s %-7s %-7s %12s %10s %9s"
          % ("shape", "mode", "prec", "max_abs_err", "over_tol%", "verdict"))
    for prec in PRECS:
        T = fresh(prec)
        for name, s in SHAPES:
            for mode, mkw in T._modes():
                kw = dict(k_groups=s.get("kg", 1), v_groups=1)
                try:
                    got, want = T._run_both(s["B"], s["hq"], s["hkv"], s["d"],
                                            s["seq"], mode_kw=mkw, **kw)
                except Exception as e:                      # noqa: BLE001
                    print("%-17s %-7s %-7s %s" % (name, mode, prec,
                                                  type(e).__name__ + ": " + str(e)[:60]))
                    rows.append(dict(shape=name, mode=mode, prec=prec,
                                     error=type(e).__name__))
                    continue
                diff = (got - want).abs()
                allowed = want.abs() * TOL + TOL
                over = float((diff > allowed).float().mean()) * 100.0
                mx = float(diff.max())
                rows.append(dict(shape=name, mode=mode, prec=prec,
                                 max_abs_err=mx, over_tol_pct=over))
                print("%-17s %-7s %-7s %12.6f %9.2f%% %9s"
                      % (name, mode, prec, mx, over,
                         "PASS" if over == 0.0 else "FAIL"))
    return rows


BT = 16


def build(T, B, Tlen, hq, hkv, d, kg=1):
    from fp8_kv import pack_kv_block, quantize_kv_fp8
    g = torch.Generator().manual_seed(0)
    nb = (Tlen + BT - 1) // BT
    k_row = BT * hkv * d + 4 * BT * hkv * kg
    v_row = BT * hkv * d + 4 * BT * hkv
    kp = torch.zeros(B * nb * k_row, dtype=torch.uint8)
    vp = torch.zeros(B * nb * v_row, dtype=torch.uint8)
    tab = torch.zeros(B, nb, dtype=torch.int32)
    row = 0
    for b in range(B):
        kt = torch.randn(Tlen, hkv, d, generator=g)
        vt = torch.randn(Tlen, hkv, d, generator=g)
        for i in range(nb):
            tab[b, i] = row
            qk, sk = quantize_kv_fp8(kt[i * BT:(i + 1) * BT], group=d // kg)
            pack_kv_block(qk, sk, kp[row * k_row:(row + 1) * k_row])
            qv, sv = quantize_kv_fp8(vt[i * BT:(i + 1) * BT], group=d)
            pack_kv_block(qv, sv, vp[row * v_row:(row + 1) * v_row])
            row += 1
    q = torch.randn(B, hq, d, generator=g).to(torch.bfloat16)
    lens = torch.full((B,), Tlen, dtype=torch.int32)
    return (q.cuda(), kp.cuda(), vp.cuda(), tab.cuda(), lens.cuda(),
            B * nb * (k_row + v_row))


def throughput():
    rows = []
    print("\n== throughput: f32 split path, B=25 T=4096 H=32/8 D=128 ==")
    print("%-7s %10s %12s %9s" % ("prec", "ms", "KV GB/s", "vs tf32"))
    base = None
    for prec in PRECS:
        T = fresh(prec)
        from fp8_paged_attn import fp8_paged_decode_attention as F
        q, kp, vp, tab, lens, kvb = build(T, 25, 4096, 32, 8, 128)
        kw = dict(n_kv_heads=8, head_dim=128, compute="f32")
        for _ in range(10):
            F(q, kp, vp, tab, lens, **kw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            F(q, kp, vp, tab, lens, **kw)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 30 * 1e3
        gbs = kvb / (ms * 1e-3) / 1e9
        base = base or gbs
        rows.append(dict(prec=prec, ms=ms, gbs=gbs, rel=gbs / base))
        print("%-7s %10.3f %12.1f %8.2fx" % (prec, ms, gbs, gbs / base))
    return rows


def lowering():
    """What Triton actually emits for the score dot at each precision --
    the mechanism, not an inference from the error. Counts the tensor-core
    and fallback instructions in the compiled split kernel's PTX."""
    out = {}
    print("\n== lowering: tensor-core / fallback instructions in the split kernel PTX ==")
    for prec in PRECS:
        T = fresh(prec)
        try:
            T._run_both(2, 16, 4, 64, [64, 80], mode_kw={})
            from fp8_paged_attn import _fp8_paged_decode_split as K
            compiled = []
            for _dev, entry in K.device_caches.items():
                kcache = entry[0] if isinstance(entry, (tuple, list)) else entry
                compiled.extend(kcache.values())
            ptx = max((c.asm["ptx"] for c in compiled), key=len)
            kinds = {}
            for ln in ptx.splitlines():
                op = ln.strip().split()[0] if ln.strip() else ""
                for tag in ("mma.sync", "wgmma", "tcgen05", "fma.rn.f32",
                            "mul.f32", "add.f32"):
                    if op.startswith(tag):
                        kinds[tag] = kinds.get(tag, 0) + 1
            out[prec] = kinds
            print("%-7s %s" % (prec, kinds or "(none matched)"))
        except Exception as e:                              # noqa: BLE001
            out[prec] = {"error": "%s: %s" % (type(e).__name__, str(e)[:80])}
            print("%-7s %s" % (prec, out[prec]))
    return out


def main():
    cap = torch.cuda.get_device_capability()
    hdr = dict(device=torch.cuda.get_device_name(0),
               cap="sm_%d%d" % cap, torch=torch.__version__,
               triton=triton.__version__,
               smem=torch.cuda.get_device_properties(0).shared_memory_per_block_optin
               if hasattr(torch.cuda.get_device_properties(0),
                          "shared_memory_per_block_optin") else None)
    print("== %s %s | torch %s | triton %s ==" % (hdr["device"], hdr["cap"],
                                                  hdr["torch"], hdr["triton"]))
    acc = accuracy()
    thr = throughput()
    low = lowering()
    print("\n=== JSON ===")
    print(json.dumps(dict(header=hdr, accuracy=acc, throughput=thr,
                          lowering=low)))


if __name__ == "__main__":
    main()
