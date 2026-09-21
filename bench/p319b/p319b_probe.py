"""gnf4#319 round 2: WHERE the sm_120 divergence is, given that it is not
the dot precision and not the compute mode.

Round 1 (p319-f32prec-1, RTX 5090) found the error byte-identical across
tf32 / tf32x3 / ieee while throughput moved 24x, and byte-identical across
split/f8dot and across packed/pf8. Either the compute mode genuinely does
not change the output on this card -- which would mean the fp8 serving
path carries the same divergence and passes only on a 7.5x looser
tolerance -- or the probe's dispatch was not doing what it claimed.

Each check below is designed so that BOTH answers are informative.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import triton  # noqa: E402
import test_fp8_paged_attn as T  # noqa: E402
from fp8_paged_attn import (fp8_paged_decode_attention as F,  # noqa: E402
                            paged_attn_ref, compute_counts,
                            reset_compute_counts)

OK = "ok"


def hdr():
    print("== %s sm_%d%d | torch %s | triton %s ==" % (
        torch.cuda.get_device_name(0), *torch.cuda.get_device_capability(),
        torch.__version__, triton.__version__))


def build(B=3, hq=16, hkv=4, d=64, seq=(64, 128, 96), kg=1):
    q, kp, vp, tab, lens = T._build(B, hq, hkv, d, list(seq), k_groups=kg)
    return q, kp, vp, tab, lens


def c1_modes_differ():
    """Do the four compute/geometry modes actually produce different bytes?
    If f32 and fp8 agree bit for bit, the mode is not reaching the kernel
    OR the paths are numerically identical -- either is a finding."""
    print("\n-- C1: do the compute modes produce different output? --")
    q, kp, vp, tab, lens = build()
    dev = lambda t: t.cuda()  # noqa: E731
    args = (dev(q), dev(kp), dev(vp), dev(tab), dev(lens))
    kw = dict(n_kv_heads=4, head_dim=64)
    outs = {}
    reset_compute_counts()
    for name, extra in (("split", {}), ("packed", {"pack_heads": True}),
                        ("f8dot", {"compute": "fp8"}),
                        ("pf8", {"pack_heads": True, "compute": "fp8"})):
        outs[name] = F(*args, **kw, **extra).cpu().float()
    print("compute_counts after four calls:", compute_counts())
    base = outs["split"]
    for name, o in outs.items():
        same = bool(torch.equal(o, base))
        print("  %-6s identical_to_split=%-5s  max|o-split|=%.6f"
              % (name, same, float((o - base).abs().max())))
    print("  packed vs pf8 identical:",
          bool(torch.equal(outs["packed"], outs["pf8"])))


def c2_oracle_device():
    """Is the ORACLE the outlier? It is pure torch and runs on whatever
    device its tensors are on; the suite runs it on CPU. If CPU and CUDA
    oracles disagree by the same magnitude as the kernel does, the kernel
    is not the thing that moved."""
    print("\n-- C2: oracle on CPU vs oracle on CUDA --")
    q, kp, vp, tab, lens = build()
    ref_cpu = paged_attn_ref(q, kp, vp, tab, lens, n_kv_heads=4,
                             head_dim=64).float()
    ref_gpu = paged_attn_ref(q.cuda(), kp.cuda(), vp.cuda(), tab.cuda(),
                             lens.cuda(), n_kv_heads=4,
                             head_dim=64).cpu().float()
    d = (ref_cpu - ref_gpu).abs()
    print("  max|cpu-gpu| = %.6f   mean = %.8f   >2e-2: %.2f%%"
          % (float(d.max()), float(d.mean()),
             float((d > 2e-2).float().mean()) * 100))
    got = F(q.cuda(), kp.cuda(), vp.cuda(), tab.cuda(), lens.cuda(),
            n_kv_heads=4, head_dim=64).cpu().float()
    for nm, ref in (("cpu", ref_cpu), ("gpu", ref_gpu)):
        e = (got - ref).abs()
        allowed = ref.abs() * 2e-2 + 2e-2
        print("  kernel vs oracle-%s: max %.6f  over_tol %.2f%%"
              % (nm, float(e.max()), float((e > allowed).float().mean()) * 100))


def c3_isolate():
    """Narrow the kernel-side suspects: one split (no cross-split combine),
    one block of context (one tile-loop iteration), one token."""
    print("\n-- C3: isolating the combine and the tile loop --")
    for label, seq, nsplit, fuse in (("T=64  n_split=auto", (64,), None, None),
                                     ("T=64  n_split=1", (64,), 1, None),
                                     ("T=64  n_split=1 fuse=off", (64,), 1, False),
                                     ("T=16  n_split=1", (16,), 1, None),
                                     ("T=1   n_split=1", (1,), 1, None)):
        q, kp, vp, tab, lens = build(B=1, hq=16, hkv=4, d=64, seq=seq)
        kw = dict(n_kv_heads=4, head_dim=64)
        if nsplit is not None:
            kw["n_split"] = nsplit
        if fuse is not None:
            kw["fuse_combine"] = fuse
        got = F(q.cuda(), kp.cuda(), vp.cuda(), tab.cuda(), lens.cuda(),
                **kw).cpu().float()
        want = paged_attn_ref(q, kp, vp, tab, lens, n_kv_heads=4,
                              head_dim=64).float()
        e = (got - want).abs()
        allowed = want.abs() * 2e-2 + 2e-2
        print("  %-22s max %.6f  over_tol %.2f%%  |want|max %.4f"
              % (label, float(e.max()), float((e > allowed).float().mean()) * 100,
                 float(want.abs().max())))


def c4_worst_element():
    """Print the worst element with its neighbours. A structural bug (a
    mis-folded scale, a wrong row) looks different from a rounding cloud."""
    print("\n-- C4: the worst element, in context --")
    q, kp, vp, tab, lens = build()
    got = F(q.cuda(), kp.cuda(), vp.cuda(), tab.cuda(), lens.cuda(),
            n_kv_heads=4, head_dim=64).cpu().float()
    want = paged_attn_ref(q, kp, vp, tab, lens, n_kv_heads=4,
                          head_dim=64).float()
    e = (got - want).abs()
    idx = torch.argmax(e.flatten()).item()
    b, h, d = (idx // (16 * 64), (idx // 64) % 16, idx % 64)
    print("  worst at [b=%d h=%d d=%d]: got %.6f want %.6f diff %.6f"
          % (b, h, d, float(got[b, h, d]), float(want[b, h, d]),
             float(e[b, h, d])))
    print("  that row, d=%d..%d:" % (max(0, d - 3), min(63, d + 3)))
    for dd in range(max(0, d - 3), min(64, d + 4)):
        print("    d=%2d got %+.6f want %+.6f diff %+.6f"
              % (dd, float(got[b, h, dd]), float(want[b, h, dd]),
                 float(got[b, h, dd] - want[b, h, dd])))
    print("  error distribution: mean %.6f  p50 %.6f  p99 %.6f  max %.6f"
          % (float(e.mean()), float(e.flatten().median()),
             float(torch.quantile(e.flatten(), 0.99)), float(e.max())))
    print("  relative Frobenius: %.6f"
          % float((got - want).norm() / want.norm()))


def c5_q_dtype():
    """The oracle casts its fp32 result to q.dtype (bf16) at the end, and
    the kernel writes bf16. Compare in fp32 by giving the oracle nothing to
    round to: if the gap collapses, the comparison -- not the kernel -- is
    where the 5% lives."""
    print("\n-- C5: is the gap output-rounding or arithmetic? --")
    q, kp, vp, tab, lens = build()
    got = F(q.cuda(), kp.cuda(), vp.cuda(), tab.cuda(), lens.cuda(),
            n_kv_heads=4, head_dim=64).cpu().float()
    want_bf16 = paged_attn_ref(q, kp, vp, tab, lens, n_kv_heads=4,
                               head_dim=64).float()
    qf = q.float()
    want_f32 = paged_attn_ref(qf, kp, vp, tab, lens, n_kv_heads=4,
                              head_dim=64).float()
    for nm, w in (("oracle->bf16 (as the suite does)", want_bf16),
                  ("oracle kept fp32", want_f32)):
        e = (got - w).abs()
        allowed = w.abs() * 2e-2 + 2e-2
        print("  %-34s max %.6f  over_tol %.2f%%"
              % (nm, float(e.max()), float((e > allowed).float().mean()) * 100))
    d = (want_bf16 - want_f32).abs()
    print("  oracle bf16 vs oracle fp32: max %.6f  mean %.8f"
          % (float(d.max()), float(d.mean())))


def main():
    hdr()
    for fn in (c1_modes_differ, c2_oracle_device, c3_isolate,
               c4_worst_element, c5_q_dtype):
        try:
            fn()
        except Exception as e:                               # noqa: BLE001
            import traceback
            print("  %s RAISED %s: %s" % (fn.__name__, type(e).__name__, e))
            traceback.print_exc()


if __name__ == "__main__":
    main()
