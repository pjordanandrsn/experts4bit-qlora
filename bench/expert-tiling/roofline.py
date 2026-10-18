"""Expert tiling / streaming: the break-even arithmetic, from existing records only. CPU, no GPU, no run.

Reproduces every number in README.md. Inputs are the records README.md cites; nothing here is measured.

    python roofline.py [--h2d 21.76 --h2d 40 --h2d 53]
"""

import argparse

# Qwen3-235B-A22B (bench/q235/PREREG-q235.md): hidden, expert intermediate, experts, top-k, layers.
H, INTER, E, TOPK, LAYERS = 4096, 1536, 128, 8, 94
PACKED = 3 * H * INTER // 2  # NF4: two values per byte (gate_up [2I, H] + down [H, I])
ABSMAX = 3 * H * INTER // 64 * 4  # one fp32 absmax per 64 values (offloaded stacks keep fp32)
EXPERT = PACKED + ABSMAX  # 10,616,832 B (flagship: 7,983,857,664 B / 752 copies of 4)
LAYER = E * EXPERT  # 1,358,954,496 B (box 1: 254,124,490,752 B / 187 copies)

# bench/q235/staged-bytes (#1681): whole-layer stagings per Shape B pass (4 micro-batches, 94 layers).
STAGINGS = {"(a) as-is": 748.0, "(b) layer-major": 187.0, "(c) routed": 548.8, "(b)+(c)": 167.9}
TRAVERSALS = {"(a) as-is": 8, "(b) layer-major": 2, "(c) routed": 8, "(b)+(c)": 2}  # layer sweeps per pass
ROUTED_UNION = 0.898  # #1681: mean fraction of experts hit by all four micro-batches
EXPERTS_COMPUTE_S = 15.13e-3 * 4 * LAYERS  # #1680: experts fwd + recompute + bwd per micro-batch per layer
# #1680's per-phase split (A2000 means / measured PRO 6000 ratios), per micro-batch per layer, seconds. Experts only.
FWD_S = 47.3e-3 / 13.30
REC_BWD_S = 45.9e-3 / 13.30 + 58.7e-3 / 7.23
# Measured: Q235 box q235-pro6000-3 (Shape B, Amendment 7) box/box.json moe_timing, 8 sampled layers x 8 calls, means.
# Experts compute per micro-batch per layer: forward 4.76 ms, recompute 4.62, backward 4.68. Staging per whole-layer call
# (layer 0's resident forward calls excluded): forward 51.40 ms, recompute 50.81 -> 26.6 GB/s, the staging rate of record;
# the same box's hostrate.json (one 1 GiB pinned buffer, copied back to back) under-reads this host at 20.39 GB/s.
A7_FWD_S, A7_REC_BWD_S = 4.76e-3, 4.62e-3 + 4.68e-3
A7_STAGE_S = (51.40e-3 + 50.81e-3) / 2
A7_HOSTRATE = 20.39

# Per-copy fixed costs on record (README.md, "DMA size"): seconds.
COPY_LATENCY = 3.3e-6  # gnf4 cold-engine phase 0: 8 KiB pinned H2D, gen 5
ENQUEUE = 19.8e-6 / 2  # gnf4 cold-engine p2-g1b: 19.8 us per row = 2 cudaMemcpyAsync + staging write


def sweep_schedules(rates, fwd_s=FWD_S, rec_bwd_s=REC_BWD_S, label="#1680 predicted"):
    """Layer-major (two sweeps per pass: the forward, then the recompute + backward) under five staging schedules.

    A forward cannot prefetch layer L+1's ROUTED experts during layer L: L+1's routes come from L's output. The
    recompute can, because the forward fixed every route. Whole-layer prefetch needs no routes, so it works in both.
    """
    cf, crb = fwd_s * 4 * LAYERS, rec_bwd_s * 4 * LAYERS
    print(
        f"\nlayer-major sweeps, {label} (experts compute only, a lower bound): forward {cf:.3f} s,"
        f" recompute + backward {crb:.3f} s"
    )
    for L in rates:
        w = LAYER / (L * 1e9)
        r, x = ROUTED_UNION * w, EXPERT / (L * 1e9)
        fw, fr, rw, rr = (LAYERS - 1) * w, (LAYERS - 1) * r, LAYERS * w, LAYERS * r  # layer 0 stays resident
        rows = {
            "(b) serial": (fw + cf + rw + crb, max(fw, cf) + max(rw, crb)),
            "(b) whole-layer double buffer": (max(fw, cf) + w + max(rw, crb) + w, max(fw, cf) + max(rw, crb)),
            "(b)+(c) fwd routed serial, rec routed buffered": (fr + cf + max(rr, crb) + r, max(fr, cf) + max(rr, crb)),
            "hybrid: fwd whole buffered, rec routed buffered": (
                max(fw, cf) + w + max(rr, crb) + r,
                max(fw, cf) + max(rr, crb),
            ),
            "(b)+(c) fwd per-expert pipelined, rec buffered": (
                max(fr, cf) + LAYERS * x + max(rr, crb) + r,
                max(fr, cf) + max(rr, crb),
            ),
        }
        floor = max(fr, cf) + max(rr, crb)
        print(f"  H2D {L} GB/s (byte floor, routed both sweeps, overlapped: {floor:.2f} s)")
        for name, (t, bound) in rows.items():
            print(f"    {name:48s} {t:6.2f} s  / own bound {t / bound:.3f}  / floor {t / floor:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2d", type=float, action="append")
    rates = ap.parse_args().h2d or [21.76, 40.0, 53.0]
    print(f"expert {EXPERT:,} B (packed {PACKED:,} + absmax {ABSMAX:,}); layer {LAYER:,} B")
    flop_row = 2 * 3 * H * INTER
    print(
        f"forward FLOP per routed row per expert {flop_row / 1e6:.1f} M; FLOP per staged byte per row {flop_row / EXPERT:.3f}"
    )
    print(f"experts compute per Shape B pass (#1680): {EXPERTS_COMPUTE_S:.2f} s")
    for L in rates:
        print(f"\nH2D {L} GB/s: expert copy {EXPERT / L / 1e6:.3f} ms, layer staging {LAYER / L / 1e6:.2f} ms")
        print(
            f"  {'layout':18s} {'staging s':>9s} {'serial s':>8s} {'overlap s':>9s} {'serial/bound':>12s} {'overlap/bound':>13s}"
        )
        for name, n in STAGINGS.items():
            s = n * LAYER / (L * 1e9)
            fill = TRAVERSALS[name] * (LAYER * (ROUTED_UNION if "(c)" in name else 1.0)) / (L * 1e9)
            serial, overlap = s + EXPERTS_COMPUTE_S, max(s, EXPERTS_COMPUTE_S) + fill
            bound = max(s, EXPERTS_COMPUTE_S)
            print(f"  {name:18s} {s:9.2f} {serial:8.2f} {overlap:9.2f} {serial / bound:12.3f} {overlap / bound:13.3f}")
    print(
        "\nbound = max(staging, compute). overlap = layer-granular double buffering: max(staging, compute) plus one layer's"
        " staging per layer sweep (the pipeline fill)."
    )
    sweep_schedules(rates)
    eff = LAYER / A7_STAGE_S / 1e9
    print(
        f"\nA7 measured staging: {A7_STAGE_S * 1e3:.2f} ms per whole-layer call = {eff:.2f} GB/s effective"
        f" ({eff / A7_HOSTRATE:.2f}x the box's own hostrate probe, {A7_HOSTRATE} GB/s)"
    )
    sweep_schedules([A7_HOSTRATE, round(eff, 2)], A7_FWD_S, A7_REC_BWD_S, "A7 measured")
    print(
        "\nI-tile sizes (t of I = 1536) and a SERIALIZED per-copy-cost floor on copy efficiency (inference; async"
        " back-to-back copies hide it, see README):"
    )
    print(f"  {'t':>5s} {'tile MB':>8s} {'down row segment B':>18s}" + "".join(f"  eff@{L}" for L in rates))
    for t in (64, 128, 256, 512, 768, 1536):
        b = EXPERT * t / INTER
        effs = []
        for L in rates:
            x = b / (L * 1e9)
            effs.append(f"  {x / (x + COPY_LATENCY + ENQUEUE):7.3f}")
        print(f"  {t:5d} {b / 1e6:8.3f} {t // 2:18d}" + "".join(effs))
    print("\ntraining bytes per layer per micro-batch, whole expert vs streamed I-tiles evicted after use:")
    print(
        "  whole layer resident through its backward: 2 stagings (forward, recompute); tiles evicted: 3 (forward,"
        " recompute, backward) -> +50 % bytes, unless every tile stays resident (no memory saved)"
    )


if __name__ == "__main__":
    main()
