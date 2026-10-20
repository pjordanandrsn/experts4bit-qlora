#!/usr/bin/env python3
"""Tile fill and its amortization (bench/tile-fill/CONSTRUCTION.md, committed first).

    python tilefill.py --replay bench/p153/router-replay/rr.json.pt --out tilefill.json

(a') Width, stratified (Amendment 1): 16 requests x j tokens at distinct random steps, B = 16 j, the same rows at every
layer (a batch's tokens pass every layer). The first construction's uniform B = 16 composition is kept as its failed
control. (b) Verify blocks: the 16 requests' k+1 consecutive tokens as one step's
rows. D = distinct experts; T = tiles, sum over experts of ceil(m_e / 16). 1,000 draws per cell, one generator per
stream seeded at 20261020. Positive control: B = 16 composed D within 0.2 of e4b#1869's null A mean.
"""
import argparse
import json

import numpy as np

SEED, DRAWS, E, TILE = 20261020, 1000, 128, 16
WIDTHS_J = (1, 2, 4, 8)                            # B = 16 j (Amendment 1)
VERIFY = (2, 4, 8)
NULL_A_D16 = {"S0": 57.257, "S1": 57.418}          # bench/routing-clustering/README.md


def tiles(ids):
    """ids [L, rows, K] -> (D [L], T [L]) for one batch."""
    L = ids.shape[0]
    m = np.zeros((L, E), dtype=np.int64)
    np.add.at(m, (np.arange(L)[:, None], ids.reshape(L, -1)), 1)
    return (m > 0).sum(1), np.ceil(m / TILE).sum(1)


def cell(samples, rows):
    d = np.array([s[0] for s in samples])          # [draws] layer-mean D
    t = np.array([s[1] for s in samples])
    q = lambda x: [float(v) for v in np.quantile(x, [0.025, 0.975])]  # noqa: E731
    return {"rows": rows, "D": float(d.mean()), "D_interval": q(d), "T": float(t.mean()), "T_interval": q(t),
            "T_per_row": float(t.mean() / rows), "fill": float(8 * rows / (TILE * t.mean()))}


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--replay", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--draws", type=int, default=DRAWS)
    args = a.parse_args(argv)
    import torch
    picks = torch.load(args.replay, map_location="cpu", weights_only=False)["picks"].numpy().astype(np.int64)
    assert picks.shape == (48, 32, 128, 8), picks.shape
    out = {"seed": SEED, "draws": args.draws, "streams": {}}
    for name, rows in (("S0", slice(0, 16)), ("S1", slice(16, 32))):
        p = picks[:, rows]                                   # [L, 16, S, K]
        L, R, S, K = p.shape
        pairs = p.reshape(L, R * S, K)                       # pair index = request * S + step
        rng = np.random.default_rng(SEED)
        # the failed control of the first construction, kept: B = 16 uniform (request, step) pairs (duplicates allowed)
        samp = []
        for _ in range(args.draws):
            sel = rng.choice(R * S, size=16, replace=False)
            d, t = tiles(pairs[:, sel])
            samp.append((d.mean(), t.mean()))
        uniform16 = cell(samp, 16)
        # Amendment 1, (a'): 16 requests x j tokens each at distinct random steps, B = 16 j
        width = {}
        for j in WIDTHS_J:
            samp = []
            for _ in range(args.draws):
                steps = np.argsort(rng.random((R, S)), axis=1)[:, :j]          # j distinct steps per request
                sel = (np.arange(R)[:, None] * S + steps).reshape(-1)
                d, t = tiles(pairs[:, sel])
                samp.append((d.mean(), t.mean()))
            width[R * j] = cell(samp, R * j)
        assert abs(width[16]["D"] - NULL_A_D16[name]) < 0.2, (name, width[16]["D"])      # the positive control
        for B in width:
            width[B]["T_per_row_vs_16"] = width[B]["T_per_row"] / width[16]["T_per_row"]
        verify = {}
        for k1 in VERIFY:
            samp = []
            for _ in range(args.draws):
                s = rng.integers(0, S - k1 + 1)
                d, t = tiles(p[:, :, s:s + k1].reshape(L, R * k1, K))
                samp.append((d.mean(), t.mean()))
            c = cell(samp, R * k1)
            ref = width[R * k1]                    # the same requests, the same tokens each, at random steps
            c.update(T_vs_random_steps=c["T"] / ref["T"], D_vs_random_steps=c["D"] / ref["D"],
                     T_vs_random_steps_interval=[c["T_interval"][0] / ref["T"], c["T_interval"][1] / ref["T"]])
            verify[k1] = c
        out["streams"][name] = {"uniform16_failed_control": uniform16,
                                "request_affinity_D16": uniform16["D"] / width[16]["D"],
                                "width_stratified": {str(b): v for b, v in width.items()},
                                "verify": {str(k): v for k, v in verify.items()}}
    json.dump(out, open(args.out, "w"), indent=1)
    for name, r in out["streams"].items():
        print(f"{name} uniform B=16 (failed control): D {r['uniform16_failed_control']['D']:.2f}; request affinity "
              f"D ratio {r['request_affinity_D16']:.4f}")
        for b, c in r["width_stratified"].items():
            print(f"{name} width B={b:>3}: D {c['D']:.2f} T {c['T']:.2f} T/row {c['T_per_row']:.3f} "
                  f"(x{c['T_per_row_vs_16']:.3f} of B=16) fill {c['fill']:.3f}")
        for k, c in r["verify"].items():
            print(f"{name} verify k+1={k} ({c['rows']} rows): D {c['D']:.2f} T {c['T']:.2f} vs random steps "
                  f"x{c['T_vs_random_steps']:.4f} [{c['T_vs_random_steps_interval'][0]:.4f}, "
                  f"{c['T_vs_random_steps_interval'][1]:.4f}] fill {c['fill']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
