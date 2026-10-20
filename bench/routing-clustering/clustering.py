#!/usr/bin/env python3
"""Cross-request routing clustering against shuffled nulls (bench/routing-clustering/NULL.md, committed first).

    python clustering.py --replay bench/p153/router-replay/rr.json.pt --out clustering.json

D = distinct experts among a step's 16 rows' top-8 picks, per layer-step (at 16 rows, K25's 16-row tile count). Null A
rotates each row's 128-step sequence by its own random offset (per layer); null B permutes it. 1,000 draws each, seed
20261020, for S0 (rows 0-15, primary) and S1 (rows 16-31). The positive control: D_obs reproduces P153's published means
(57.107 S0, 57.052 S1) or the script stops.
"""
import argparse
import json

import numpy as np

SEED, DRAWS, E = 20261020, 1000, 128
PUBLISHED = {"S0": 57.107, "S1": 57.052}          # bench/p153/router-replay/README.md


def distinct(p):
    """p: [L, R, S, K] expert ids -> D [L, S], the distinct experts over the R rows' K picks at each layer-step."""
    L, R, S, K = p.shape
    pres = np.zeros((L, S, E), dtype=bool)
    ids = p.transpose(0, 2, 1, 3).reshape(L, S, R * K)
    li = np.arange(L)[:, None, None]
    si = np.arange(S)[None, :, None]
    pres[li, si, ids] = True
    return pres.sum(-1)


def shift(p, rng):
    L, R, S, _ = p.shape
    off = rng.integers(0, S, size=(L, R))
    idx = (np.arange(S)[None, None, :] + off[:, :, None]) % S
    return np.take_along_axis(p, idx[..., None].repeat(p.shape[3], axis=3), axis=2)


def permute(p, rng):
    L, R, S, _ = p.shape
    idx = np.argsort(rng.random((L, R, S)), axis=2)
    return np.take_along_axis(p, idx[..., None].repeat(p.shape[3], axis=3), axis=2)


def summarise(obs, nulls):
    """obs [L, S]; nulls [draws, L] per-layer means -> overall and per-layer ratio with the null's 2.5-97.5 % interval."""
    o_all, n_all = obs.mean(), nulls.mean(axis=1)
    q_lo, q_hi = np.quantile(n_all, [0.025, 0.975])
    per = []
    for layer in range(obs.shape[0]):
        ol, nl = obs[layer].mean(), nulls[:, layer]
        lo, hi = np.quantile(nl, [0.025, 0.975])
        per.append({"layer": layer, "d_obs": float(ol), "d_null_mean": float(nl.mean()), "ratio": float(ol / nl.mean()),
                    "ratio_interval": [float(ol / hi), float(ol / lo)]})
    return {"d_obs": float(o_all), "d_null_mean": float(n_all.mean()), "d_null_interval": [float(q_lo), float(q_hi)],
            "ratio": float(o_all / n_all.mean()), "ratio_interval": [float(o_all / q_hi), float(o_all / q_lo)],
            "fill_obs": float(128 / o_all), "fill_null": float(128 / n_all.mean()), "per_layer": per}


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--replay", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--draws", type=int, default=DRAWS)
    args = a.parse_args(argv)
    import torch
    picks = torch.load(args.replay, map_location="cpu", weights_only=False)["picks"].numpy().astype(np.int64)
    assert picks.shape == (48, 32, 128, 8), picks.shape
    out = {"replay_shape": list(picks.shape), "seed": SEED, "draws": args.draws, "streams": {}}
    for name, rows in (("S0", slice(0, 16)), ("S1", slice(16, 32))):
        p = picks[:, rows]
        obs = distinct(p)
        assert round(float(obs.mean()), 3) == PUBLISHED[name], (name, obs.mean())   # the positive control
        rng = np.random.default_rng(SEED)
        res = {}
        for null, fn in (("A_circular_shift", shift), ("B_permutation", permute)):
            nulls = np.stack([distinct(fn(p, rng)).mean(axis=1) for _ in range(args.draws)])
            res[null] = summarise(obs, nulls)
        out["streams"][name] = res
    json.dump(out, open(args.out, "w"), indent=1)
    for name, res in out["streams"].items():
        for null, s in res.items():
            print(f"{name} {null}: D_obs {s['d_obs']:.3f} D_null {s['d_null_mean']:.3f} "
                  f"[{s['d_null_interval'][0]:.3f}, {s['d_null_interval'][1]:.3f}] ratio {s['ratio']:.4f} "
                  f"[{s['ratio_interval'][0]:.4f}, {s['ratio_interval'][1]:.4f}] fill {s['fill_obs']:.3f} vs "
                  f"{s['fill_null']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
