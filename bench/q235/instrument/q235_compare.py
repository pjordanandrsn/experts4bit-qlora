# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Compare two runs' per-layer outputs, router top-k and loss over the REAL tokens (attention mask 1; pads carry no meaning):
the per-layer relative error ||a - b|| / ||b|| (fp32), the fraction of tokens whose top-k expert SET agrees, the mean overlap
|A & B| / k, and the loss gap a - b. ``b`` is the reference."""
from __future__ import annotations

import argparse
import json
import os

import torch


def _layers(d):
    return sorted(f for f in os.listdir(d) if f.startswith("layer") and f.endswith(".pt"))


def _topk(d):
    for name in ("reference.pt", "nf4.pt"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return torch.load(p, weights_only=True)["router_topk"]
    raise FileNotFoundError(f"no router top-k file in {d}")


def _loss(d):
    for name in ("reference.json", "nf4.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return json.load(open(p))["loss"]
    raise FileNotFoundError(f"no loss record in {d}")


def compare(a_dir, b_dir, mask):
    m = mask.bool()
    flat = m.reshape(-1)
    la, lb = _layers(a_dir), _layers(b_dir)
    assert la == lb, "the two runs hold different layer sets"
    rel = []
    for f in la:
        a = torch.load(os.path.join(a_dir, f), weights_only=True).float()[m]
        b = torch.load(os.path.join(b_dir, f), weights_only=True).float()[m]
        rel.append(float((a - b).norm() / b.norm().clamp_min(1e-30)))
    ta, tb = _topk(a_dir), _topk(b_dir)
    exact, overlap = [], []
    for x, y in zip(ta, tb):
        x, y = x[flat], y[flat]
        xs, ys = x.sort(dim=-1).values, y.sort(dim=-1).values
        exact.append(float((xs == ys).all(dim=-1).float().mean()))
        k = x.shape[-1]
        ov = (x.unsqueeze(-1) == y.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float() / k
        overlap.append(float(ov.mean()))
    s = sorted(rel)
    return {"layers": len(rel), "rel_err": rel, "rel_err_median": s[len(s) // 2], "rel_err_max": max(rel),
            "router_set_agree": exact, "router_set_agree_min": min(exact), "router_overlap": overlap,
            "router_overlap_min": min(overlap), "loss_a": _loss(a_dir), "loss_b": _loss(b_dir),
            "loss_gap": _loss(a_dir) - _loss(b_dir), "real_tokens": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", help="the reference")
    ap.add_argument("tokens")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = compare(a.a, a.b, torch.load(a.tokens, weights_only=True)["attention_mask"])
    json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("layers", "rel_err_median", "rel_err_max", "router_set_agree_min", "router_overlap_min", "loss_gap")}))


if __name__ == "__main__":
    main()
