"""D0 Step 1: per-expert retention of the trained expert delta under repacking (PREREG-d0.md, "Step 1").

For every layer, projection and expert, and each cell T x S (T in NF4 / INT4 / MXFP4; S in Q = Q_train(W) from
fixture-base, W = the raw checkpoint):  D = deq(T(S + dW)) - deq(T(S)).  One CSV row per (layer, which, expert, cell)
carries the sums the summaries need (dot = <D,dW>, dd = |dW|^2, DD = |D|^2, ee = |D-dW|^2, changed, numel, and the
base conversion error bb = |deq(T(S)) - Q|^2 with qq = |Q|^2). One row per (layer, which, expert) records the two
reported checks: deq(NF4(Q)) == Q exactly, and the CPU rebuild's share of codes differing from fixture-base.

    python d0_retention.py --ckpt DIR --fixture-base DIR [--dq] --adapter DIR --out DIR [--layers 0-15] [--experts N]

Resumable: a layer whose CSVs exist is skipped.
"""
import argparse
import csv
import os
import time

import torch

import d0lib


def parse_layers(s: str):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def code_diff_share(w: torch.Tensor, packed: torch.Tensor) -> tuple:
    """The CPU rebuild (bitsandbytes CPU quantize_4bit of the raw weight) against fixture-base's stored codes."""
    import bitsandbytes.functional as F
    q, _ = F.quantize_4bit(w.contiguous(), blocksize=d0lib.BLOCK, quant_type="nf4", compress_statistics=False)
    a, b = q.reshape(-1), packed.reshape(-1)
    diff = int(((a >> 4) != (b >> 4)).sum()) + int(((a & 0xF) != (b & 0xF)).sum())
    return diff, 2 * a.numel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fixture-base", required=True)
    ap.add_argument("--dq", action="store_true", help="the fixture trained at the double-quantized absmax")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-15")
    ap.add_argument("--experts", type=int, default=0, help="first N experts only (plumbing tests)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck, fb, ad = d0lib.Checkpoint(a.ckpt), d0lib.FixtureBase(a.fixture_base, a.dq), d0lib.Adapter(a.adapter)
    T = d0lib.targets()
    cfg = d0lib.load_cfg(a.ckpt)
    H, inter = cfg["hidden_size"], cfg["intermediate_size"]
    shapes = {"gate_up": (2 * inter, H), "down": (H, inter)}
    for layer in parse_layers(a.layers):
        cell_path = os.path.join(a.out, f"cells_L{layer:02d}.csv")
        chk_path = os.path.join(a.out, f"checks_L{layer:02d}.csv")
        if os.path.exists(cell_path) and os.path.exists(chk_path):
            continue
        t0 = time.time()
        cell_rows, chk_rows = [], []
        for which in d0lib.WHICH:
            n, k = shapes[which]
            Q = fb.q_train(layer, which, n, k)
            dW = ad.delta(layer, which)
            packed = fb.codes(layer, which)
            if Q.shape != dW.shape:
                raise ValueError(f"L{layer} {which}: Q {tuple(Q.shape)} vs dW {tuple(dW.shape)}")
            E = Q.shape[0] if not a.experts else a.experts
            for e in range(E):
                W = ck.expert(layer, e, which)
                q, d = Q[e], dW[e]
                diff, ncodes = code_diff_share(W, packed[e])
                chk_rows.append({"layer": layer, "which": which, "expert": e,
                                 "nf4_q_exact": bool(torch.equal(T["NF4"](q), q)),
                                 "cpu_codes_diff": diff, "codes": ncodes,
                                 "dW_norm2": float(d.double().pow(2).sum()), "Q_norm2": float(q.double().pow(2).sum())})
                for sname, src in (("Q", q), ("W", W)):
                    for tname, fn in T.items():
                        base = fn(src)
                        D = fn(src + d) - base
                        r = {"layer": layer, "which": which, "expert": e, "T": tname, "S": sname,
                             "dot": float((D.double() * d.double()).sum()), "dd": float(d.double().pow(2).sum()),
                             "DD": float(D.double().pow(2).sum()), "ee": float((D - d).double().pow(2).sum()),
                             "changed": int((D != 0).sum()), "numel": D.numel(),
                             "bb": float((base - q).double().pow(2).sum()), "qq": float(q.double().pow(2).sum())}
                        cell_rows.append(r)
            del Q, dW
        for path, rows in ((cell_path, cell_rows), (chk_path, chk_rows)):
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, path)
        print(f"L{layer:02d} done in {time.time() - t0:.0f} s ({len(cell_rows)} cell rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
