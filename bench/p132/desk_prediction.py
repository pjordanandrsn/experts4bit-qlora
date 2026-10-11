"""Desk prediction for (b), the 5090 timing split of the 1 -> n row increment (e4b#1313). $0, receipts only.

Every number in the note is printed here, from the receipts it reads:
  --e4b DIR     an e4b checkout holding bench/sd2/receipts/sd2-5090-3/v.json, bench/sd2/sd2_reduce.py (TAU_SD1, G_BAR,
                verify_cost) and bench/p123/receipts/p123-5090-1/census_b{1,16}.json
  --k14 FILE    gnf4 kernel/receipts-k14/k14_rows.json (cuBLAS bf16 per shape at M = 1 and 4, RTX 5090)
  --traces F..  gnf4 bench/cold-engine/routing-trace/qwen3_{prose,code,math,dialogue}.jsonl (the distinct-expert ratio)

Target: I(n) = verify_ms(n) - anchor, n = k + 1 rows (sd2-5090-3 stage V1). Components, each lo / central / hi:
  bytes(n)     = (rho(n) - 1) x moe_expert(B=1)                                   one value (lo = central = hi)
  route(n)     lo 48 x the 5090 graph launch floor (one tile table per layer); hi = P123 moe_route(B16) - moe_route(B1)
               (all of the route growth to 16 rows, which includes the 48 tile tables); central = their mean
  dense(n)     lo 0 (K14's q_proj is flat at M = 4); central = (K14 attention-projection time ratio M4/M1 - 1) x the
               attention projections' byte share of P123's B = 1 dense time (router and lm_head held flat);
               hi = P123 dense(B16) - dense(B1)
  attn_norm(n) = (n - 1) x [P123 (attn + norm_elem)(B16) - (B1)] / 15                one value
The hole is I(n) minus the components (lo / central / hi swap ends). H's reading (registered): in (b), the share of the
hole that lies in the expert path, [dExp(n) - bytes(n) - route_c(n)] / [I_b(n) - known_c(n)] with dExp = the measured
moe_expert + moe_route increment, I_b the measured increment and _c the central terms here: SUPPORT >= 0.70,
INCONCLUSIVE 0.50-0.70, REFUTE < 0.50. Savable ceiling S(n) = expert-path increment - bytes(n); the build line: BUILD a
2-4-row expert kernel iff the MEASURED S(n) >= need(n) for some n, need(n) = the second-smallest over the three
workloads of full_ms(k) - TAU_SD1[w][k] x anchor / G_BAR (SD2's own S_V at its FASTER bar, 2 of 3 workloads).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

TRACE_SHA = {"prose": "1293f390", "code": "64647ca1", "math": "b432a869", "dialogue": "1dc7aea0"}
NS = (2, 3, 4)
# Qwen3-30B-A3B's dense projections (bf16 bytes = 2 N K): fused q/k/v and o per layer x 48, the router x 48, lm_head
ATTN_BYTES = 48 * 2 * (5120 * 2048 + 2048 * 4096)
ROUTER_BYTES = 48 * 2 * 128 * 2048
LMHEAD_BYTES = 2 * 151936 * 2048


def sha8(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]


def rho_from_traces(paths):
    pooled = {n: [0, 0] for n in (1,) + NS}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            meta = json.loads(f.readline())["meta"]
            steps = sorted((json.loads(line) for line in f if line.strip()), key=lambda s: s["step"])
        assert meta["top_k"] == 8 and meta["n_experts"] == 128
        assert sha8(path) == TRACE_SHA[meta["prompt"]], f"{path}: sha256 is not the registered trace"
        for layer in steps[0]["routed"]:
            sets = [frozenset(s["routed"][layer]) for s in steps]
            for n in pooled:
                for t in range(len(sets) - n + 1):
                    pooled[n][0] += len(frozenset().union(*sets[t:t + n]))
                    pooled[n][1] += 1
    e = {n: s / c for n, (s, c) in pooled.items()}
    assert abs(e[1] - 8.0) < 1e-9
    return {n: e[n] / 8.0 for n in NS}, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4b", required=True)
    ap.add_argument("--k14", required=True)
    ap.add_argument("--traces", nargs=4, required=True)
    a = ap.parse_args()
    e4b = Path(a.e4b)
    spec = importlib.util.spec_from_file_location("sd2_reduce", e4b / "bench/sd2/sd2_reduce.py")
    red = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(red)
    v1 = json.loads((e4b / "bench/sd2/receipts/sd2-5090-3/v.json").read_text())["v1"]
    vc = red.verify_cost(v1)                              # the registered reading: anchor, S_V
    anchor = vc["anchor_ms"]
    full = {k: v1[f"k{k}"]["full_ms"] for k in (1, 2, 3)}
    ver = {k: v1[f"k{k}"]["verify_ms"] for k in (1, 2, 3)}
    b1 = json.loads((e4b / "bench/p123/receipts/p123-5090-1/census_b1.json").read_text())["terms"]
    b16 = json.loads((e4b / "bench/p123/receipts/p123-5090-1/census_b16.json").read_text())["terms"]
    k14 = json.loads(Path(a.k14).read_text())
    rows = {(r["shape"], r["M"]): r["ms"]["bf16"] for r in k14["rows"]}
    shapes = ("q_proj", "k_proj", "v_proj", "o_proj")
    r4 = sum(rows[(s, 4)] for s in shapes) / sum(rows[(s, 1)] for s in shapes)
    floor = k14["launch_floor_ms"]
    rho, e_d = rho_from_traces(a.traces)

    attn_share = ATTN_BYTES / (ATTN_BYTES + ROUTER_BYTES + LMHEAD_BYTES)
    route_hi = b16["moe_route"] - b1["moe_route"]
    route_lo = 48 * floor
    dense_hi = b16["dense_gemm"] - b1["dense_gemm"]
    dense_c = (r4 - 1.0) * attn_share * b1["dense_gemm"]
    per_row = ((b16["attn"] + b16["norm_elem"]) - (b1["attn"] + b1["norm_elem"])) / 15.0
    print(f"inputs: anchor {anchor:.4f} ms (verify_cost); moe_expert(B1) {b1['moe_expert']:.4f}; moe_route B1/B16 "
          f"{b1['moe_route']:.4f}/{b16['moe_route']:.4f}; dense B1/B16 {b1['dense_gemm']:.4f}/{b16['dense_gemm']:.4f}; "
          f"attn+norm per extra row {per_row:.4f}; K14 M4/M1 ratio (q,k,v,o) {r4:.4f}; launch floor {floor * 1e3:.3f} us; "
          f"attention-projection byte share {attn_share:.4f}")
    print("E[D_n] =", {n: round(x, 4) for n, x in e_d.items()}, " rho =", {n: round(x, 4) for n, x in rho.items()})
    out = {"anchor_ms": anchor, "rho": rho, "per_n": {}, "need": {}}
    print(f"\n{'n':>2} {'I(n)':>7} | {'bytes':>6} | {'route lo/c/hi':>20} | {'dense lo/c/hi':>20} | {'attn+norm':>9} |"
          f" {'known lo/c/hi':>20} | {'HOLE lo/c/hi':>20} | {'S under H lo/c/hi':>20}")
    for n in NS:
        k = n - 1
        I = ver[k] - anchor
        by = (rho[n] - 1.0) * b1["moe_expert"]
        route = (route_lo, (route_lo + route_hi) / 2, route_hi)
        dense = (0.0, dense_c, dense_hi)
        an = per_row * (n - 1)
        known = tuple(by + route[i] + dense[i] + an for i in range(3))
        hole = (I - known[2], I - known[1], I - known[0])
        sav = (I - dense[2] - an - by, I - dense[1] - an - by, I - dense[0] - an - by)    # (i) - bytes if the hole is expert-path
        out["per_n"][n] = {"I": I, "bytes": by, "route": route, "dense": dense, "attn_norm": an,
                           "known": known, "hole": hole, "S_under_H": sav, "S_c_share_of_I": sav[1] / I}
        f3 = lambda t: "/".join(f"{x:.3f}" for x in t)  # noqa: E731
        print(f"{n:>2} {I:7.3f} | {by:6.3f} | {f3(route):>20} | {f3(dense):>20} | {an:9.3f} | {f3(known):>20} |"
              f" {f3(hole):>20} | {f3(sav):>20}")
    print("\nbuild thresholds: need(n) = second-smallest over workloads of full_ms(k) - TAU_SD1 x anchor / G_BAR"
          f" (G_BAR {red.G_BAR})")
    for n in NS:
        k = n - 1
        need = {w: full[k] - red.TAU_SD1[w][k] * anchor / red.G_BAR for w in red.WORKLOADS}
        second = sorted(need.values())[1]
        out["need"][n] = {"per_workload": need, "threshold": second}
        print(f"  n={n} (k={k}): full {full[k]:.4f}; need " + ", ".join(f"{w} {x:.3f}" for w, x in need.items())
              + f"  -> threshold {second:.3f} ms")
    print("\n" + json.dumps(out, default=lambda x: list(x) if isinstance(x, tuple) else x, sort_keys=True))


if __name__ == "__main__":
    main()
