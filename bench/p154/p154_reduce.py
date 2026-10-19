#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""p154_reduce.py -- P154's rule (bench/p154/PREREG-p154.md): the W16 expert byte floor of Qwen3-30B-A3B's served step,
on the replay's own route, against the grouped NF4 small-M GEMM times P152's census RECORDED.

Inputs: the routing pass's record (``route_b16.json``, from ``p154_route.py``) and P152's committed verdict
(``p152_verdict.json``, staged at its pinned bytes). Nothing is re-timed.

- VOID: the record missing; any fault in it (the route not device-grouped K25 on an NF4 store on every MoE layer, the
  K25 count not 2 a layer a step, the arena not this run's own, a router or step fault); other than 64 steps; bytes per
  expert unread; P152's verdict not at its pinned bytes or its two GEMM lines missing.
- READ: for gate_up and down, the floor = the mean unique experts a step (summed over the layers) x that projection's
  bytes per expert / P141's copy rate (1,498.9 GB/s, as P152 registered it); the GEMM's recorded time; their ratio
  (the GEMM's fraction of the copy rate); and the headroom (recorded minus floor). Reported beside it: the same floor at
  this box's own copy rate, and at the step totals' min and max. REPORTING only: it names no lever.

    python p154_reduce.py --route route_b16.json --p152 p152_verdict.json --out verdict_p154.json
    python p154_reduce.py --self-test
"""
import argparse
import hashlib
import json
import sys

COPY_GBPS = 1498.9            # P141's measured copy rate, as P152's floor used it
P152_VERDICT_SHA256 = "a99d8b0fec82ad27645e79e0aa49ccada904e7788ba1d91e560652663305bc6e"   # p152-5090-2's verdict_p152.json
GEMM_KEYS = {"gate_up": "_gemm_nf4_grouped_smallm|136x48x1", "down": "_gemm_nf4_grouped_smallm|136x64x1"}
STEPS = 64
BYTES_KEY = {"gate_up": "gate_up_bytes_per_expert", "down": "down_bytes_per_expert"}


def recorded_gemm(p152: dict) -> dict:
    """The two GEMM lines' removable time, as P152's W16 census recorded them (ms a step), or {} when absent."""
    ranked = {r["key"]: r for r in (p152.get("W16") or {}).get("ranked") or []}
    out = {}
    for proj, key in GEMM_KEYS.items():
        r = ranked.get(key)
        if r is None or r.get("n_per_step") != 48:
            return {}
        out[proj] = float(r["removable_ms"])
    return out


def floor_ms(unique_total: float, bytes_per_expert: float, gbps: float) -> float:
    return unique_total * bytes_per_expert / (gbps * 1e9) * 1e3


def reduce(route: dict | None, p152: dict | None, p152_sha: str | None) -> dict:
    why = []
    if route is None:
        why.append("route_b16.json missing")
    if p152 is None or p152_sha != P152_VERDICT_SHA256:
        why.append("P152's verdict is not at its pinned bytes")
    gemm = recorded_gemm(p152 or {})
    if p152 is not None and not gemm:
        why.append("P152's verdict lacks the two grouped NF4 GEMM lines at 48 launches a step")
    if route is not None:
        if route.get("faults"):
            why.append(f"{len(route['faults'])} routing-pass faults: {route['faults'][:3]}")
        if (route.get("summary") or {}).get("steps") != STEPS:
            why.append(f"{(route.get('summary') or {}).get('steps')} steps recorded, {STEPS} registered")
        st = route.get("store") or {}
        if st.get("kind") != "nf4" or not all(isinstance(st.get(BYTES_KEY[p]), (int, float)) and st[BYTES_KEY[p]] > 0
                                             for p in BYTES_KEY):
            why.append("the NF4 store's bytes per expert are unread")
    v = {"lane": "P154", "verdict": "VOID" if why else "READ", "why": why, "copy_gbps": COPY_GBPS,
         "p152_verdict_sha256": p152_sha, "recorded_gemm_ms": gemm, "reporting_only": True}
    if why:
        return v
    s, st = route["summary"], route["store"]
    u, lo, hi = (s["unique_per_step_total"][k] for k in ("mean", "min", "max"))
    box = (route.get("copy_rate") or {}).get("gbps")
    proj = {}
    for p in ("gate_up", "down"):
        b = float(st[BYTES_KEY[p]])
        f = floor_ms(u, b, COPY_GBPS)
        proj[p] = {"bytes_per_expert": b, "floor_ms": round(f, 4), "recorded_ms": gemm[p],
                   "fraction_of_copy": round(f / gemm[p], 4), "headroom_ms": round(gemm[p] - f, 4),
                   "floor_ms_step_range": [round(floor_ms(lo, b, COPY_GBPS), 4), round(floor_ms(hi, b, COPY_GBPS), 4)],
                   "floor_ms_at_box_copy_rate": round(floor_ms(u, b, box), 4) if box else None}
    both_f = sum(proj[p]["floor_ms"] for p in proj)
    both_r = sum(gemm.values())
    v.update({"unique_per_step_total": s["unique_per_step_total"], "unique_per_layer_step": s["unique_per_layer_step"],
              "box_copy_gbps": box, "projections": proj,
              "both": {"floor_ms": round(both_f, 4), "recorded_ms": round(both_r, 4),
                       "fraction_of_copy": round(both_f / both_r, 4), "headroom_ms": round(both_r - both_f, 4)}})
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route")
    ap.add_argument("--p152")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    def load(path):
        try:
            with open(path, "rb") as f:
                raw = f.read()
            return json.loads(raw), hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, TypeError):
            return None, None
    route, _ = load(a.route)
    p152, sha = load(a.p152)
    v = reduce(route, p152, sha)
    with open(a.out, "w", newline="\n") as f:
        json.dump(v, f, indent=1)
        f.write("\n")
    print("P154 " + v["verdict"] + " " + json.dumps({"why": v["why"], "both": v.get("both")}), flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------- self-test --
def _route(**over):
    r = {"faults": [], "summary": {"steps": 64, "unique_per_step_total": {"mean": 2736.81, "min": 2507, "max": 3035},
                                   "unique_per_layer_step": {"mean": 57.0169, "min": 40, "max": 81}},
         "store": {"kind": "nf4", "gate_up_bytes_per_expert": 1769472.0, "down_bytes_per_expert": 884736.0},
         "copy_rate": {"gbps": 1500.0}}
    r.update(over)
    return r


def _p152(gu=8.395073, dn=3.35296, n=48):
    return {"W16": {"ranked": [{"key": GEMM_KEYS["gate_up"], "n_per_step": n, "removable_ms": gu},
                               {"key": GEMM_KEYS["down"], "n_per_step": n, "removable_ms": dn},
                               {"key": "Kernel2|8x16x1", "n_per_step": 48, "removable_ms": 1.111104}]}}


def self_test() -> int:
    cases = []
    sha = P152_VERDICT_SHA256
    v = reduce(_route(), _p152(), sha)
    gu = v.get("projections", {}).get("gate_up", {})
    cases.append(("a clean record READs", v["verdict"] == "READ" and not v["why"]))
    cases.append(("gate_up's floor is the mean unique a step x its bytes over P141's rate",
                  abs(gu.get("floor_ms", 0) - 2736.81 * 1769472 / 1498.9e9 * 1e3) < 1e-3))
    cases.append(("the fraction of copy is floor over the RECORDED time", gu.get("recorded_ms") == 8.395073
                  and abs(gu["fraction_of_copy"] - gu["floor_ms"] / 8.395073) < 1e-4))
    cases.append(("the headroom is recorded minus floor", abs(gu["headroom_ms"] - (8.395073 - gu["floor_ms"])) < 1e-4))
    cases.append(("the box's own copy rate is reported beside", gu["floor_ms_at_box_copy_rate"] is not None
                  and gu["floor_ms_at_box_copy_rate"] < gu["floor_ms"]))
    cases.append(("a routing fault VOIDs", reduce(_route(faults=["route: x"]), _p152(), sha)["verdict"] == "VOID"))
    cases.append(("other than 64 steps VOIDs", reduce(_route(summary=dict(_route()["summary"], steps=63)), _p152(),
                                                      sha)["verdict"] == "VOID"))
    cases.append(("a store that is not NF4 VOIDs",
                  reduce(_route(store={"kind": None}), _p152(), sha)["verdict"] == "VOID"))
    cases.append(("unread bytes per expert VOID",
                  reduce(_route(store={"kind": "nf4", "gate_up_bytes_per_expert": None, "down_bytes_per_expert": 1.0}),
                         _p152(), sha)["verdict"] == "VOID"))
    cases.append(("P152's verdict at other bytes VOIDs", reduce(_route(), _p152(), "0" * 64)["verdict"] == "VOID"))
    cases.append(("P152's GEMM lines missing VOIDs", reduce(_route(), {"W16": {"ranked": []}}, sha)["verdict"] == "VOID"))
    cases.append(("a GEMM line at other than 48 launches VOIDs", reduce(_route(), _p152(n=47), sha)["verdict"] == "VOID"))
    cases.append(("a missing record VOIDs", reduce(None, _p152(), sha)["verdict"] == "VOID"))
    cases.append(("an unread box copy rate leaves the floor READ, its column empty",
                  reduce(_route(copy_rate={"error": "x"}), _p152(), sha)["projections"]["down"]["floor_ms_at_box_copy_rate"]
                  is None))
    bad = [n for n, ok in cases if not ok]
    print(f"p154_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
