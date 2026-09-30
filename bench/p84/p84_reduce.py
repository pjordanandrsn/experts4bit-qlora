#!/usr/bin/env python3
"""Lane P84's reducer (bench/p84/PREREG-p84.md; e4b#674). Which factor moved the calibrated int4 recipe's fp32-router
wikitext K8 from P70's build (mean NLL 1.8511420498367808, 6.36709) to P82's (1.8506507749113845, 6.36396)? P83 showed
each value bit-reproducible across machines, so the two known floats anchor a chain of one-factor steps built on ONE box:

    O (known)  e4b 0.37.4 + gnf4 0.33.0 + P70's harness
      | KERNEL   grouped-nf4-gemm 0.33.0 -> 0.33.7
    H2         e4b 0.37.4 + gnf4 0.33.7 + P70's harness
      | PACKAGE  e4b 0.37.4 -> the launch commit
    H1         e4b launch + gnf4 0.33.7 + P70's harness
      | HARNESS  bench/p39 step_decomp + hook v6 + P70's env -> bench/p81 step_decomp + hook v7 + P82's env
    C          e4b launch + gnf4 0.33.7 + P82's harness  == P82's build, which must read N's known float (the control)

Reads k8_{C,H1,H2}_{build,rep}.json, their .router.json stamps, and packs.json.

  VOID    the control C_build does not equal N's known mean NLL bit for bit (this box does not reproduce it, so no
          comparison with a known float holds); or a reading is missing, off the registered window or step count, not
          on the fp32 router, or not from its stack; or a build does not repeat itself bit for bit;
  else    the verdict names every step whose two ends differ, in chain order: some of KERNEL, PACKAGE, HARNESS joined
          by '+'. At least one must differ, because O and N differ.
Reported: each step's difference in nats (they sum to N - O), and the packs.

    python p84_reduce.py --dir <dir> --out verdict.json
    python p84_reduce.py --control-ok <k8_C_build.json>     # exit 0 iff it equals N's known mean NLL
    python p84_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

WINDOW = "9ef10d760ad9"
STEPS = 2048
O_KNOWN = 1.8511420498367808     # P70 / P64 / P83-O: e4b 0.37.4 + gnf4 0.33.0 + P70's harness, fp32 router
N_KNOWN = 1.8506507749113845     # P82 / P83-N: e4b 0.37.8 + gnf4 0.33.7 + P82's harness, fp32 router
BUILDS = ("C", "H1", "H2")
READINGS = tuple(f"{b}_{k}" for b in BUILDS for k in ("build", "rep"))
STACK = {"C": "B", "H1": "B", "H2": "A"}
STACK_VERSIONS = {"B": (None, "0.33.7"), "A": ("0.37.4", "0.33.7")}   # (e4b, gnf4); B's e4b is the launch commit's
LICENSED_FP = "sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42"
ATTN_FP = "sha256:d7cfa1f496d75120ef50b72fae0a31928b6ffb04ac0c5ec49c522a673068a422"


def _valid(k: str, r: dict, st: dict) -> str | None:
    if not str(r.get("text_sha", "")).startswith(WINDOW) or int(r.get("tokens_scored", r.get("steps", -1))) != STEPS:
        return f"{k}: not the registered window/steps (sha {str(r.get('text_sha'))[:12]}, steps {r.get('steps')})"
    if st.get("router_epi_cast_weights") is not False:
        return f"{k}: not on the fp32 router (stamp {st.get('router_epi_cast_weights')!r})"
    e4b, gnf4 = STACK_VERSIONS[STACK[k.split("_")[0]]]
    if st.get("gnf4") != gnf4 or (e4b is not None and st.get("e4b") != e4b) or (e4b is None and st.get("e4b") == "0.37.4"):
        return f"{k}: stamped e4b {st.get('e4b')} / gnf4 {st.get('gnf4')}, not its stack's"
    return None


def reduce(recs: dict, stamps: dict, packs: dict | None = None) -> dict:
    v: dict = {"lane": "P84", "reasons": [], "verdict": None, "known": {"O": O_KNOWN, "N": N_KNOWN}}
    nll = {k: float(recs[k]["mean_nll"]) for k in READINGS if isinstance(recs.get(k), dict) and "mean_nll" in recs[k]}
    v["mean_nll"] = nll
    v["ppl"] = {k: math.exp(x) for k, x in nll.items()}
    if packs:
        v["packs"] = packs
    # the control first: it decides whether any known float may be used on this box
    for k in ("C_build", "C_rep"):
        if k not in nll:
            v["verdict"] = "VOID"
            v["reasons"].append(f"the control reading {k} is missing")
            return v
    if nll["C_build"] != N_KNOWN:
        v["verdict"] = "VOID"
        v["reasons"].append(f"CONTROL: C_build {nll['C_build']!r} != N's known {N_KNOWN!r} -- this box does not reproduce "
                            "P82's build, so no comparison with a known float holds")
        return v
    missing = [k for k in READINGS if k not in nll]
    if missing:
        v["verdict"] = "VOID"
        v["reasons"].append(f"reading(s) without a K8 receipt: {missing}")
        return v
    for k in READINGS:
        bad = _valid(k, recs[k], stamps.get(k) or {})
        if bad:
            v["verdict"] = "VOID"
            v["reasons"].append(bad)
            return v
    for b in BUILDS:
        if nll[f"{b}_build"] != nll[f"{b}_rep"]:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{b} does not repeat its K8 on this box: {nll[f'{b}_build']!r} vs {nll[f'{b}_rep']!r}")
            return v
    steps = {"KERNEL": nll["H2_build"] - O_KNOWN, "PACKAGE": nll["H1_build"] - nll["H2_build"],
             "HARNESS": nll["C_build"] - nll["H1_build"]}
    v["step_nats"] = steps
    moved = [f for f in ("KERNEL", "PACKAGE", "HARNESS") if steps[f] != 0.0]
    v["verdict"] = "+".join(moved)
    v["reasons"].append("moved by: " + ", ".join(f"{f} {steps[f]:+.6f} nats" for f in moved)
                        + f"; the steps sum to N - O = {N_KNOWN - O_KNOWN:+.6f} nats")
    if packs:
        v["reported"] = {f"{b}_expert_is_licensed": packs.get(f"{b}_expert") == LICENSED_FP for b in BUILDS}
        v["reported"]["C_attention_equals_d7cfa1f4"] = packs.get("C_attention") == ATTN_FP
    return v


def _load(d: Path):
    recs, stamps = {}, {}
    for k in READINGS:
        for store, f in ((recs, d / f"k8_{k}.json"), (stamps, d / f"k8_{k}.router.json")):
            if f.is_file():
                try:
                    store[k] = json.loads(f.read_text())
                except json.JSONDecodeError:
                    store[k] = {}
    p = d / "packs.json"
    return recs, stamps, (json.loads(p.read_text()) if p.is_file() else None)


def _synthetic(nll: dict):
    recs = {k: {"steps": STEPS, "tokens_scored": STEPS, "mean_nll": x, "ppl": math.exp(x), "text_sha": WINDOW + "f" * 52}
            for k, x in nll.items()}
    stamps = {k: {"router_epi_cast_weights": False, "e4b": "0.37.4" if k.startswith("H2") else "0.37.8", "gnf4": "0.33.7"}
              for k in nll}
    return recs, stamps


def _chain(h2: float, h1: float, c: float = N_KNOWN) -> dict:
    return {"C_build": c, "C_rep": c, "H1_build": h1, "H1_rep": h1, "H2_build": h2, "H2_rep": h2}


def self_test() -> None:
    # one factor at a time
    assert reduce(*_synthetic(_chain(O_KNOWN, N_KNOWN)))["verdict"] == "PACKAGE"
    assert reduce(*_synthetic(_chain(O_KNOWN, O_KNOWN)))["verdict"] == "HARNESS"
    assert reduce(*_synthetic(_chain(N_KNOWN, N_KNOWN)))["verdict"] == "KERNEL"
    # two factors, and the steps sum to N - O
    got = reduce(*_synthetic(_chain(1.8509, N_KNOWN)))
    assert got["verdict"] == "KERNEL+PACKAGE", got
    assert abs(sum(got["step_nats"].values()) - (N_KNOWN - O_KNOWN)) < 1e-15
    assert reduce(*_synthetic(_chain(1.8509, 1.8508)))["verdict"] == "KERNEL+PACKAGE+HARNESS"
    # the control: off N's known float, or missing, voids everything -- even with every other reading present
    assert reduce(*_synthetic(_chain(O_KNOWN, N_KNOWN, c=N_KNOWN + 1e-12)))["verdict"] == "VOID"
    r, s = _synthetic(_chain(O_KNOWN, N_KNOWN))
    del r["C_rep"]
    assert reduce(r, s)["verdict"] == "VOID"
    # a build that does not repeat, a missing reading, the wrong window, a cast router, the wrong stack: VOID
    x = _chain(O_KNOWN, N_KNOWN)
    x["H1_rep"] = N_KNOWN + 1e-12
    assert reduce(*_synthetic(x))["verdict"] == "VOID"
    r, s = _synthetic(_chain(O_KNOWN, N_KNOWN))
    del r["H2_build"]
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_chain(O_KNOWN, N_KNOWN))
    r["H1_build"]["text_sha"] = "0" * 64
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_chain(O_KNOWN, N_KNOWN))
    s["C_rep"]["router_epi_cast_weights"] = None
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_chain(O_KNOWN, N_KNOWN))
    s["H2_build"]["e4b"] = "0.37.8"
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_chain(O_KNOWN, N_KNOWN))
    s["H1_rep"]["gnf4"] = "0.33.0"
    assert reduce(r, s)["verdict"] == "VOID"
    # the control gate the runner calls
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "c.json"
        f.write_text(json.dumps({"mean_nll": N_KNOWN}))
        assert control_ok(f) is True
        f.write_text(json.dumps({"mean_nll": N_KNOWN + 1e-12}))
        assert control_ok(f) is False
        assert control_ok(Path(t) / "missing.json") is False
    print("p84_reduce self-test OK (15 cases)")


def control_ok(f: Path) -> bool:
    try:
        return float(json.loads(Path(f).read_text())["mean_nll"]) == N_KNOWN
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--control-ok")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if a.control_ok:
        ok = control_ok(Path(a.control_ok))
        print(f"P84_CONTROL {'OK' if ok else 'FAILED'}: C_build vs N's known mean NLL {N_KNOWN!r}")
        return 0 if ok else 1
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(*_load(Path(a.dir)))
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P84_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    for k, x in sorted((v.get("mean_nll") or {}).items()):
        print(f"  K8 {k} = {math.exp(x):.5f} ({x!r})")
    for k, x in (v.get("step_nats") or {}).items():
        print(f"  step {k} = {x:+.6f} nats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
