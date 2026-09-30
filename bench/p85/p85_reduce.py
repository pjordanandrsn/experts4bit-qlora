#!/usr/bin/env python3
"""Lane P85's reducer (bench/p85/PREREG-p85.md; e4b#674). Is grouped-nf4-gemm#413 (0.33.7: the fused fp8 KV append's
quotient IEEE-rounded) the whole KERNEL step lane P84 found? P84, on one box: grouped-nf4-gemm 0.33.0 -> 0.33.7 moved the
calibrated int4 recipe's fp32-router wikitext K8 from O's known mean NLL 1.8511420498367808 to N's 1.8506507749113845;
the e4b release and the harness moved it by zero. Every reading here is e4b c77aab6 (0.37.4) through P70's harness:

    O_build, O_rep   gnf4 0.33.0, the fused append on (P70's build; the control: must be O's known float)
    F_1, F_2         gnf4 0.33.0, E4B_FUSED_KV_APPEND=0 (every append through quantize_kv_fp8)   predicted: N's float
    S_1, S_2         gnf4 0.33.6, the fused append on (every release of the cut but #413's)      predicted: O's float

Reads k8_<reading>.json, their .router.json stamps (router, versions, the fused append's resolution), and packs.json.

  VOID          O_build is not O's known mean NLL bit for bit (this box does not reproduce P70's build); or a reading
                is missing, off the registered window or step count, not on the fp32 router, not from its stack, or
                not stamped with its append setting; or a pair does not repeat bit for bit
  PATH-REFUTED  F equals O's known float: turning the fused append off changes nothing, so it is not on K8's path
  CONFIRMED     F equals N's known float and S equals O's: #413 is the whole KERNEL step
  MIXED         F equals N's known float but S does not equal O's: the append reproduces N, and something in
                0.33.1-0.33.6 also moves K8
  REFUTED       otherwise (F is neither known float): the append moves K8, but its bytes are not the whole step
Reported: F - O, F - N and S - O in nats, and the expert pack.

    python p85_reduce.py --dir <dir> --out verdict.json
    python p85_reduce.py --control-ok <k8_O_build.json>     # exit 0 iff it equals O's known mean NLL
    python p85_reduce.py --self-test
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
N_KNOWN = 1.8506507749113845     # P82 / P83-N / P84 C, H1, H2: gnf4 0.33.7, fp32 router
PAIRS = (("O_build", "O_rep"), ("F_1", "F_2"), ("S_1", "S_2"))
READINGS = tuple(k for p in PAIRS for k in p)
# per reading: (e4b, gnf4, the fused append's resolution, the E4B_FUSED_KV_APPEND env)
WANT = {"O": ("0.37.4", "0.33.0", True, None), "F": ("0.37.4", "0.33.0", False, "0"), "S": ("0.37.4", "0.33.6", True, None)}
LICENSED_FP = "sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42"


def _valid(k: str, r: dict, st: dict) -> str | None:
    if not str(r.get("text_sha", "")).startswith(WINDOW) or int(r.get("tokens_scored", r.get("steps", -1))) != STEPS:
        return f"{k}: not the registered window/steps (sha {str(r.get('text_sha'))[:12]}, steps {r.get('steps')})"
    if st.get("router_epi_cast_weights") is not False:
        return f"{k}: not on the fp32 router (stamp {st.get('router_epi_cast_weights')!r})"
    e4b, gnf4, resolved, env = WANT[k[0]]
    if st.get("e4b") != e4b or st.get("gnf4") != gnf4:
        return f"{k}: stamped e4b {st.get('e4b')} / gnf4 {st.get('gnf4')}, not its stack's ({e4b} / {gnf4})"
    if st.get("fp8_kv_has_413") is not False:
        return f"{k}: the installed fp8_kv is not the pre-#413 kernel (stamp {st.get('fp8_kv_has_413')!r})"
    if st.get("fused_kv_append_resolved") is not resolved or st.get("fused_kv_append_env") != env:
        return (f"{k}: the fused append resolved {st.get('fused_kv_append_resolved')!r} under env "
                f"{st.get('fused_kv_append_env')!r}, not {resolved!r} under {env!r}")
    return None


def reduce(recs: dict, stamps: dict, packs: dict | None = None) -> dict:
    v: dict = {"lane": "P85", "reasons": [], "verdict": None, "known": {"O": O_KNOWN, "N": N_KNOWN}}
    nll = {k: float(recs[k]["mean_nll"]) for k in READINGS if isinstance(recs.get(k), dict) and "mean_nll" in recs[k]}
    v["mean_nll"] = nll
    v["ppl"] = {k: math.exp(x) for k, x in nll.items()}
    if packs:
        v["packs"] = packs
    # the control first: it decides whether any known float may be used on this box
    if "O_build" not in nll:
        v["verdict"] = "VOID"
        v["reasons"].append("the control reading O_build is missing")
        return v
    if nll["O_build"] != O_KNOWN:
        v["verdict"] = "VOID"
        v["reasons"].append(f"CONTROL: O_build {nll['O_build']!r} != O's known {O_KNOWN!r} -- this box does not reproduce "
                            "P70's build, so no comparison with a known float holds")
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
    for a, b in PAIRS:
        if nll[a] != nll[b]:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{a} and {b} differ on this box: {nll[a]!r} vs {nll[b]!r}")
            return v
    f, s = nll["F_1"], nll["S_1"]
    v["nats"] = {"F_minus_O": f - O_KNOWN, "F_minus_N": f - N_KNOWN, "S_minus_O": s - O_KNOWN}
    if f == O_KNOWN:
        v["verdict"] = "PATH-REFUTED"
        v["reasons"].append("F equals O's known float: turning the fused append off changes nothing on K8")
    elif f == N_KNOWN and s == O_KNOWN:
        v["verdict"] = "CONFIRMED"
        v["reasons"].append("F equals N's known float and S equals O's: #413 is the whole KERNEL step")
    elif f == N_KNOWN:
        v["verdict"] = "MIXED"
        v["reasons"].append(f"F equals N's known float, but S is {s!r}, not O's: 0.33.1-0.33.6 also move K8")
    else:
        v["verdict"] = "REFUTED"
        v["reasons"].append(f"F is {f!r}, neither known float: the append moves K8, but its bytes are not the whole step")
    if packs:
        v["reported"] = {"O_expert_is_licensed": packs.get("O_expert") == LICENSED_FP}
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
    stamps = {}
    for k in nll:
        e4b, gnf4, resolved, env = WANT[k[0]]
        stamps[k] = {"router_epi_cast_weights": False, "e4b": e4b, "gnf4": gnf4, "fp8_kv_has_413": False,
                     "fused_kv_append_resolved": resolved, "fused_kv_append_env": env}
    return recs, stamps


def _set(f: float, s: float, o: float = O_KNOWN) -> dict:
    return {"O_build": o, "O_rep": o, "F_1": f, "F_2": f, "S_1": s, "S_2": s}


def self_test() -> None:
    # the four outcomes
    assert reduce(*_synthetic(_set(N_KNOWN, O_KNOWN)))["verdict"] == "CONFIRMED"
    assert reduce(*_synthetic(_set(O_KNOWN, O_KNOWN)))["verdict"] == "PATH-REFUTED"
    assert reduce(*_synthetic(_set(O_KNOWN, N_KNOWN)))["verdict"] == "PATH-REFUTED"
    assert reduce(*_synthetic(_set(N_KNOWN, 1.8509)))["verdict"] == "MIXED"
    got = reduce(*_synthetic(_set(1.8508, O_KNOWN)))
    assert got["verdict"] == "REFUTED", got
    assert abs(got["nats"]["F_minus_O"] - (1.8508 - O_KNOWN)) < 1e-15
    # the control: off O's known float, or missing, voids everything -- even with every other reading present
    assert reduce(*_synthetic(_set(N_KNOWN, O_KNOWN, o=O_KNOWN + 1e-12)))["verdict"] == "VOID"
    r, s = _synthetic(_set(N_KNOWN, O_KNOWN))
    del r["O_build"]
    assert reduce(r, s)["verdict"] == "VOID"
    # a pair that does not repeat, a missing reading, the wrong window, a cast router: VOID
    x = _set(N_KNOWN, O_KNOWN)
    x["F_2"] = N_KNOWN + 1e-12
    assert reduce(*_synthetic(x))["verdict"] == "VOID"
    x = _set(N_KNOWN, O_KNOWN)
    x["O_rep"] = N_KNOWN
    assert reduce(*_synthetic(x))["verdict"] == "VOID"
    r, s = _synthetic(_set(N_KNOWN, O_KNOWN))
    del r["S_2"]
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_set(N_KNOWN, O_KNOWN))
    r["S_1"]["text_sha"] = "0" * 64
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_set(N_KNOWN, O_KNOWN))
    s["O_rep"]["router_epi_cast_weights"] = None
    assert reduce(r, s)["verdict"] == "VOID"
    # the wrong stack, a #413 kernel, and the append not resolving as the reading registers: VOID
    r, s = _synthetic(_set(N_KNOWN, O_KNOWN))
    s["S_1"]["gnf4"] = "0.33.0"
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_set(N_KNOWN, O_KNOWN))
    s["F_1"]["gnf4"] = "0.33.7"
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_set(N_KNOWN, O_KNOWN))
    s["S_2"]["fp8_kv_has_413"] = True
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_set(N_KNOWN, O_KNOWN))
    s["F_2"]["fused_kv_append_resolved"] = True
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(_set(N_KNOWN, O_KNOWN))
    s["O_build"]["fused_kv_append_env"] = "0"
    assert reduce(r, s)["verdict"] == "VOID"
    # the control gate the runner calls
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "o.json"
        f.write_text(json.dumps({"mean_nll": O_KNOWN}))
        assert control_ok(f) is True
        f.write_text(json.dumps({"mean_nll": O_KNOWN + 1e-12}))
        assert control_ok(f) is False
        assert control_ok(Path(t) / "missing.json") is False
    print("p85_reduce self-test OK (20 cases)")


def control_ok(f: Path) -> bool:
    try:
        return float(json.loads(Path(f).read_text())["mean_nll"]) == O_KNOWN
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
        print(f"P85_CONTROL {'OK' if ok else 'FAILED'}: O_build vs O's known mean NLL {O_KNOWN!r}")
        return 0 if ok else 1
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(*_load(Path(a.dir)))
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P85_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    for k, x in sorted((v.get("mean_nll") or {}).items()):
        print(f"  K8 {k} = {math.exp(x):.5f} ({x!r})")
    for k, x in (v.get("nats") or {}).items():
        print(f"  {k} = {x:+.6f} nats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
