#!/usr/bin/env python3
"""Lane P83's reducer (bench/p83/PREREG-p83.md; e4b#674). A same-box A/B of the licensed build's wikitext K8 under two
software stacks, both with the fp32 router:
  O = P70's build as P70 ran it (e4b 0.37.4 + grouped-nf4-gemm 0.33.0), and its repeat (P55x's "lic" arm);
  N = P82's build as P82 ran it (e4b at the launch commit + grouped-nf4-gemm 0.33.7), and its repeat (P82's K32).
Reads k8_{O_build,O_rep,N_build,N_rep}.json, their .router.json stamps, and packs.json.

  VOID       a reading is missing, off the registered window or step count, not on the fp32 router, or not from its
             stack; or a stack does not REPEAT its own K8 exactly on this box (then the A/B cannot separate the box from
             the software);
  SAME       O's build and N's build read the identical K8 (mean NLL bit-equal): on this box the software between them
             does not move K8, so the spread between boxes is the box;
  DIFFERENT  they differ: the software moved K8 on this box.
Reported, never decisive: each stack's K8 against the licensed 6.36709 and P82's 6.36396 (five decimals), the
difference in nats, and the packs.

    python p83_reduce.py --dir <dir> --out verdict.json
    python p83_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

READINGS = ("O_build", "O_rep", "N_build", "N_rep")
WINDOW = "9ef10d760ad9"
STEPS = 2048
STACK_VERSIONS = {"O": ("0.37.4", "0.33.0"), "N": (None, "0.33.7")}   # (e4b, gnf4); N's e4b is the launch commit's
LICENSED_K8 = 6.36709      # P55x / P64 / P70's builds (e4b <= 0.37.4, fp32 router)
P82_K8 = 6.36396           # P82's build and K32 on its box (e4b 0.37.8, fp32 router)
LICENSED_FP = "sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42"
P81_P82_ATTN_FP = "sha256:d7cfa1f496d75120ef50b72fae0a31928b6ffb04ac0c5ec49c522a673068a422"


def _stack(name: str) -> str:
    return name.split("_", 1)[0]


def reduce(recs: dict, stamps: dict, packs: dict | None = None) -> dict:
    """``recs`` maps a reading to its K8 receipt, ``stamps`` to its router stamp (either may be missing)."""
    v: dict = {"lane": "P83", "reasons": [], "verdict": None}
    missing = [k for k in READINGS if not isinstance(recs.get(k), dict) or "mean_nll" not in recs[k]]
    if missing:
        v["verdict"] = "VOID"
        v["reasons"].append(f"reading(s) without a K8 receipt: {missing}")
        return v
    for k in READINGS:
        r, st = recs[k], stamps.get(k) or {}
        if not str(r.get("text_sha", "")).startswith(WINDOW) or int(r.get("tokens_scored", r.get("steps", -1))) != STEPS:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k}: not the registered window/steps (sha {str(r.get('text_sha'))[:12]}, steps {r.get('steps')})")
            return v
        if st.get("router_epi_cast_weights") is not False:
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k}: not on the fp32 router (stamp {st.get('router_epi_cast_weights')!r})")
            return v
        e4b, gnf4 = STACK_VERSIONS[_stack(k)]
        if st.get("gnf4") != gnf4 or (e4b is not None and st.get("e4b") != e4b) or (e4b is None and st.get("e4b") == "0.37.4"):
            v["verdict"] = "VOID"
            v["reasons"].append(f"{k}: stamped e4b {st.get('e4b')} / gnf4 {st.get('gnf4')}, not stack {_stack(k)}'s")
            return v
    nll = {k: float(recs[k]["mean_nll"]) for k in READINGS}
    ppl = {k: float(recs[k]["ppl"]) for k in READINGS}
    v["ppl"] = ppl
    v["mean_nll"] = nll
    for s in ("O", "N"):
        if nll[f"{s}_build"] != nll[f"{s}_rep"]:
            v["verdict"] = "VOID"
            v["reasons"].append(f"stack {s} does not repeat its K8 on this box: build {ppl[f'{s}_build']!r} vs repeat "
                                f"{ppl[f'{s}_rep']!r} -- the A/B cannot separate the box from the software")
            return v
    d = nll["N_build"] - nll["O_build"]
    v["delta_nats_N_minus_O"] = d
    if d == 0.0:
        v["verdict"] = "SAME"
        v["reasons"].append(f"O and N read the identical K8 on this box ({ppl['O_build']!r}): the software between them "
                            "does not move K8 here; the spread between boxes is the box")
    else:
        v["verdict"] = "DIFFERENT"
        v["reasons"].append(f"O {ppl['O_build']:.5f} vs N {ppl['N_build']:.5f} ({d:+.5f} nats): the software moved K8 on this box")
    v["reported"] = {
        "O_reproduces_licensed_5dp": round(ppl["O_build"], 5) == LICENSED_K8,
        "N_reproduces_p82_box_5dp": round(ppl["N_build"], 5) == P82_K8,
        "O_minus_licensed_nats": nll["O_build"] - math.log(LICENSED_K8),
        "N_minus_p82_box_nats": nll["N_build"] - math.log(P82_K8),
    }
    if packs:
        v["reported"].update({
            "O_expert_is_licensed": packs.get("O_expert") == LICENSED_FP,
            "N_expert_is_licensed": packs.get("N_expert") == LICENSED_FP,
            "N_attention_equals_p81_p82": packs.get("N_attention") == P81_P82_ATTN_FP,
            "packs": packs,
        })
    return v


def _load(d: Path):
    recs, stamps = {}, {}
    for k in READINGS:
        f = d / f"k8_{k}.json"
        if f.is_file():
            try:
                recs[k] = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                recs[k] = {"status": f"unreadable: {e}"}
        g = d / f"k8_{k}.router.json"
        if g.is_file():
            try:
                stamps[k] = json.loads(g.read_text())
            except json.JSONDecodeError:
                stamps[k] = {}
    p = d / "packs.json"
    packs = json.loads(p.read_text()) if p.is_file() else None
    return recs, stamps, packs


def _synthetic(nll: dict):
    recs = {k: {"k8": "ppl", "steps": STEPS, "tokens_scored": STEPS, "mean_nll": x, "ppl": math.exp(x),
                "text_sha": WINDOW + "f" * 52} for k, x in nll.items()}
    stamps = {k: {"router_epi_cast_env": "0", "router_epi_cast_weights": False,
                  "e4b": "0.37.4" if k.startswith("O") else "0.37.8",
                  "gnf4": "0.33.0" if k.startswith("O") else "0.33.7"} for k in nll}
    return recs, stamps


def self_test() -> None:
    same = {"O_build": 1.851, "O_rep": 1.851, "N_build": 1.851, "N_rep": 1.851}
    assert reduce(*_synthetic(same))["verdict"] == "SAME"
    o, n = math.log(6.367091), math.log(6.363960)   # the licensed build's and P82's readings, to their printed digits
    diff = {"O_build": o, "O_rep": o, "N_build": n, "N_rep": n}
    got = reduce(*_synthetic(diff))
    assert got["verdict"] == "DIFFERENT" and got["delta_nats_N_minus_O"] < 0, got
    assert got["reported"]["O_reproduces_licensed_5dp"] and got["reported"]["N_reproduces_p82_box_5dp"], got["reported"]
    # a stack that does not repeat itself voids the A/B, whichever stack it is
    for bad in ("O_rep", "N_rep"):
        x = dict(diff)
        x[bad] += 1e-9
        assert reduce(*_synthetic(x))["verdict"] == "VOID", bad
    # a missing reading, the wrong window, a cast router, or the wrong stack's versions void it
    r, s = _synthetic(same)
    del r["N_rep"]
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(same)
    r["O_build"]["text_sha"] = "0" * 64
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(same)
    r["N_build"]["tokens_scored"] = 1024
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(same)
    s["N_rep"]["router_epi_cast_weights"] = None
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(same)
    s["O_rep"]["gnf4"] = "0.33.7"
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(same)
    s["N_build"]["e4b"] = "0.37.4"
    assert reduce(r, s)["verdict"] == "VOID"
    r, s = _synthetic(same)
    del s["O_build"]
    assert reduce(r, s)["verdict"] == "VOID"
    # packs are reported, never decisive
    r, s = _synthetic(diff)
    got = reduce(r, s, {"O_expert": LICENSED_FP, "N_expert": "sha256:" + "e" * 64, "N_attention": P81_P82_ATTN_FP})
    assert got["verdict"] == "DIFFERENT" and got["reported"]["N_expert_is_licensed"] is False
    assert got["reported"]["N_attention_equals_p81_p82"] is True
    print("p83_reduce self-test OK (13 cases)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not a.dir or not a.out:
        ap.error("--dir and --out are required")
    v = reduce(*_load(Path(a.dir)))
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P83_VERDICT {v['verdict']} " + " | ".join(v["reasons"]))
    for k, x in (v.get("ppl") or {}).items():
        print(f"  K8 {k} = {x:.5f} ({x!r})")
    for k, x in (v.get("reported") or {}).items():
        if k != "packs":
            print(f"  {k} = {x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
