"""bench/dq4/dq4_reduce.py -- lane DQ4's registered rule (bench/dq4/DQ4-PREREG.md). Pure: a directory of dq4_cap.py receipts in,
readings and verdicts out. ``python dq4_reduce.py <dir>`` or ``python dq4_reduce.py --self-test``.

Per configuration (c_def graded, c_exp secondary, s_def descriptive), per arm X in {R, S}:
  L*_X = the ladder's largest passing sequence; CONFIRMED when a fresh process passes at L*_X and fails at the ladder's
  first OOM (or the ladder reached its top: CEILING, L*_X is a lower bound).
  G = L*_S / L*_R.  CAP_REAL if G >= 1.5, CAP_MARGINAL if 1.1 <= G < 1.5, CAP_NONE if G < 1.1.
Order: VOID, then FUNCTION_FAIL, then NOISY (a confirmation disagrees with its ladder), then the CAP verdict.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

REGISTERED_DEVICE = "NVIDIA GeForce RTX 5090"
N_LAYERS = 64
WRAPPED = 7 * N_LAYERS
G_REAL, G_MARGINAL = 1.5, 1.1
CONFIGS = {"c_def": ("chunked", "default", True), "c_exp": ("chunked", "expandable_segments:True", True),
           "s_def": ("stock", "default", False)}
ROLE = {"c_def": "graded", "c_exp": "secondary", "s_def": "descriptive"}


def _void(tag, a, loss, alloc):
    out = []
    if a.get("rehearsal"):
        out.append(f"{tag}: rehearsal receipt")
    if a.get("device") != REGISTERED_DEVICE:
        out.append(f"{tag}: device {a.get('device')!r}")
    if a.get("config_overrides") != {"num_hidden_layers": N_LAYERS}:
        out.append(f"{tag}: not the registered subject ({a.get('config_overrides')})")
    e = a.get("engagement") or {}
    if (e.get("wrapped") != WRAPPED or e.get("wrapper_kinds") != ["peft.tuners.lora.bnb.Linear4bit"]
            or e.get("lora_dtypes") != ["torch.float32"] or not e.get("gradient_checkpointing")):
        out.append(f"{tag}: engagement {e}")
    if a.get("loss") != loss:
        out.append(f"{tag}: loss {a.get('loss')!r} != {loss!r}")
    if loss == "chunked" and not (a.get("chunked_loss") or {}).get("patched"):
        out.append(f"{tag}: the chunked loss did not engage ({a.get('chunked_loss')})")
    if a.get("alloc_conf") != alloc:
        out.append(f"{tag}: allocator {a.get('alloc_conf')!r} != {alloc!r}")
    if a.get("arm") == "S":
        off = a.get("offload") or {}
        if off.get("layers") != N_LAYERS or off.get("late_bound_4bit") != WRAPPED:
            out.append(f"{tag}: offload {off} (want {N_LAYERS} layers, late_bound_4bit {WRAPPED})")
    if not a.get("finished_at"):
        out.append(f"{tag}: did not finish")
    return out


def _rung(a, seq):
    return next((r for r in a.get("rungs", []) if r["seq"] == seq), None)


def reduce_config(cfg: str, rec: dict) -> dict:
    """rec: tag-suffix -> receipt for this configuration (R_ladder, S_ladder, R_ok, R_oom, S_ok, S_oom)."""
    loss, alloc, confirm = CONFIGS[cfg]
    out = {"role": ROLE[cfg], "void": [], "function_fail": [], "noisy": [], "arms": {}}
    for arm in ("R", "S"):
        lad = rec.get(f"{arm}_ladder")
        if lad is None:
            out["void"].append(f"{arm}: no ladder receipt")
            continue
        out["void"] += _void(f"{arm}_ladder", lad, loss, alloc)
        if lad.get("arm") != arm or lad.get("mode") != "ladder":
            out["void"].append(f"{arm}_ladder: role mismatch")
        for r in lad.get("rungs", []):
            if r["ok"] and not all(s["finite"] for s in r["steps"]):
                out["function_fail"].append(f"{arm} ladder seq {r['seq']}: non-finite loss")
        L, F = lad.get("max_ok"), lad.get("first_oom")
        arm_out = {"L": L, "first_oom": F, "ceiling": F is None and L is not None}
        if L is None:
            out["void"].append(f"{arm}: OOM at the ladder's first rung ({lad.get('ladder', {}).get('start')})")
        if confirm and L is not None:
            ok, oom = rec.get(f"{arm}_ok"), rec.get(f"{arm}_oom")
            if ok is None:
                out["void"].append(f"{arm}: no fresh confirmation at L*")
            else:
                out["void"] += _void(f"{arm}_ok", ok, loss, alloc)
                r = _rung(ok, L)
                if r is None or not r["ok"]:
                    out["noisy"].append(f"{arm}: L*={L} failed fresh")
                else:
                    if not all(s["finite"] for s in r["steps"]):
                        out["function_fail"].append(f"{arm} confirm seq {L}: non-finite loss")
                    arm_out["at_L"] = {"peak_alloc": r["peak_alloc"], "peak_reserved": r["peak_reserved"],
                                       "step_s": [s["s"] for s in r["steps"]]}
            if F is not None:
                if oom is None:
                    out["void"].append(f"{arm}: no fresh confirmation at the first OOM")
                else:
                    out["void"] += _void(f"{arm}_oom", oom, loss, alloc)
                    r = _rung(oom, F)
                    if r is None or r["ok"]:
                        out["noisy"].append(f"{arm}: first OOM {F} passed fresh")
        if not confirm and L is not None:
            r = _rung(lad, L)
            arm_out["at_L"] = {"peak_alloc": r["peak_alloc"], "peak_reserved": r["peak_reserved"],
                               "step_s": [s["s"] for s in r["steps"]]}
        out["arms"][arm] = arm_out
    shas = {a.get("env", {}).get("E4B_SHA") for a in rec.values()}
    if len(shas) > 1:
        out["void"].append(f"software differs across receipts: {sorted(map(str, shas))}")
    R, S = out["arms"].get("R", {}), out["arms"].get("S", {})
    out["L_R"], out["L_S"] = R.get("L"), S.get("L")
    out["G"] = (S["L"] / R["L"]) if (R.get("L") and S.get("L")) else None
    if R.get("ceiling"):
        out["void"].append("R reached the ladder's top: no boundary to compare against")
    out["G_is_lower_bound"] = bool(S.get("ceiling"))
    # descriptive: T(S)/T(R) at the largest sequence both ladders passed
    lr, ls = rec.get("R_ladder"), rec.get("S_ladder")
    if lr and ls:
        common = sorted({r["seq"] for r in lr.get("rungs", []) if r["ok"]} & {r["seq"] for r in ls.get("rungs", []) if r["ok"]})
        if common:
            c = common[-1]
            tr = sum(s["s"] for s in _rung(lr, c)["steps"]) / len(_rung(lr, c)["steps"])
            ts = sum(s["s"] for s in _rung(ls, c)["steps"]) / len(_rung(ls, c)["steps"])
            out["step_ratio_at"] = {"seq": c, "S_over_R": ts / tr}
    if out["void"]:
        out["verdict"] = "VOID"
    elif out["function_fail"]:
        out["verdict"] = "FUNCTION_FAIL"
    elif out["noisy"]:
        out["verdict"] = "NOISY"
    elif out["G"] is None:
        out["verdict"] = "VOID"
    elif out["G"] >= G_REAL:
        out["verdict"] = "CAP_REAL"
    elif out["G"] >= G_MARGINAL:
        out["verdict"] = "CAP_MARGINAL" if not out["G_is_lower_bound"] else "CAP_MARGINAL_OR_MORE"
    else:
        out["verdict"] = "CAP_NONE" if not out["G_is_lower_bound"] else "CAP_UNRESOLVED"
    return out


def reduce_dir(d: str) -> dict:
    by_cfg: dict = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        name = os.path.basename(f)[:-5]
        cfg, _, suffix = name.partition("_")
        cfg = f"{cfg}_{suffix.split('_')[0]}" if suffix else cfg
        if cfg not in CONFIGS:
            continue
        rest = name[len(cfg) + 1:]
        by_cfg.setdefault(cfg, {})[rest] = json.load(open(f))
    configs = {c: reduce_config(c, by_cfg[c]) for c in CONFIGS if c in by_cfg}
    verdicts = {c: v["verdict"] for c, v in configs.items()}
    lane = verdicts.get("c_def", "VOID")
    return {"schema": "dq4-read/1", "verdicts": {"lane": lane, **verdicts}, "configs": configs,
            "skipped": sorted(set(CONFIGS) - set(configs))}


# ------------------------------------------------------------------------------------------------------------- self-test --
def synth_receipt(arm, mode, *, loss="chunked", alloc="default", seqs=(), oom_at=None, seq=None, finite=True, device=None,
                  rehearsal=False, layers=N_LAYERS, late=WRAPPED, patched=True, sha="x" * 40, wrapped=WRAPPED):
    rungs = []
    run = [seq] if mode == "confirm" else list(seqs)
    for s in run:
        ok = oom_at is None or s < oom_at
        rungs.append({"seq": s, "ok": ok, "steps": ([{"s": s / 1000, "loss": 1.0 if finite else float("nan"),
                                                       "finite": finite}] * 2 if ok else []),
                      "peak_alloc": s * 10, "peak_reserved": s * 11})
        if not ok:
            break
    oks = [r["seq"] for r in rungs if r["ok"]]
    a = {"arm": arm, "mode": mode, "loss": loss, "alloc_conf": alloc, "rehearsal": rehearsal,
         "device": device or REGISTERED_DEVICE, "config_overrides": {"num_hidden_layers": layers},
         "engagement": {"wrapped": wrapped, "wrapper_kinds": ["peft.tuners.lora.bnb.Linear4bit"],
                        "lora_dtypes": ["torch.float32"], "gradient_checkpointing": True},
         "chunked_loss": {"patched": patched} if loss == "chunked" else None, "env": {"E4B_SHA": sha},
         "rungs": rungs, "max_ok": max(oks) if oks else None,
         "first_oom": next((r["seq"] for r in rungs if not r["ok"]), None), "finished_at": "t",
         "ladder": {"start": 2048, "step": 1024, "top": 32768}}
    if arm == "S":
        a["offload"] = {"layers": layers, "late_bound_4bit": late}
    return a


def synth_config(*, r_oom=9216, s_oom=20480, top=32768, cfg="c_def", r_ok_fresh=True, s_oom_fresh=True, **kw):
    loss, alloc, confirm = CONFIGS[cfg]
    seqs = range(2048, top + 1, 1024)
    rec = {"R_ladder": synth_receipt("R", "ladder", loss=loss, alloc=alloc, seqs=seqs, oom_at=r_oom, **kw),
           "S_ladder": synth_receipt("S", "ladder", loss=loss, alloc=alloc, seqs=seqs, oom_at=s_oom, **kw)}
    if confirm:
        for arm, oom in (("R", r_oom), ("S", s_oom)):
            L = rec[f"{arm}_ladder"]["max_ok"]
            fresh_ok = r_ok_fresh if arm == "R" else True
            if L is not None:
                rec[f"{arm}_ok"] = synth_receipt(arm, "confirm", loss=loss, alloc=alloc, seq=L,
                                                 oom_at=None if fresh_ok else L, **kw)
            if rec[f"{arm}_ladder"]["first_oom"] is not None:
                fresh_oom = s_oom_fresh if arm == "S" else True
                rec[f"{arm}_oom"] = synth_receipt(arm, "confirm", loss=loss, alloc=alloc, seq=oom,
                                                  oom_at=oom if fresh_oom else None, **kw)
    return rec


def self_test() -> int:
    fails = []

    def case(name, cfg, rec, check):
        try:
            o = reduce_config(cfg, rec)
            if not check(o):
                fails.append(f"{name}: {o['verdict']} G={o['G']} void={o['void'][:2]} noisy={o['noisy']}")
        except Exception as exc:  # a crash is a failure of the self-test
            fails.append(f"{name}: crashed {exc!r}")

    V = lambda o: o["verdict"]  # noqa: E731
    case("the predicted read is CAP_REAL", "c_def", synth_config(), lambda o: V(o) == "CAP_REAL" and o["L_R"] == 8192
         and o["L_S"] == 19456 and abs(o["G"] - 19456 / 8192) < 1e-9)
    case("G exactly 1.5 is CAP_REAL", "c_def", synth_config(r_oom=5120, s_oom=7168), lambda o: V(o) == "CAP_REAL")
    case("G 1.25 is CAP_MARGINAL", "c_def", synth_config(r_oom=9216, s_oom=11264), lambda o: V(o) == "CAP_MARGINAL")
    case("G 1.0 is CAP_NONE", "c_def", synth_config(r_oom=9216, s_oom=9216), lambda o: V(o) == "CAP_NONE")
    case("S at the ceiling is a lower bound", "c_def", synth_config(s_oom=None), lambda o: V(o) == "CAP_REAL"
         and o["G_is_lower_bound"])
    case("R at the ceiling is VOID", "c_def", synth_config(r_oom=None, s_oom=None), lambda o: V(o) == "VOID")
    case("a fresh failure at L* is NOISY", "c_def", synth_config(r_ok_fresh=False), lambda o: V(o) == "NOISY")
    case("a fresh pass at the first OOM is NOISY", "c_def", synth_config(s_oom_fresh=False), lambda o: V(o) == "NOISY")
    case("a non-finite loss is FUNCTION_FAIL", "c_def", synth_config(finite=False), lambda o: V(o) == "FUNCTION_FAIL")
    rec = synth_config()
    rec["S_ladder"]["rungs"][0]["steps"][0] = {"s": 1.0, "loss": float("nan"), "finite": False}
    case("a non-finite loss in a ladder rung alone is FUNCTION_FAIL", "c_def", rec, lambda o: V(o) == "FUNCTION_FAIL")
    case("the wrong device is VOID", "c_def", synth_config(device="NVIDIA RTX A2000 12GB"), lambda o: V(o) == "VOID")
    case("a rehearsal receipt is VOID", "c_def", synth_config(rehearsal=True), lambda o: V(o) == "VOID")
    case("a reduced subject is VOID", "c_def", synth_config(layers=8), lambda o: V(o) == "VOID")
    rec = synth_config()
    rec["R_ladder"]["config_overrides"] = {"num_hidden_layers": 8}
    case("a reduced R subject alone is VOID", "c_def", rec, lambda o: V(o) == "VOID")
    case("an un-routed S is VOID", "c_def", synth_config(late=0), lambda o: V(o) == "VOID")
    case("a refused chunked loss is VOID", "c_def", synth_config(patched=False), lambda o: V(o) == "VOID")
    case("short engagement is VOID", "c_def", synth_config(wrapped=7), lambda o: V(o) == "VOID")
    case("OOM at the first rung is VOID", "c_def", synth_config(r_oom=2048), lambda o: V(o) == "VOID")
    rec = synth_config()
    rec["S_ladder"]["env"]["E4B_SHA"] = "y" * 40
    case("software differing across receipts is VOID", "c_def", rec, lambda o: V(o) == "VOID")
    rec = synth_config()
    del rec["S_ok"]
    case("a missing confirmation is VOID", "c_def", rec, lambda o: V(o) == "VOID")
    rec = synth_config()
    rec["R_ladder"]["alloc_conf"] = "expandable_segments:True"
    case("the wrong allocator is VOID", "c_def", rec, lambda o: V(o) == "VOID")
    case("the expandable config grades its own allocator", "c_exp", synth_config(cfg="c_exp"), lambda o: V(o) == "CAP_REAL")
    case("the stock config needs no confirmations", "s_def", synth_config(cfg="s_def", r_oom=4096, s_oom=9216),
         lambda o: V(o) == "CAP_REAL" and abs(o["G"] - 8192 / 3072) < 1e-9)
    case("reserved and allocated at L* are reported", "c_def", synth_config(),
         lambda o: o["arms"]["S"]["at_L"]["peak_reserved"] == 19456 * 11 and o["arms"]["R"]["at_L"]["peak_alloc"] == 81920)
    case("the step ratio is read at the largest common rung", "c_def", synth_config(),
         lambda o: o["step_ratio_at"]["seq"] == 8192 and abs(o["step_ratio_at"]["S_over_R"] - 1.0) < 1e-9)
    if fails:
        print("self-test FAILED:")
        print("\n".join(fails))
        return 1
    print("self-test OK (25 cases)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(self_test())
    json.dump(reduce_dir(sys.argv[1]), sys.stdout, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    print()
