"""bench/dq3/dq3_reduce.py -- lane DQ3's registered rule (bench/dq3/DQ3-PREREG.md, "The rule", with Amendments 0 and 1).
Pure: the six arm receipts (palindrome R S S0 S0 S R) in, readings and a verdict out. ``python dq3_reduce.py a1.json ...``
prints the read; ``--self-test`` runs the synthetic cases; tests/test_dq3_lane.py pins them and kills the rule's mutants.
"""
from __future__ import annotations

import json
import statistics
import sys

ORDER = ("R", "S", "S0", "S0", "S", "R")
REGISTERED_DEVICE = "NVIDIA GeForce RTX 5090"
N_LAYERS = 64
WRAPPED = 7 * N_LAYERS
LAYER_STREAMED_BYTES = 243_793_920            # the seven projections' packed NF4 bytes (Qwen3-32B), quant_state resident
SLOT_PREDICTION = (N_LAYERS - 2) * LAYER_STREAMED_BYTES
STEP_MAX = 1.10
SAVING_FRACTION = 0.9
BLOCKING_MAX = 2
ISSUED_PER_STEP = 2 * (N_LAYERS - 2)
SELF_PAIR_BAND = (0.97, 1.03)


def _step_time(arm):
    return statistics.mean(s["s"] for s in arm["timing"]["timed"])


def _per_step_counts(arm):
    """Per timed step, the counter deltas of the one device chain (S only)."""
    prev = None
    out = []
    for s in arm["timing"]["warm"][-1:] + arm["timing"]["timed"]:
        c = next(iter((s.get("counters") or {}).values()), None)
        if c is None:
            return None
        if prev is not None:
            out.append({k: c[k] - prev[k] for k in c if k != "hwm_resident"} | {"hwm_resident": c["hwm_resident"]})
        prev = c
    return out


def reduce(arms: list) -> dict:
    out = {"void": [], "function_fail": [], "noisy": False, "readings": {}, "verdicts": {}}
    if [a.get("arm") for a in arms] != list(ORDER):
        out["void"].append(f"arms {[a.get('arm') for a in arms]} are not the palindrome {ORDER}")
        out["verdicts"]["lane"] = "VOID"
        return out
    for k, a in enumerate(arms):
        tag = f"{a['arm']}{k}"
        if a.get("rehearsal"):
            out["void"].append(f"{tag}: rehearsal receipt")
        if a.get("device") != REGISTERED_DEVICE:
            out["void"].append(f"{tag}: device {a.get('device')!r}")
        if not a.get("finished_at"):
            out["void"].append(f"{tag}: did not finish")
        if a.get("config_overrides") != {"num_hidden_layers": N_LAYERS}:
            out["void"].append(f"{tag}: not the registered subject ({a.get('config_overrides')})")
        e = a.get("engagement", {})
        if (e.get("wrapped") != WRAPPED or e.get("wrapper_kinds") != ["peft.tuners.lora.bnb.Linear4bit"]
                or e.get("lora_dtypes") != ["torch.float32"] or e.get("lora_params") != 2 * WRAPPED
                or not e.get("gradient_checkpointing")):
            out["void"].append(f"{tag}: engagement {e}")
        off = a.get("offload")
        if a["arm"] == "R" and off is not None:
            out["void"].append(f"{tag}: the resident arm has dense offload")
        if a["arm"] != "R" and (off is None or off.get("layers") != N_LAYERS):
            out["void"].append(f"{tag}: dense offload not over {N_LAYERS} layers ({off})")
    if out["void"]:
        out["verdicts"]["lane"] = "VOID"
        return out
    R1, S1, Z1, Z2, S2, R2 = arms
    # deterministic baseline: R against R, bitwise
    if [p.get("loss_bits") for p in R1["parity"]] != [p.get("loss_bits") for p in R2["parity"]] or \
            [p.get("grads") for p in R1["parity"]] != [p.get("grads") for p in R2["parity"]]:
        out["void"].append("R1 and R2 parity passes differ: the deterministic baseline is not deterministic")
        out["verdicts"]["lane"] = "VOID"
        return out
    ref = R1["parity"]
    for a, tag in ((S1, "S1"), (Z1, "S0a"), (Z2, "S0b"), (S2, "S2")):
        for k, (p, q) in enumerate(zip(a["parity"], ref)):
            if p.get("loss_bits") != q.get("loss_bits"):
                out["function_fail"].append(f"{tag} parity step {k + 1}: loss differs ({p.get('loss')} vs {q.get('loss')})")
            bad = [n for n in q["grads"] if p["grads"].get(n) != q["grads"][n]]
            if bad:
                out["function_fail"].append(f"{tag} parity step {k + 1}: {len(bad)} LoRA gradients differ, first {bad[0]}")
        if len(a["parity"]) != len(ref):
            out["function_fail"].append(f"{tag}: {len(a['parity'])} parity steps, R has {len(ref)}")
    for a in arms:
        if not all(s["loss"] == s["loss"] and abs(s["loss"]) != float("inf") for s in a["timing"]["timed"]):
            out["function_fail"].append(f"{a['arm']}: a non-finite timing-pass loss")
    if out["function_fail"]:
        out["verdicts"]["lane"] = "FUNCTION_FAIL"
        return out
    tR1, tR2 = _step_time(R1), _step_time(R2)
    sp = tR2 / tR1
    out["noisy"] = not (SELF_PAIR_BAND[0] <= sp <= SELF_PAIR_BAND[1])
    tR, tS, tZ = (tR1 + tR2) / 2, (_step_time(S1) + _step_time(S2)) / 2, (_step_time(Z1) + _step_time(Z2)) / 2
    peakR = min(R1["timing"]["peak_alloc"], R2["timing"]["peak_alloc"])
    peakS = max(S1["timing"]["peak_alloc"], S2["timing"]["peak_alloc"])
    rd = out["readings"]
    rd.update({"T_R_s": tR, "T_S_s": tS, "T_S0_s": tZ, "S_over_R": tS / tR, "S0_over_R": tZ / tR, "R_self_pair": sp,
               "saving_bytes": peakR - peakS, "slot_prediction_bytes": SLOT_PREDICTION,
               "saving_fraction": (peakR - peakS) / SLOT_PREDICTION})
    steps = [c for a in (S1, S2) for c in (_per_step_counts(a) or [])]
    if not steps:
        out["void"].append("S has no per-step counters")
        out["verdicts"]["lane"] = "VOID"
        return out
    rd["S_steps"] = steps
    if out["noisy"]:
        out["verdicts"]["lane"] = "NOISY"
        return out
    V = out["verdicts"]
    V["lane"] = "READ"
    V["step"] = "PASS" if tS / tR <= STEP_MAX else "SLOW"
    cov_ok = all(c["fwd_blocking"] + c["bwd_blocking"] <= BLOCKING_MAX
                 and c["fwd_prefetch_issued"] == c["bwd_prefetch_issued"] == ISSUED_PER_STEP // 2
                 and sum(c[f"{p}_{k}"] for p in ("fwd", "bwd") for k in ("overlapped", "waited"))
                 == c["fwd_prefetch_issued"] + c["bwd_prefetch_issued"]
                 and c["hwm_resident"] <= 2 and c["unscheduled"] == 0 for c in steps)
    V["coverage"] = "PASS" if cov_ok else "SCHEDULE_FAIL"
    V["capacity"] = "PASS" if peakR - peakS >= SAVING_FRACTION * SLOT_PREDICTION else "SHORT"
    if V["step"] == V["coverage"] == V["capacity"] == "PASS":
        V["lane"] = "PROTO_PASS"
    return out


# ------------------------------------------------------------------ self-test (synthetic receipts; no GPU)
def synth_arm(arm, *, t=3.0, peak=24e9, loss_bits="L", grads=None, fwd_issued=62, bwd_issued=62, blocking=0,
              waited=2, hwm=2, device=REGISTERED_DEVICE, layers=N_LAYERS, wrapper="peft.tuners.lora.bnb.Linear4bit",
              dtype="torch.float32", finished=True, overrides=None, rehearsal=False):
    grads = grads if grads is not None else {"a": "g1", "b": "g2"}
    if arm == "S":
        def c(k):
            return {"cuda:0": {"uses": 0, "resident_hits": 0, "unscheduled": 0, "hwm_resident": hwm,
                               "fwd_prefetch_issued": fwd_issued * k, "bwd_prefetch_issued": bwd_issued * k,
                               "fwd_overlapped": (fwd_issued - waited // 2) * k, "bwd_overlapped": (bwd_issued - waited // 2) * k,
                               "fwd_waited": (waited // 2) * k, "bwd_waited": (waited // 2) * k,
                               "fwd_blocking": blocking * k, "bwd_blocking": 0}}
    a = {"arm": arm, "rehearsal": rehearsal, "device": device, "config_overrides": overrides or {"num_hidden_layers": N_LAYERS},
         "engagement": {"wrapped": WRAPPED, "wrapper_kinds": [wrapper], "lora_params": 2 * WRAPPED,
                        "lora_dtypes": [dtype], "gradient_checkpointing": True},
         "offload": None if arm == "R" else {"layers": layers},
         "parity": [{"loss": 1.0, "loss_bits": loss_bits, "grads": dict(grads)} for _ in range(2)],
         "timing": {"warm": [{"s": t, "loss": 1.0, "counters": c(2) if arm == "S" else None}],
                    "timed": [{"s": t, "loss": 1.0, "counters": c(3 + i) if arm == "S" else None} for i in range(6)],
                    "peak_alloc": peak}}
    if finished:
        a["finished_at"] = "x"
    return a


def synth(**kw):
    """A passing six-arm read; keyword groups override one arm kind: R_*, S_*, S0_*, or all_*."""
    def pick(prefix):
        return {k[len(prefix):]: v for k, v in kw.items() if k.startswith(prefix)}
    base = pick("all_")
    r = {"t": 3.0, "peak": 24e9, **base, **pick("R_")}
    s = {"t": 3.1, "peak": 24e9 - 14.5e9, **base, **pick("S_")}
    z = {"t": 3.7, "peak": 24e9 - 14.5e9, **base, **pick("S0_")}
    r2 = dict(r)
    if "R2_t" in kw:
        r2["t"] = kw["R2_t"]
    if "R2_loss_bits" in kw:
        r2["loss_bits"] = kw["R2_loss_bits"]
    return [synth_arm("R", **r), synth_arm("S", **s), synth_arm("S0", **z), synth_arm("S0", **z), synth_arm("S", **s),
            synth_arm("R", **r2)]


def _self_test() -> int:
    cases = []

    def case(name, arms, check):
        o = reduce(arms)
        ok = bool(check(o))
        cases.append(ok)
        if not ok:
            print("FAIL", name, o["verdicts"], o["void"][:2], o["function_fail"][:2])

    V = lambda o: o["verdicts"]  # noqa: E731
    case("pass", synth(), lambda o: V(o) == {"lane": "PROTO_PASS", "step": "PASS", "coverage": "PASS", "capacity": "PASS"})
    case("S at 1.09x is PASS", synth(S_t=3.0 * 1.09), lambda o: V(o)["step"] == "PASS")
    case("S at 1.11x is SLOW", synth(S_t=3.0 * 1.11), lambda o: V(o)["step"] == "SLOW" and V(o)["lane"] == "READ")
    case("saving 14.0 GB passes (>= 0.9 x 15.12)", synth(S_peak=24e9 - 14.0e9), lambda o: V(o)["capacity"] == "PASS")
    case("saving 13.0 GB is SHORT", synth(S_peak=24e9 - 13.0e9), lambda o: V(o)["capacity"] == "SHORT")
    case("3 blocking per step fails coverage", synth(S_blocking=3), lambda o: V(o)["coverage"] == "SCHEDULE_FAIL")
    case("2 blocking per step passes coverage", synth(S_blocking=2), lambda o: V(o)["coverage"] == "PASS")
    case("forward-only schedule fails coverage", synth(S_bwd_issued=0, S_waited=0),
         lambda o: V(o)["coverage"] == "SCHEDULE_FAIL")
    case("one prefetch too few fails coverage", synth(S_fwd_issued=61), lambda o: V(o)["coverage"] == "SCHEDULE_FAIL")
    case("three resident fails coverage", synth(S_hwm=3), lambda o: V(o)["coverage"] == "SCHEDULE_FAIL")
    case("124 issued but split 63/61 fails coverage", synth(S_fwd_issued=63, S_bwd_issued=61),
         lambda o: V(o)["coverage"] == "SCHEDULE_FAIL")
    case("a rehearsal receipt is VOID", synth(all_rehearsal=True), lambda o: V(o)["lane"] == "VOID")
    case("S loss bits differ is FUNCTION_FAIL", synth(S_loss_bits="X"), lambda o: V(o)["lane"] == "FUNCTION_FAIL")
    case("S0 grads differ is FUNCTION_FAIL", synth(S0_grads={"a": "g1", "b": "zz"}), lambda o: V(o)["lane"] == "FUNCTION_FAIL")
    case("R1 != R2 is VOID", synth(R2_loss_bits="Y"), lambda o: V(o)["lane"] == "VOID")
    case("R self-pair 1.04 is NOISY", synth(R2_t=3.12), lambda o: V(o)["lane"] == "NOISY")
    case("R self-pair 1.02 is read", synth(R2_t=3.06), lambda o: V(o)["lane"] != "NOISY")
    case("wrong card is VOID", synth(all_device="NVIDIA RTX A2000 12GB"), lambda o: V(o)["lane"] == "VOID")
    case("generic PEFT wrapper is VOID", synth(all_wrapper="peft.tuners.lora.layer.Linear"), lambda o: V(o)["lane"] == "VOID")
    case("bf16 adapters are VOID", synth(all_dtype="torch.bfloat16"), lambda o: V(o)["lane"] == "VOID")
    case("an unfinished arm is VOID", synth(S_finished=False), lambda o: V(o)["lane"] == "VOID")
    case("a reduced subject is VOID", synth(all_overrides={"num_hidden_layers": 4}), lambda o: V(o)["lane"] == "VOID")
    case("offload over too few layers is VOID", synth(S_layers=4), lambda o: V(o)["lane"] == "VOID")
    a = synth()
    a[0], a[1] = a[1], a[0]
    case("arms out of palindrome order are VOID", a, lambda o: V(o)["lane"] == "VOID")
    case("readings", synth(), lambda o: abs(o["readings"]["S_over_R"] - 3.1 / 3.0) < 1e-12
         and abs(o["readings"]["saving_fraction"] - 14.5e9 / SLOT_PREDICTION) < 1e-12)
    bad = sum(1 for ok in cases if not ok)
    print(f"self-test {'OK' if not bad else 'FAILED'} ({len(cases)} cases{', ' + str(bad) + ' failed' if bad else ''})")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(_self_test())
    arms = [json.load(open(p)) for p in sys.argv[1:]]
    print(json.dumps(reduce(arms), indent=1, default=str))
