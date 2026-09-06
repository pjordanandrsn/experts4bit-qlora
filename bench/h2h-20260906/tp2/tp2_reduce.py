#!/usr/bin/env python3
"""tp2_reduce.py -- the per-family support / position / quality table for lane tp2 in the pre-registered form (P40-PREREG.md).

Reads <fam>_<fw>_<tag>.json (tp2_arm.py receipts and stubs) plus, when present, summary.txt (one `fam/fw/tag rc=N` line
per attempt), versions.txt and ds_manifest.json, and prints per family:

  1. the SUPPORT table -- every expected arm a row (and every unexpected receipt too), status vocabulary EXACTLY
     OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN, mapped from the receipt's status
     (ok, c1_failed -> OK; refused -> REFUSED; oom -> OOM; install_failed -> INSTALL_FAILED; load_fault, verify_failed ->
     LOAD_FAULT; alarm -> ALARM; not_run -> NOT_RUN; tokens_mismatch / void_trainable / void_attn4 / harness_error /
     unreadable -> HARNESS_ERROR; no receipt: a summary rc line -> HARNESS_ERROR (crash before the first write), else NOT_RUN);
  2. VALIDITY per OK row (P40 "Validity rules", VOID never enters a ratio), with the registered n_layers table
     (Granite 32, OLMoE 16, Qwen3 48, Gemma-4 30, Mixtral 32; gpt-oss 24): e4b fused `n_patched == n_layers` and
     `kernel_calls_per_step_min >= 2*n_layers`, `n_attn4 == 4*n_layers` when attn-4bit; Unsloth census
     `Params4bit_expert_stacks >= 2*n_layers`, `experts_forward_calls_per_step_min >= n_layers`, `n_bnb4bit_unwrapped >=
     n_layers`, the "Enabling LoRA on MoE parameters" banner; both: C1 bit-exact (bytes > 0, empties 0, control fired),
     step count == N, same tokens sha, same trainable count (vs the family's e4b primary);
  3. the POSITION from the primary pair (e4b fused_attn4 vs unsloth ckpt_unsloth; gpt-oss has no primary pair by
     registration): s/step ratio unsloth/e4b (median of steps 11..N), peak-VRAM delta, J/step ratio, tok/s both --
     quoted only when both VALID; the QUALITY reading (held-out |delta| at N <= 0.05 nats -> COMPARABLE, else FLAGGED and
     the ratio carries the flag);
  4. e4b internal parity (fused_attn4 vs reference_attn4) in tp1's rule (VOID: init_sha differs, C1, n_patched == 0,
     kernel calls < 2*n_patched, step counts; else PASS iff |delta final| <= 0.05 AND median step-wise |delta| <= 0.05), informational;
  5. the cross-lane anchor: Qwen3's ratio vs P38's 1.413 (+-10 %: AGREES, else DIFFERS -- a finding, stated, not averaged);
  6. the P1-P7 prediction scoring (HELD / FALSIFIED / UNTESTED with the evidence).
It licenses nothing and quotes no cross-box number. stdlib only.  Usage: tp2_reduce.py <dir> [--md out.md]
"""
import argparse
import glob
import json
import os
import re
import statistics

PREREG = "tp2/P40-PREREG.md"
READ = 0.05                       # P40: held-out |delta| at N <= 0.05 nats reads COMPARABLE (a reading threshold, not a gate)
BAND = 0.05                       # tp1's rule (B2/C2): |delta final train loss| <= 0.05 AND median step-wise |delta| <= 0.05
P38_ANCHOR = 1.413                # P38's primary-pair s/step ratio (Unsloth/e4b, Qwen3-30B-A3B, RTX 5090)
ANCHOR_TOL = 0.10                 # P40: a difference beyond +-10 % is a finding about box/stack variance
P38_FIXTURE = {"lr": 1e-4, "accum": 1, "autocast": False}   # P38 as run (P38-UNSLOTH-PREREG "Optimizer", "Loss"); P40 registers lr 2e-4 / accum 4 / bf16 autocast

FAMS = ["granite", "olmoe", "gptoss", "qwen3", "gemma4", "mixtral"]
NAMES = {"granite": "Granite-3.1-3B-A800M-instruct", "olmoe": "OLMoE-1B-7B-0924-Instruct", "gptoss": "gpt-oss-20b",
         "qwen3": "Qwen3-30B-A3B", "gemma4": "Gemma-4-26B-A4B-it", "mixtral": "Mixtral-8x7B-Instruct-v0.1"}
N_LAYERS = {"granite": 32, "olmoe": 16, "qwen3": 48, "gemma4": 30, "mixtral": 32, "gptoss": 24}
PRIMARY = {"e4b": "fused_attn4", "unsloth": "ckpt_unsloth"}
EXPECTED = {fam: [("e4b", "reference_attn4"), ("e4b", "fused_attn4"), ("unsloth", "ckpt_unsloth")] for fam in FAMS}
EXPECTED["gptoss"] = [("e4b", "reference_attn4"), ("e4b", "fused_attn4"), ("e4b", "attn_only"), ("unsloth", "ckpt_unsloth")]
EXPECTED["qwen3"] = EXPECTED["qwen3"] + [("unsloth", "ckpt_hf")]
VOCAB = ("OK", "REFUSED", "OOM", "INSTALL_FAILED", "LOAD_FAULT", "HARNESS_ERROR", "ALARM", "NOT_RUN")
STATUS_MAP = {"ok": "OK", "c1_failed": "OK", "refused": "REFUSED", "oom": "OOM", "install_failed": "INSTALL_FAILED",
              "load_fault": "LOAD_FAULT", "verify_failed": "LOAD_FAULT", "alarm": "ALARM", "not_run": "NOT_RUN",
              "tokens_mismatch": "HARNESS_ERROR", "void_trainable": "HARNESS_ERROR", "void_attn4": "HARNESS_ERROR",
              "harness_error": "HARNESS_ERROR", "unreadable": "HARNESS_ERROR"}


# ----------------------------------------------------------------------------- inputs
def load(d):
    """{fam: {(fw, tag): receipt}} from <fam>_<fw>_<tag>.json (tokens_*.json and other files are skipped)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_*_*.json"))):
        base = os.path.basename(p)[:-5]
        m = re.match(r"^([a-z0-9]+)_(e4b|unsloth)_(.+)$", base)
        if not m:
            continue
        fam, fw, tag = m.groups()
        try:
            r = json.load(open(p))
        except Exception as e:
            r = {"framework": fw, "fam": fam, "tag": tag, "status": "unreadable", "reason": str(e)}
        out.setdefault(fam, {})[(fw, tag)] = r
    return out


def load_summary(d):
    """{(fam, fw, tag): [rc, ...]} from summary.txt's `fam/fw/tag rc=N` lines (every attempt)."""
    out = {}
    p = os.path.join(d, "summary.txt")
    if not os.path.exists(p):
        return out
    for ln in open(p, errors="replace"):
        m = re.match(r"^([a-z0-9]+)/(e4b|unsloth)/(\S+) rc=(\d+)", ln)
        if m:
            out.setdefault((m.group(1), m.group(2), m.group(3)), []).append(int(m.group(4)))
    return out


def f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ratio(num, den):
    try:
        return num / den if den else None
    except Exception:
        return None


# ----------------------------------------------------------------------------- 1. status, 2. validity
def status_of(r, rcs):
    """(STATUS from VOCAB, reason)."""
    if r is None:
        if rcs:
            return "HARNESS_ERROR", f"rc={rcs[-1]} and no receipt (the process died before its first write; attempts {rcs})"
        return "NOT_RUN", "no receipt and no attempt line"
    st = STATUS_MAP.get(r.get("status"), "HARNESS_ERROR")
    assert st in VOCAB
    reason = r.get("reason", "") or ""
    if r.get("status") == "c1_failed":
        reason = "C1 FAILED: frozen bytes changed -- the arm trained but is VOID"
    if r.get("cited") and r.get("probe_reason"):
        reason = f"{reason} | probe on this box: {r['probe_reason']}"
    return st, reason[:220]


def c1_ok(r):
    return bool(r.get("C1_bit_exact")) and r.get("C1_bytes_hashed", 0) > 0 and r.get("C1_empties_skipped", 1) == 0 \
        and bool(r.get("C1_control_detects_flipped_byte"))


def validity(fam, r, tokens_sha, e4b_trainable, n_steps):
    """P40 validity rules for an OK row -> (VALID|VOID, why). The n_layers table is the registered one."""
    if r is None or STATUS_MAP.get(r.get("status")) != "OK":
        return "—", ""
    L = N_LAYERS.get(fam)
    why = []
    if L is None:
        why.append(f"no registered n_layers for family {fam}")
        L = r.get("n_layers") or 0
    elif r.get("n_layers") not in (None, L):
        why.append(f"config n_layers {r.get('n_layers')} != registered {L}")
    if not c1_ok(r):
        why.append("C1 not clean")
    if len(r.get("losses", [])) != r.get("steps") or (n_steps and r.get("steps") != n_steps):
        why.append(f"step count {len(r.get('losses', []))}/{r.get('steps')} != N={n_steps or r.get('steps')}")
    if tokens_sha and r.get("tokens", {}).get("sha256") != tokens_sha:
        why.append("tokens sha differs from the family's")
    if e4b_trainable is not None and r.get("trainable_params") != e4b_trainable:
        g = r.get("trainable_by_group") or {}
        why.append(f"trainable {r.get('trainable_params')} != e4b's {e4b_trainable} (by group: {g})")
    elif r.get("trainable_mismatch"):
        tm = r["trainable_mismatch"]
        why.append(f"harness recorded a trainable mismatch: expected {tm.get('expected')} got {tm.get('got')} (by group: {tm.get('by_group')})")
    if r.get("framework") == "e4b":
        if r.get("arm") == "fused":
            if r.get("n_patched") != L:
                why.append(f"n_patched {r.get('n_patched')} != {L}")
            if r.get("kernel_calls_per_step_min", 0) < 2 * L:
                why.append(f"kernel calls/step min {r.get('kernel_calls_per_step_min')} < 2*{L}")
        if r.get("attn_4bit") and r.get("n_attn4") != 4 * L:
            why.append(f"n_attn4 {r.get('n_attn4')} != 4*{L}")
    else:
        c = r.get("census", {}) or {}
        if c.get("Params4bit_expert_stacks", 0) < 2 * L:
            why.append(f"Params4bit expert stacks {c.get('Params4bit_expert_stacks')} < 2*{L}")
        if r.get("experts_forward_calls_per_step_min", 0) < L:
            why.append(f"experts forward calls/step min {r.get('experts_forward_calls_per_step_min')} < {L}")
        b = r.get("unsloth_bnb4bit_modules")
        nb = b.get("n_bnb4bit_unwrapped") if isinstance(b, dict) else None
        if nb is None or nb < L:
            why.append(f"bnb4bit expert modules (innermost) {nb} < {L} (silent fallback?)")
        if not any("Enabling LoRA on MoE parameters" in s for s in r.get("engagement_banners", []) or []):
            why.append("no 'Enabling LoRA on MoE parameters' banner")
    return ("VOID" if why else "VALID"), "; ".join(why)


# ----------------------------------------------------------------------------- 4. e4b internal parity (tp1's rule)
def parity(ref, arm):
    """(verdict, d_final, med, why) in tp1's registered units; informational here."""
    if arm is None or STATUS_MAP.get(arm.get("status")) != "OK":
        return "NO-ARM", None, None, ""
    if ref is None or STATUS_MAP.get(ref.get("status")) != "OK":
        return "NO-REF", None, None, "reference arm missing or not OK"
    why = []
    if arm.get("init_sha") != ref.get("init_sha"):
        why.append("init_sha differs from reference (arms did not start identical)")
    for tag, r in (("ref", ref), ("arm", arm)):
        if not c1_ok(r):
            why.append(f"C1 not clean on {tag}")
    if arm.get("n_patched", 0) == 0:
        why.append("n_patched == 0")
    if arm.get("kernel_calls_per_step_min", 0) < 2 * arm.get("n_patched", 0):
        why.append(f"kernel calls/step min {arm.get('kernel_calls_per_step_min')} < 2*n_patched={2 * arm.get('n_patched', 0)}")
    if len(arm.get("losses", [])) != len(ref.get("losses", [])) or not arm.get("losses"):
        why.append("step counts differ")
    if why:
        return "VOID", None, None, "; ".join(why)
    d_final = abs(arm["loss_last"] - ref["loss_last"])
    med = statistics.median(abs(x - y) for x, y in zip(arm["losses"], ref["losses"]))
    return ("PASS" if d_final <= BAND and med <= BAND else "FAIL"), d_final, med, ""


def curve_at(r, step):
    for c in r.get("eval_curve", []) or []:
        if c["step"] == step:
            return c["heldout_loss"]
    return None


# ----------------------------------------------------------------------------- the per-family reduction (one dict; the printer and the tests read it)
def reduce_family(fam, recs, rcs_all, n_steps=None):
    exp = list(EXPECTED.get(fam, [("e4b", "reference_attn4"), ("e4b", "fused_attn4"), ("unsloth", "ckpt_unsloth")]))
    keys = exp + [k for k in recs if k not in exp]
    e_primary = recs.get(("e4b", PRIMARY["e4b"]))
    e4b_trainable = e_primary.get("trainable_params") if e_primary and STATUS_MAP.get(e_primary.get("status")) == "OK" else None
    if e4b_trainable is None:                        # gpt-oss: the e4b row is attn_only; other families: fall back to the reference arm
        for k in (("e4b", "attn_only"), ("e4b", "reference_attn4")):
            r = recs.get(k)
            if r and STATUS_MAP.get(r.get("status")) == "OK":
                e4b_trainable = r.get("trainable_params")
                break
    tokens_sha = None
    for k in keys:
        r = recs.get(k)
        if r and r.get("tokens", {}).get("sha256"):
            tokens_sha = r["tokens"]["sha256"]
            break
    N = n_steps or next((r.get("steps") for r in recs.values() if r and r.get("steps")), None)
    rows = []
    for fw, tag in keys:
        r = recs.get((fw, tag))
        st, reason = status_of(r, rcs_all.get((fam, fw, tag)))
        v, why = validity(fam, r, tokens_sha, e4b_trainable if fw == "unsloth" else None, N)
        rows.append({"fw": fw, "tag": tag, "status": st, "reason": reason, "validity": v, "why": why, "r": r})
    V = {(x["fw"], x["tag"]): x["validity"] for x in rows}
    # 4. internal parity (informational)
    ref, fu = recs.get(("e4b", "reference_attn4")), recs.get(("e4b", "fused_attn4"))
    pv, d_final, med, pwhy = parity(ref, fu)
    par = {"verdict": pv, "d_final": d_final, "median": med, "why": pwhy}
    if pv in ("PASS", "FAIL"):
        par["speed_x"] = ratio(ref.get("s_per_step_median_11plus"), fu.get("s_per_step_median_11plus"))
        par["peak_x"] = ratio(fu.get("peak_vram_gb"), ref.get("peak_vram_gb"))
    # 3. position from the primary pair
    e, u = recs.get(("e4b", PRIMARY["e4b"])), recs.get(("unsloth", PRIMARY["unsloth"]))
    pos = {"quoted": False, "why": ""}
    if fam == "gptoss":
        pos["why"] = "no primary pair by registration (e4b fused REFUSED on this family; attn_only is a secondary row; no 4-bit ratio either way)"
    elif V.get(("e4b", PRIMARY["e4b"])) == "VALID" and V.get(("unsloth", PRIMARY["unsloth"])) == "VALID":
        pos.update({"quoted": True, "ratio": ratio(u["s_per_step_median_11plus"], e["s_per_step_median_11plus"]),
                    "e4b_s": e["s_per_step_median_11plus"], "unsloth_s": u["s_per_step_median_11plus"],
                    "peak_e4b": e["peak_vram_gb"], "peak_unsloth": u["peak_vram_gb"], "peak_delta": u["peak_vram_gb"] - e["peak_vram_gb"],
                    "j_e4b": e.get("joules_per_step"), "j_unsloth": u.get("joules_per_step"), "j_ratio": ratio(u.get("joules_per_step"), e.get("joules_per_step")),
                    "tok_e4b": e.get("tokens_per_s"), "tok_unsloth": u.get("tokens_per_s")})
        le, lu = curve_at(e, e["steps"]), curve_at(u, u["steps"])
        pos["heldout_e4b"], pos["heldout_unsloth"] = le, lu
        pos["heldout_delta"] = (lu - le) if (le is not None and lu is not None) else None
        pos["quality"] = "COMPARABLE" if pos["heldout_delta"] is not None and abs(pos["heldout_delta"]) <= READ else "FLAGGED"
        pos["shared_evals"] = [(s, curve_at(e, s), curve_at(u, s)) for s in sorted({c["step"] for c in e.get("eval_curve", [])} & {c["step"] for c in u.get("eval_curve", [])})]
        pos["step0_delta"] = (u.get("eval_loss_step0") or 0) - (e.get("eval_loss_step0") or 0) if e.get("eval_loss_step0") is not None and u.get("eval_loss_step0") is not None else None
    else:
        pos["why"] = f"primary arms not both VALID: e4b {PRIMARY['e4b']} {V.get(('e4b', PRIMARY['e4b']), 'missing')} / unsloth {PRIMARY['unsloth']} {V.get(('unsloth', PRIMARY['unsloth']), 'missing')}"
    # 5. the anchor (Qwen3 only)
    anchor = None
    if fam == "qwen3" and pos.get("quoted"):
        dev = pos["ratio"] / P38_ANCHOR - 1
        fixture = {"lr": e.get("lr"), "accum": e.get("accum"), "autocast": e.get("autocast")}
        same = all(fixture.get(k) == v for k, v in P38_FIXTURE.items())
        anchor = {"ratio": pos["ratio"], "p38": P38_ANCHOR, "deviation": dev, "within": abs(dev) <= ANCHOR_TOL,
                  "verdict": "AGREES (within ±10 %)" if abs(dev) <= ANCHOR_TOL else "DIFFERS beyond ±10 % -- a finding about box/stack variance, stated, not averaged",
                  "fixture": fixture, "same_fixture_as_p38": same}
    return {"fam": fam, "rows": rows, "V": V, "parity": par, "position": pos, "anchor": anchor, "tokens_sha": tokens_sha,
            "e4b_trainable": e4b_trainable, "N": N, "e": e, "u": u, "ref": ref}


# ----------------------------------------------------------------------------- 6. predictions
def score_predictions(F):
    def st(fam, fw, tag):
        R = F.get(fam)
        if not R:
            return None
        for x in R["rows"]:
            if (x["fw"], x["tag"]) == (fw, tag):
                return x["status"]
        return None

    def rat(fam):
        R = F.get(fam)
        return R["position"].get("ratio") if R and R["position"].get("quoted") else None

    out = []
    # P1 Granite & OLMoE: both frameworks train; ratio in [0.9, 1.3]
    for fam in ("granite", "olmoe"):
        both = st(fam, "e4b", "fused_attn4") == "OK" and st(fam, "unsloth", "ckpt_unsloth") == "OK"
        r = rat(fam)
        if not F.get(fam):
            out.append(("P1", fam, "UNTESTED", "no receipts"))
        elif not both:
            out.append(("P1", fam, "FALSIFIED", f"both train: e4b {st(fam, 'e4b', 'fused_attn4')} / unsloth {st(fam, 'unsloth', 'ckpt_unsloth')}"))
        elif r is None:
            out.append(("P1", fam, "UNTESTED", "both OK but no VALID ratio: " + F[fam]["position"]["why"]))
        else:
            out.append(("P1", fam, "HELD" if 0.9 <= r <= 1.3 else "FALSIFIED", f"ratio {r:.3f} vs [0.9, 1.3]"))
    # P2 Qwen3: ratio within +-10 % of 1.413
    A = F.get("qwen3", {}).get("anchor") if F.get("qwen3") else None
    if A:
        out.append(("P2", "qwen3", "HELD" if A["within"] else "FALSIFIED", f"ratio {A['ratio']:.3f} vs {P38_ANCHOR} ({A['deviation']:+.1%}); same fixture as P38: {A['same_fixture_as_p38']}"))
    else:
        out.append(("P2", "qwen3", "UNTESTED", F["qwen3"]["position"]["why"] if F.get("qwen3") else "no receipts"))
    # P3 Gemma-4: Unsloth loads and engages its 4-bit MoE path; ratio in [1.2, 1.6]
    G = F.get("gemma4")
    if not G:
        out.append(("P3", "gemma4", "UNTESTED", "no receipts"))
    else:
        us, uv = st("gemma4", "unsloth", "ckpt_unsloth"), G["V"].get(("unsloth", "ckpt_unsloth"))
        r = rat("gemma4")
        if us != "OK":
            out.append(("P3", "gemma4", "FALSIFIED", f"Unsloth {us}: " + next(x["reason"] for x in G["rows"] if (x["fw"], x["tag"]) == ("unsloth", "ckpt_unsloth"))))
        elif uv != "VALID":
            out.append(("P3", "gemma4", "FALSIFIED", "Unsloth OK but not VALID (4-bit MoE path not engaged as registered): " + next(x["why"] for x in G["rows"] if (x["fw"], x["tag"]) == ("unsloth", "ckpt_unsloth"))))
        elif r is None:
            out.append(("P3", "gemma4", "UNTESTED", "Unsloth engaged; no ratio: " + G["position"]["why"]))
        else:
            out.append(("P3", "gemma4", "HELD" if 1.2 <= r <= 1.6 else "FALSIFIED", f"Unsloth engaged; ratio {r:.3f} vs [1.2, 1.6]"))
    # P4 Mixtral: Unsloth resident OOMs at seq 512; e4b offload trains -- no ratio
    M = F.get("mixtral")
    if not M:
        out.append(("P4", "mixtral", "UNTESTED", "no receipts"))
    else:
        us, es = st("mixtral", "unsloth", "ckpt_unsloth"), st("mixtral", "e4b", "fused_attn4")
        held = us == "OOM" and es == "OK"
        out.append(("P4", "mixtral", "HELD" if held else "FALSIFIED", f"unsloth {us} / e4b fused (offload) {es}" + (f"; ratio {rat('mixtral'):.3f} was quoted" if rat("mixtral") else "")))
    # P5 gpt-oss: e4b fused REFUSED; Unsloth refuses MXFP4 4-bit training or dequantises to bf16 (no 4-bit ratio either way)
    O = F.get("gptoss")
    if not O:
        out.append(("P5", "gptoss", "UNTESTED", "no receipts"))
    else:
        es, us = st("gptoss", "e4b", "fused_attn4"), st("gptoss", "unsloth", "ckpt_unsloth")
        ur = O["u"]
        stacks = ((ur or {}).get("census") or {}).get("Params4bit_expert_stacks", 0)
        no4 = us in ("REFUSED", "LOAD_FAULT", "OOM", "INSTALL_FAILED") or (us == "OK" and stacks < 2 * N_LAYERS["gptoss"])
        out.append(("P5", "gptoss", "HELD" if es == "REFUSED" and no4 else "FALSIFIED",
                    f"e4b fused {es}; unsloth {us}" + (f" with {stacks} Params4bit expert stacks (need 2*{N_LAYERS['gptoss']} for a 4-bit MoE path)" if us == "OK" else "")))
    # P6 quality COMPARABLE at N=60 on every family where both train
    qs = []
    for fam in FAMS:
        R = F.get(fam)
        if R and R["position"].get("quoted"):
            qs.append((fam, R["position"]["quality"], R["position"]["heldout_delta"]))
    if not qs:
        out.append(("P6", "all", "UNTESTED", "no family with both primary arms VALID"))
    else:
        bad = [q for q in qs if q[1] != "COMPARABLE"]
        out.append(("P6", "all", "HELD" if not bad else "FALSIFIED", "; ".join(f"{q[0]} {q[1]} (Δ {q[2]:+.4f})" for q in qs)))
    # P7 e4b internal parity PASS on the five fused families
    ps = []
    for fam in ("granite", "olmoe", "qwen3", "gemma4", "mixtral"):
        R = F.get(fam)
        ps.append((fam, R["parity"]["verdict"] if R else "NO-RECEIPTS"))
    if all(v == "PASS" for _, v in ps):
        out.append(("P7", "five", "HELD", "; ".join(f"{k} {v}" for k, v in ps)))
    elif any(v in ("FAIL",) for _, v in ps):
        out.append(("P7", "five", "FALSIFIED", "; ".join(f"{k} {v}" for k, v in ps)))
    else:
        out.append(("P7", "five", "UNTESTED" if not any(v == "PASS" for _, v in ps) else "PARTIAL", "; ".join(f"{k} {v}" for k, v in ps)))
    return out


# ----------------------------------------------------------------------------- the printer
def family_block(R):
    fam = R["fam"]
    lines = [f"\n### {NAMES.get(fam, fam)} (`{fam}`, registered n_layers {N_LAYERS.get(fam, '?')})"]
    src = R["e"] or R["ref"] or R["u"] or next((x["r"] for x in R["rows"] if x["r"]), None)
    if src:
        env = src.get("env", {}) or {}
        lines.append(f"- model `{src.get('model')}` @ `{str(src.get('revision', ''))[:12]}`; tokens sha `{str(R['tokens_sha'] or '')[:12]}`; N={R['N']}; "
                     f"fixture lr {src.get('lr')} accum {src.get('accum')} autocast {src.get('autocast')} seq {src.get('seq')} r {src.get('r')} α {src.get('alpha')}; "
                     f"e4b trainable {R['e4b_trainable']}; box_class {env.get('box_class')} gpu {env.get('gpu')}")
    lines.append("| framework | arm | status | validity | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | n_patched / kcalls min | n_attn4 | trainable | adapter MB (dtypes) | reason / why |")
    lines.append("|" + "---|" * 16)
    for x in R["rows"]:
        r = x["r"] or {}
        ad = r.get("adapter", {}) or {}
        b = r.get("unsloth_bnb4bit_modules")
        eng = (f"{r.get('n_patched', '—')} / {r.get('kernel_calls_per_step_min', '—')}" if x["fw"] == "e4b"
               else f"stacks {(r.get('census') or {}).get('Params4bit_expert_stacks', '—')} / fwd {r.get('experts_forward_calls_per_step_min', '—')} / u8 {b.get('n_bnb4bit_unwrapped') if isinstance(b, dict) else '—'}")
        note = " — ".join(s for s in (x["reason"], x["why"]) if s)
        lines.append(f"| {x['fw']} | {x['tag']} | **{x['status']}** | {x['validity']} | {r.get('steps', '—')} | {f(r.get('s_per_step_median_11plus'))} | {f(r.get('tokens_per_s'), 1)} | "
                     f"{f(r.get('peak_vram_gb'))} | {f(r.get('joules_per_step'), 1)} | {f(r.get('loss_first'), 4)}→{f(r.get('loss_last'), 4)} | "
                     f"{f(r.get('eval_loss_step0'), 4)}→{f(r.get('eval_loss_final'), 4)} | {eng} | {r.get('n_attn4', '—')} | {r.get('trainable_params', '—')} | "
                     f"{(ad.get('bytes') or 0) / 1e6:.1f} ({','.join(ad.get('dtypes', []) or ['?'])}) | {note} |")
    p = R["parity"]
    if p["verdict"] in ("PASS", "FAIL"):
        lines.append(f"- e4b internal parity (tp1's rule, informational): fused_attn4 vs reference_attn4 Δfinal {p['d_final']:.5f}, median step |Δ| {p['median']:.5f} → **{p['verdict']}** "
                     f"(band {BAND}/{BAND}); ×{f(p.get('speed_x'), 2)} faster per step, peak ×{f(p.get('peak_x'), 3)}")
    else:
        lines.append(f"- e4b internal parity: {p['verdict']}{(' — ' + p['why']) if p['why'] else ''}")
    pos = R["position"]
    if pos.get("quoted"):
        flag = "" if pos["quality"] == "COMPARABLE" else " **[QUALITY FLAGGED: not a clean position]**"
        lines.append(f"- **POSITION: s/step ratio Unsloth/e4b = {pos['ratio']:.3f}**{flag} ({pos['unsloth_s']:.3f} vs {pos['e4b_s']:.3f} s; {'e4b faster' if pos['ratio'] > 1 else 'Unsloth faster'} per step); "
                     f"peak VRAM unsloth {pos['peak_unsloth']:.2f} vs e4b {pos['peak_e4b']:.2f} GB (Δ {pos['peak_delta']:+.2f}); J/step unsloth {f(pos['j_unsloth'], 1)} vs e4b {f(pos['j_e4b'], 1)} (×{f(pos['j_ratio'], 3)}); "
                     f"tok/s unsloth {f(pos['tok_unsloth'], 1)} vs e4b {f(pos['tok_e4b'], 1)}")
        lines.append(f"- quality reading at N={R['N']}: held-out e4b {f(pos['heldout_e4b'], 4)} / unsloth {f(pos['heldout_unsloth'], 4)} (Δ {f(pos['heldout_delta'], 4)}) → **{pos['quality']}** (|Δ| ≤ {READ} reads COMPARABLE; a reading threshold, not a gate); "
                     + "shared evals: " + "; ".join(f"{s}: {f(a, 4)}/{f(b, 4)}" for s, a, b in pos["shared_evals"])
                     + (f"; step-0 Δ {pos['step0_delta']:+.4f} (the two quantisers on the same bytes)" if pos.get("step0_delta") is not None else ""))
    else:
        lines.append(f"- **NO POSITION QUOTED** — {pos['why']}")
    if R["anchor"]:
        A = R["anchor"]
        lines.append(f"- cross-lane anchor: ratio {A['ratio']:.3f} vs P38's {A['p38']} ({A['deviation']:+.1%}) → **{A['verdict']}**"
                     + ("" if A["same_fixture_as_p38"] else f"; NOTE the fixture differs from P38's as run (this lane {A['fixture']} vs P38 {P38_FIXTURE}) — the comparison crosses fixtures and is stated as such"))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--md", default=None)
    ap.add_argument("--steps", type=int, default=None, help="registered N (default: read from the receipts)")
    a = ap.parse_args()
    recs, rcs = load(a.dir), load_summary(a.dir)
    F = {fam: reduce_family(fam, recs[fam], rcs, a.steps) for fam in list(FAMS) + sorted(set(recs) - set(FAMS)) if fam in recs}
    out = [f"# tp2 — e4b vs Unsloth per MoE family, one box, one fixture ({os.path.abspath(a.dir)})",
           f"Rule ({PREREG}): status per attempt in the vocabulary {' / '.join(VOCAB)}; VALID/VOID per P40's validity rules with the registered n_layers "
           f"({', '.join(f'{k} {v}' for k, v in N_LAYERS.items())}); the position = s/step ratio Unsloth/e4b from the primary pair (median of steps 11..N), quoted only when both arms are VALID, "
           f"with the quality reading (held-out |Δ| at N ≤ {READ} nats → COMPARABLE, else the ratio carries the flag); e4b internal parity in tp1's B2/C2 units, informational; "
           f"Qwen3's ratio vs P38's {P38_ANCHOR} within ±{int(ANCHOR_TOL * 100)} % or a stated finding. VOID never enters a ratio. Nothing is licensed; no cross-box number is divided into these."]
    vp = os.path.join(a.dir, "versions.txt")
    if os.path.exists(vp):
        out.append("```\n" + open(vp).read().strip() + "\n```")
    for fam in list(FAMS) + sorted(set(F) - set(FAMS)):
        if fam in F:
            out += family_block(F[fam])
    out += ["\n## Cross-family summary", "| family | e4b primary | unsloth primary | position (Unsloth/e4b s/step) | quality | e4b internal parity | notes |", "|---|---|---|---|---|---|---|"]
    for fam in FAMS:
        R = F.get(fam)
        if not R:
            out.append(f"| {NAMES[fam]} | (no receipts) | | | | | |")
            continue
        ek = ("e4b", "attn_only") if fam == "gptoss" else ("e4b", PRIMARY["e4b"])
        es = next((f"{x['status']} {x['validity']}" for x in R["rows"] if (x["fw"], x["tag"]) == ek), "—")
        us = next((f"{x['status']} {x['validity']}" for x in R["rows"] if (x["fw"], x["tag"]) == ("unsloth", PRIMARY["unsloth"])), "—")
        pos = R["position"]
        pstr = f"**{pos['ratio']:.3f}**" + ("" if pos.get("quality") == "COMPARABLE" else " (flagged)") if pos.get("quoted") else "not quoted"
        others = "; ".join(f"{x['fw']}/{x['tag']}: {x['status']}" for x in R["rows"] if (x["fw"], x["tag"]) not in (ek, ("unsloth", PRIMARY["unsloth"])))
        out.append(f"| {NAMES[fam]} | {es}{' (attn_only)' if fam == 'gptoss' else ''} | {us} | {pstr} | {pos.get('quality', '—')} | {R['parity']['verdict']} | {others or '—'} |")
    out += ["\n## Predictions P1–P7 (P40, scored mechanically)", "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
    for pid, fam, v, ev in score_predictions(F):
        out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
