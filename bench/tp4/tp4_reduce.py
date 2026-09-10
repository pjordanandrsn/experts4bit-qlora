#!/usr/bin/env python3
"""tp4_reduce.py -- the per-family support / position / quality table for lane tp4 in the pre-registered form (TP4-PREREG.md).

Base: bench/h2h-20260906/tp2/tp2_reduce.py (a committed receipt, never edited), generalised to THREE frameworks and the tp4
arm set. Reads <fam>_<fw>_<tag>.json (tp4_arm.py receipts and stubs) plus, when present, summary.txt (one `fam/fw/tag rc=N`
line per attempt), versions.txt and box.json, and prints per family:

  1. the SUPPORT table -- every expected arm a row (and every unexpected receipt too), status vocabulary EXACTLY
     OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN (tp2's mapping);
  2. VALIDITY per OK row (TP4 "Validity rules", VOID never enters a ratio), accum-aware (a kernel call per micro-batch per
     layer): e4b fused `n_patched == L` and `kernel_calls_per_step_min >= 2*L*accum`, attn-4bit `n_attn4 == the structural
     census` (tp3 T10; 4*L when the receipt carries no census); Unsloth `Params4bit_expert_stacks >= 2*L`,
     `experts_forward_calls_per_step_min >= L*accum`, `n_bnb4bit_unwrapped >= L`, the "Enabling LoRA on MoE parameters"
     banner; HF `experts_forward_calls_per_step_min >= L*accum` (its regime -- how many expert stacks are 4-bit -- is
     RECORDED beside the row, never a VOID: the arm did what the field's stack does); all: C1 bit-exact, step count == N,
     same tokens sha, same trainable count as the family's e4b primary;
  3. the POSITIONS from the primary triple -- e4b fused_attn4 vs unsloth ckpt_unsloth and vs hf hf_peft -- as s/step ratios
     other/e4b (median of steps 11..N; > 1 = e4b faster), peak-VRAM delta, J/step ratio, tok/s both; quoted only when both
     arms VALID; the QUALITY reading (held-out |delta| at N <= 0.05 nats -> COMPARABLE, else FLAGGED); the secondary
     `_mb1` pair (micro-batch 1 x accum 8, run only when a primary arm OOMed) the same way, labelled secondary;
  4. e4b internal parity (fused_attn4 vs reference_attn4) in tp1's rule, informational;
  5. the cross-lane anchor: Qwen3's `_p38` pair (tp2's fixture) vs tp2's 1.457 and P38's 1.413 (+-10 %);
  6. the P1-P9 prediction scoring (HELD / FALSIFIED / UNTESTED / PARTIAL with the evidence).
It licenses nothing and quotes no cross-box number. stdlib only.  Usage: tp4_reduce.py <dir> [--md out.md] [--steps N]
"""
import argparse
import glob
import json
import os
import re
import statistics

PREREG = "tp4/TP4-PREREG.md"
READ = 0.05                       # held-out |delta| at N <= 0.05 nats reads COMPARABLE (a reading threshold, not a gate; P38/P40)
BAND = 0.05                       # tp1's rule (B2/C2): |delta final train loss| <= 0.05 AND median step-wise |delta| <= 0.05
TP2_ANCHOR, P38_ANCHOR = 1.457, 1.413   # the Qwen3 primary-pair ratios of tp2 (2026-09-06) and P38 (2026-09-05), same fixture, RTX 5090
ANCHOR_TOL = 0.10
P38_FIXTURE = {"lr": 1e-4, "accum": 1, "autocast": False, "seq": 512, "r": 8, "micro_batch": 1}

FAMS = ["granite", "olmoe", "gptoss", "qwen3", "qwen3_5", "gemma4", "mixtral", "deepseek_v4", "kimi_k3"]
NAMES = {"granite": "Granite-3.1-3B-A800M-instruct", "olmoe": "OLMoE-1B-7B-0924-Instruct", "gptoss": "gpt-oss-20b",
         "qwen3": "Qwen3-30B-A3B", "qwen3_5": "Qwen3.6-35B-A3B (qwen3_5_moe)", "gemma4": "Gemma-4-26B-A4B-it",
         "mixtral": "Mixtral-8x7B-Instruct-v0.1", "deepseek_v4": "DeepSeek-V4-Flash", "kimi_k3": "Kimi-K3"}
N_LAYERS = {"granite": 32, "olmoe": 16, "qwen3": 48, "qwen3_5": 40, "gemma4": 30, "mixtral": 32, "gptoss": 24, "deepseek_v4": 43, "kimi_k3": None}
FW = ("e4b", "unsloth", "hf")
PRIMARY = {"e4b": "fused_attn4", "unsloth": "ckpt_unsloth", "hf": "hf_peft"}
SECONDARY = {"e4b": "fused_attn4_mb1", "unsloth": "ckpt_unsloth_mb1", "hf": "hf_peft_mb1"}
ANCHOR = {"e4b": "fused_attn4_p38", "unsloth": "ckpt_unsloth_p38"}
EXPECTED = {fam: [("e4b", "fused_attn4"), ("unsloth", "ckpt_unsloth"), ("hf", "hf_peft"), ("e4b", "reference_attn4")] for fam in FAMS}
EXPECTED["gptoss"] = [("e4b", "fused_attn4"), ("e4b", "attn_only"), ("unsloth", "ckpt_unsloth"), ("hf", "hf_peft"), ("e4b", "reference_attn4")]
EXPECTED["qwen3"] = EXPECTED["qwen3"] + [("e4b", ANCHOR["e4b"]), ("unsloth", ANCHOR["unsloth"])]
VOCAB = ("OK", "REFUSED", "OOM", "INSTALL_FAILED", "LOAD_FAULT", "HARNESS_ERROR", "ALARM", "NOT_RUN")
STATUS_MAP = {"ok": "OK", "c1_failed": "OK", "refused": "REFUSED", "oom": "OOM", "install_failed": "INSTALL_FAILED",
              "load_fault": "LOAD_FAULT", "verify_failed": "LOAD_FAULT", "alarm": "ALARM", "not_run": "NOT_RUN",
              "tokens_mismatch": "HARNESS_ERROR", "void_trainable": "HARNESS_ERROR", "void_attn4": "HARNESS_ERROR",
              "harness_error": "HARNESS_ERROR", "unreadable": "HARNESS_ERROR"}


# ----------------------------------------------------------------------------- inputs
def load(d):
    """{fam: {(fw, tag): receipt}} from <fam>_<fw>_<tag>.json (tokens_*.json, box.json and other files are skipped)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_*_*.json"))):
        base = os.path.basename(p)[:-5]
        m = re.match(r"^([a-z0-9_]+?)_(e4b|unsloth|hf)_(.+)$", base)
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
        m = re.match(r"^([a-z0-9_]+)/(e4b|unsloth|hf)/(\S+) rc=(\d+)", ln)
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
    if r.get("loader_fallback_reason"):
        reason = (reason + " | " if reason else "") + f"loader fallback: {r.get('loader_used')}"
    return st, reason[:260]


def c1_ok(r):
    return bool(r.get("C1_bit_exact")) and r.get("C1_bytes_hashed", 0) > 0 and r.get("C1_empties_skipped", 1) == 0 \
        and bool(r.get("C1_control_detects_flipped_byte"))


def regime_of(fam, r):
    """What is 4-bit in this arm, from the census: the recorded regime label (never a VOID)."""
    L = N_LAYERS.get(fam) or r.get("n_layers") or 0
    c = r.get("census", {}) or {}
    if r.get("framework") == "e4b":
        return "4-bit experts (e4b NF4) + " + ("NF4 attention" if r.get("attn_4bit") else "bf16 attention")
    stacks = c.get("Params4bit_expert_stacks", 0) or 0
    attn4 = c.get("Linear4bit", 0) or 0
    exp = "4-bit expert stacks" if L and stacks >= 2 * L else ("PARTIAL 4-bit experts" if stacks else "bf16 experts (NOT the 4-bit MoE regime)")
    return f"{exp} + {'bnb-4bit' if attn4 else 'bf16'} attention (Params4bit stacks {stacks}, Linear4bit {attn4})"


def validity(fam, r, tokens_sha, e4b_trainable, n_steps):
    """TP4 validity rules for an OK row -> (VALID|VOID, why). The n_layers table is the registered one; accum-aware."""
    if r is None or STATUS_MAP.get(r.get("status")) != "OK":
        return "—", ""
    L = N_LAYERS.get(fam)
    A = int(r.get("accum") or 1)
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
    fw = r.get("framework")
    if fw == "e4b":
        if r.get("arm") == "fused":
            if r.get("n_patched") != L:
                why.append(f"n_patched {r.get('n_patched')} != {L}")
            if r.get("kernel_calls_per_step_min", 0) < 2 * L * A:
                why.append(f"kernel calls/step min {r.get('kernel_calls_per_step_min')} < 2*{L}*accum {A}")
        if r.get("attn_4bit"):
            want = r.get("structural_expected_n_attn4")
            if want is None:
                want = 4 * L
            if r.get("n_attn4") != want:
                why.append(f"n_attn4 {r.get('n_attn4')} != {want} ({'structural census' if r.get('structural_expected_n_attn4') is not None else '4*L'})")
    elif fw == "unsloth":
        c = r.get("census", {}) or {}
        if c.get("Params4bit_expert_stacks", 0) < 2 * L:
            why.append(f"Params4bit expert stacks {c.get('Params4bit_expert_stacks')} < 2*{L}")
        if r.get("experts_forward_calls_per_step_min", 0) < L * A:
            why.append(f"experts forward calls/step min {r.get('experts_forward_calls_per_step_min')} < {L}*accum {A}")
        b = r.get("unsloth_bnb4bit_modules")
        nb = b.get("n_bnb4bit_unwrapped") if isinstance(b, dict) else None
        if nb is None or nb < L:
            why.append(f"bnb4bit expert modules (innermost) {nb} < {L} (silent fallback?)")
        if not any("Enabling LoRA on MoE parameters" in s for s in r.get("engagement_banners", []) or []):
            why.append("no 'Enabling LoRA on MoE parameters' banner")
    elif fw == "hf":
        if r.get("experts_forward_calls_per_step_min", 0) < L * A:
            why.append(f"experts forward calls/step min {r.get('experts_forward_calls_per_step_min')} < {L}*accum {A}")
        if not (r.get("hf_targets") or {}).get("n_target_parameters"):
            why.append("PEFT adapted no expert parameter (target_parameters empty) -- attention-only, not the registered adapter set")
    return ("VOID" if why else "VALID"), "; ".join(why)


# ----------------------------------------------------------------------------- 4. e4b internal parity (tp1's rule)
def parity(ref, arm):
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
    A = int(arm.get("accum") or 1)
    if arm.get("kernel_calls_per_step_min", 0) < 2 * arm.get("n_patched", 0) * A:
        why.append(f"kernel calls/step min {arm.get('kernel_calls_per_step_min')} < 2*n_patched*accum={2 * arm.get('n_patched', 0) * A}")
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


def position(e, o, ve, vo, label):
    """The pair (e4b arm e, other arm o): quoted only when both VALID; the quality reading rides along."""
    pos = {"label": label, "quoted": False, "why": ""}
    if ve == "VALID" and vo == "VALID":
        pos.update({"quoted": True, "ratio": ratio(o["s_per_step_median_11plus"], e["s_per_step_median_11plus"]),
                    "e4b_s": e["s_per_step_median_11plus"], "other_s": o["s_per_step_median_11plus"],
                    "peak_e4b": e["peak_vram_gb"], "peak_other": o["peak_vram_gb"], "peak_delta": o["peak_vram_gb"] - e["peak_vram_gb"],
                    "j_e4b": e.get("joules_per_step"), "j_other": o.get("joules_per_step"), "j_ratio": ratio(o.get("joules_per_step"), e.get("joules_per_step")),
                    "tok_e4b": e.get("tokens_per_s"), "tok_other": o.get("tokens_per_s"), "other_regime": None})
        le, lo = curve_at(e, e["steps"]), curve_at(o, o["steps"])
        pos["heldout_e4b"], pos["heldout_other"] = le, lo
        pos["heldout_delta"] = (lo - le) if (le is not None and lo is not None) else None
        pos["quality"] = "COMPARABLE" if pos["heldout_delta"] is not None and abs(pos["heldout_delta"]) <= READ else "FLAGGED"
        pos["shared_evals"] = [(s, curve_at(e, s), curve_at(o, s)) for s in sorted({c["step"] for c in e.get("eval_curve", [])} & {c["step"] for c in o.get("eval_curve", [])})]
        pos["step0_delta"] = (o.get("eval_loss_step0") or 0) - (e.get("eval_loss_step0") or 0) if e.get("eval_loss_step0") is not None and o.get("eval_loss_step0") is not None else None
    else:
        pos["why"] = f"arms not both VALID: e4b {ve or 'missing'} / {label} {vo or 'missing'}"
    return pos


# ----------------------------------------------------------------------------- the per-family reduction (one dict; the printer and the tests read it)
def reduce_family(fam, recs, rcs_all, n_steps=None):
    exp = list(EXPECTED.get(fam, [("e4b", "fused_attn4"), ("unsloth", "ckpt_unsloth"), ("hf", "hf_peft"), ("e4b", "reference_attn4")]))
    keys = exp + [k for k in recs if k not in exp]
    e_primary = recs.get(("e4b", PRIMARY["e4b"]))
    e4b_trainable = e_primary.get("trainable_params") if e_primary and STATUS_MAP.get(e_primary.get("status")) == "OK" else None
    if e4b_trainable is None:                        # gpt-oss: the e4b row is attn_only; other families: fall back to the reference arm
        for k in (("e4b", "attn_only"), ("e4b", "reference_attn4")):
            r = recs.get(k)
            if r and STATUS_MAP.get(r.get("status")) == "OK":
                e4b_trainable = r.get("trainable_params")
                break
    # the secondary and anchor pairs have their OWN e4b arm as the trainable reference (different r / recipe)
    def trainable_ref(tag):
        if tag.endswith("_mb1"):
            r = recs.get(("e4b", SECONDARY["e4b"]))
        elif tag.endswith("_p38"):
            r = recs.get(("e4b", ANCHOR["e4b"]))
        else:
            return e4b_trainable
        return r.get("trainable_params") if r and STATUS_MAP.get(r.get("status")) == "OK" else None
    tokens_by_tag = {}
    for k in keys:
        r = recs.get(k)
        if r and r.get("tokens", {}).get("sha256"):
            grp = "p38" if k[1].endswith("_p38") else "main"
            tokens_by_tag.setdefault(grp, r["tokens"]["sha256"])
    N = n_steps or next((r.get("steps") for r in recs.values() if r and r.get("steps")), None)
    rows = []
    for fw, tag in keys:
        r = recs.get((fw, tag))
        st, reason = status_of(r, rcs_all.get((fam, fw, tag)))
        grp = "p38" if tag.endswith("_p38") else "main"
        v, why = validity(fam, r, tokens_by_tag.get(grp), trainable_ref(tag) if fw != "e4b" else None, N)
        rows.append({"fw": fw, "tag": tag, "status": st, "reason": reason, "validity": v, "why": why, "r": r,
                     "regime": regime_of(fam, r) if (r and st == "OK") else None})
    V = {(x["fw"], x["tag"]): x["validity"] for x in rows}
    ref, fu = recs.get(("e4b", "reference_attn4")), recs.get(("e4b", "fused_attn4"))
    pv, d_final, med, pwhy = parity(ref, fu)
    par = {"verdict": pv, "d_final": d_final, "median": med, "why": pwhy}
    if pv in ("PASS", "FAIL"):
        par["speed_x"] = ratio(ref.get("s_per_step_median_11plus"), fu.get("s_per_step_median_11plus"))
        par["peak_x"] = ratio(fu.get("peak_vram_gb"), ref.get("peak_vram_gb"))
    e = recs.get(("e4b", PRIMARY["e4b"]))
    positions = {}
    for other in ("unsloth", "hf"):
        o = recs.get((other, PRIMARY[other]))
        if fam == "gptoss":
            positions[other] = {"label": other, "quoted": False,
                                "why": "no primary pair by registration (e4b fused REFUSED on this family; attn_only is a secondary row; no 4-bit ratio either way)"}
        else:
            positions[other] = position(e, o, V.get(("e4b", PRIMARY["e4b"])), V.get((other, PRIMARY[other])), other) if (e or o) else \
                {"label": other, "quoted": False, "why": "no receipts for the pair"}
        if positions[other].get("quoted"):
            positions[other]["other_regime"] = regime_of(fam, o)
    secondary = {}
    e2 = recs.get(("e4b", SECONDARY["e4b"]))
    for other in ("unsloth", "hf"):
        o2 = recs.get((other, SECONDARY[other]))
        if e2 or o2:
            secondary[other] = position(e2, o2, V.get(("e4b", SECONDARY["e4b"])), V.get((other, SECONDARY[other])), other + " (mb1)")
    anchor = None
    ea, ua = recs.get(("e4b", ANCHOR["e4b"])), recs.get(("unsloth", ANCHOR["unsloth"]))
    if fam == "qwen3" and (ea or ua):
        ap = position(ea, ua, V.get(("e4b", ANCHOR["e4b"])), V.get(("unsloth", ANCHOR["unsloth"])), "unsloth (p38 fixture)")
        anchor = {"position": ap}
        if ap.get("quoted"):
            fixture = {k: ea.get(k) for k in P38_FIXTURE}
            same = all(fixture.get(k) == v for k, v in P38_FIXTURE.items())
            dev2, dev8 = ap["ratio"] / TP2_ANCHOR - 1, ap["ratio"] / P38_ANCHOR - 1
            anchor.update({"ratio": ap["ratio"], "tp2": TP2_ANCHOR, "p38": P38_ANCHOR, "dev_tp2": dev2, "dev_p38": dev8,
                           "within_tp2": abs(dev2) <= ANCHOR_TOL, "within_p38": abs(dev8) <= ANCHOR_TOL,
                           "fixture": fixture, "same_fixture": same,
                           "verdict": ("AGREES with tp2 (within ±10 %)" if abs(dev2) <= ANCHOR_TOL else "DIFFERS from tp2 beyond ±10 % -- a finding about box/stack variance, stated, not averaged")})
    return {"fam": fam, "rows": rows, "V": V, "parity": par, "positions": positions, "secondary": secondary, "anchor": anchor,
            "tokens_sha": tokens_by_tag.get("main"), "e4b_trainable": e4b_trainable, "N": N, "e": e, "ref": ref,
            "u": recs.get(("unsloth", PRIMARY["unsloth"])), "h": recs.get(("hf", PRIMARY["hf"]))}


# ----------------------------------------------------------------------------- 6. predictions (TP4-PREREG.md, scored mechanically)
def score_predictions(F):
    def st(fam, fw, tag):
        R = F.get(fam)
        if not R:
            return None
        for x in R["rows"]:
            if (x["fw"], x["tag"]) == (fw, tag):
                return x["status"]
        return None

    def rat(fam, other="unsloth"):
        R = F.get(fam)
        p = R["positions"].get(other) if R else None
        return p.get("ratio") if p and p.get("quoted") else None

    def why(fam, other):
        R = F.get(fam)
        return R["positions"][other]["why"] if R and other in R["positions"] else "no receipts"

    out = []
    # P1 Granite & OLMoE: all three frameworks train at the field recipe; Unsloth/e4b in [0.9, 1.4]; HF trains with bf16 experts
    for fam in ("granite", "olmoe"):
        if not F.get(fam):
            out.append(("P1", fam, "UNTESTED", "no receipts"))
            continue
        s = {fw: st(fam, fw, PRIMARY[fw]) for fw in FW}
        r = rat(fam)
        hf_row = next((x for x in F[fam]["rows"] if (x["fw"], x["tag"]) == ("hf", PRIMARY["hf"])), None)
        hf_bf16 = bool(hf_row and hf_row["regime"] and "bf16 experts" in hf_row["regime"])
        if any(v != "OK" for v in s.values()):
            out.append(("P1", fam, "FALSIFIED", "all three train: " + ", ".join(f"{k} {v}" for k, v in s.items())))
        elif r is None:
            out.append(("P1", fam, "UNTESTED", "all OK but no VALID unsloth/e4b ratio: " + why(fam, "unsloth")))
        else:
            v = "HELD" if (0.9 <= r <= 1.4 and hf_bf16) else "FALSIFIED"
            out.append(("P1", fam, v, f"unsloth/e4b {r:.3f} vs [0.9, 1.4]; hf regime {hf_row['regime'] if hf_row else '?'}; hf/e4b {f(rat(fam, 'hf'))}"))
    # P2 Qwen3 (field recipe): e4b fused + Unsloth OK, HF OOM; unsloth/e4b in [1.1, 1.8]
    Q = F.get("qwen3")
    if not Q:
        out.append(("P2", "qwen3", "UNTESTED", "no receipts"))
    else:
        s = {fw: st("qwen3", fw, PRIMARY[fw]) for fw in FW}
        r = rat("qwen3")
        if s["e4b"] == "OK" and s["unsloth"] == "OK" and s["hf"] == "OOM" and r is not None:
            out.append(("P2", "qwen3", "HELD" if 1.1 <= r <= 1.8 else "FALSIFIED", f"e4b OK, unsloth OK, hf OOM; unsloth/e4b {r:.3f} vs [1.1, 1.8]"))
        elif s["e4b"] == "OK" and s["unsloth"] == "OK" and r is None:
            out.append(("P2", "qwen3", "UNTESTED", "both OK but no VALID ratio: " + why("qwen3", "unsloth") + f"; hf {s['hf']}"))
        else:
            out.append(("P2", "qwen3", "FALSIFIED", ", ".join(f"{k} {v}" for k, v in s.items()) + (f"; ratio {r:.3f}" if r else "")))
    # P3 Qwen3 anchor pair (tp2's fixture): within +-10 % of tp2's 1.457
    A = Q.get("anchor") if Q else None
    if A and A.get("ratio") is not None:
        out.append(("P3", "qwen3", "HELD" if A["within_tp2"] else "FALSIFIED",
                    f"p38-fixture ratio {A['ratio']:.3f} vs tp2 {TP2_ANCHOR} ({A['dev_tp2']:+.1%}) / P38 {P38_ANCHOR} ({A['dev_p38']:+.1%}); same fixture: {A['same_fixture']}"))
    else:
        out.append(("P3", "qwen3", "UNTESTED", (A["position"]["why"] if A else "no anchor-pair receipts")))
    # P4 Qwen3.6 (qwen3_5): e4b and Unsloth both train and engage; trainable counts match; ratio in [1.0, 1.8]; HF OOM
    R5 = F.get("qwen3_5")
    if not R5:
        out.append(("P4", "qwen3_5", "UNTESTED", "no receipts"))
    else:
        s = {fw: st("qwen3_5", fw, PRIMARY[fw]) for fw in FW}
        r = rat("qwen3_5")
        if s["e4b"] == "OK" and s["unsloth"] == "OK" and r is not None:
            out.append(("P4", "qwen3_5", "HELD" if (1.0 <= r <= 1.8 and s["hf"] == "OOM") else "FALSIFIED", f"unsloth/e4b {r:.3f} vs [1.0, 1.8]; hf {s['hf']} (predicted OOM)"))
        elif s["e4b"] == "OK" and s["unsloth"] == "OK":
            out.append(("P4", "qwen3_5", "UNTESTED", "both OK but no VALID ratio: " + why("qwen3_5", "unsloth")))
        else:
            out.append(("P4", "qwen3_5", "FALSIFIED", ", ".join(f"{k} {v}" for k, v in s.items())))
    # P5 Gemma-4: e4b fused trains (post-#435 census), Unsloth OK, ratio in [1.0, 1.6], HF OOM
    G = F.get("gemma4")
    if not G:
        out.append(("P5", "gemma4", "UNTESTED", "no receipts"))
    else:
        s = {fw: st("gemma4", fw, PRIMARY[fw]) for fw in FW}
        r = rat("gemma4")
        if s["e4b"] == "OK" and s["unsloth"] == "OK" and r is not None:
            out.append(("P5", "gemma4", "HELD" if (1.0 <= r <= 1.6 and s["hf"] == "OOM") else "FALSIFIED", f"unsloth/e4b {r:.3f} vs [1.0, 1.6]; hf {s['hf']} (predicted OOM)"))
        elif s["e4b"] == "OK" and s["unsloth"] == "OK":
            out.append(("P5", "gemma4", "UNTESTED", "both OK but no VALID ratio: " + why("gemma4", "unsloth")))
        else:
            out.append(("P5", "gemma4", "FALSIFIED", ", ".join(f"{k} {v}" for k, v in s.items())))
    # P6 Mixtral: Unsloth resident OOMs at the field recipe AND at mb1; e4b (offload) trains both; HF OOM -> no ratio
    M = F.get("mixtral")
    if not M:
        out.append(("P6", "mixtral", "UNTESTED", "no receipts"))
    else:
        us, es, hs = st("mixtral", "unsloth", PRIMARY["unsloth"]), st("mixtral", "e4b", PRIMARY["e4b"]), st("mixtral", "hf", PRIMARY["hf"])
        us2 = st("mixtral", "unsloth", SECONDARY["unsloth"])
        held = us == "OOM" and es == "OK" and hs == "OOM" and us2 in ("OOM", None)
        out.append(("P6", "mixtral", "HELD" if held else "FALSIFIED",
                    f"unsloth {us} (mb1: {us2}) / e4b fused offload {es} / hf {hs}" + (f"; ratio {rat('mixtral'):.3f} was quoted" if rat("mixtral") else "")))
    # P7 gpt-oss: e4b fused REFUSED (bare experts), attn_only OK; Unsloth refuses or has no 4-bit MoE path; HF refuses or OOMs
    GO = F.get("gptoss")
    if not GO:
        out.append(("P7", "gptoss", "UNTESTED", "no receipts"))
    else:
        es, ao, us, hs = st("gptoss", "e4b", "fused_attn4"), st("gptoss", "e4b", "attn_only"), st("gptoss", "unsloth", PRIMARY["unsloth"]), st("gptoss", "hf", PRIMARY["hf"])
        ur = GO["u"]
        stacks = ((ur or {}).get("census") or {}).get("Params4bit_expert_stacks", 0) if ur else 0
        no4 = us in ("REFUSED", "LOAD_FAULT", "OOM", "INSTALL_FAILED") or (us == "OK" and stacks < 2 * N_LAYERS["gptoss"])
        hf_ok = hs in ("REFUSED", "OOM", "LOAD_FAULT")
        out.append(("P7", "gptoss", "HELD" if (es == "REFUSED" and ao == "OK" and no4 and hf_ok) else "FALSIFIED",
                    f"e4b fused {es}, attn_only {ao}; unsloth {us}" + (f" with {stacks} Params4bit expert stacks" if us == "OK" else "") + f"; hf {hs}"))
    # P8 quality COMPARABLE on every quoted pair
    qs = []
    for fam in FAMS:
        R = F.get(fam)
        if not R:
            continue
        for other, p in R["positions"].items():
            if p.get("quoted"):
                qs.append((fam, other, p["quality"], p["heldout_delta"]))
    if not qs:
        out.append(("P8", "all", "UNTESTED", "no pair with both arms VALID"))
    else:
        bad = [q for q in qs if q[2] != "COMPARABLE"]
        out.append(("P8", "all", "HELD" if not bad else "FALSIFIED", "; ".join(f"{q[0]}/{q[1]} {q[2]} (Δ {q[3]:+.4f})" for q in qs)))
    # P9 e4b internal parity PASS wherever both e4b arms ran
    ps = [(fam, F[fam]["parity"]["verdict"]) for fam in FAMS if F.get(fam) and F[fam]["parity"]["verdict"] not in ("NO-ARM",)]
    if not ps:
        out.append(("P9", "all", "UNTESTED", "no family with both e4b arms"))
    elif all(v == "PASS" for _, v in ps):
        out.append(("P9", "all", "HELD", "; ".join(f"{k} {v}" for k, v in ps)))
    elif any(v == "FAIL" for _, v in ps):
        out.append(("P9", "all", "FALSIFIED", "; ".join(f"{k} {v}" for k, v in ps)))
    else:
        out.append(("P9", "all", "PARTIAL", "; ".join(f"{k} {v}" for k, v in ps)))
    return out


# ----------------------------------------------------------------------------- the printer
def pos_lines(pos, N, prefix="POSITION"):
    if not pos.get("quoted"):
        return [f"- **NO {prefix} QUOTED ({pos['label']})** — {pos['why']}"]
    flag = "" if pos["quality"] == "COMPARABLE" else " **[QUALITY FLAGGED: not a clean position]**"
    who = pos["label"]
    lines = [f"- **{prefix}: s/step ratio {who}/e4b = {pos['ratio']:.3f}**{flag} ({pos['other_s']:.3f} vs {pos['e4b_s']:.3f} s; {'e4b faster' if pos['ratio'] > 1 else who + ' faster'} per step); "
             f"peak VRAM {who} {pos['peak_other']:.2f} vs e4b {pos['peak_e4b']:.2f} GB (Δ {pos['peak_delta']:+.2f}); J/step {who} {f(pos['j_other'], 1)} vs e4b {f(pos['j_e4b'], 1)} (×{f(pos['j_ratio'], 3)}); "
             f"tok/s {who} {f(pos['tok_other'], 1)} vs e4b {f(pos['tok_e4b'], 1)}" + (f"; {who} regime: {pos['other_regime']}" if pos.get("other_regime") else "")]
    lines.append(f"- quality reading at N={N} ({who}): held-out e4b {f(pos['heldout_e4b'], 4)} / {who} {f(pos['heldout_other'], 4)} (Δ {f(pos['heldout_delta'], 4)}) → **{pos['quality']}** "
                 f"(|Δ| ≤ {READ} reads COMPARABLE; a reading threshold, not a gate); shared evals: " + "; ".join(f"{s}: {f(a, 4)}/{f(b, 4)}" for s, a, b in pos["shared_evals"])
                 + (f"; step-0 Δ {pos['step0_delta']:+.4f} (the two quantisers on the same bytes)" if pos.get("step0_delta") is not None else ""))
    return lines


def family_block(R):
    fam = R["fam"]
    lines = [f"\n### {NAMES.get(fam, fam)} (`{fam}`, registered n_layers {N_LAYERS.get(fam, '?')})"]
    src = R["e"] or R["ref"] or R["u"] or R["h"] or next((x["r"] for x in R["rows"] if x["r"]), None)
    if src:
        env = src.get("env", {}) or {}
        lines.append(f"- model `{src.get('model')}` @ `{str(src.get('revision', ''))[:12]}`; tokens sha `{str(R['tokens_sha'] or '')[:12]}`; N={R['N']}; "
                     f"fixture template {src.get('template')} seq {src.get('seq')} micro-batch {src.get('micro_batch')} × accum {src.get('accum')} lr {src.get('lr')} r {src.get('r')} α {src.get('alpha')} "
                     f"optimizer {src.get('optimizer')} autocast {src.get('autocast')}; e4b trainable {R['e4b_trainable']}; box_class {env.get('box_class')} gpu {env.get('gpu')}")
    lines.append("| framework | arm | status | validity | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | engagement | n_attn4 | trainable | adapter MB (dtypes) | regime | reason / why |")
    lines.append("|" + "---|" * 17)
    for x in R["rows"]:
        r = x["r"] or {}
        ad = r.get("adapter", {}) or {}
        b = r.get("unsloth_bnb4bit_modules")
        if x["fw"] == "e4b":
            eng = f"patched {r.get('n_patched', '—')} / kcalls {r.get('kernel_calls_per_step_min', '—')}"
        elif x["fw"] == "unsloth":
            eng = f"stacks {(r.get('census') or {}).get('Params4bit_expert_stacks', '—')} / fwd {r.get('experts_forward_calls_per_step_min', '—')} / u8 {b.get('n_bnb4bit_unwrapped') if isinstance(b, dict) else '—'}"
        else:
            ht = r.get("hf_targets") or {}
            eng = f"peft mods {ht.get('n_target_modules', '—')} / params {ht.get('n_target_parameters', '—')} / fwd {r.get('experts_forward_calls_per_step_min', '—')}"
        note = " — ".join(s for s in (x["reason"], x["why"]) if s)
        lines.append(f"| {x['fw']} | {x['tag']} | **{x['status']}** | {x['validity']} | {r.get('steps', '—')} | {f(r.get('s_per_step_median_11plus'))} | {f(r.get('tokens_per_s'), 1)} | "
                     f"{f(r.get('peak_vram_gb'))} | {f(r.get('joules_per_step'), 1)} | {f(r.get('loss_first'), 4)}→{f(r.get('loss_last'), 4)} | "
                     f"{f(r.get('eval_loss_step0'), 4)}→{f(r.get('eval_loss_final'), 4)} | {eng} | {r.get('n_attn4', '—')} | {r.get('trainable_params', '—')} | "
                     f"{(ad.get('bytes') or 0) / 1e6:.1f} ({','.join(ad.get('dtypes', []) or ['?'])}) | {x['regime'] or '—'} | {note} |")
    p = R["parity"]
    if p["verdict"] in ("PASS", "FAIL"):
        lines.append(f"- e4b internal parity (tp1's rule, informational): fused_attn4 vs reference_attn4 Δfinal {p['d_final']:.5f}, median step |Δ| {p['median']:.5f} → **{p['verdict']}** "
                     f"(band {BAND}/{BAND}); ×{f(p.get('speed_x'), 2)} faster per step, peak ×{f(p.get('peak_x'), 3)}")
    else:
        lines.append(f"- e4b internal parity: {p['verdict']}{(' — ' + p['why']) if p['why'] else ''}")
    for other in ("unsloth", "hf"):
        if other in R["positions"]:
            lines += pos_lines(R["positions"][other], R["N"])
    for other, sp in R["secondary"].items():
        lines += pos_lines(sp, R["N"], prefix="SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed)")
    if R["anchor"]:
        A = R["anchor"]
        lines += pos_lines(A["position"], R["N"], prefix="ANCHOR PAIR (tp2/P38 fixture)")
        if A.get("ratio") is not None:
            lines.append(f"- cross-lane anchor: ratio {A['ratio']:.3f} vs tp2's {A['tp2']} ({A['dev_tp2']:+.1%}) and P38's {A['p38']} ({A['dev_p38']:+.1%}) → **{A['verdict']}**"
                         + ("" if A["same_fixture"] else f"; NOTE the fixture differs from P38's as run ({A['fixture']} vs {P38_FIXTURE}) — the comparison crosses fixtures and is stated as such"))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--md", default=None)
    ap.add_argument("--steps", type=int, default=None, help="registered N (default: read from the receipts)")
    a = ap.parse_args()
    recs, rcs = load(a.dir), load_summary(a.dir)
    F = {fam: reduce_family(fam, recs[fam], rcs, a.steps) for fam in list(FAMS) + sorted(set(recs) - set(FAMS)) if fam in recs}
    out = [f"# tp4 — e4b (GitHub main) vs Unsloth vs plain HF+PEFT+bnb per MoE family, same box, one fixture ({os.path.abspath(a.dir)})",
           f"Rule ({PREREG}): status per attempt in the vocabulary {' / '.join(VOCAB)}; VALID/VOID per TP4's validity rules with the registered n_layers "
           f"({', '.join(f'{k} {v}' for k, v in N_LAYERS.items())}); positions = s/step ratios other/e4b from the primary triple (median of steps 11..N; > 1 = e4b faster), "
           f"quoted only when both arms are VALID, with the quality reading (held-out |Δ| at N ≤ {READ} nats → COMPARABLE, else the ratio carries the flag) and the other arm's "
           f"4-bit REGIME beside it; e4b internal parity in tp1's B2/C2 units, informational; Qwen3's tp2-fixture anchor pair vs tp2's {TP2_ANCHOR} / P38's {P38_ANCHOR} within "
           f"±{int(ANCHOR_TOL * 100)} % or a stated finding. VOID never enters a ratio. Nothing is licensed; no cross-box number is divided into these."]
    for fn in ("versions.txt", "box.json"):
        vp = os.path.join(a.dir, fn)
        if os.path.exists(vp):
            out.append(f"`{fn}`\n```\n" + open(vp).read().strip() + "\n```")
    for fam in list(FAMS) + sorted(set(F) - set(FAMS)):
        if fam in F:
            out += family_block(F[fam])
    out += ["\n## Cross-family summary", "| family | e4b primary | unsloth primary | hf primary | unsloth/e4b s/step | hf/e4b s/step | quality | e4b internal parity | notes |", "|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMS:
        R = F.get(fam)
        if not R:
            out.append(f"| {NAMES[fam]} | (no receipts) | | | | | | | |")
            continue
        ek = ("e4b", "attn_only") if fam == "gptoss" else ("e4b", PRIMARY["e4b"])

        def cell(k, R=R):
            return next((f"{x['status']} {x['validity']}" for x in R["rows"] if (x["fw"], x["tag"]) == k), "—")

        def pstr(p):
            return (f"**{p['ratio']:.3f}**" + ("" if p.get("quality") == "COMPARABLE" else " (flagged)")) if p.get("quoted") else "not quoted"
        pu, ph = R["positions"].get("unsloth", {}), R["positions"].get("hf", {})
        q = "; ".join(f"{k} {p['quality']}" for k, p in R["positions"].items() if p.get("quoted")) or "—"
        others = "; ".join(f"{x['fw']}/{x['tag']}: {x['status']}" for x in R["rows"] if (x["fw"], x["tag"]) not in (ek, ("unsloth", PRIMARY["unsloth"]), ("hf", PRIMARY["hf"])))
        out.append(f"| {NAMES[fam]} | {cell(ek)}{' (attn_only)' if fam == 'gptoss' else ''} | {cell(('unsloth', PRIMARY['unsloth']))} | {cell(('hf', PRIMARY['hf']))} | {pstr(pu)} | {pstr(ph)} | {q} | {R['parity']['verdict']} | {others or '—'} |")
    out += ["\n## Predictions P1–P9 (TP4-PREREG.md, scored mechanically)", "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
    for pid, fam, v, ev in score_predictions(F):
        out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
