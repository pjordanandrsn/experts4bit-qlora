#!/usr/bin/env python3
"""tc1_reduce.py -- the per-family support / validity / VERDICT / draws / position / equivalence / frozen-base / prediction
table for lane TC1 in the pre-registered form (TC1-PREREG.md).

Base: bench/tp4/tp4_reduce.py @ e4b main 10ce711d (a committed lane's reducer, never edited): its inputs (<fam>_<fw>_<tag>.json,
summary.txt, versions.txt, box.json), its status vocabulary, its validity predicates, its parity rule and its printer shape are
kept. What TC1 adds, named so the files can be diffed:

  R1 DRAWS: an arm registered with a second draw (`_d2`, a fresh process, same everything) is read as ONE arm with two
     draws: every quoted metric is the median over both draws; STABILITY = |d1 - d2| / mean <= 0.05 for e4b and <= 0.10 for
     every other framework, else the arm is UNSTABLE (reported, never quoted). A missing or non-VALID second draw leaves the
     stability UNMEASURED and the position unquoted (a single draw cannot show the agreement the registration asks for).
  R2 VERDICT per attempt, exactly one of VALID / VOID / QUALITY_FAIL / OOM / UNSUPPORTED / HARNESS_ERROR / ALARM / NOT_RUN:
     OK + validity VALID + quality COMPARABLE (held-out |delta| at N <= 0.05 nats against e4b/fused_attn4_m) -> VALID;
     OK + a validity predicate failed -> VOID; OK + VALID + |delta| > 0.05 -> QUALITY_FAIL; OOM -> OOM;
     REFUSED / INSTALL_FAILED / LOAD_FAULT -> UNSUPPORTED; HARNESS_ERROR / ALARM / NOT_RUN as named.
  R3 MATCHED-SET validity (on top of tp4's predicates): a registered matched arm must have run `--lora-init matched:<seed>`
     with the anchor's seed, `matched_init.complete` true, `adapter_dtypes_after` == {fp32}, and a step-0 held-out loss
     within 0.005 nats of e4b/reference_attn4_m (the same base and the same init by construction) -- else VOID.
  R4 EQUIVALENCE for each matched arm against e4b/reference_attn4_m (both VALID by predicates): median per-step
     |delta train loss| <= 0.02 AND |delta held-out at N| <= 0.02 reads EQUIVALENT; both <= 0.05 reads COMPARABLE; else
     DIVERGENT (a finding that stops the position on that pair).
  R5 FROZEN BASE across arms: per registered slot (gate_up, down, q_proj) each arm's sha against e4b/fused_attn4_m's:
     SAME-BYTES / DIFFERENT when the regimes are equal, N-A when they differ or a slot is missing; each arm's own byte-flip
     control shown.
  R6 PREDICTIONS P1-P10 (TC1-PREREG "Predictions") scored HELD / FALSIFIED / UNTESTED, mechanically.
  R7 `--selftest`: hand-built receipts (a complete OK receipt per arm with every field this file reads), >= 16 cases,
     including one receipt per validity predicate that MUST read VOID, one QUALITY_FAIL, one UNSTABLE, one DIVERGENT.

It licenses nothing and quotes no cross-box number. stdlib only.  Usage: tc1_reduce.py <dir> [--md out.md] [--steps N] | --selftest
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

PREREG = "tc1/TC1-PREREG.md"
READ = 0.05                       # quality: held-out |delta| at N vs e4b/fused_attn4_m <= 0.05 nats reads COMPARABLE, else QUALITY_FAIL
BAND = 0.05                       # tp1's parity rule (B2/C2): |delta final train loss| <= 0.05 AND median step-wise |delta| <= 0.05
EQUIV, COMPARABLE = 0.02, 0.05    # R4: equivalence bands on BOTH the median per-step |delta train| and |delta held-out at N|
STEP0_TOL = 0.005                 # R3: the matched set agrees on step-0 held-out loss to 0.005 nats, by construction
STABILITY = {"e4b": 0.05}         # R1: |d1 - d2| / mean; every other framework 0.10
STABILITY_OTHER = 0.10

FAMS = ["qwen3"]
NAMES = {"qwen3": "Qwen3-30B-A3B"}
N_LAYERS = {"qwen3": 48}
FW = ("e4b", "unsloth", "hf", "axolotl")
QUALITY_ANCHOR = ("e4b", "fused_attn4_m")        # the quality reading's anchor and the matched position's e4b side
EQUIV_ANCHOR = ("e4b", "reference_attn4_m")      # the equivalence anchor (and the step-0 SAME-BASE anchor)
PRIMARY = {"e4b": "fused_attn4_m", "unsloth": "ckpt_unsloth_m", "hf": "hf_peft_m", "axolotl": "ckpt_axolotl_m"}
SECONDARY = {fw: PRIMARY[fw] + "_mb1" for fw in PRIMARY}
DRAW2 = {("e4b", "fused_attn4_m"): ("e4b", "fused_attn4_m_d2"), ("unsloth", "ckpt_unsloth_m"): ("unsloth", "ckpt_unsloth_m_d2")}
NATIVE = {"e4b": "fused_attn4_shipped", "unsloth": "ckpt_unsloth_best", "axolotl": "ckpt_axolotl_best"}
PROF = ("unsloth", "ckpt_unsloth_prof")
# TC1-PREREG "Arms, in this order": the registered arm set, in the registered order (the support table's row order)
EXPECTED = {"qwen3": [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "fused_attn4_m_d2"), ("unsloth", "ckpt_unsloth_m_d2"),
                      ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_best"), ("unsloth", "ckpt_unsloth_best"),
                      ("e4b", "fused_attn4_shipped"), ("e4b", "reference_attn4_m"), ("unsloth", "ckpt_unsloth_prof")]}
# the arms registered MATCHED (fp32 adapters, --lora-init matched:<seed>): R3 applies to these, native rows carry no R3
MATCHED = {"fused_attn4_m", "fused_attn4_m_d2", "ckpt_unsloth_m", "ckpt_unsloth_m_d2", "hf_peft_m", "ckpt_axolotl_m",
           "reference_attn4_m", "ckpt_unsloth_prof", "fused_attn4_m_mb1", "ckpt_unsloth_m_mb1", "hf_peft_m_mb1", "ckpt_axolotl_m_mb1"}
VOCAB = ("OK", "REFUSED", "OOM", "INSTALL_FAILED", "LOAD_FAULT", "HARNESS_ERROR", "ALARM", "NOT_RUN")
VERDICTS = ("VALID", "VOID", "QUALITY_FAIL", "OOM", "UNSUPPORTED", "HARNESS_ERROR", "ALARM", "NOT_RUN")
STATUS_MAP = {"ok": "OK", "c1_failed": "OK", "refused": "REFUSED", "oom": "OOM", "install_failed": "INSTALL_FAILED",
              "load_fault": "LOAD_FAULT", "verify_failed": "LOAD_FAULT", "alarm": "ALARM", "not_run": "NOT_RUN",
              "phase_alarm": "ALARM",             # e4b#548: the same registered word as the SIGALRM it replaces; the row can name the phase
              "tokens_mismatch": "HARNESS_ERROR", "void_trainable": "HARNESS_ERROR", "void_attn4": "HARNESS_ERROR",
              "harness_error": "HARNESS_ERROR", "unreadable": "HARNESS_ERROR"}
FW_RE = "|".join(FW)


# ----------------------------------------------------------------------------- inputs
def load(d):
    """{fam: {(fw, tag): receipt}} from <fam>_<fw>_<tag>.json (tokens_*.json, box.json and other files are skipped)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_*_*.json"))):
        base = os.path.basename(p)[:-5]
        m = re.match(rf"^([a-z0-9_]+?)_({FW_RE})_(.+)$", base)
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
        m = re.match(rf"^([a-z0-9_]+)/({FW_RE})/(\S+) rc=(\d+)", ln)
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


def is_ok(r):
    return r is not None and STATUS_MAP.get(r.get("status")) == "OK"


# ----------------------------------------------------------------------------- 1. status, 2. validity, R2 verdict
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


def prologue_lines(rows, top=4):
    """e4b#548: what the arm spent before step 1, from the receipt's own `phase_seconds`; the residual printed ALWAYS."""
    out = []
    for x in rows:
        r = x["r"] or {}
        ph, tot = r.get("phase_seconds"), r.get("prologue_s")
        if not ph or tot is None:
            continue
        big = sorted(ph.items(), key=lambda kv: -kv[1])[:top]
        share = f"{100 * tot / (tot + (r.get('window_wall_s') or 0)):.0f}% of the arm" if r.get("window_wall_s") else "—"
        out.append(f"- prologue `{x['fw']}/{x['tag']}` **{tot:.1f} s** before step 1 ({share}): "
                   + ", ".join(f"{k} {v:.1f}" for k, v in big)
                   + f"; unattributed {r.get('prologue_unattributed_s', '—')}"
                   + (f"; budget {r['phase_budget_s']}" if r.get("phase_budget_s") else ""))
    return out


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


def matched_seed_of(r):
    m = re.fullmatch(r"matched:(\d+)", str((r or {}).get("lora_init") or ""))
    return int(m.group(1)) if m else None


def validity(fam, r, tokens_sha, e4b_trainable, n_steps, matched=False, ref_step0=None, anchor_seed=None, is_ref=False):
    """tp4's validity rules for an OK row (+ R3 for a registered matched arm) -> (VALID|VOID, why). The n_layers table is
    the registered one; accum-aware (a kernel call per micro-batch per layer)."""
    if r is None or not is_ok(r):
        return "—", ""
    L = N_LAYERS.get(fam)
    A = int(r.get("accum") or 1)
    why = []
    if L is None:
        why.append(f"no registered n_layers for family {fam}")
        L = r.get("n_layers") or 0
    elif r.get("n_layers") not in (None, L):
        why.append(f"config n_layers {r.get('n_layers')} != registered {L}")
    if not c1_ok(r) or r.get("status") == "c1_failed":
        why.append("C1 not clean" if r.get("status") != "c1_failed" else "C1 FAILED (the arm's own status)")
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
    elif fw in ("hf", "axolotl"):
        if r.get("experts_forward_calls_per_step_min", 0) < L * A:
            why.append(f"experts forward calls/step min {r.get('experts_forward_calls_per_step_min')} < {L}*accum {A}")
        key = "hf_targets" if fw == "hf" else "axolotl_targets"
        if not (r.get(key) or r.get("hf_targets") or {}).get("n_target_parameters"):
            why.append("adapted no expert parameter (target_parameters empty) -- attention-only, not the registered adapter set")
    if matched:                                   # R3: the matched set's own predicates (TC1-PREREG "Validity", new in this lane)
        seed = matched_seed_of(r)
        if seed is None:
            why.append(f"registered matched but ran lora_init={r.get('lora_init')!r}")
        elif anchor_seed is not None and seed != anchor_seed:
            why.append(f"matched seed {seed} != the anchor's {anchor_seed}")
        mi = r.get("matched_init") or {}
        if not mi.get("complete"):
            why.append(f"matched_init.complete is {mi.get('complete')} (set {mi.get('n_slots_set')} of {mi.get('n_slots_expected')}; unmapped {str((mi.get('unmapped') or [])[:2])[:160]})")
        dts = list((r.get("adapter_dtypes_after") or {}).keys())
        if dts != ["torch.float32"]:
            why.append(f"adapter_dtypes_after {r.get('adapter_dtypes_after')} is not fp32-only")
        if not is_ref and ref_step0 is not None and r.get("eval_loss_step0") is not None and abs(r["eval_loss_step0"] - ref_step0) > STEP0_TOL:
            why.append(f"step-0 held-out {r['eval_loss_step0']:.4f} differs from reference_attn4_m's {ref_step0:.4f} by {abs(r['eval_loss_step0'] - ref_step0):.4f} > {STEP0_TOL}")
    return ("VOID" if why else "VALID"), "; ".join(why)


def verdict_of(st, v, qdelta):
    """R2: the one-word verdict. `qdelta` is the held-out delta at N against the quality anchor (None = no anchor to read)."""
    if st == "OK":
        if v == "VOID":
            return "VOID"
        if qdelta is not None and abs(qdelta) > READ:
            return "QUALITY_FAIL"
        return "VALID"
    if st == "OOM":
        return "OOM"
    if st in ("REFUSED", "INSTALL_FAILED", "LOAD_FAULT"):
        return "UNSUPPORTED"
    assert st in ("HARNESS_ERROR", "ALARM", "NOT_RUN"), st
    return st


# ----------------------------------------------------------------------------- 4. e4b internal parity (tp1's rule)
def parity(ref, arm):
    if arm is None or not is_ok(arm):
        return "NO-ARM", None, None, ""
    if ref is None or not is_ok(ref):
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


def heldout_at_N(r):
    return curve_at(r, r.get("steps")) if r else None


# ----------------------------------------------------------------------------- R1: draws
def draws_of(recs, verdicts, key):
    """One arm, one or two draws -> {usable, s, peak, j, tok, draws, stability, threshold, verdict, why}. Medians over the
    usable draws; a registered second draw that is missing or not VALID leaves the stability UNMEASURED (not usable)."""
    fw, tag = key
    r1, k2 = recs.get(key), DRAW2.get(key)
    v1 = verdicts.get(key)
    out = {"key": key, "usable": False, "draws": 0, "stability": None, "threshold": STABILITY.get(fw, STABILITY_OTHER), "verdict": "—", "why": ""}
    if v1 != "VALID":
        out["why"] = f"{fw}/{tag} is {v1 or 'missing'}"
        return out
    rs = [r1]
    if k2 is not None:
        r2, v2 = recs.get(k2), verdicts.get(k2)
        if v2 != "VALID":
            out.update({"draws": 1, "verdict": "UNMEASURED", "why": f"second draw {k2[1]} is {v2 or 'missing'}: stability unmeasured, position not quoted"})
            return out
        s1, s2 = r1["s_per_step_median_11plus"], r2["s_per_step_median_11plus"]
        stab = abs(s1 - s2) / statistics.mean([s1, s2]) if statistics.mean([s1, s2]) else None
        out.update({"draws": 2, "stability": stab, "s1": s1, "s2": s2, "verdict": "STABLE" if (stab is not None and stab <= out["threshold"]) else "UNSTABLE"})
        rs.append(r2)
        if out["verdict"] != "STABLE":
            out["why"] = f"draws {s1:.3f} / {s2:.3f} s/step differ by {100 * (stab or 0):.1f}% > {100 * out['threshold']:.0f}% (UNSTABLE: reported, not quoted)"
            return out
    else:
        out.update({"draws": 1, "verdict": "SINGLE", "why": "single draw (no second draw registered)"})
    out.update({"usable": True, "s": statistics.median([r["s_per_step_median_11plus"] for r in rs]),
                "peak": statistics.median([r["peak_vram_gb"] for r in rs]),
                "j": (statistics.median([r["joules_per_step"] for r in rs]) if all(r.get("joules_per_step") is not None for r in rs) else None),
                "tok": (statistics.median([r["tokens_per_s"] for r in rs]) if all(r.get("tokens_per_s") is not None for r in rs) else None),
                "heldout": heldout_at_N(r1), "step0": r1.get("eval_loss_step0"), "regime": None})
    return out


def position(e, o, label):
    """The pair (e4b draws e, other draws o): quoted only when both are usable (VALID verdict, STABLE or single draw)."""
    pos = {"label": label, "quoted": False, "why": ""}
    if e.get("usable") and o.get("usable"):
        pos.update({"quoted": True, "ratio": ratio(o["s"], e["s"]), "e4b_s": e["s"], "other_s": o["s"], "e4b_draws": e["draws"], "other_draws": o["draws"],
                    "e4b_stability": e.get("stability"), "other_stability": o.get("stability"),
                    "peak_e4b": e["peak"], "peak_other": o["peak"], "peak_delta": o["peak"] - e["peak"],
                    "j_e4b": e.get("j"), "j_other": o.get("j"), "j_ratio": ratio(o.get("j"), e.get("j")),
                    "tok_e4b": e.get("tok"), "tok_other": o.get("tok"), "other_regime": None,
                    "heldout_e4b": e.get("heldout"), "heldout_other": o.get("heldout"),
                    "heldout_delta": (o["heldout"] - e["heldout"]) if (e.get("heldout") is not None and o.get("heldout") is not None) else None,
                    "step0_delta": (o["step0"] - e["step0"]) if (e.get("step0") is not None and o.get("step0") is not None) else None})
        pos["quality"] = "COMPARABLE" if pos["heldout_delta"] is not None and abs(pos["heldout_delta"]) <= READ else "FLAGGED"
    else:
        pos["why"] = f"not quoted: e4b {e.get('why') or e.get('verdict')} / {label} {o.get('why') or o.get('verdict')}"
    return pos


# ----------------------------------------------------------------------------- R4: equivalence, R5: frozen base
def equivalence(ref, r, vref, vr):
    """{reading, med_train, d_heldout, d_step0, why} for one matched arm against e4b/reference_attn4_m."""
    if r is None or not is_ok(r):
        return {"reading": "—", "why": "no OK receipt"}
    if ref is None or not is_ok(ref):
        return {"reading": "N-A", "why": "reference_attn4_m missing or not OK"}
    if vref != "VALID" or vr != "VALID":
        return {"reading": "N-A", "why": f"validity: reference {vref}, arm {vr} (VOID never enters an equivalence reading)"}
    if len(r.get("losses", [])) != len(ref.get("losses", [])) or not r.get("losses"):
        return {"reading": "N-A", "why": "step counts differ"}
    med = statistics.median(abs(x - y) for x, y in zip(r["losses"], ref["losses"]))
    hr, hf_ = heldout_at_N(r), heldout_at_N(ref)
    if hr is None or hf_ is None:
        return {"reading": "N-A", "why": "no held-out loss at N on one side", "med_train": med}
    dh = abs(hr - hf_)
    d0 = abs((r.get("eval_loss_step0") or 0) - (ref.get("eval_loss_step0") or 0)) if (r.get("eval_loss_step0") is not None and ref.get("eval_loss_step0") is not None) else None
    reading = "EQUIVALENT" if (med <= EQUIV and dh <= EQUIV) else ("COMPARABLE" if (med <= COMPARABLE and dh <= COMPARABLE) else "DIVERGENT")
    return {"reading": reading, "med_train": med, "d_heldout": dh, "d_step0": d0, "why": ""}


def frozen_base(anchor, r):
    """Per registered slot: SAME-BYTES / DIFFERENT when both arms carry the slot in the SAME regime, N-A otherwise;
    plus each side's own control. Nothing is compared across regimes."""
    out = {}
    pa = ((anchor or {}).get("frozen_base_probe") or {}).get("slots") or {}
    pr = ((r or {}).get("frozen_base_probe") or {}).get("slots") or {}
    for slot in ("gate_up", "down", "q_proj"):
        a, b = pa.get(slot), pr.get(slot)
        if not a or not b:
            out[slot] = {"reading": "N-A", "why": "slot missing on " + ("both" if not a and not b else ("the anchor" if not a else "this arm")), "regime": (b or {}).get("regime")}
        elif a.get("regime") != b.get("regime"):
            out[slot] = {"reading": "N-A", "why": f"regime differs: anchor {a.get('regime')} vs {b.get('regime')}", "regime": b.get("regime")}
        else:
            out[slot] = {"reading": "SAME-BYTES" if a.get("sha") == b.get("sha") else "DIFFERENT", "regime": b.get("regime"), "why": ""}
        out[slot]["control"] = bool((b or {}).get("control_detects_flip"))
    return out


# ----------------------------------------------------------------------------- the per-family reduction (one dict; the printer and the tests read it)
def reduce_family(fam, recs, rcs_all, n_steps=None):
    exp = list(EXPECTED.get(fam, EXPECTED["qwen3"]))
    keys = exp + [k for k in recs if k not in exp]
    e_anchor = recs.get(QUALITY_ANCHOR)
    e4b_trainable = e_anchor.get("trainable_params") if is_ok(e_anchor) else None
    if e4b_trainable is None:                        # fall back to the e4b reference arm (same recipe, same adapter set)
        rr = recs.get(EQUIV_ANCHOR)
        if is_ok(rr):
            e4b_trainable = rr.get("trainable_params")

    def trainable_ref(tag):                          # the mb1 pair has its OWN e4b arm as the trainable reference
        if tag.endswith("_mb1"):
            r = recs.get(("e4b", SECONDARY["e4b"]))
            return r.get("trainable_params") if is_ok(r) else None
        return e4b_trainable
    tokens_sha = next((r["tokens"]["sha256"] for k in keys if (r := recs.get(k)) and r.get("tokens", {}).get("sha256")), None)
    N = n_steps or next((r.get("steps") for r in recs.values() if r and r.get("steps")), None)
    ref = recs.get(EQUIV_ANCHOR)
    ref_step0 = ref.get("eval_loss_step0") if is_ok(ref) else None
    anchor_seed = matched_seed_of(e_anchor) if is_ok(e_anchor) else (matched_seed_of(ref) if is_ok(ref) else None)
    rows, V = [], {}
    for fw, tag in keys:
        r = recs.get((fw, tag))
        st, reason = status_of(r, rcs_all.get((fam, fw, tag)))
        v, why = validity(fam, r, tokens_sha, trainable_ref(tag) if fw != "e4b" else None, N,
                          matched=(tag in MATCHED), ref_step0=ref_step0, anchor_seed=anchor_seed, is_ref=((fw, tag) == EQUIV_ANCHOR))
        rows.append({"fw": fw, "tag": tag, "status": st, "reason": reason, "validity": v, "why": why, "r": r,
                     "regime": regime_of(fam, r) if (r and st == "OK") else None, "matched": tag in MATCHED})
        V[(fw, tag)] = v
    # R2: quality against the anchor (read only when the anchor is VALID by predicates), then the verdict
    anchor_ok = is_ok(e_anchor) and V.get(QUALITY_ANCHOR) == "VALID"
    ha = heldout_at_N(e_anchor) if anchor_ok else None
    verdicts = {}
    for x in rows:
        r = x["r"]
        q = None
        if x["status"] == "OK" and ha is not None and heldout_at_N(r) is not None:
            q = heldout_at_N(r) - ha
        x["quality_delta"] = q
        x["quality"] = ("COMPARABLE" if abs(q) <= READ else "FLAGGED") if q is not None else ("N-A (anchor " + ("VOID" if is_ok(e_anchor) else "missing") + ")" if x["status"] == "OK" else None)
        x["verdict"] = verdict_of(x["status"], x["validity"], q)
        assert x["verdict"] in VERDICTS
        verdicts[(x["fw"], x["tag"])] = x["verdict"]
    # tp1's parity (fused_m vs reference_m), informational
    fu = recs.get(QUALITY_ANCHOR)
    pv, d_final, med, pwhy = parity(ref, fu)
    par = {"verdict": pv, "d_final": d_final, "median": med, "why": pwhy}
    if pv in ("PASS", "FAIL"):
        par["speed_x"] = ratio(ref.get("s_per_step_median_11plus"), fu.get("s_per_step_median_11plus"))
        par["peak_x"] = ratio(fu.get("peak_vram_gb"), ref.get("peak_vram_gb"))
    # R1 draws per arm, then the positions
    draws = {k: draws_of(recs, verdicts, k) for k in keys if k not in DRAW2.values()}
    e_d = draws.get(QUALITY_ANCHOR, {"usable": False, "why": "no e4b/fused_attn4_m"})
    positions = {}
    for other in ("unsloth", "hf", "axolotl"):
        k = (other, PRIMARY[other])
        positions[other] = position(e_d, draws.get(k, {"usable": False, "why": "no receipt"}), other)
        if positions[other].get("quoted"):
            positions[other]["other_regime"] = regime_of(fam, recs[k])
    secondary = {}
    e2 = draws.get(("e4b", SECONDARY["e4b"]))
    for other in ("unsloth", "hf", "axolotl"):
        k2 = (other, SECONDARY[other])
        if e2 or recs.get(k2):
            secondary[other] = position(e2 or {"usable": False, "why": "no receipt"}, draws.get(k2, {"usable": False, "why": "no receipt"}), other + " (mb1)")
    native = {}
    sh = draws.get(("e4b", NATIVE["e4b"]), {"usable": False, "why": "no receipt"})
    for other in ("unsloth", "axolotl"):
        k = (other, NATIVE[other])
        if recs.get(k) or recs.get(("e4b", NATIVE["e4b"])):
            native[other] = position(sh, draws.get(k, {"usable": False, "why": "no receipt"}), f"{other} native-best vs e4b shipped")
            if native[other].get("quoted"):
                native[other]["other_regime"] = regime_of(fam, recs[k])
    # R4: equivalence of every matched arm against the reference; R5: frozen base against the quality anchor
    equiv = {}
    for fw, tag in keys:
        if tag in MATCHED and (fw, tag) != EQUIV_ANCHOR and recs.get((fw, tag)) is not None:
            equiv[(fw, tag)] = equivalence(ref, recs[(fw, tag)], V.get(EQUIV_ANCHOR), V.get((fw, tag)))
    frozen = {}
    for fw, tag in keys:
        r = recs.get((fw, tag))
        if is_ok(r) and (fw, tag) != QUALITY_ANCHOR:
            frozen[(fw, tag)] = frozen_base(e_anchor if is_ok(e_anchor) else None, r)
    anchor_probe = ((e_anchor or {}).get("frozen_base_probe") or {}) if is_ok(e_anchor) else {}
    prof = recs.get(PROF)
    profile = (prof.get("profile") or None) if is_ok(prof) else None
    return {"fam": fam, "rows": rows, "V": V, "verdicts": verdicts, "parity": par, "draws": draws, "positions": positions, "secondary": secondary,
            "native": native, "equivalence": equiv, "frozen": frozen, "anchor_probe": anchor_probe, "profile": profile,
            "tokens_sha": tokens_sha, "e4b_trainable": e4b_trainable, "N": N, "e": e_anchor, "ref": ref,
            "u": recs.get(("unsloth", PRIMARY["unsloth"])), "h": recs.get(("hf", PRIMARY["hf"])), "ax": recs.get(("axolotl", PRIMARY["axolotl"]))}


# ----------------------------------------------------------------------------- R6: predictions (TC1-PREREG.md, scored mechanically)
def score_predictions(F):
    R = F.get("qwen3")
    out = []
    if not R:
        return [(f"P{i}", "qwen3", "UNTESTED", "no receipts") for i in range(1, 11)]
    vd = R["verdicts"]
    pos = R["positions"]
    dr = R["draws"]

    def vof(fw, tag):
        return vd.get((fw, tag), "missing")
    # P1 matched position Unsloth/e4b in [2.0, 5.0]; below 1.5 refutes the standing position
    pu = pos.get("unsloth", {})
    if pu.get("quoted"):
        r = pu["ratio"]
        held = 2.0 <= r <= 5.0
        out.append(("P1", "qwen3", "HELD" if held else "FALSIFIED",
                    f"matched unsloth/e4b {r:.3f} vs [2.0, 5.0]" + ("" if held else ("; BELOW 1.5: the standing position is refuted and superseded" if r < 1.5 else "; outside the band"))))
    else:
        out.append(("P1", "qwen3", "UNTESTED", "no quoted matched position: " + pu.get("why", "no pair")))
    # P2 draws agree within 5 % (e4b) / 10 % (Unsloth)
    ev = []
    for k in (QUALITY_ANCHOR, ("unsloth", PRIMARY["unsloth"])):
        d = dr.get(k, {})
        if d.get("draws") == 2 and d.get("stability") is not None:
            ev.append((k[0], d["stability"], d["threshold"], d["verdict"]))
    if not ev:
        out.append(("P2", "qwen3", "UNTESTED", "no arm with two VALID draws: " + "; ".join(f"{k[0]} {dr.get(k, {}).get('why', 'no receipt')}" for k in (QUALITY_ANCHOR, ("unsloth", PRIMARY["unsloth"])))))
    else:
        bad = [e for e in ev if e[3] != "STABLE"]
        out.append(("P2", "qwen3", "HELD" if not bad else "FALSIFIED", "; ".join(f"{fw} |d1-d2|/mean {100 * s:.1f}% vs {100 * t:.0f}% -> {v}" for fw, s, t, v in ev)))
    # P3 the matched set is EQUIVALENT: e4b fused_m and unsloth_m against reference_m
    eq = {k: R["equivalence"].get(k, {}).get("reading", "—") for k in (QUALITY_ANCHOR, ("unsloth", PRIMARY["unsloth"]))}
    if any(v in ("—", "N-A") for v in eq.values()):
        out.append(("P3", "qwen3", "UNTESTED", "; ".join(f"{k[0]}/{k[1]} {v} ({R['equivalence'].get(k, {}).get('why', 'no receipt')})" for k, v in eq.items())))
    else:
        out.append(("P3", "qwen3", "HELD" if all(v == "EQUIVALENT" for v in eq.values()) else "FALSIFIED",
                    "; ".join(f"{k[0]}/{k[1]} {v} (median step |Δ| {f(R['equivalence'][k].get('med_train'), 4)}, |Δ held-out at N| {f(R['equivalence'][k].get('d_heldout'), 4)})" for k, v in eq.items())
                    + ("" if all(v == "EQUIVALENT" for v in eq.values()) else ("; DIVERGENT: no position is quoted from this lane (decision rule)" if any(v == "DIVERGENT" for v in eq.values()) else "; COMPARABLE, not EQUIVALENT, and not DIVERGENT: the decision rule does not fire"))))
    # P4 shipped is 1.05-1.3x faster per step than fused_m and its held-out at N is lower by >= 0.01
    sh, fm = dr.get(("e4b", NATIVE["e4b"]), {}), dr.get(QUALITY_ANCHOR, {})
    if sh.get("usable") and fm.get("usable") and sh.get("heldout") is not None and fm.get("heldout") is not None:
        x = fm["s"] / sh["s"] if sh["s"] else None
        gain = fm["heldout"] - sh["heldout"]
        held = x is not None and 1.05 <= x <= 1.3 and gain >= 0.01
        out.append(("P4", "qwen3", "HELD" if held else "FALSIFIED", f"fused_m/shipped s/step {f(x)} vs [1.05, 1.3]; held-out matched {fm['heldout']:.4f} - shipped {sh['heldout']:.4f} = {gain:+.4f} (>= 0.01 predicted)"))
    else:
        out.append(("P4", "qwen3", "UNTESTED", f"shipped {vof('e4b', NATIVE['e4b'])} / fused_m {vof(*QUALITY_ANCHOR)}: both must be VALID"))
    # P5 HF OOMs at the primary recipe and at mb1
    hp, hm = vof("hf", PRIMARY["hf"]), vof("hf", SECONDARY["hf"])
    if hp == "missing" and hm == "missing":
        out.append(("P5", "qwen3", "UNTESTED", "no hf receipts"))
    elif hp == "OOM" and hm == "OOM":
        out.append(("P5", "qwen3", "HELD", "hf_peft_m OOM and hf_peft_m_mb1 OOM"))
    elif hp in ("VALID", "QUALITY_FAIL", "VOID") or hm in ("VALID", "QUALITY_FAIL", "VOID"):
        out.append(("P5", "qwen3", "FALSIFIED", f"hf_peft_m {hp}, hf_peft_m_mb1 {hm}: HF trained at least one recipe"))
    else:
        out.append(("P5", "qwen3", "UNTESTED", f"hf_peft_m {hp}, hf_peft_m_mb1 {hm}: neither an OOM pair nor a trained arm"))
    # P6 axolotl: INSTALL_FAILED/UNSUPPORTED/OOM, or trains with axolotl/e4b in [1.5, 6]
    av = vof("axolotl", PRIMARY["axolotl"])
    pa = pos.get("axolotl", {})
    if av in ("UNSUPPORTED", "OOM"):
        out.append(("P6", "qwen3", "HELD", f"axolotl {av}"))
    elif av in ("VALID", "QUALITY_FAIL", "VOID"):
        if pa.get("quoted"):
            out.append(("P6", "qwen3", "HELD" if 1.5 <= pa["ratio"] <= 6 else "FALSIFIED", f"axolotl trained; axolotl/e4b {pa['ratio']:.3f} vs [1.5, 6]"))
        else:
            out.append(("P6", "qwen3", "UNTESTED", f"axolotl {av} but no quoted position: {pa.get('why')}"))
    else:
        out.append(("P6", "qwen3", "UNTESTED", f"axolotl {av} (NOT_RUN / HARNESS_ERROR / ALARM is not a reading)"))
    # P7 unsloth_best at most 1.5x faster than unsloth_m, and e4b_m >= 2x faster than unsloth_best
    ub, um, em = dr.get(("unsloth", NATIVE["unsloth"]), {}), dr.get(("unsloth", PRIMARY["unsloth"]), {}), dr.get(QUALITY_ANCHOR, {})
    if ub.get("usable") and um.get("usable") and em.get("usable"):
        gain_x, e_x = um["s"] / ub["s"], ub["s"] / em["s"]
        out.append(("P7", "qwen3", "HELD" if (gain_x <= 1.5 and e_x >= 2.0) else "FALSIFIED", f"unsloth_m/unsloth_best {gain_x:.3f} (<= 1.5 predicted); unsloth_best/e4b_m {e_x:.3f} (>= 2 predicted)"))
    else:
        out.append(("P7", "qwen3", "UNTESTED", f"unsloth_best {vof('unsloth', NATIVE['unsloth'])} / unsloth_m {vof('unsloth', PRIMARY['unsloth'])} / e4b_m {vof(*QUALITY_ANCHOR)}: all three must be VALID"))
    # P8 the profiled Unsloth arm is host-bound: device busy fraction < 0.4
    pr = R.get("profile")
    if pr and pr.get("device_busy_fraction") is not None:
        out.append(("P8", "qwen3", "HELD" if pr["device_busy_fraction"] < 0.4 else "FALSIFIED", f"device_busy_fraction {pr['device_busy_fraction']:.3f} (< 0.4 predicted); device events/step {pr.get('device_events_per_step')}"))
    else:
        out.append(("P8", "qwen3", "UNTESTED", f"ckpt_unsloth_prof {vof(*PROF)} carries no profile summary"))
    # P9 e4b internal parity PASS
    pv = R["parity"]["verdict"]
    out.append(("P9", "qwen3", {"PASS": "HELD", "FAIL": "FALSIFIED"}.get(pv, "UNTESTED"), f"fused_m vs reference_m {pv}" + (f" (Δfinal {R['parity']['d_final']:.5f}, median {R['parity']['median']:.5f})" if pv in ("PASS", "FAIL") else f" {R['parity']['why']}")))
    # P10 C1 bit-exact in every OK arm; expert slots SAME-BYTES e4b vs Unsloth; attention slot not SAME-BYTES
    oks = [x for x in R["rows"] if x["status"] == "OK"]
    c1 = [f"{x['fw']}/{x['tag']}" for x in oks if not c1_ok(x["r"])]
    fb = R["frozen"].get(("unsloth", PRIMARY["unsloth"]))
    if not oks:
        out.append(("P10", "qwen3", "UNTESTED", "no OK arm"))
    elif c1:
        out.append(("P10", "qwen3", "FALSIFIED", "C1 not bit-exact on " + ", ".join(c1)))
    elif not fb:
        out.append(("P10", "qwen3", "UNTESTED", "C1 bit-exact on every OK arm; no unsloth_m probe to read the slots against e4b"))
    else:
        ex = [fb[s]["reading"] for s in ("gate_up", "down")]
        at = fb["q_proj"]["reading"]
        if "N-A" in ex:
            out.append(("P10", "qwen3", "UNTESTED", f"C1 bit-exact; expert slots {ex} ({fb['gate_up'].get('why') or fb['down'].get('why')}); attention {at}"))
        else:
            held = all(v == "SAME-BYTES" for v in ex) and at != "SAME-BYTES"
            out.append(("P10", "qwen3", "HELD" if held else "FALSIFIED", f"C1 bit-exact on every OK arm; expert slots e4b vs unsloth {ex}; attention {at} ({fb['q_proj'].get('why') or fb['q_proj'].get('regime')})"))
    return out


# ----------------------------------------------------------------------------- the printer
def pos_lines(pos, N, prefix="POSITION"):
    if not pos.get("quoted"):
        return [f"- **NO {prefix} QUOTED ({pos['label']})** — {pos['why']}"]
    flag = "" if pos["quality"] == "COMPARABLE" else " **[QUALITY FLAGGED: not a clean position]**"
    who = pos["label"]
    lines = [f"- **{prefix}: s/step ratio {who}/e4b = {pos['ratio']:.3f}**{flag} ({pos['other_s']:.3f} vs {pos['e4b_s']:.3f} s, medians over {pos['other_draws']}/{pos['e4b_draws']} draws; "
             f"{'e4b faster' if pos['ratio'] > 1 else who + ' faster'} per step); "
             f"peak VRAM {who} {pos['peak_other']:.2f} vs e4b {pos['peak_e4b']:.2f} GB (Δ {pos['peak_delta']:+.2f}); J/step {who} {f(pos['j_other'], 1)} vs e4b {f(pos['j_e4b'], 1)} (×{f(pos['j_ratio'], 3)}); "
             f"tok/s {who} {f(pos['tok_other'], 1)} vs e4b {f(pos['tok_e4b'], 1)}" + (f"; {who} regime: {pos['other_regime']}" if pos.get("other_regime") else "")]
    lines.append(f"- quality reading at N={N} ({who}): held-out e4b {f(pos['heldout_e4b'], 4)} / {who} {f(pos['heldout_other'], 4)} (Δ {f(pos['heldout_delta'], 4)}) → **{pos['quality']}** "
                 f"(|Δ| ≤ {READ} reads COMPARABLE)" + (f"; step-0 Δ {pos['step0_delta']:+.4f}" if pos.get("step0_delta") is not None else ""))
    return lines


def family_block(R):
    fam = R["fam"]
    lines = [f"\n### {NAMES.get(fam, fam)} (`{fam}`, registered n_layers {N_LAYERS.get(fam, '?')})"]

    def _ran(*cands):
        for c in cands:
            if c and is_ok(c):
                return c
        return None
    src = (_ran(R["e"], R["ref"], R["u"], R["h"], R["ax"], *[x["r"] for x in R["rows"]])
           or R["e"] or R["ref"] or R["u"] or R["h"] or next((x["r"] for x in R["rows"] if x["r"]), None))
    if src:
        env = src.get("env", {}) or {}
        lines.append(f"- model `{src.get('model')}` @ `{str(src.get('revision', ''))[:12]}`; tokens sha `{str(R['tokens_sha'] or '')[:12]}`; N={R['N']}; "
                     f"fixture template {src.get('template')} seq {src.get('seq')} micro-batch {src.get('micro_batch')} × accum {src.get('accum')} lr {src.get('lr')} r {src.get('r')} α {src.get('alpha')} "
                     f"optimizer {src.get('optimizer')} autocast {src.get('autocast')}; e4b trainable {R['e4b_trainable']}; box_class {env.get('box_class')} gpu {env.get('gpu')}")
    lines.append("| framework | arm | status | validity | **VERDICT** | matched | init / dtype | N | s/step med(11+) | tok/s | peak GB | J/step | train first→last | held-out 0→final | quality Δ | engagement | trainable | regime | reason / why |")
    lines.append("|" + "---|" * 19)
    for x in R["rows"]:
        r = x["r"] or {}
        b = r.get("unsloth_bnb4bit_modules")
        if x["fw"] == "e4b":
            eng = f"patched {r.get('n_patched', '—')} / kcalls {r.get('kernel_calls_per_step_min', '—')}"
        elif x["fw"] == "unsloth":
            eng = f"stacks {(r.get('census') or {}).get('Params4bit_expert_stacks', '—')} / fwd {r.get('experts_forward_calls_per_step_min', '—')} / u8 {b.get('n_bnb4bit_unwrapped') if isinstance(b, dict) else '—'}"
        else:
            ht = r.get("hf_targets") or r.get("axolotl_targets") or {}
            eng = f"peft mods {ht.get('n_target_modules', '—')} / params {ht.get('n_target_parameters', '—')} / fwd {r.get('experts_forward_calls_per_step_min', '—')}"
        mi = r.get("matched_init") or {}
        init = f"{r.get('lora_init', '—')}" + (f" ({'complete' if mi.get('complete') else 'INCOMPLETE'} {mi.get('n_slots_set')}/{mi.get('n_slots_expected')})" if mi else "") \
            + f" / {','.join(sorted(k.replace('torch.', '') for k in (r.get('adapter_dtypes_after') or {}))) or '—'}"
        note = " — ".join(s for s in (x["reason"], x["why"]) if s)
        lines.append(f"| {x['fw']} | {x['tag']} | **{x['status']}** | {x['validity']} | **{x['verdict']}** | {'yes' if x['matched'] else 'native'} | {init} | {r.get('steps', '—')} | "
                     f"{f(r.get('s_per_step_median_11plus'))} | {f(r.get('tokens_per_s'), 1)} | {f(r.get('peak_vram_gb'))} | {f(r.get('joules_per_step'), 1)} | "
                     f"{f(r.get('loss_first'), 4)}→{f(r.get('loss_last'), 4)} | {f(r.get('eval_loss_step0'), 4)}→{f(r.get('eval_loss_final'), 4)} | "
                     f"{f(x.get('quality_delta'), 4) if x.get('quality_delta') is not None else (x.get('quality') or '—')} | {eng} | {r.get('trainable_params', '—')} | {x['regime'] or '—'} | {note} |")
    for line in prologue_lines(R["rows"]):
        lines.append(line)
    lines.append("- draws (R1): " + "; ".join(
        f"`{k[0]}/{k[1]}` {d['verdict']}" + (f" ({d['s1']:.3f}/{d['s2']:.3f} s, |Δ|/mean {100 * d['stability']:.1f}% vs {100 * d['threshold']:.0f}%)" if d.get("draws") == 2 and d.get("stability") is not None else (f" ({d['why']})" if d.get("why") else ""))
        for k, d in R["draws"].items() if k in DRAW2 or k in DRAW2.values() or d.get("draws")))
    p = R["parity"]
    if p["verdict"] in ("PASS", "FAIL"):
        lines.append(f"- e4b internal parity (tp1's rule, informational): fused_attn4_m vs reference_attn4_m Δfinal {p['d_final']:.5f}, median step |Δ| {p['median']:.5f} → **{p['verdict']}** "
                     f"(band {BAND}/{BAND}); ×{f(p.get('speed_x'), 2)} faster per step, peak ×{f(p.get('peak_x'), 3)}")
    else:
        lines.append(f"- e4b internal parity: {p['verdict']}{(' — ' + p['why']) if p['why'] else ''}")
    for other in ("unsloth", "hf", "axolotl"):
        if other in R["positions"]:
            lines += pos_lines(R["positions"][other], R["N"], prefix="MATCHED POSITION")
    for other, sp in R["secondary"].items():
        lines += pos_lines(sp, R["N"], prefix="SECONDARY POSITION (mb1 × accum 8, run because a primary arm OOMed)")
    for other, np_ in R["native"].items():
        lines += pos_lines(np_, R["N"], prefix="NATIVE-BEST (reported beside, never instead of, the matched position)")
    if R["equivalence"]:
        lines.append("- equivalence vs `e4b/reference_attn4_m` (R4; EQUIVALENT ≤ 0.02 / COMPARABLE ≤ 0.05 on both the median per-step |Δ train| and |Δ held-out at N|): "
                     + "; ".join(f"`{k[0]}/{k[1]}` **{e['reading']}**" + (f" (median {f(e.get('med_train'), 4)}, held-out {f(e.get('d_heldout'), 4)}, step-0 {f(e.get('d_step0'), 4)})" if e.get("med_train") is not None else "") + (f" — {e['why']}" if e.get("why") else "")
                                 for k, e in R["equivalence"].items()))
    if R["anchor_probe"]:
        ap = R["anchor_probe"].get("slots") or {}
        lines.append("- frozen base (R5), anchor `e4b/fused_attn4_m`: " + ", ".join(f"{s} {v.get('regime')} sha {str(v.get('sha'))[:12]} control {'detects' if v.get('control_detects_flip') else 'FAILS'}" for s, v in ap.items())
                     + ("" if R["anchor_probe"].get("control_detects_flip") else " **[anchor control did not detect its flip]**"))
    for k, fbz in R["frozen"].items():
        lines.append(f"- frozen base `{k[0]}/{k[1]}`: " + ", ".join(f"{s} **{v['reading']}**" + (f" ({v['why']})" if v.get("why") else f" ({v.get('regime')})") + f" control {'detects' if v['control'] else 'FAILS'}" for s, v in fbz.items()))
    if R["profile"]:
        pr = R["profile"]
        lines.append(f"- where Unsloth's step goes (`unsloth/ckpt_unsloth_prof`, descriptive): device busy fraction {f(pr.get('device_busy_fraction'), 3)}, device events/step {pr.get('device_events_per_step')}, "
                     f"CPU ops/step {pr.get('cpu_ops_per_step')}, CPU self by family {pr.get('cpu_self_by_family_fraction')}")
    return lines


def render(F, d):
    out = [f"# TC1 — Qwen3-30B-A3B on one RTX 5090: e4b vs Unsloth vs HF+PEFT vs axolotl at matched work, init and adapter precision ({os.path.abspath(d)})",
           f"Rule ({PREREG}): status per attempt in the vocabulary {' / '.join(VOCAB)}; VALID/VOID per tp4's predicates plus the matched-set predicates (R3: matched init complete, "
           f"fp32 adapters, step-0 held-out within {STEP0_TOL} of e4b/reference_attn4_m); VERDICT exactly one of {' / '.join(VERDICTS)} (QUALITY_FAIL = held-out |Δ| at N > {READ} "
           f"vs e4b/fused_attn4_m); positions = s/step ratios other/e4b from the medians over both draws (stability |d1-d2|/mean ≤ 5 % e4b / 10 % others, else UNSTABLE), quoted only "
           f"when both arms are VALID; equivalence vs e4b/reference_attn4_m EQUIVALENT ≤ {EQUIV} / COMPARABLE ≤ {COMPARABLE} / else DIVERGENT; frozen base SAME-BYTES / DIFFERENT / N-A per slot; "
           f"predictions P1–P10 scored HELD / FALSIFIED / UNTESTED. VOID never enters a ratio or an equivalence reading. Nothing is licensed; no cross-box number is divided into these."]
    for fn in ("versions.txt", "box.json"):
        vp = os.path.join(d, fn)
        if os.path.exists(vp):
            out.append(f"`{fn}`\n```\n" + open(vp).read().strip() + "\n```")
    for fam in list(FAMS) + sorted(set(F) - set(FAMS)):
        if fam in F:
            out += family_block(F[fam])
    out += ["\n## Verdicts", "| family | arm | VERDICT | validity | quality | note |", "|---|---|---|---|---|---|"]
    for fam in list(FAMS) + sorted(set(F) - set(FAMS)):
        R = F.get(fam)
        if not R:
            continue
        for x in R["rows"]:
            out.append(f"| {fam} | {x['fw']}/{x['tag']} | **{x['verdict']}** | {x['validity']} | {f(x.get('quality_delta'), 4) if x.get('quality_delta') is not None else (x.get('quality') or '—')} | {(x['why'] or x['reason'])[:160]} |")
    out += ["\n## Predictions P1–P10 (TC1-PREREG.md, scored mechanically)", "| prediction | family | verdict | evidence |", "|---|---|---|---|"]
    for pid, fam, v, ev in score_predictions(F):
        out.append(f"| {pid} | {fam} | **{v}** | {ev} |")
    return "\n".join(out)


def reduce_dir(d, n_steps=None):
    recs, rcs = load(d), load_summary(d)
    return {fam: reduce_family(fam, recs[fam], rcs, n_steps) for fam in list(FAMS) + sorted(set(recs) - set(FAMS)) if fam in recs}


# ----------------------------------------------------------------------------- R7: the selftest (hand-built receipts)
def _receipt(fw, tag, arm, steps=20, s=1.0, heldout0=2.0000, heldout_n=1.8000, losses=None, matched=True, seed=3407, **over):
    """A complete OK receipt carrying every field this reducer reads, valid by every predicate for qwen3 (L=48, accum 4)."""
    L, A = 48, 4
    losses = losses if losses is not None else [round(2.0 - 0.01 * i, 5) for i in range(steps)]
    r = {"framework": fw, "fam": "qwen3", "arm": arm, "tag": tag, "status": "ok", "steps": steps, "seq": 2048, "accum": A, "micro_batch": 2,
         "model": "Qwen/Qwen3-30B-A3B", "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "n_layers": L, "r": 16, "alpha": 16, "lr": 2e-4, "autocast": False,
         "template": "alpaca", "optimizer": "adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5", "env": {"box_class": "RTX 5090", "gpu": "NVIDIA GeForce RTX 5090"},
         "tokens": {"sha256": "t" * 64}, "trainable_params": 642514944, "trainable_mismatch": None, "trainable_by_group": {"attention": 1, "experts": 2, "other": 0},
         "losses": losses, "loss_first": losses[0], "loss_last": losses[-1],
         "eval_curve": [{"step": 0, "heldout_loss": heldout0}, {"step": steps, "heldout_loss": heldout_n}], "eval_loss_step0": heldout0, "eval_loss_final": heldout_n,
         "s_per_step_median_11plus": s, "peak_vram_gb": 20.0, "joules_per_step": 100.0, "tokens_per_s": 1000.0,
         "C1_bit_exact": True, "C1_bytes_hashed": 10, "C1_empties_skipped": 0, "C1_control_detects_flipped_byte": True, "init_sha": "i" * 64,
         "attn_4bit": fw == "e4b", "n_attn4": 4 * L if fw == "e4b" else 0, "structural_expected_n_attn4": 4 * L if fw == "e4b" else None,
         "n_patched": L if arm == "fused" else 0, "kernel_calls_per_step_min": 2 * L * A if arm == "fused" else 0,
         "census": {"Params4bit_expert_stacks": 2 * L if fw in ("unsloth", "axolotl") else 0, "Linear4bit": 4 * L if fw != "hf" else 0},
         "experts_forward_calls_per_step_min": L * A if fw != "e4b" else 0,
         "unsloth_bnb4bit_modules": {"n_bnb4bit_unwrapped": L} if fw == "unsloth" else None,
         "engagement_banners": ["Unsloth: Enabling LoRA on MoE parameters"] if fw == "unsloth" else [],
         "hf_targets": {"n_target_modules": 4 * L, "n_target_parameters": 2 * L} if fw in ("hf", "axolotl") else None,
         "adapter_dtype": "fp32" if matched else "native", "adapter_dtypes_after": {"torch.float32": 576} if matched else {"torch.bfloat16": 192, "torch.float32": 384},
         "lora_init": f"matched:{seed}" if matched else "native",
         "matched_init": {"seed": seed, "complete": True, "n_slots_set": 192 + 2 * L * 128, "n_slots_expected": 192 + 2 * L * 128, "unmapped": []} if matched else None,
         "frozen_base_probe": {"slots": {"gate_up": {"sha": "g" * 64, "regime": "nf4/64", "control_detects_flip": True},
                                         "down": {"sha": "d" * 64, "regime": "nf4/64", "control_detects_flip": True},
                                         "q_proj": {"sha": ("q" if fw == "e4b" else "u") * 64, "regime": "nf4/64+dq" if fw == "e4b" else "nf4/64", "control_detects_flip": True}},
                               "control_detects_flip": True, "errors": []},
         "adapter": {"bytes": 1, "dtypes": ["torch.float32"]}, "profile": None}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(r.get(k), dict) and not k.startswith("="):
            r[k] = {**r[k], **v}
        else:
            r[k.lstrip("=")] = v
    return r


def _stub(fw, tag, arm, status, reason="x"):
    return {"framework": fw, "fam": "qwen3", "arm": arm, "tag": tag, "status": status, "reason": reason, "steps": 20}


def _good_set():
    """The baseline scenario: everything that can be VALID is, Unsloth/e4b 3.0 on both draws, HF OOM twice, axolotl INSTALL_FAILED."""
    R = {}
    R[("e4b", "fused_attn4_m")] = _receipt("e4b", "fused_attn4_m", "fused", s=1.00)
    R[("e4b", "fused_attn4_m_d2")] = _receipt("e4b", "fused_attn4_m_d2", "fused", s=1.02)
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.00, heldout_n=1.8100)
    R[("unsloth", "ckpt_unsloth_m_d2")] = _receipt("unsloth", "ckpt_unsloth_m_d2", "unsloth", s=3.06, heldout_n=1.8100)
    R[("hf", "hf_peft_m")] = _stub("hf", "hf_peft_m", "hf", "oom", "OOM at step 1")
    R[("hf", "hf_peft_m_mb1")] = _stub("hf", "hf_peft_m_mb1", "hf", "oom", "OOM at step 1")
    R[("axolotl", "ckpt_axolotl_m")] = _stub("axolotl", "ckpt_axolotl_m", "axolotl", "install_failed", "pip rc=1")
    R[("axolotl", "ckpt_axolotl_best")] = _stub("axolotl", "ckpt_axolotl_best", "axolotl", "install_failed", "pip rc=1")
    R[("unsloth", "ckpt_unsloth_best")] = _receipt("unsloth", "ckpt_unsloth_best", "unsloth", s=2.40, heldout_n=1.8300, matched=False)
    R[("e4b", "fused_attn4_shipped")] = _receipt("e4b", "fused_attn4_shipped", "fused", s=0.85, heldout_n=1.7800, matched=False)
    R[("e4b", "reference_attn4_m")] = _receipt("e4b", "reference_attn4_m", "reference", s=2.00, heldout_n=1.8050)
    R[("unsloth", "ckpt_unsloth_prof")] = _receipt("unsloth", "ckpt_unsloth_prof", "unsloth", s=3.30, heldout_n=1.8100, profile={"device_busy_fraction": 0.31, "device_events_per_step": 5000, "cpu_ops_per_step": 90000, "cpu_self_by_family_fraction": {"routing": 0.4}})
    return R


def selftest():
    import tempfile
    cases = 0

    def run(R, n_steps=20):
        return reduce_family("qwen3", R, {}, n_steps)

    def row(F, fw, tag):
        return next(x for x in F["rows"] if (x["fw"], x["tag"]) == (fw, tag))

    # 1. baseline: every verdict, the quoted position, every prediction HELD
    F = run(_good_set())
    V = F["verdicts"]
    assert V[("e4b", "fused_attn4_m")] == "VALID" and V[("unsloth", "ckpt_unsloth_m")] == "VALID" and V[("e4b", "reference_attn4_m")] == "VALID", V
    assert V[("hf", "hf_peft_m")] == "OOM" and V[("axolotl", "ckpt_axolotl_m")] == "UNSUPPORTED" and V[("unsloth", "ckpt_unsloth_best")] == "VALID" and V[("e4b", "fused_attn4_shipped")] == "VALID"
    assert F["draws"][("e4b", "fused_attn4_m")]["verdict"] == "STABLE" and F["draws"][("unsloth", "ckpt_unsloth_m")]["verdict"] == "STABLE"
    pu = F["positions"]["unsloth"]
    assert pu["quoted"] and abs(pu["ratio"] - 3.03 / 1.01) < 1e-9 and pu["e4b_draws"] == 2 and pu["other_draws"] == 2 and pu["quality"] == "COMPARABLE", pu
    assert F["equivalence"][("unsloth", "ckpt_unsloth_m")]["reading"] == "EQUIVALENT" and F["equivalence"][("e4b", "fused_attn4_m")]["reading"] == "EQUIVALENT"
    fb = F["frozen"][("unsloth", "ckpt_unsloth_m")]
    assert fb["gate_up"]["reading"] == "SAME-BYTES" and fb["down"]["reading"] == "SAME-BYTES" and fb["q_proj"]["reading"] == "N-A" and "regime differs" in fb["q_proj"]["why"], fb
    assert F["native"]["unsloth"]["quoted"] and abs(F["native"]["unsloth"]["ratio"] - 2.40 / 0.85) < 1e-9
    assert F["parity"]["verdict"] == "PASS"
    P = {p: v for p, _, v, _ in score_predictions({"qwen3": F})}
    assert P == {f"P{i}": "HELD" for i in range(1, 11)}, P
    cases += 1
    # 2-7. one receipt per R3 predicate that MUST read VOID (the matched-set predicates)
    for name, over in (("incomplete", {"matched_init": {"complete": False, "n_slots_set": 100, "n_slots_expected": 192, "unmapped": ["x"]}}),
                       ("bf16", {"adapter_dtypes_after": {"torch.float32": 384, "torch.bfloat16": 192}}),
                       ("step0", {"eval_loss_step0": 2.0200, "eval_curve": [{"step": 0, "heldout_loss": 2.02}, {"step": 20, "heldout_loss": 1.81}]}),
                       ("native-on-matched", {"lora_init": "native", "matched_init": None}),
                       ("seed", {"lora_init": "matched:1", "matched_init": {"seed": 1}})):
        R = _good_set()
        R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.81, **over)
        F = run(R)
        x = row(F, "unsloth", "ckpt_unsloth_m")
        assert x["verdict"] == "VOID", (name, x["verdict"], x["why"])
        assert not F["positions"]["unsloth"]["quoted"] and F["equivalence"][("unsloth", "ckpt_unsloth_m")]["reading"] == "N-A", (name, F["positions"]["unsloth"])
        P = {p: v for p, _, v, _ in score_predictions({"qwen3": F})}
        assert P["P1"] == "UNTESTED" and P["P3"] == "UNTESTED", (name, P)
        cases += 1
    # 8-12. tp4's predicates still VOID: n_patched, C1, trainable count, unsloth banner/stacks, hf attention-only
    for name, key, over in (("n_patched", ("e4b", "fused_attn4_m"), {"n_patched": 47}),
                            ("c1", ("e4b", "fused_attn4_m"), {"C1_bit_exact": False}),
                            ("trainable", ("unsloth", "ckpt_unsloth_m"), {"trainable_params": 1}),
                            ("banner", ("unsloth", "ckpt_unsloth_m"), {"engagement_banners": []}),
                            ("stacks", ("unsloth", "ckpt_unsloth_m"), {"census": {"Params4bit_expert_stacks": 10}})):
        R = _good_set()
        base = R[key]
        R[key] = {**base, **over}
        F = run(R)
        assert row(F, *key)["verdict"] == "VOID", (name, row(F, *key)["why"])
        assert not F["positions"]["unsloth"]["quoted"], name
        cases += 1
    R = _good_set()
    R[("hf", "hf_peft_m")] = _receipt("hf", "hf_peft_m", "hf", s=5.0, hf_targets={"n_target_modules": 192, "n_target_parameters": 0})
    F = run(R)
    assert row(F, "hf", "hf_peft_m")["verdict"] == "VOID" and "attention-only" in row(F, "hf", "hf_peft_m")["why"]
    cases += 1
    # 13. QUALITY_FAIL: validity holds, held-out at N is 0.08 above the anchor -> QUALITY_FAIL, position not quoted
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.8800)
    R[("unsloth", "ckpt_unsloth_m_d2")] = _receipt("unsloth", "ckpt_unsloth_m_d2", "unsloth", s=3.0, heldout_n=1.8800)
    F = run(R)
    assert row(F, "unsloth", "ckpt_unsloth_m")["verdict"] == "QUALITY_FAIL" and row(F, "unsloth", "ckpt_unsloth_m")["validity"] == "VALID"
    assert not F["positions"]["unsloth"]["quoted"] and "QUALITY_FAIL" in F["positions"]["unsloth"]["why"]
    assert F["equivalence"][("unsloth", "ckpt_unsloth_m")]["reading"] == "DIVERGENT"     # 0.075 vs the reference's 1.805: a finding too
    cases += 1
    # 14. UNSTABLE: the e4b draws differ by 12 % -> UNSTABLE, no position, P2 FALSIFIED
    R = _good_set()
    R[("e4b", "fused_attn4_m_d2")] = _receipt("e4b", "fused_attn4_m_d2", "fused", s=1.13)
    F = run(R)
    d = F["draws"][("e4b", "fused_attn4_m")]
    assert d["verdict"] == "UNSTABLE" and abs(d["stability"] - 0.13 / 1.065) < 1e-9 and not d["usable"], d
    assert not F["positions"]["unsloth"]["quoted"] and "UNSTABLE" in F["positions"]["unsloth"]["why"]
    P = {p: v for p, _, v, _ in score_predictions({"qwen3": F})}
    assert P["P2"] == "FALSIFIED" and P["P1"] == "UNTESTED", P
    cases += 1
    # 15. DIVERGENT: unsloth_m's train losses sit 0.10 above the reference's -> DIVERGENT, P3 FALSIFIED (position still arithmetic-quoted; the decision rule is stated)
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.81, losses=[round(2.1 - 0.01 * i, 5) for i in range(20)])
    F = run(R)
    e = F["equivalence"][("unsloth", "ckpt_unsloth_m")]
    assert e["reading"] == "DIVERGENT" and abs(e["med_train"] - 0.1) < 1e-9, e
    P = {p: v for p, _, v, _ in score_predictions({"qwen3": F})}
    assert P["P3"] == "FALSIFIED", P
    cases += 1
    # 16. COMPARABLE (0.03 on both): not EQUIVALENT, not DIVERGENT -> P3 FALSIFIED with the decision-rule note
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = _receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.8350, losses=[round(2.03 - 0.01 * i, 5) for i in range(20)])
    F = run(R)
    assert F["equivalence"][("unsloth", "ckpt_unsloth_m")]["reading"] == "COMPARABLE"
    pe = next(x for x in score_predictions({"qwen3": F}) if x[0] == "P3")
    assert pe[2] == "FALSIFIED" and "does not fire" in pe[3], pe
    cases += 1
    # 17. the status->verdict map: OOM, UNSUPPORTED (refused / install_failed / load_fault), ALARM (alarm + phase_alarm), HARNESS_ERROR, NOT_RUN, c1_failed -> VOID
    for st, want in (("oom", "OOM"), ("refused", "UNSUPPORTED"), ("install_failed", "UNSUPPORTED"), ("load_fault", "UNSUPPORTED"), ("verify_failed", "UNSUPPORTED"),
                     ("alarm", "ALARM"), ("phase_alarm", "ALARM"), ("harness_error", "HARNESS_ERROR"), ("tokens_mismatch", "HARNESS_ERROR"), ("not_run", "NOT_RUN")):
        R = _good_set()
        R[("unsloth", "ckpt_unsloth_m")] = _stub("unsloth", "ckpt_unsloth_m", "unsloth", st)
        assert row(run(R), "unsloth", "ckpt_unsloth_m")["verdict"] == want, (st, want)
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")] = {**_receipt("unsloth", "ckpt_unsloth_m", "unsloth", s=3.0, heldout_n=1.81), "status": "c1_failed"}
    assert row(run(R), "unsloth", "ckpt_unsloth_m")["verdict"] == "VOID"
    assert row(run({}), "unsloth", "ckpt_unsloth_m")["verdict"] == "NOT_RUN"
    cases += 1
    # 18. a missing second draw: stability UNMEASURED, position not quoted, P2 UNTESTED
    R = _good_set()
    del R[("unsloth", "ckpt_unsloth_m_d2")]
    F = run(R)
    assert F["draws"][("unsloth", "ckpt_unsloth_m")]["verdict"] == "UNMEASURED" and not F["positions"]["unsloth"]["quoted"]
    P = {p: v for p, _, v, _ in score_predictions({"qwen3": F})}
    assert P["P1"] == "UNTESTED" and P["P2"] == "HELD", P   # e4b's pair still agrees; unsloth's cannot be read
    cases += 1
    # 19. frozen base: DIFFERENT expert bytes and SAME attention bytes -> P10 FALSIFIED; a missing slot -> N-A
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_m")]["frozen_base_probe"]["slots"]["gate_up"]["sha"] = "z" * 64
    F = run(R)
    assert F["frozen"][("unsloth", "ckpt_unsloth_m")]["gate_up"]["reading"] == "DIFFERENT"
    P = {p: v for p, _, v, _ in score_predictions({"qwen3": F})}
    assert P["P10"] == "FALSIFIED", P
    R = _good_set()
    del R[("unsloth", "ckpt_unsloth_m")]["frozen_base_probe"]["slots"]["down"]
    F = run(R)
    assert F["frozen"][("unsloth", "ckpt_unsloth_m")]["down"]["reading"] == "N-A"
    assert {p: v for p, _, v, _ in score_predictions({"qwen3": F})}["P10"] == "UNTESTED"
    cases += 1
    # 20. the other predictions' FALSIFIED legs: P1 below 1.5, P4 matched as good, P5 hf trains, P7 best too fast, P8 device-bound
    R = _good_set()
    for k in (("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_m_d2")):
        R[k]["s_per_step_median_11plus"] = 1.2
    pe = next(x for x in score_predictions({"qwen3": run(R)}) if x[0] == "P1")
    assert pe[2] == "FALSIFIED" and "BELOW 1.5" in pe[3], pe
    R = _good_set()
    R[("e4b", "fused_attn4_shipped")]["eval_curve"][1]["heldout_loss"] = 1.8000
    assert {p: v for p, _, v, _ in score_predictions({"qwen3": run(R)})}["P4"] == "FALSIFIED"
    R = _good_set()
    R[("hf", "hf_peft_m")] = _receipt("hf", "hf_peft_m", "hf", s=6.0, heldout_n=1.81)
    assert {p: v for p, _, v, _ in score_predictions({"qwen3": run(R)})}["P5"] == "FALSIFIED"
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_best")]["s_per_step_median_11plus"] = 1.5
    assert {p: v for p, _, v, _ in score_predictions({"qwen3": run(R)})}["P7"] == "FALSIFIED"
    R = _good_set()
    R[("unsloth", "ckpt_unsloth_prof")]["profile"]["device_busy_fraction"] = 0.6
    assert {p: v for p, _, v, _ in score_predictions({"qwen3": run(R)})}["P8"] == "FALSIFIED"
    cases += 1
    # 21. axolotl trains: P6 reads the ratio; the mb1 secondary pair when the primary OOMed
    R = _good_set()
    R[("axolotl", "ckpt_axolotl_m")] = _receipt("axolotl", "ckpt_axolotl_m", "axolotl", s=4.0, heldout_n=1.81)
    F = run(R)
    assert F["positions"]["axolotl"]["quoted"] and {p: v for p, _, v, _ in score_predictions({"qwen3": F})}["P6"] == "HELD"
    R[("axolotl", "ckpt_axolotl_m")]["s_per_step_median_11plus"] = 9.0
    assert {p: v for p, _, v, _ in score_predictions({"qwen3": run(R)})}["P6"] == "FALSIFIED"
    R = _good_set()
    R[("e4b", "fused_attn4_m_mb1")] = _receipt("e4b", "fused_attn4_m_mb1", "fused", s=1.1, accum=8, micro_batch=1, kernel_calls_per_step_min=2 * 48 * 8)
    R[("hf", "hf_peft_m_mb1")] = _receipt("hf", "hf_peft_m_mb1", "hf", s=7.7, heldout_n=1.81, accum=8, micro_batch=1, experts_forward_calls_per_step_min=48 * 8)
    F = run(R)
    assert F["secondary"]["hf"]["quoted"] and abs(F["secondary"]["hf"]["ratio"] - 7.0) < 1e-9, F["secondary"]["hf"]
    cases += 1
    # 22. end to end through the files: the printer renders every section and the markdown says VERDICT
    d = tempfile.mkdtemp(prefix="tc1_reduce_selftest_")
    for (fw, tag), r in _good_set().items():
        json.dump(r, open(os.path.join(d, f"qwen3_{fw}_{tag}.json"), "w"))
    open(os.path.join(d, "summary.txt"), "w").write("qwen3/hf/hf_peft_m rc=5 CELL OOM\n")
    F = reduce_dir(d, 20)
    text = render(F, d)
    for needle in ("**VALID**", "**OOM**", "**UNSUPPORTED**", "MATCHED POSITION: s/step ratio unsloth/e4b = 3.000", "EQUIVALENT", "SAME-BYTES", "NATIVE-BEST", "| P10 | qwen3 | **HELD** |", "draws (R1)"):
        assert needle in text, needle
    assert "qwen3/hf/hf_peft_m rc=5" not in text   # the summary line feeds status_of only when the receipt is missing
    cases += 1
    # 23. the registered arm order is the support table's row order
    F = run(_good_set())
    order = [(x["fw"], x["tag"]) for x in F["rows"]]
    assert order[:len(EXPECTED["qwen3"])] == EXPECTED["qwen3"] and order[len(EXPECTED["qwen3"]):] == [("hf", "hf_peft_m_mb1")], order   # unexpected receipts follow, never vanish
    assert set(VERDICTS) == {"VALID", "VOID", "QUALITY_FAIL", "OOM", "UNSUPPORTED", "HARNESS_ERROR", "ALARM", "NOT_RUN"}
    cases += 1
    print(f"REDUCE SELFTEST OK cases={cases} dir={d}")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--md", default=None)
    ap.add_argument("--steps", type=int, default=None, help="registered N (default: read from the receipts)")
    ap.add_argument("--selftest", action="store_true", help="R7: hand-built receipts through every reading; exit 0 iff every case reads as registered")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.dir:
        ap.error("dir is required (or --selftest)")
    F = reduce_dir(a.dir, a.steps)
    text = render(F, a.dir)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
