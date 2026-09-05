#!/usr/bin/env python3
"""p38_reduce.py -- the position table for lane p38 in the pre-registered form (P38-UNSLOTH-PREREG.md, "Verdict rule").

Reads <framework>_train_<tag>.json (p38_arm.py receipts and stubs) and prints: the arm table with VALID/VOID and the
reason (trainable count, adapters-only, tokens sha, engagement, C1, step count), the e4b internal parity (fused_attn4
vs reference_attn4, B2/C2 units, informational), the PRIMARY PAIR block (unsloth_ckpt_unsloth vs e4b_fused_attn4:
s/step ratio, peak VRAM, tok/s, J/step with the perturbation caveat, held-out loss at every shared eval with the
0.05 reading threshold, time-to-target, adapter bytes), the secondary rows, and the 200-step curves. It licenses
nothing and quotes no cross-box number. stdlib only.  Usage: p38_reduce.py <dir> [--md out.md] [--target 0.32]
"""
import argparse
import glob
def _set_u8_proof(path):
    global U8_PROOF
    U8_PROOF = json.load(open(path)) if path else None
U8_PROOF = None  # amendment 4: u8_proof.json (wrapper vs innermost predicate), loaded from --u8-proof
import json
import os
import statistics

EXPECTED = 321_257_472
BAND = 0.05
READ = 0.05


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_train_*.json"))):
        base = os.path.basename(p)[:-5]
        fw, _, tag = base.partition("_train_")
        try:
            out[(fw, tag)] = json.load(open(p))
        except Exception as e:
            out[(fw, tag)] = {"framework": fw, "tag": tag, "status": "unreadable", "reason": str(e)}
    return out


def f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def validity(r, tokens_sha):
    if r.get("status") != "ok":
        return r.get("status", "?").upper(), r.get("reason", "")[:200]
    why = []
    if r.get("trainable_params") != EXPECTED:
        why.append(f"trainable {r.get('trainable_params')} != {EXPECTED}")
    if tokens_sha and r.get("tokens", {}).get("sha256") != tokens_sha:
        why.append("tokens sha differs")
    if not (r.get("C1_bit_exact") and r.get("C1_bytes_hashed", 0) > 0 and r.get("C1_empties_skipped", 1) == 0 and r.get("C1_control_detects_flipped_byte")):
        why.append("C1 not clean")
    if r["framework"] == "e4b":
        if r.get("arm") == "fused" and (r.get("n_patched") != 48 or r.get("kernel_calls_per_step_min", 0) < 96):
            why.append(f"fused engagement: n_patched {r.get('n_patched')} kcalls/step min {r.get('kernel_calls_per_step_min')} (need 48 / >=96)")
        if r.get("attn_4bit") and r.get("n_attn4") != 192:
            why.append(f"attn4 {r.get('n_attn4')} != 192")
    else:
        c = r.get("census", {})
        if c.get("Params4bit_expert_stacks", 0) < 96:
            why.append(f"Params4bit expert stacks {c.get('Params4bit_expert_stacks')} < 96")
        if r.get("experts_forward_calls_per_step_min", 0) < 48:
            why.append(f"experts forward calls/step min {r.get('experts_forward_calls_per_step_min')} < 48")
        b = r.get("unsloth_bnb4bit_modules")
        if isinstance(b, dict):
            if "n_bnb4bit_unwrapped" in b:            # amendment 4: the predicate on the innermost experts module
                if b["n_bnb4bit_unwrapped"] < 48:
                    why.append(f"bnb4bit expert modules (innermost) {b['n_bnb4bit_unwrapped']} < 48 (silent fallback?)")
            elif b.get("n_bnb4bit", 0) < 48:
                proof = U8_PROOF or {}
                inner_ok = (proof.get("after_peft_innermost") or {}).get("n_pred_true", 0) >= 48
                wrapper_zero = (proof.get("after_peft_wrapper") or {}).get("n_pred_true", 1) == 0
                own = c.get("Params4bit_expert_stacks", 0) >= 96 and r.get("kernel_calls_per_step_min", 0) >= 96
                if not (inner_ok and wrapper_zero and own):
                    why.append(f"bnb4bit expert modules {b.get('n_bnb4bit')} < 48 (silent fallback?)")
        if not any("Enabling LoRA on MoE parameters" in s for s in r.get("engagement_banners", [])):
            why.append("no 'Enabling LoRA on MoE parameters' banner (census-only evidence)")
    if len(r.get("losses", [])) != r.get("steps"):
        why.append("step count differs")
    return ("VOID" if why else "VALID"), "; ".join(why)


def curve_at(r, step):
    for c in r.get("eval_curve", []):
        if c["step"] == step:
            return c["heldout_loss"]
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--u8-proof", default=None, help="amendment 4: u8_proof.json for receipts written before the unwrapped U8 count")
    ap.add_argument("--md", default=None); ap.add_argument("--target", type=float, default=0.32)
    a = ap.parse_args(); _set_u8_proof(a.u8_proof); R = load(a.dir)
    tsha = None
    for r in R.values():
        tsha = tsha or r.get("tokens", {}).get("sha256")
    out = [f"# p38 -- e4b vs Unsloth QLoRA end-to-end, one box, one training problem ({os.path.abspath(a.dir)})",
           f"Rule: positions in registered units (s/step median of steps 11..N, GB allocator peak, tok/s, J/step, held-out loss in nats); VOID arms never enter a ratio; "
           f"cross-framework held-out |Δ| ≤ {READ} reads 'comparable quality' (a reading threshold, not a gate) and outside it time-to-target ({a.target}) is the headline; "
           "the e4b fused-vs-reference band (0.05/0.05) is informational here (tp1 owns it). Nothing is licensed; no cross-box number is divided into these.",
           f"tokens sha: {tsha}"]
    vp = os.path.join(a.dir, "versions.txt")
    if os.path.exists(vp):
        out.append("```\n" + open(vp).read().strip() + "\n```")
    out.append("| framework | arm | status | validity | N | s/step med(11+) | s/step mean | tok/s | peak GB | J/step | train first→last | held-out 0→final | t→target s | adapter MB (dtype) | trainable | ckpt | notes |")
    out.append("|" + "---|" * 17)
    V = {}
    for (fw, tag), r in R.items():
        st, why = validity(r, tsha); V[(fw, tag)] = st
        ad = r.get("adapter", {}) or {}
        out.append(f"| {fw} | {tag} | {r.get('status')} | **{st}** | {r.get('steps')} | {f(r.get('s_per_step_median_11plus'))} | {f(r.get('s_per_step'))} | {f(r.get('tokens_per_s'),1)} | "
                   f"{f(r.get('peak_vram_gb'))} | {f(r.get('joules_per_step'),1)} | {f(r.get('loss_first'),4)}→{f(r.get('loss_last'),4)} | {f(r.get('eval_loss_step0'),4)}→{f(r.get('eval_loss_final'),4)} | "
                   f"{f(r.get('time_to_target_s'),1) if r.get('time_to_target_s') is not None else (r.get('time_to_target_note') or '—')} | "
                   f"{(ad.get('bytes') or 0)/1e6:.1f} ({','.join(ad.get('dtypes', []) or ['?'])}) | {r.get('trainable_params')} | {r.get('grad_ckpt')} | {why} |")
    # energy perturbation control
    ns = R.get(("e4b", "fused_attn4_nosamp")); pr = R.get(("e4b", "fused_attn4"))
    caveat = ""
    if ns and pr and ns.get("status") == "ok" and pr.get("status") == "ok":
        m_ns = statistics.median(ns["step_ms"][10:20]) if len(ns.get("step_ms", [])) >= 20 else None
        m_pr = statistics.median(pr["step_ms"][10:20]) if len(pr.get("step_ms", [])) >= 20 else None
        if m_ns and m_pr:
            d = abs(m_pr / m_ns - 1)
            caveat = f"sampler perturbation {d*100:.2f}% (steps 11..20: sampled {m_pr:.0f} ms vs unsampled {m_ns:.0f} ms) -> " + ("energy reported without caveat" if d <= 0.02 else "ENERGY CARRIES A CAVEAT (>2%) on every row")
            out.append(f"\n- energy control: {caveat}")
    # e4b internal parity (informational)
    ref = R.get(("e4b", "reference_attn4")); fu = R.get(("e4b", "fused_attn4"))
    if ref and fu and ref.get("status") == "ok" and fu.get("status") == "ok" and V[("e4b", "reference_attn4")] == "VALID" and V[("e4b", "fused_attn4")] == "VALID":
        d_final = abs(fu["loss_last"] - ref["loss_last"]); med = statistics.median(abs(x - y) for x, y in zip(fu["losses"], ref["losses"]))
        out.append(f"- e4b internal parity (informational; tp1 owns the licence): fused_attn4 vs reference_attn4 Δfinal {d_final:.5f}, median step |Δ| {med:.5f} -> "
                   f"**{'PASS' if d_final <= BAND and med <= BAND else 'FAIL'}** in the B2/C2 band; cost ×{ref['s_per_step_median_11plus']/fu['s_per_step_median_11plus']:.2f} faster per step, peak ×{fu['peak_vram_gb']/ref['peak_vram_gb']:.3f}")
    # the primary pair
    e = R.get(("e4b", "fused_attn4")); u = R.get(("unsloth", "ckpt_unsloth"))
    out.append("\n## Primary pair -- Unsloth (4-bit MoE, 'unsloth' checkpointing) vs e4b (fused dgrad path + NF4 attention), N=60")
    if e and u and V.get(("e4b", "fused_attn4")) == "VALID" and V.get(("unsloth", "ckpt_unsloth")) == "VALID":
        rs = u["s_per_step_median_11plus"] / e["s_per_step_median_11plus"]
        out.append(f"- **s/step ratio unsloth/e4b = {rs:.3f}** ({u['s_per_step_median_11plus']:.3f} vs {e['s_per_step_median_11plus']:.3f} s; {'e4b faster' if rs > 1 else 'Unsloth faster'} per step at this workload)")
        out.append(f"- peak VRAM: unsloth {u['peak_vram_gb']:.2f} GB vs e4b {e['peak_vram_gb']:.2f} GB (Δ {u['peak_vram_gb']-e['peak_vram_gb']:+.2f} GB)")
        out.append(f"- tokens/s: unsloth {u['tokens_per_s']:.1f} vs e4b {e['tokens_per_s']:.1f}; J/step: unsloth {f(u.get('joules_per_step'),1)} vs e4b {f(e.get('joules_per_step'),1)} {('(' + caveat + ')') if caveat else ''}")
        rows = []
        for s in sorted({c['step'] for c in e.get('eval_curve', [])} & {c['step'] for c in u.get('eval_curve', [])}):
            le, lu = curve_at(e, s), curve_at(u, s)
            rows.append(f"{s}: e4b {le:.4f} / unsloth {lu:.4f} (Δ {lu-le:+.4f}{' ≤' if abs(lu-le) <= READ else ' >'} {READ})")
        out.append("- held-out loss at shared evals: " + "; ".join(rows))
        d60 = None
        if curve_at(e, e['steps']) is not None and curve_at(u, u['steps']) is not None:
            d60 = curve_at(u, u['steps']) - curve_at(e, e['steps'])
        out.append(f"- reading: {'COMPARABLE QUALITY at N=60 (|Δ| ≤ ' + str(READ) + ') -- the s/step ratio is the position' if d60 is not None and abs(d60) <= READ else 'held-out losses differ by more than the reading threshold -- time-to-target is the headline, s/step is quoted WITH the loss delta'}")
        e2 = R.get(("e4b", "fused_attn4_200")); u2 = R.get(("unsloth", "ckpt_unsloth_200"))
        tt = []
        for name, r0, r2 in (("e4b", e, e2), ("unsloth", u, u2)):
            t = r0.get("time_to_target_s") if r0.get("time_to_target_s") is not None else (r2.get("time_to_target_s") if r2 else None)
            src = "60-step arm" if r0.get("time_to_target_s") is not None else ("200-step arm" if r2 and r2.get("time_to_target_s") is not None else "not reached")
            tt.append(f"{name} {f(t,1)} s ({src})")
        out.append(f"- time-to-target (held-out ≤ {a.target}, cumulative training wall, evals excluded): " + "; ".join(tt))
        out.append(f"- adapters: e4b {(e.get('adapter',{}).get('bytes') or 0)/1e6:.1f} MB {e.get('adapter',{}).get('dtypes')} / unsloth {(u.get('adapter',{}).get('bytes') or 0)/1e6:.1f} MB {u.get('adapter',{}).get('dtypes')}; both {EXPECTED} parameters (asserted)")
        out.append(f"- step-0 held-out (the two quantisers on the same bytes, B=0): e4b {e['eval_loss_step0']:.4f} vs unsloth {u['eval_loss_step0']:.4f} (Δ {u['eval_loss_step0']-e['eval_loss_step0']:+.4f})")
    else:
        out.append(f"- **NO POSITION QUOTED** -- primary arms: e4b fused_attn4 {V.get(('e4b','fused_attn4'),'missing')}, unsloth ckpt_unsloth {V.get(('unsloth','ckpt_unsloth'),'missing')} (see the table; a refusal/OOM is a row, the workload is not weakened)")
    # secondary rows
    out.append("\n## Secondary rows (same columns; never the headline)")
    for fw, tag, label in (("e4b", "fused", "e4b tp1 fixture (bf16 attention)"), ("unsloth", "ckpt_hf", "Unsloth with HF checkpointing (the semantic match to e4b's mode)")):
        r = R.get((fw, tag))
        if r and r.get("status") == "ok":
            base = e if fw == "e4b" else u
            rel = f"; vs the primary {fw} arm: s/step ×{r['s_per_step_median_11plus']/base['s_per_step_median_11plus']:.3f}, peak {r['peak_vram_gb']-base['peak_vram_gb']:+.2f} GB" if base and base.get("status") == "ok" else ""
            out.append(f"- {label}: {V.get((fw,tag))} s/step {r['s_per_step_median_11plus']:.3f}, peak {r['peak_vram_gb']:.2f} GB, held-out {r['eval_loss_step0']:.4f}→{r['eval_loss_final']:.4f}{rel}")
        elif r:
            out.append(f"- {label}: {r.get('status','?').upper()} {r.get('reason','')[:160]}")
    # curves
    out.append("\n## 200-step curves (held-out loss at every eval; cumulative training wall)")
    for fw, tag in (("e4b", "fused_attn4_200"), ("unsloth", "ckpt_unsloth_200")):
        r = R.get((fw, tag))
        if r and r.get("eval_curve"):
            out.append(f"- {fw} ({V.get((fw,tag))}): " + " · ".join(f"{c['step']}:{c['heldout_loss']:.4f}@{c['train_wall_s']:.0f}s" for c in r["eval_curve"]))
        elif r:
            out.append(f"- {fw}: {r.get('status','?').upper()} {r.get('reason','')[:160]}")
    text = "\n".join(out); print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
