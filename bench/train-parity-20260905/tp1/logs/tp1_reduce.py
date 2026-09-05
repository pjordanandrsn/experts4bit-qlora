#!/usr/bin/env python3
"""tp1_reduce.py -- the parity table for lane tp1, verdicts in the REGISTERED units.

Reads <fam>_train_<arm>.json (tp1_train_smoke.py receipts and stubs) plus the gpt-oss
EXPERIMENTAL envelope gptoss_train_mxfp4.json, and prints, per family, the load row, one
row per arm, and the verdict column computed exactly as P36-PREREG.md registers it:

  VOID  if init_sha != the reference's, C1 not bit-exact / 0 bytes / empties > 0 / control
        silent (either arm), n_patched == 0 on fused/batched, kernel_calls_per_step_min
        < 2 * n_patched (the patched forward did not reach the kernel on every layer --
        batched.py's silent _PAD_WASTE_LIMIT fallback), or a different step count.
  PASS  iff |loss_last_arm - loss_last_ref| <= 0.05 AND median_i |loss_i_arm - loss_i_ref|
        <= 0.05 (PREREG-flagship-matrix B2 / -model2 C2).  FAIL otherwise.
Cost (s/step, tok/s, peak GB, J/step) is reported, never gated. Eval deltas are printed
labelled "not the band". stdlib only. Usage: tp1_reduce.py <dir> [--md out.md]
"""
import argparse
import glob
import json
import os
import statistics

BAND = 0.05
FAMS = ["granite", "olmoe", "gptoss", "qwen3", "gemma4", "mixtral"]
NAMES = {"granite": "Granite-3.1-3B-A800M", "olmoe": "OLMoE-1B-7B-0924-Instruct",
         "gptoss": "gpt-oss-20b", "qwen3": "Qwen3-30B-A3B", "gemma4": "Gemma-4-26B-A4B-it",
         "mixtral": "Mixtral-8x7B-Instruct-v0.1"}
ARM_ORDER = ["load", "reference", "fused", "batched", "attn_only", "resident_probe", "mxfp4"]


def load_dir(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_train_*.json"))):
        base = os.path.basename(p)[:-5]
        fam, _, arm = base.partition("_train_")
        try:
            out.setdefault(fam, {})[arm] = json.load(open(p))
        except Exception as e:
            out.setdefault(fam, {})[arm] = {"fam": fam, "arm": arm, "status": "unreadable", "reason": str(e)}
    return out


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def c1_ok(r):
    return bool(r.get("C1_bit_exact")) and r.get("C1_bytes_hashed", 0) > 0 \
        and r.get("C1_empties_skipped", 1) == 0 and bool(r.get("C1_control_detects_flipped_byte"))


def verdict(ref, arm):
    """(verdict, d_final, med, why) in registered units. Never touches eval loss."""
    st = arm.get("status")
    if st != "ok":
        return st.upper(), None, None, arm.get("reason", "")[:160]
    if ref is None or ref.get("status") != "ok":
        return "NO-REF", None, None, "reference arm missing or not ok"
    why = []
    if arm.get("init_sha") != ref.get("init_sha"):
        why.append("init_sha differs from reference (arms did not start identical)")
    for tag, r in (("ref", ref), ("arm", arm)):
        if not c1_ok(r):
            why.append(f"C1 not clean on {tag}")
    if arm["arm"] in ("fused", "batched"):
        if arm.get("n_patched", 0) == 0:
            why.append("n_patched == 0")
        need = 2 * arm.get("n_patched", 0)
        if arm.get("kernel_calls_per_step_min", 0) < need:
            why.append(f"kernel calls/step min {arm.get('kernel_calls_per_step_min')} < 2*n_patched={need} (not engaged on every layer)")
    if len(arm.get("losses", [])) != len(ref.get("losses", [])) or not arm.get("losses"):
        why.append("step counts differ")
    if why:
        return "VOID", None, None, "; ".join(why)
    d_final = abs(arm["loss_last"] - ref["loss_last"])
    med = statistics.median(abs(x - y) for x, y in zip(arm["losses"], ref["losses"]))
    ok = d_final <= BAND and med <= BAND
    return ("PASS" if ok else "FAIL"), d_final, med, ""


def ratio(num, den):
    try:
        return num / den if den else None
    except Exception:
        return None


def family_block(fam, recs):
    lines = []
    lines.append(f"\n### {NAMES.get(fam, fam)} (`{fam}`)")
    ld = recs.get("load") or next((r for a, r in recs.items() if r.get("status") == "ok"), None)
    if ld:
        g = ld.get("geometry", {})
        lines.append(f"- load: status={ld.get('status')} model_type={ld.get('model_type')} load_s={fmt(ld.get('load_s'),1)} "
                     f"loaded_gb={fmt(ld.get('loaded_gb'))} verify={ld.get('verify')} wrapped={ld.get('n_lora_wrapped')} "
                     f"bare={ld.get('n_bare_experts')} classes={ld.get('expert_classes')} offload={ld.get('offload')} "
                     f"geometry(H/I/L/E/k)={g.get('hidden_size')}/{g.get('moe_intermediate_size') or g.get('intermediate_size')}/"
                     f"{g.get('num_hidden_layers')}/{g.get('num_experts') or g.get('num_local_experts')}/{g.get('num_experts_per_tok')}")
        env = ld.get("env", {})
        if env:
            lines.append(f"- env: e4b {env.get('e4b')} gnf4 {env.get('gnf4')} torch {env.get('torch')} transformers {env.get('transformers')} "
                         f"gpu {env.get('gpu')} box_class {env.get('box_class')} host_mem_gib {env.get('host', {}).get('host_mem_gib')}")
    else:
        lines.append("- load: (no receipt)")
    ref = recs.get("reference")
    hdr = ("| arm | status | n_patched | kcalls/step min (need) | C1 | steps | s/step | tok/s | peak GB | J/step | "
           "loss first→last | eval 0→final (not the band) | Δ final train | median step \\|Δ\\| | verdict (registered units) |")
    lines.append(hdr)
    lines.append("|" + "---|" * 15)
    for arm in ARM_ORDER:
        if arm == "load" or arm not in recs:
            continue
        r = recs[arm]
        if arm == "mxfp4":
            art = r.get("artifact", {}) or {}
            can = art.get("canary", {}) or {}
            prov = art.get("provenance", {}) or {}
            ev = art.get("eval_loss", {}) or {}
            evk = sorted(ev, key=lambda k: int(k))
            lines.append(f"| mxfp4 (gnf4 ExpertsMxfp4LoRA, own text) | {r.get('status')} | — | — | "
                         f"prov pre==post: {prov.get('pre_equals_post')} | {len(r.get('losses', []))} | "
                         f"{fmt(statistics.mean(r['step_s']) if r.get('step_s') else None)} | — | {fmt(r.get('peak_vram_gb'))} | — | "
                         f"{fmt(r['losses'][0] if r.get('losses') else None)}→{fmt(r['losses'][-1] if r.get('losses') else None)} | "
                         f"{fmt(ev.get(evk[0]) if evk else None)}→{fmt(ev.get(evk[-1]) if evk else None)} | — | — | "
                         f"**EXPERIMENTAL — NOT LICENSED** (canary top1={fmt(can.get('top1'))} kl={fmt(can.get('kl'),5)}) |")
            continue
        v, d_final, med, why = verdict(ref, r) if arm in ("fused", "batched") else (
            (r.get("status", "?").upper() if r.get("status") != "ok" else ("REF" if arm == "reference" else "NO-PAIR")), None, None, r.get("reason", "")[:160])
        need = 2 * r.get("n_patched", 0) if arm in ("fused", "batched") else 0
        c1 = "ok" if c1_ok(r) else ("—" if r.get("status") != "ok" else "**FAIL**")
        sp = r.get("s_per_step"); pk = r.get("peak_vram_gb"); jj = r.get("joules_per_step")
        cost = ""
        if ref and ref.get("status") == "ok" and r.get("status") == "ok" and arm != "reference":
            cost = (f" (×{fmt(ratio(ref.get('s_per_step'), sp),2)} faster, peak ×{fmt(ratio(pk, ref.get('peak_vram_gb')),3)}"
                    + (f", J ×{fmt(ratio(jj, ref.get('joules_per_step')),3)}" if jj and ref.get('joules_per_step') else "") + ")")
        ev_d = ""
        if ref and r.get("status") == "ok" and arm != "reference" and ref.get("eval_loss_final") is not None and r.get("eval_loss_final") is not None:
            ev_d = f" (Δ vs ref {r['eval_loss_final'] - ref['eval_loss_final']:+.4f})"
        lines.append(f"| {arm} | {r.get('status')} | {r.get('n_patched', 0)} | {r.get('kernel_calls_per_step_min', '—')} ({need}) | {c1} | "
                     f"{len(r.get('losses', [])) or r.get('steps_done', 0)} | {fmt(sp, 3)} | {fmt(r.get('tokens_per_s'), 1)} | {fmt(pk)} | {fmt(jj, 1)} | "
                     f"{fmt(r.get('loss_first'), 4)}→{fmt(r.get('loss_last'), 4)} | {fmt(r.get('eval_loss_step0'), 4)}→{fmt(r.get('eval_loss_final'), 4)}{ev_d} | "
                     f"{fmt(d_final, 5)} | {fmt(med, 5)} | **{v}**{cost}{(' — ' + why) if why else ''} |")
    if ref and ref.get("status") == "ok":
        lines.append(f"- init_sha (reference) `{ref.get('init_sha', '')[:16]}`; dataset `{ref.get('dataset', {}).get('path')}` sha "
                     f"`{ref.get('dataset', {}).get('sha256', '')[:12]}`; trainable {ref.get('trainable_params')} params in {ref.get('trainable_tensors')} tensors")
    return lines


def summary(all_recs):
    lines = ["\n## Cross-family summary (verdicts in registered units; cost ratios within family, this box)",
             "| family | reference | fused | batched | other rows |", "|---|---|---|---|---|"]
    for fam in FAMS:
        recs = all_recs.get(fam)
        if not recs:
            lines.append(f"| {NAMES[fam]} | (no receipts) | | | |")
            continue
        ref = recs.get("reference")
        cells = []
        for arm in ("reference", "fused", "batched"):
            r = recs.get(arm)
            if r is None:
                cells.append("not run")
                continue
            if arm == "reference":
                cells.append(f"{r.get('status')} {fmt(r.get('s_per_step'), 2)} s/step, {fmt(r.get('peak_vram_gb'), 2)} GB" if r.get("status") == "ok" else r.get("status", "?").upper())
                continue
            v, d, m, why = verdict(ref, r)
            extra = f" ({fmt(d, 4)}/{fmt(m, 4)}; ×{fmt(ratio(ref.get('s_per_step'), r.get('s_per_step')), 2)})" if v in ("PASS", "FAIL") else ""
            cells.append(f"**{v}**{extra}")
        others = [f"{a}: {r.get('status')}" + (" EXPERIMENTAL" if a == "mxfp4" else "") for a, r in recs.items()
                  if a not in ("load", "reference", "fused", "batched")]
        lines.append(f"| {NAMES[fam]} | " + " | ".join(cells) + f" | {'; '.join(others) or '—'} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    all_recs = load_dir(a.dir)
    out = [f"# tp1 — training parity table ({os.path.abspath(a.dir)})",
           f"Verdict rule: PASS iff |Δ final train loss| ≤ {BAND} AND median step-wise |Δ| ≤ {BAND} vs the family's reference arm "
           "(PREREG-flagship-matrix B2 / -model2 C2); VOID when the arm cannot be read (init_sha, C1, n_patched, kernel engagement, steps). "
           "Cost is reported, not gated. Eval deltas are not the band. No cross-family, cross-lane or cross-card ratio appears here."]
    for fam in FAMS:
        if fam in all_recs:
            out += family_block(fam, all_recs[fam])
    for fam in sorted(set(all_recs) - set(FAMS)):
        out += family_block(fam, all_recs[fam])
    out += summary(all_recs)
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
