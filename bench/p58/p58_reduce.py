#!/usr/bin/env python3
"""p58_reduce.py -- the head-to-head table for lane P58 (bench/p58/P58-PREREG.md, "Verdict rule"), P37's reducer
re-shaped for P58's arms: e4b `nf4_*` (control) and `int4_*` (P54's current int4 stack: fused q/k/v at B=1, unfused
at B=16), and vLLM receipts from TWO builds, `vllm_primary_*` (latest PyPI at launch) and `vllm_secondary_*`
(0.29.0 pinned). Prints per batch size the arm table (VALID/VOID with the pre-registered reason), the self-pairs,
the RATIO vLLM/e4b for each vLLM build from its primary pair (only when both sides' self-pairs are inside 1.03x and
both receipts carry the same prompts_sha), the build-to-build delta, e4b's three axes, and the secondary rows. It
licenses nothing and quotes no cross-box number. stdlib only.
Usage: p58_reduce.py <dir> [--md out.md]
"""
import argparse
import glob
import json
import os
import re

SELF_PAIR = 1.03
ANCHOR_B1 = 159.2          # tok/s, the NF4 anchor-class ceiling -- a PROJECTION base only; the class was never certified
P54 = {1: 3.693, 16: 11.420}   # P54's same-box int4 medians (ms/step; B=1 fused, B=16 unfused), cited beside, never divided into
NEED_STEPS = {1: 127, 16: 70}
BUILDS = ("primary", "secondary")


def jload(path):
    try:
        return json.load(open(path))
    except Exception as e:  # noqa: BLE001
        return {"status": "unreadable", "reason": str(e)}


def grep(path, pat):
    try:
        return [line.rstrip() for line in open(path, errors="replace") if re.search(pat, line)]
    except Exception:  # noqa: BLE001
        return []


def vram_max(path):
    try:
        vals = [float(line.split()[1].rstrip(",")) for line in open(path) if len(line.split()) > 1]
        return round(max(vals) / 1024, 2) if vals else None
    except Exception:  # noqa: BLE001
        return None


def e4b_row(d, B, arm, log):
    """(tok/s, ms, status, why). VOID reasons are the pre-registered engagement rules."""
    if d.get("status") not in (None, "ok"):
        return None, None, d.get("status", "?").upper(), d.get("reason", "")[:160]
    ms = d.get("step_ms_clean")
    if ms is None:
        return None, None, "VOID", "no step_ms_clean in receipt"
    ms_r = round(ms, 3)
    tok = 1000.0 / ms_r if B == 1 else d.get("aggregate_tok_s")
    why = []
    if d.get("recompiles_in_window", 0):
        why.append(f"recompiles_in_window={d.get('recompiles_in_window')}")
    if d.get("n_steps") != NEED_STEPS[B]:
        why.append(f"n_steps {d.get('n_steps')} != {NEED_STEPS[B]}")
    if arm.startswith("int4"):
        if not grep(log, r"INT4EXP.*48 layers"):
            why.append("no INT4EXP 48-layer banner")
        if not grep(log, r"ATTNINT4.*192"):
            why.append("no ATTNINT4 192 banner")
        fused = bool(grep(log, r"fused q/k/v projections on 48 attention modules"))
        if B == 1 and not fused:
            why.append("B=1 int4 arm without the 48-module q/k/v fusion banner (P54's licensed lever not engaged)")
        if B == 16 and grep(log, r"fused q/k/v projections on"):
            why.append("B=16 int4 arm shows the q/k/v fusion (no default there, P54)")
    elif grep(log, r"INT4EXP|ATTNINT4|fused q/k/v"):
        why.append("control arm shows int4/fusion banners")
    return tok, ms_r, ("VOID" if why else "VALID"), "; ".join(why)


def vllm_row(d, arm, log):
    if d.get("status") not in (None, "ok") and "decode_tok_s" not in d:
        return None, None, d.get("status", "?").upper(), d.get("reason", "")[:200]
    tok = d.get("decode_tok_s")
    ms = round(d["decode_ms_per_step"], 3) if d.get("decode_ms_per_step") is not None else None
    why, label = [], []
    if not grep(log, r"Marlin|MARLIN"):
        label.append("KERNEL-CHANGED (no Marlin line)")
    cap = bool(grep(log, r"Capturing CUDA graphs"))
    if arm.startswith("graph") and not cap:
        why.append("graph arm without CUDA-graph capture")
    if arm == "eager" and cap:
        why.append("eager arm captured graphs")
    return tok, ms, ("VOID" if why else ("VALID " + " ".join(label)).strip()), "; ".join(why)


def pair(x, y):
    return (max(x, y) / min(x, y)) if (x and y) else None


def valid(row):
    return bool(row and row[0] and str(row[2]).startswith("VALID"))


def batch_section(D, B, out, ratios):
    pf = jload(os.path.join(D, f"prompts_b{B}.json"))
    psha = pf.get("prompts_sha256")
    out.append(f"\n## B={B}  (prompts_sha {str(psha)[:16]}, rows {pf.get('batch')}, row step {pf.get('row_step')} tokens)")
    out.append("| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |")
    out.append("|---|---|---|---|---|---|---|")
    e4b = {}
    vl = {vt: {} for vt in BUILDS}
    for path in sorted(glob.glob(os.path.join(D, f"e4b_b{B}_*.json"))):
        arm = os.path.basename(path)[len(f"e4b_b{B}_"):-5]
        d = jload(path)
        tok, ms, st, why = e4b_row(d, B, arm, os.path.join(D, "logs", f"run_e4b_b{B}_{arm}.log"))
        vr = vram_max(os.path.join(D, f"vram_e4b_b{B}_{arm}.txt"))
        e4b[arm] = (tok, ms, st)
        out.append(f"| e4b | {arm} | {st} | {tok and round(tok, 1)} | {ms} | {vr} | {why} |")
    for vt in BUILDS:
        for path in sorted(glob.glob(os.path.join(D, f"vllm_{vt}_b{B}_*.json"))):
            arm = os.path.basename(path)[len(f"vllm_{vt}_b{B}_"):-5]
            d = jload(path)
            log = os.path.join(D, "logs", f"run_vllm_{vt}_b{B}_{arm}.log")
            tok, ms, st, why = vllm_row(d, arm, log)
            if d.get("prompts_sha256") and psha and d.get("prompts_sha256") != psha:
                st = "VOID"
                why = (why + "; " if why else "") + "prompts_sha differs from the e4b side"
            vr = vram_max(os.path.join(D, f"vram_vllm_{vt}_b{B}_{arm}.txt"))
            acct = grep(log, r"Actual usage is")
            note = (f"vllm {d.get('vllm_version')}; min-of-3; median-of-3 {d.get('decode_tok_s_median')} tok/s; e2e incl. prefill "
                    f"{d.get('end_to_end_tok_s_long')}; " + (acct[-1].split("]")[-1].strip()[:160] if acct else "no accounting line"))
            vl[vt][arm] = (tok, ms, st, d.get("vllm_version"))
            out.append(f"| vllm-{vt} | {arm} | {st} | {tok} | {ms} | {vr} (policy: reserves 0.90) | {why or note} |")
    sp_nf4 = pair(e4b.get("nf4_r1", (None,))[0], e4b.get("nf4_r2", (None,))[0])
    sp_int4 = pair(e4b.get("int4_r1", (None,))[0], e4b.get("int4_r2", (None,))[0])
    sp_v = {vt: pair(vl[vt].get("graph_r1", (None,))[0], vl[vt].get("graph_r2", (None,))[0]) for vt in BUILDS}
    out.append(f"- self-pairs: e4b nf4 {sp_nf4 and round(sp_nf4, 4)}; e4b int4 {sp_int4 and round(sp_int4, 4)}; "
               + "; ".join(f"vllm-{vt} graph {v and round(v, 4)}" for vt, v in sp_v.items()) + f"  (rule: inside {SELF_PAIR}x or DRIFT)")
    drift_e4b = [n for n, v in (("e4b nf4", sp_nf4), ("e4b int4", sp_int4)) if v and v > SELF_PAIR]
    nf4_row = e4b.get("nf4_r1", (None, None, ""))
    nf4 = nf4_row[0] if nf4_row[2] == "VALID" else None
    i4 = e4b.get("int4_r1", (None, None, ""))
    if nf4 and valid(i4):
        r = i4[0] / nf4
        axes = f"x{r:.3f} over same-box NF4; rental-measured {i4[0]:.1f} tok/s ({i4[1]} ms/step) on this box"
        if B == 1:
            axes += f"; anchor-class PROJECTION {ANCHOR_B1 * r:.0f} tok/s = {ANCHOR_B1} x {r:.3f} (uncertified class)"
        else:
            axes += "; no anchor projection at B=16"
        stop1 = "within" if abs(i4[1] / P54[B] - 1) <= 0.08 else "OUTSIDE"
        out.append(f"- e4b three axes: {axes}. P54's median on its box, cited beside and never divided: {P54[B]} ms/step ({stop1} +-8 %, STOP-1 informational).")
    for vt in BUILDS:
        v = vl[vt].get("graph_r1")
        drift = drift_e4b + ([f"vllm-{vt} graph"] if sp_v[vt] and sp_v[vt] > SELF_PAIR else [])
        if valid(v) and valid(i4) and not drift:
            ratio = v[0] / i4[0]
            ratios[(vt, B)] = ratio
            out.append(f"- **RATIO vLLM {v[3]} / e4b int4 at B={B}: {ratio:.3f}  ({v[0]:.1f} vs {i4[0]:.1f} tok/s; "
                       f"{'vLLM ahead' if ratio > 1 else 'e4b ahead'}; primary pair, min-of-3 slope vs {NEED_STEPS[B]}-step graph window, identical prompt ids)**")
            vm = jload(os.path.join(D, f"vllm_{vt}_b{B}_graph_r1.json")).get("decode_tok_s_median")
            extra = f"  - with the median-of-3 slope: {vm / i4[0]:.3f}" if vm else "  -"
            if e4b.get("int4_r2", (None,))[0]:
                extra += f"; with e4b int4_r2: {v[0] / e4b['int4_r2'][0]:.3f}"
            if nf4:
                extra += f"; vLLM / e4b NF4 control: {v[0] / nf4:.3f}"
            out.append(extra)
        else:
            if drift:
                why = "DRIFT: " + ", ".join(drift)
            else:
                missing = [n for n, x in ((f"vllm-{vt} graph_r1", v), ("e4b int4_r1", i4)) if not valid(x)]
                why = "missing/VOID primary arm: " + ", ".join(missing)
            out.append(f"- **NO RATIO QUOTED for vllm-{vt} at B={B} -- {why}** (readings above; a re-run is a new lane)")
    p_, s_ = vl["primary"].get("graph_r1"), vl["secondary"].get("graph_r1")
    pairs_ok = all(sp_v[vt] is not None and sp_v[vt] <= SELF_PAIR for vt in BUILDS)
    if valid(p_) and valid(s_) and not pairs_ok:
        out.append(f"- build-to-build at B={B}: NOT QUOTED -- a vLLM self-pair is outside {SELF_PAIR}x (first draws {p_[0]} vs {s_[0]} tok/s, printed, not a position)")
    if valid(p_) and valid(s_) and pairs_ok:
        out.append(f"- build-to-build (same box, same prompts): vLLM {p_[3]} / vLLM {s_[3]} = {p_[0] / s_[0]:.3f} at B={B} ({p_[0]} vs {s_[0]} tok/s)")
    if B == 1:
        ve, ee = vl["primary"].get("eager"), e4b.get("int4_eager")
        if ve and ve[0]:
            s = f"- secondary -- eager pairing: vllm-primary eager {ve[0]:.1f} tok/s"
            if valid(p_):
                s += f" ({ve[0] / p_[0]:.3f} of its graph arm)"
            if ee and ee[0]:
                s += f"; e4b int4_eager {ee[0]:.1f} tok/s"
                if i4[0]:
                    s += f" ({ee[0] / i4[0]:.3f} of e4b graph)"
                s += f"; eager-vs-eager {ve[0] / ee[0]:.3f} (secondary, never the headline)"
            out.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    D = a.dir
    out = ["# P58 -- vLLM head-to-head, same box, current vs current, two vLLM builds (" + os.path.abspath(D) + ")",
           "Rule: per vLLM build and batch size, the position is the ratio vLLM/e4b (decode tok/s) from the PRIMARY pair "
           "(vllm graph_r1 vs e4b int4_r1), quoted only when both sides' self-pairs are inside 1.03x and both receipts carry the "
           "same prompts_sha; VOID arms never enter a ratio. Nothing here is licensed; no cross-box number is divided into these."]
    vp = os.path.join(D, "versions.txt")
    out.append("```\n" + (open(vp).read().strip() if os.path.exists(vp) else "(versions.txt missing)") + "\n```")
    fx = os.path.join(D, "forensics.txt")
    if os.path.exists(fx):
        out.append("box: " + " | ".join(line.strip() for line in open(fx) if line.strip()))
    ratios = {}
    for B in (1, 16):
        batch_section(D, B, out, ratios)
    out.append("\nTTFT: informational only where present in the logs (vLLM: none from offline generate; e4b: scheduled PREFILL wall line) -- no ratio (P37 fixture).")
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
