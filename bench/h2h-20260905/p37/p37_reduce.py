#!/usr/bin/env python3
"""p37_reduce.py -- the head-to-head table for lane p37, in the pre-registered form (P37-VLLM-PREREG.md, "Verdict rule").

Reads e4b_b{B}_<arm>.json (step_decomp receipts: step_ms_clean / aggregate_tok_s) and vllm_b{B}_<arm>.json
(p37_vllm.py receipts: decode_tok_s from the slope), the run logs (engagement banners, Marlin / graph-capture
lines, vLLM's memory accounting), the vram_*.txt samplers, and prompts_b{B}.json; prints per batch size:
the arm table (status, VALID/VOID with the reason), the self-pairs, the RATIO vLLM/e4b from the primary pair
(only when both sides' self-pairs are inside 1.03x and both receipts carry the same prompts_sha), e4b's three
axes, and the secondary rows. It licenses nothing and quotes no cross-box number. stdlib only.
Usage: p37_reduce.py <dir> [--md out.md]
"""
import argparse
import glob
import json
import math
import os
import re

SELF_PAIR = 1.03
ANCHOR_B1 = 159.2          # tok/s, the NF4 anchor-class ceiling -- a PROJECTION base only; the class was never certified
BO7 = {1: 2.067, 16: 2.602}  # bo7's same-box licensed/NF4 ratios, cited beside (rule 4), never divided into


def jload(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"status": "unreadable", "reason": str(e)}


def grep(path, pat):
    try:
        return [l.rstrip() for l in open(path, errors="replace") if re.search(pat, l)]
    except Exception:
        return []


def vram_max(path):
    try:
        vals = [float(l.split()[1].rstrip(",")) for l in open(path) if len(l.split()) > 1]
        return round(max(vals) / 1024, 2) if vals else None
    except Exception:
        return None


def e4b_row(d, B, arm, log, vram):
    """(tok/s, ms, status, why) for an e4b receipt; VOID reasons are the pre-registered ones."""
    if d.get("status") not in (None, "ok"):
        return None, None, d.get("status", "?").upper(), d.get("reason", "")[:160]
    ms = d.get("step_ms_clean")
    if ms is None:
        return None, None, "VOID", "no step_ms_clean in receipt"
    ms_r = round(ms, 2)
    tok = 1000.0 / ms_r if B == 1 else d.get("aggregate_tok_s")
    why = []
    if d.get("recompiles_in_window", 0):
        why.append(f"recompiles_in_window={d.get('recompiles_in_window')}")
    if d.get("fuse_qkv"):
        why.append("fuse_qkv true")
    need = 127 if B == 1 else 70
    if d.get("n_steps") != need:
        why.append(f"n_steps {d.get('n_steps')} != {need}")
    if arm.startswith("lic"):
        if not grep(log, r"INT4EXP.*48 layers"):
            why.append("no INT4EXP 48-layer banner")
        if not grep(log, r"ATTNINT4.*192"):
            why.append("no ATTNINT4 192 banner")
        counts = grep(log, r"11512 gptq / 776 rtn")
        if not counts:
            why.append("64k pack counts 11512/776 not in log (different pack or not calibrated)")
    else:
        if grep(log, r"INT4EXP|ATTNINT4"):
            why.append("control arm shows int4 banners")
    return tok, ms_r, ("VOID" if why else "VALID"), "; ".join(why)


def vllm_row(d, B, arm, log):
    if d.get("status") not in (None, "ok") and "decode_tok_s" not in d:
        return None, None, d.get("status", "?").upper(), d.get("reason", "")[:200]
    tok, ms = d.get("decode_tok_s"), (round(d["decode_ms_per_step"], 3) if d.get("decode_ms_per_step") is not None else None)
    why, label = [], []
    if not grep(log, r"Marlin|MARLIN"):
        label.append("KERNEL-CHANGED (no Marlin line)")
    cap = bool(grep(log, r"Capturing CUDA graphs"))
    if arm.startswith("graph") or arm == "fp8kv":
        if not cap:
            why.append("graph arm without CUDA-graph capture")
    elif arm == "eager" and cap:
        why.append("eager arm captured graphs")
    return tok, ms, ("VOID" if why else ("VALID " + " ".join(label)).strip()), "; ".join(why)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--md", default=None)
    a = ap.parse_args(); D = a.dir
    out = ["# p37 -- vLLM head-to-head, same box (" + os.path.abspath(D) + ")",
           "Rule: the position is the ratio vLLM/e4b (decode tok/s) from the PRIMARY pair (vllm graph_r1 vs e4b lic_r1) per batch size, "
           "quoted only when both sides' self-pairs are inside 1.03x and both receipts carry the same prompts_sha; VOID arms never enter a ratio. "
           "Nothing here is licensed; no cross-box number is divided into these."]
    vers = open(os.path.join(D, "versions.txt")).read().strip() if os.path.exists(os.path.join(D, "versions.txt")) else "(versions.txt missing)"
    out.append("```\n" + vers + "\n```")
    fx = os.path.join(D, "forensics.txt")
    if os.path.exists(fx):
        out.append("box: " + " | ".join(l.strip() for l in open(fx) if l.strip()))
    for B in (1, 16):
        pf = jload(os.path.join(D, f"prompts_b{B}.json")); psha = pf.get("prompts_sha256")
        out.append(f"\n## B={B}  (prompts_sha {str(psha)[:16]}, rows {pf.get('batch')}, row step {pf.get('row_step')} tokens)")
        out.append("| engine | arm | status | tok/s | ms/step | peak VRAM GB (nvidia-smi max) | notes |"); out.append("|---|---|---|---|---|---|---|")
        e4b, vl = {}, {}
        for p in sorted(glob.glob(os.path.join(D, f"e4b_b{B}_*.json"))):
            arm = os.path.basename(p)[len(f"e4b_b{B}_"):-5]; d = jload(p)
            tok, ms, st, why = e4b_row(d, B, arm, os.path.join(D, "logs", f"run_e4b_b{B}_{arm}.log"), None)
            vr = vram_max(os.path.join(D, f"vram_e4b_b{B}_{arm}.txt"))
            e4b[arm] = (tok, ms, st); out.append(f"| e4b | {arm} | {st} | {tok and round(tok,1)} | {ms} | {vr} | {why} |")
        for p in sorted(glob.glob(os.path.join(D, f"vllm_b{B}_*.json"))):
            arm = os.path.basename(p)[len(f"vllm_b{B}_"):-5]; d = jload(p)
            log = os.path.join(D, "logs", f"run_vllm_b{B}_{arm}.log")
            tok, ms, st, why = vllm_row(d, B, arm, log)
            if d.get("prompts_sha256") and psha and d.get("prompts_sha256") != psha:
                st, why = "VOID", (why + "; " if why else "") + "prompts_sha differs from the e4b side"
            vr = vram_max(os.path.join(D, f"vram_vllm_b{B}_{arm}.txt"))
            acct = grep(log, r"Actual usage is"); kvl = grep(log, r"Available KV cache memory")
            note = (f"min-of-3; median-of-3 {d.get('decode_tok_s_median')} tok/s; e2e incl. prefill {d.get('end_to_end_tok_s_long')}; "
                    f"vllm {d.get('vllm_version')}; " + (acct[-1].split("]")[-1].strip()[:160] if acct else "no accounting line") + ("; " + kvl[-1].split("]")[-1].strip()[:80] if kvl else ""))
            vl[arm] = (tok, ms, st); out.append(f"| vllm | {arm} | {st} | {tok} | {ms} | {vr} (policy: reserves 0.90) | {why or note} |")
        # self-pairs
        def pair(x, y):
            return (max(x, y) / min(x, y)) if (x and y) else None
        sp_nf4 = pair(e4b.get("nf4_r1", (None,))[0], e4b.get("nf4_r2", (None,))[0])
        sp_lic = pair(e4b.get("lic_r1", (None,))[0], e4b.get("lic_r2", (None,))[0]) if B == 1 else None
        sp_vl = pair(vl.get("graph_r1", (None,))[0], vl.get("graph_r2", (None,))[0])
        out.append(f"- self-pairs: e4b nf4 {sp_nf4 and round(sp_nf4,4)}; e4b lic {sp_lic and round(sp_lic,4) if B == 1 else 'n/a at B=16 (not repeated, pre-registered)'}; vllm graph {sp_vl and round(sp_vl,4)}  (rule: inside {SELF_PAIR}x or DRIFT)")
        drift = [n for n, v in (("e4b nf4", sp_nf4), ("e4b lic", sp_lic), ("vllm graph", sp_vl)) if v and v > SELF_PAIR]
        # instrument check: e4b same-box ratio beside bo7's
        nf4 = e4b.get("nf4_r1", (None, None, ""))[0]; lic = e4b.get("lic_r1", (None, None, ""))[0]
        if nf4 and lic and e4b["nf4_r1"][2] == "VALID" and e4b["lic_r1"][2] == "VALID":
            r = lic / nf4
            out.append(f"- e4b same-box control: licensed/NF4 = x{r:.3f} on this box (bo7, cited beside, never divided: x{BO7[B]}; {'reproduced' if abs(r/BO7[B]-1) <= 0.10 else 'OUTSIDE +-10% of bo7 -- reported, not gated'})")
            axes = f"x{r:.3f} over same-box NF4; rental-measured {lic:.1f} tok/s on this box"
            axes += f"; anchor-class PROJECTION {ANCHOR_B1 * r:.0f} tok/s = {ANCHOR_B1} x {r:.3f} (uncertified class)" if B == 1 else "; no anchor projection at B=16"
            out.append(f"- e4b three axes: {axes}")
        # the number
        v = vl.get("graph_r1"); e = e4b.get("lic_r1")
        if v and e and v[0] and e[0] and v[2].startswith("VALID") and e[2] == "VALID" and not drift:
            ratio = v[0] / e[0]
            out.append(f"- **RATIO vLLM/e4b at B={B}: {ratio:.3f}  ({v[0]:.1f} vs {e[0]:.1f} tok/s; {'vLLM ahead' if ratio > 1 else 'e4b ahead'}; primary pair, min-of-3 slope vs 127/70-step graph window, identical prompt ids)**")
            vm = jload(os.path.join(D, f"vllm_b{B}_graph_r1.json")).get("decode_tok_s_median")
            if vm:
                out.append(f"  - with the median-of-3 slope: {vm / e[0]:.3f}; with e4b lic_r2: {(v[0] / e4b['lic_r2'][0]):.3f}" if B == 1 and e4b.get("lic_r2", (None,))[0] else f"  - with the median-of-3 slope: {vm / e[0]:.3f}")
        else:
            why = "DRIFT: " + ", ".join(drift) if drift else ("missing/VOID primary arm: " + ", ".join(n for n, x in (("vllm graph_r1", v), ("e4b lic_r1", e)) if not (x and x[0] and str(x[2]).startswith("VALID"))))
            out.append(f"- **NO RATIO QUOTED at B={B} -- {why}** (both readings above; a re-run is a new lane)")
        # secondary rows
        for name, vk, ek in (("eager pairing", "eager", "lic_eager" if B == 1 else None), ("fp8 KV vs kv auto (vLLM only)", "fp8kv", None)):
            vv = vl.get(vk)
            if vv and vv[0]:
                s = f"- secondary -- {name}: vllm {vk} {vv[0]} tok/s"
                if v and v[0]:
                    s += f" ({vv[0] / v[0]:.3f} of vllm graph)"
                if ek and e4b.get(ek, (None,))[0]:
                    s += f"; e4b {ek} {e4b[ek][0]:.1f} tok/s ({e4b[ek][0] / e[0]:.3f} of e4b graph)" if e and e[0] else f"; e4b {ek} {e4b[ek][0]:.1f} tok/s"
                    s += f"; eager-vs-eager ratio {vv[0] / e4b[ek][0]:.3f} (secondary, never the headline)"
                out.append(s)
    out.append("\nTTFT: informational only where present in the logs (vLLM: none from offline generate; e4b: scheduled PREFILL wall line) -- no ratio (P37 fixture).")
    text = "\n".join(out); print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
