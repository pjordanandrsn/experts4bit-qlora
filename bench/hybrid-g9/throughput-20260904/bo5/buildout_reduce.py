"""Reduce lane bo5's JSON receipts (<fam>_ppl_<arm>.json / <fam>_b1_<arm>.json / <fam>_b16_<arm>.json) into
one per-family build-out table, with the registered K8 gate applied in ITS OWN units (perplexity) on every
text the arm was scored on, nats quoted beside the verdict (they never change it), and the B=1 / B=16 numbers.

Usage: buildout_reduce.py <dir> [--ref <dir>]

  <dir>   this bundle. A `_c4val` suffix on a ppl arm marks the second text (the JSON's own `ppl_source`
          field is what is read; the suffix is only stripped to name the configuration).
  --ref   a sibling bundle (bo3) whose NF4 wikitext K8 rows are the baseline where this lane did not re-score
          NF4 on wikitext (Qwen3; Mixtral's integration-6 arms). The scored window's sha must match or the row
          gets no delta. Speed ratios never cross lanes: `x base` is against a named arm on THIS lane only.

The registered rule (experts4bit_qlora.k8_gate): an UNCALIBRATED arm passes when |delta ppl| <= 0.05 on every
text; a CALIBRATED pack passes when delta ppl <= +0.05 on every text, and an improvement may only be claimed
when it holds with the same sign on >= 2 texts (one outside the calibration domain). B=1 tok/s is 1000 / the
timed graph window's step, rounded to the 0.01 ms the lane logs print (so the table reproduces the log lines).
"""
import argparse
import glob
import json
import math
import os
import re

FLOOR = {"granite": 0.0033, "gptoss": 0.0176, "qwen3": 0.0095, "mixtral": None}
NAME = {"qwen3": "Qwen3-30B-A3B", "granite": "Granite-3.1-3B-A800M", "gptoss": "gpt-oss-20b",
        "mixtral": "Mixtral-8x7B-Instruct"}
FAMS = ["qwen3", "granite", "gptoss", "mixtral"]
BUDGET = 0.05
# The cut each configuration ran on: integration-6 (@0535930, lane script bo5_run.sh) unless listed here as
# integration-7 (@d090940, bo5b.sh / bo5c.sh). NF4 numerics do not depend on the cut; the pairing matters for
# the speed pairs and for which NF4 wikitext control Mixtral's arms are read against.
I7 = {("qwen3", a) for a in ("int4exp", "calib", "folds", "int4folds")}
I7 |= {("granite", a) for a in ("nf4_r12epi_i7", "nf4_r12epi_fq2", "nf4_r12epi_i7b", "nf4_r12epi_fq2b",
                                "nf4_r12epi_i7_cen", "nf4_r12epi_fq2_cen")}
I7 |= {("mixtral", a) for a in ("all_i7", "all_fq", "lic", "lic_fq", "nf4_i7")}
# K8 baseline per (family, text[, cut]): ("local"|"ref", configuration).
K8_BASE = {("qwen3", "wikitext"): ("ref", "nf4"), ("qwen3", "c4val1"): ("local", "nf4"),
           ("granite", "wikitext"): ("local", "nf4_r12epi"), ("granite", "c4val1"): ("local", "nf4_r12epi"),
           ("mixtral", "wikitext", "i6"): ("ref", "nf4"), ("mixtral", "wikitext", "i7"): ("local", "nf4_i7"),
           ("mixtral", "c4val1"): ("local", "nf4")}
# Speed reference per family, on this lane only.
SPEED_BASE = {"granite": "nf4_r12epi", "gptoss": "r12", "mixtral": "all"}
# A calibrated pack (C4-calibrated int4 attention and/or GPTQ int4 experts) is in the configuration.
CALIBRATED = {("qwen3", "all"), ("qwen3", "calib"), ("granite", "calibexp_r12epi"),
              ("mixtral", "all"), ("mixtral", "all_fq"), ("mixtral", "all_i7"), ("mixtral", "all_nor2")}
ORDER = {"qwen3": ["nf4", "all", "all_noglue", "all_cen", "all_noglue_cen", "int4exp", "calib", "folds", "int4folds"],
         "granite": ["nf4_r12epi", "calibexp_r12epi", "rtnexp_r12epi", "nf4_r12epi_fq", "nf4_r12epi_i7",
                     "nf4_r12epi_fq2", "nf4_r12epi_i7b", "nf4_r12epi_fq2b", "nf4_r12epi_i7_cen", "nf4_r12epi_fq2_cen"],
         "gptoss": ["r12", "store_r12", "store_r12_cen"],
         "mixtral": ["nf4", "nf4_i7", "all", "all_nor2", "all_i7", "all_fq", "lic", "lic_fq"]}
DESC = {("qwen3", "nf4"): "NF4 experts, bf16 attention (baseline; wikitext row from bo3, same window sha)",
        ("qwen3", "all"): "int4 experts + C4-calibrated int4 attention + r1 + r2 (rope-only fold) + router epilogue + #385 glue",
        ("qwen3", "all_noglue"): "all with E4B_FUSE_SWIGLU=0 E4B_FUSE_COMBINE=0 (the #385 A/B arm)",
        ("qwen3", "all_cen"): "all, kernel census run", ("qwen3", "all_noglue_cen"): "all_noglue, kernel census run",
        ("qwen3", "int4exp"): "int4 experts only (attribution, second text)",
        ("qwen3", "calib"): "int4 experts + C4-calibrated int4 attention (attribution, second text)",
        ("qwen3", "folds"): "r1 + r2 + epilogue only, NF4 experts (attribution, second text; exact arithmetic)",
        ("qwen3", "int4folds"): "int4 experts + r1 + r2 + epilogue, no calibrated attention (attribution, second text)",
        ("granite", "nf4_r12epi"): "NF4 experts + r1 + r2 (rotary-only fold, #379) + epilogue — the licensed stack (baseline)",
        ("granite", "calibexp_r12epi"): "licensed stack with C4-calibrated (GPTQ) int4 experts, #384: 2524 gptq / 36 rtn",
        ("granite", "rtnexp_r12epi"): "licensed stack with uncalibrated (RTN) int4 experts",
        ("granite", "nf4_r12epi_fq"): "licensed stack without --no-fuse-qkv on integration-6 (no-op control)",
        ("granite", "nf4_r12epi_i7"): "licensed stack on integration-7 (unfused control for #387)",
        ("granite", "nf4_r12epi_fq2"): "licensed stack + #387 fused qkv (32 modules) + fused rope-only fold, integration-7",
        ("granite", "nf4_r12epi_i7b"): "second unfused control", ("granite", "nf4_r12epi_fq2b"): "second #387 arm",
        ("granite", "nf4_r12epi_i7_cen"): "unfused, kernel census run", ("granite", "nf4_r12epi_fq2_cen"): "#387 fused, kernel census run",
        ("gptoss", "r12"): "NF4 experts + r1 + r2 folds (baseline; bo3's licensed row)",
        ("gptoss", "store_r12"): "native MXFP4 store: gemv_mxfp4_b32 for single rows, NF4 kept for batched rows (E4B_INT4_KEEP_NF4=1) + r1 + r2",
        ("gptoss", "store_r12_cen"): "store_r12, kernel census run",
        ("mixtral", "nf4"): "NF4 experts, bf16 attention (baseline; wikitext row = P30 NF4 from bo3, same window sha)",
        ("mixtral", "nf4_i7"): "NF4 experts, bf16 attention, re-baked on integration-7 (wikitext control for the i7 arms)",
        ("mixtral", "all"): "int4 experts + C4-calibrated int4 attention + r1 + r2 (rope-only fold) + epilogue",
        ("mixtral", "all_nor2"): "all without the round-2 fold (E4B_FUSE_T1_GLUE_R2=0)",
        ("mixtral", "all_i7"): "all on integration-7 (control for #387)",
        ("mixtral", "all_fq"): "all + #387 fused qkv: 0 modules fused (calibrated attention children are not Linear)",
        ("mixtral", "lic"): "int4 experts (RTN) + r1 + r2 + epilogue, no calibrated pack — the P30 'licensed stack'",
        ("mixtral", "lic_fq"): "lic + #387 fused qkv (32 modules) + fused rope-only fold"}
PAIRS = {"qwen3": [("all", "all_noglue", "#385 swiglu_rows + combine_rows glue on vs off"),
                   ("all_cen", "all_noglue_cen", "same pair, census runs")],
         "granite": [("calibexp_r12epi", "nf4_r12epi", "C4-calibrated int4 experts (#384) vs NF4 experts — speed only, the K8 verdict is FAIL"),
                     ("nf4_r12epi_fq", "nf4_r12epi", "fused-qkv flag on integration-6: no-op control"),
                     ("nf4_r12epi_fq2", "nf4_r12epi_i7", "#387 fused qkv + fused rope-only fold vs unfused, integration-7"),
                     ("nf4_r12epi_fq2b", "nf4_r12epi_i7b", "second pair"),
                     ("nf4_r12epi_fq2_cen", "nf4_r12epi_i7_cen", "census pair")],
         "gptoss": [("store_r12", "r12", "MXFP4 store route (GEMV single rows, NF4 batched rows) vs NF4 experts")],
         "mixtral": [("all", "all_nor2", "rope-only round-2 fold (#379) on vs off, calibrated stack"),
                     ("all_fq", "all_i7", "#387 on the calibrated stack (0 modules fused)"),
                     ("all_fq", "all", "the same #387 arm's K8 against integration-6 `all` (0 modules fused: identical numerics expected)"),
                     ("lic_fq", "lic", "#387 on the licensed int4-expert stack (32 modules fused)"),
                     ("all", "lic", "calibrated attention on top of the int4-expert stack (integration-6 vs -7)")]}
PAT = re.compile(r"^(?P<fam>[a-z0-9]+)_(?P<kind>ppl|b1|b16)_(?P<arm>.+)\.json$")


def fmt(v, p, sign=False):
    if v is None:
        return "—"
    return f"{v:+.{p}f}" if sign else f"{v:.{p}f}"


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        fam, kind, arm = m.group("fam"), m.group("kind"), m.group("arm")
        j = json.load(open(f))
        if kind == "ppl":
            cfg = arm[:-len("_c4val")] if arm.endswith("_c4val") else arm
            r = rows.setdefault((fam, cfg), {"ppl": {}})
            r["ppl"][j["ppl_source"]] = {"nll": j["mean_nll"], "ppl": j["ppl"], "sha": j["text_sha"][:12],
                                         "steps": j["steps"], "file": os.path.basename(f)}
        else:
            r = rows.setdefault((fam, arm), {"ppl": {}})
            if kind == "b1":
                r["b1_ms"] = round(j["step_ms_clean"], 2)
                r["b1_file"] = os.path.basename(f)
            else:
                r["b16"] = j["aggregate_tok_s"]
                r["b16_file"] = os.path.basename(f)
    return rows


def census(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "census_*.txt"))):
        txt = open(f, errors="replace").read().splitlines()
        steps = launches = 0
        cuda_total = None
        for ln in txt:
            m = re.match(r"profiled replay steps: (\d+)", ln)
            if m:
                steps = int(m.group(1))
            m = re.match(r"Self CUDA time total: ([\d.]+)ms", ln)
            if m:
                cuda_total = float(m.group(1))
            cols = re.split(r"\s{2,}", ln.strip())
            if len(cols) >= 11 and re.match(r"^\d+$", cols[-1]):
                self_cuda = cols[-5]
                if self_cuda.endswith("us") and float(self_cuda[:-2]) > 0 or self_cuda.endswith("ms"):
                    launches += int(cols[-1])
        out[os.path.basename(f)] = {"steps": steps, "cuda_ms": cuda_total,
                                    "launches": (launches / steps) if steps else None}
    return out


def k8_base(fam, text, cut):
    return K8_BASE.get((fam, text, cut)) or K8_BASE.get((fam, text))


def verdict(fam, cfg, deltas, is_base):
    """deltas: {text: dppl}. Returns (per-text cells, registered verdict) in ppl units."""
    if is_base:
        return {t: "baseline" for t in deltas}, "baseline"
    if fam == "gptoss":
        return {t: "no instrument" for t in deltas}, "no instrument (OOD regime)"
    cal = (fam, cfg) in CALIBRATED
    cells, fails = {}, []
    for t, dp in deltas.items():
        bad = (dp > BUDGET) if cal else (abs(dp) > BUDGET)
        cells[t] = "FAIL" if bad else "pass"
        if bad:
            fails.append(f"{t} {dp:+.4f}" + ("" if cal or dp > 0 else " by improving"))
    if fails:
        return cells, "FAIL as registered (" + "; ".join(fails) + ")"
    n = len(deltas)
    if cal:
        if n >= 2 and all(dp < 0 for dp in deltas.values()):
            return cells, f"pass — licensed (improves on {n} texts)"
        if n >= 2:
            return cells, f"pass on {n} texts (within +{BUDGET}; no same-sign improvement to claim)"
        return cells, "one text (needs 2)"
    return cells, f"pass ({n} text{'s' if n > 1 else ''})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    rows = load(a.dir)
    ref = load(a.ref) if a.ref else {}
    cen = census(a.dir)
    texts = ["wikitext", "c4val1"]
    for fam in FAMS:
        fams = {c: r for (f, c), r in rows.items() if f == fam}
        if not fams:
            continue
        floor = FLOOR[fam]
        sb = SPEED_BASE.get(fam)
        base_speed = fams.get(sb, {}) if sb else {}
        print(f"\n### {NAME[fam]}")
        hdr = (f"arithmetic-order floor {floor} nats" if floor else "arithmetic-order floor unmeasured")
        # per-text floor in ppl, at the baseline's ppl on that text
        fl = []
        for t in texts:
            b = k8_base(fam, t, "i6")
            if not b:
                continue
            src = ref if b[0] == "ref" else fams
            br = (src.get((fam, b[1])) if b[0] == "ref" else src.get(b[1])) or {}
            bp = br.get("ppl", {}).get(t)
            if bp and floor:
                fl.append(f"{t}: base ppl {bp['ppl']:.3f} → ±{bp['ppl'] * (math.exp(floor) - 1):.3f} ppl")
        if fl:
            hdr += " = " + "; ".join(fl)
        if sb:
            hdr += f". `× base` = ratio to `{sb}` on this lane"
        else:
            hdr += ". No NF4 speed arm on this lane, so no `× base`"
        print(hdr + ".")
        print("| arm | configuration | cut | K8 wikitext nll | Δppl (Δnats) | gate | K8 c4val1 nll | Δppl (Δnats) | gate | registered verdict (ppl) | B=1 ms | B=1 tok/s | × base | B=16 tok/s | × base |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = ORDER[fam] + sorted(c for c in fams if c not in ORDER[fam])
        for cfg in order:
            r = fams.get(cfg)
            if not r:
                continue
            cut = "i7" if (fam, cfg) in I7 else "i6"
            cells, deltas, nll_s, d_s = {}, {}, {}, {}
            is_base = False
            for t in texts:
                p = r["ppl"].get(t)
                if not p:
                    nll_s[t] = d_s[t] = "—"
                    continue
                nll_s[t] = f"{p['nll']:.5f}"
                b = k8_base(fam, t, cut)
                if b and b[1] == cfg and b[0] == "local":
                    is_base = True
                    d_s[t] = "—"
                    deltas[t] = 0.0
                    continue
                src = ref if (b and b[0] == "ref") else fams
                br = (src.get((fam, b[1])) if (b and b[0] == "ref") else (src.get(b[1]) if b else None)) or {}
                bp = br.get("ppl", {}).get(t)
                if not bp or bp["sha"] != p["sha"]:
                    d_s[t] = "— (no matching baseline)"
                    continue
                dn = p["nll"] - bp["nll"]
                dp = p["ppl"] - bp["ppl"]
                deltas[t] = dp
                tag = ""
                if floor is not None:
                    tag = " sub-floor" if abs(dn) <= floor else f" {abs(dn) / floor:.1f}× floor"
                d_s[t] = f"{dp:+.4f} ({dn:+.4f}{tag})" + (" ᵇ" if b[0] == "ref" else "")
            cells, ver = verdict(fam, cfg, deltas, is_base)
            if not deltas:
                ver = "—"
            b1 = r.get("b1_ms")
            tps = 1000 / b1 if b1 else None
            x1 = (base_speed["b1_ms"] / b1) if (b1 and base_speed.get("b1_ms")) else None
            b16 = r.get("b16")
            x16 = (b16 / base_speed["b16"]) if (b16 and base_speed.get("b16")) else None
            if cfg == sb:
                x1 = x16 = None
            print(f"| `{cfg}` | {DESC.get((fam, cfg), cfg)} | {cut} | {nll_s['wikitext']} | {d_s['wikitext']} | "
                  f"{cells.get('wikitext', '—')} | {nll_s['c4val1']} | {d_s['c4val1']} | {cells.get('c4val1', '—')} | {ver} | "
                  f"{fmt(b1, 2)} | {fmt(tps, 1)} | {fmt(x1, 3)} | {fmt(b16, 1)} | {fmt(x16, 3)} |")
        pairs = [(n, d, why) for n, d, why in PAIRS.get(fam, []) if n in fams and d in fams]
        if pairs:
            print(f"\nPairs on this lane ({NAME[fam]}; B=1 ratio = ms(den)/ms(num), B=16 ratio = tok/s(num)/tok/s(den)):")
            for n, d, why in pairs:
                rn, rd = fams[n], fams[d]
                s = f"- `{n}` vs `{d}` — {why}:"
                if rn.get("b1_ms") and rd.get("b1_ms"):
                    s += f" B=1 {rd['b1_ms']:.2f} → {rn['b1_ms']:.2f} ms = ×{rd['b1_ms'] / rn['b1_ms']:.3f}"
                if rn.get("b16") and rd.get("b16"):
                    s += f"; B=16 {rd['b16']:.1f} → {rn['b16']:.1f} tok/s = ×{rn['b16'] / rd['b16']:.3f}"
                pn, pd = rn["ppl"].get("wikitext"), rd["ppl"].get("wikitext")
                if pn and pd and pn["sha"] == pd["sha"]:
                    s += f"; K8 wikitext {pn['nll'] - pd['nll']:+.4f} nats ({pn['ppl'] - pd['ppl']:+.4f} ppl)"
                print(s)
        if fam == "qwen3" and all(c in fams and "c4val1" in fams[c]["ppl"] for c in ("nf4", "int4exp", "calib", "folds", "all")):
            g = lambda c: fams[c]["ppl"]["c4val1"]["nll"] - fams["nf4"]["ppl"]["c4val1"]["nll"]  # noqa: E731
            pred = g("int4exp") + (g("calib") - g("int4exp")) + g("folds")
            print(f"\nAttribution on c4val1 (nats vs NF4): int4 experts {g('int4exp'):+.4f}; calibrated attention on top "
                  f"{g('calib') - g('int4exp'):+.4f}; exact folds {g('folds'):+.4f}; sum {pred:+.4f} vs measured `all` "
                  f"{g('all'):+.4f} — non-additive by {abs(g('all') - pred):.4f} nats (floor {FLOOR['qwen3']}).")
        fc = {k: v for k, v in cen.items() if k.startswith(f"census_{fam}_")}
        if fc:
            print(f"\nKernel census ({NAME[fam]}, `--replay-profile-out`, Self CUDA over the profiled replay steps; launches = CUDA-kernel rows' calls per step):")
            for k, v in fc.items():
                arm = k[len(f"census_{fam}_"):-4]
                r = fams.get(arm, {})
                print(f"- `{k}`: {v['steps']} steps, Self CUDA total {fmt(v['cuda_ms'], 3)} ms "
                      f"({fmt(v['cuda_ms'] / v['steps'] if v['cuda_ms'] and v['steps'] else None, 3)} ms/step), "
                      f"{fmt(v['launches'], 0)} launches/step; timed wall {fmt(r.get('b1_ms'), 2)} ms/step")
    print("\nᵇ = baseline row taken from the `--ref` bundle (bo3) on the same window sha; every other delta is against an arm on this lane."
          "\nGate cells and the registered verdict are in perplexity, the registered unit; the nats beside them are read against the family's"
          " arithmetic-order floor and never change the verdict. `one text (needs 2)` = a calibrated pack within +0.05 ppl on the one text scored.")


if __name__ == "__main__":
    main()
