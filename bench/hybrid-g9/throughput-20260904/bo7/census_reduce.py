"""Reduce lane bo7's JSON receipts (<fam>_b1_<arm>.json / <fam>_b16_<arm>.json) into the THROUGHPUT CENSUS tables:
per family every arm with its configuration, its LICENCE LABEL as the register (docs/claims.json + the K8 verdicts on
record at bundle time: bo3, bo5, bo6, bo6c) states it, its B=1 and B=16 speed and the ratio to the family's own NF4
arm on THIS box; then the licensed best per family on the three axes; then a cross-family summary of ratios.

Usage: census_reduce.py <dir>

  <dir>   this bundle: the receipts flat, the lane console at logs/outer.log, the per-arm run logs at logs/run_*.log.

bo7 measures SPEED ONLY. It licenses nothing: every label in the `licence (register)` column is copied from the register
as it stood when the bundle was written, with the claim id that carries the verdict. Reading rules, pre-registered (P35):
  (1) every ratio is the family's own NF4 arm on this box, same session -- never a cross-lane ratio; bo3/bo5/bo6 numbers
      are cited beside, never divided into, bo7's;
  (2) the licence label of each arm comes from the register -- where the register is silent the label is `no quality
      verdict on record`, never `licensed`;
  (3) three axes for every LICENSED best only: ratio x N over the family's NF4 on this box, rental-measured tok/s on this
      box, and the anchor-class projection marked as a projection -- an anchor projection exists ONLY for Qwen3-30B at
      B=1 (159.2 tok/s x the ratio; no anchor-class box was ever certified: 12 refusals); every other family and B=16
      print `no anchor projection`;
  (4) unlicensed arms appear labelled `measured, not licensed` (with the verdict and the claim id) and never as a position;
  (5) no cross-lane ratio.

Speed: B=1 tok/s is 1000 / the timed graph window's step in ms rounded to the 0.01 ms the lane logs print (bo5's
convention, so the table reproduces the log lines; the ratio is computed from the same rounded values -- +-0.25% at a
2 ms step); B=16 tok/s is the receipt's `aggregate_tok_s`. Ratio B=1 = ms(NF4) / ms(arm); ratio B=16 = tok/s(arm) /
tok/s(NF4). gpt-oss's NF4 arm on this lane is `nf4_r12` (NF4 experts + the exact round-1/2 folds), so its ratios are to that.

Status is read from logs/outer.log: an arm line with a receipt is `ok`; an arm line followed by `Alarm clock` is `alarm`
(killed by its own per-arm alarm: 5400 s for calibrated arms, 3600 s otherwise -- a harness limit, not a model result);
`REFUSED` / `Traceback` / `RuntimeError` is `refused` (with the sentence); a `BAKE FAILED` family is `bake failed`; an
arm the script runs with no line yet is `pending (arrives before merge)`. What each arm ENGAGED is read from its run log
(the hook's `INT4EXP` / `ATTNINT4` banners; the E4B_FUSE_* fusions print no banner on this harness and refuse aloud when
nothing matches, so an arm that ran to a receipt engaged them).
"""
import argparse
import glob
import json
import os
import re

FAMS = ["granite", "olmoe", "gptoss", "qwen3", "gemma4", "mixtral"]
NAME = {"granite": "Granite-3.1-3B-A800M", "olmoe": "OLMoE-1B-7B", "gptoss": "gpt-oss-20b", "qwen3": "Qwen3-30B-A3B",
        "gemma4": "Gemma-4-26B-A4B", "mixtral": "Mixtral-8x7B-Instruct"}
NF4 = {"granite": "nf4", "olmoe": "nf4", "gptoss": "nf4_r12", "qwen3": "nf4", "gemma4": "nf4", "mixtral": "nf4"}
ORDER = {"granite": ["nf4", "r12epi", "int4_r12epi", "calibexp_r12epi"],
         "olmoe": ["nf4", "folds", "calattn", "int4all"],
         "gptoss": ["nf4_r12", "store_r12"],
         "qwen3": ["nf4", "folds", "calattn", "int4all", "calibexp_all", "calibexp_folds"],
         "gemma4": ["nf4", "r1epi", "int4_r1epi", "calattn_r1epi"],
         "mixtral": ["nf4", "folds", "lic", "all"]}
DROPPED = {"mixtral": [("calibexp_lic", "DROPPED under P35 amendment 1 before the box was rented: FAIL as registered on bo6b "
                                        "(wikitext +0.077 ppl, `e4b.serve.buildout.bo6.mixtral.lic-calibexp-streamed.k8.2026-09-04`) "
                                        "and ~85 min of streamed calibration per arm inside a 10-h guard; not run, no number")]}
ANCHOR_NF4 = 159.2  # tok/s: the certified single-stream NF4 ceiling on the anchor class (Qwen3-30B-A3B, B=1); never re-certified
DESC = {
    ("granite", "nf4"): "NF4 experts, bf16 attention, no folds (this lane's reference)",
    ("granite", "r12epi"): "NF4 experts + round-1 + round-2 (rotary-only) folds + router epilogue",
    ("granite", "int4_r12epi"): "RTN int4-b32 experts + round-1/2 folds + router epilogue",
    ("granite", "calibexp_r12epi"): "streamed GPTQ-calibrated int4 experts (16k C4 tokens) + round-1/2 folds + router epilogue",
    ("olmoe", "nf4"): "NF4 experts, bf16 attention, no folds (this lane's reference)",
    ("olmoe", "folds"): "NF4 experts + round-1/2 folds + router epilogue (exact arithmetic)",
    ("olmoe", "calattn"): "NF4 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue",
    ("olmoe", "int4all"): "RTN int4-b32 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue",
    ("gptoss", "nf4_r12"): "NF4 experts + round-1/2 folds, bf16 attention (this lane's reference)",
    ("gptoss", "store_r12"): "native MXFP4 store: gemv_mxfp4_b32 for single rows, NF4 kept for batched rows (E4B_INT4_KEEP_NF4=1) + round-1/2 folds",
    ("qwen3", "nf4"): "NF4 experts, bf16 attention, no folds (this lane's reference)",
    ("qwen3", "folds"): "NF4 experts + round-1/2 (rope-only) folds + router epilogue (exact arithmetic)",
    ("qwen3", "calattn"): "NF4 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue",
    ("qwen3", "int4all"): "RTN int4-b32 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue + #385 glue (bo5's `all`)",
    ("qwen3", "calibexp_all"): "streamed GPTQ-calibrated int4 experts at the hook's 16k default + C4-calibrated int4 attention + round-1/2 folds + router epilogue + glue",
    ("qwen3", "calibexp_folds"): "streamed GPTQ-calibrated int4 experts (16k) + round-1/2 folds + router epilogue, bf16 attention",
    ("gemma4", "nf4"): "NF4 experts, bf16 attention, no folds (this lane's reference)",
    ("gemma4", "r1epi"): "NF4 experts + round-1 norm fold + router epilogue (round 2 refuses by design on this family; exact arithmetic)",
    ("gemma4", "int4_r1epi"): "RTN int4-b32 experts + round-1 fold + router epilogue (bo3's `stack`)",
    ("gemma4", "calattn_r1epi"): "NF4 experts + C4-calibrated int4 attention + round-1 fold + router epilogue",
    ("mixtral", "nf4"): "NF4 experts, bf16 attention, no folds (this lane's reference)",
    ("mixtral", "folds"): "NF4 experts + round-1/2 (rope-only) folds + router epilogue (exact arithmetic)",
    ("mixtral", "lic"): "RTN int4-b32 experts + round-1/2 folds + router epilogue (bo5's `lic`, the withdrawn 'licensed stack')",
    ("mixtral", "all"): "RTN int4-b32 experts + C4-calibrated int4 attention + round-1/2 folds + router epilogue (bo5's `all`)",
}
# The register's label for each arm, verbatim in substance, with the claim id that carries it. Nothing here is decided
# by this lane.
LICENCE = {
    ("granite", "nf4"): "baseline (reference); paged decode at parity with the model's own attention, 0.00229 nats (`e4b.parity.granite.paged-vs-own-attention`)",
    ("granite", "r12epi"): "LICENSED -- K8 +0.019 ppl vs NF4 on wikitext, pass under the uncalibrated rule (bo3, `e4b.serve.buildout.granite.b1.5090.2026-09-04` / `.b16`; re-measured on bo5, `e4b.serve.buildout.bo5.granite.b1.5090.2026-09-04` / `.b16`)",
    ("granite", "int4_r12epi"): "measured, not licensed -- RTN int4 experts FAIL as registered: +0.063 ppl on wikitext, over the 0.05 budget (bo3 `int4exp`, retracted in #381; bo5 `rtnexp_r12epi` +0.0634; `e4b.serve.tp.granite.b1.5090.2026-09-04` CORRECTION and `e4b.serve.buildout.granite.b1.5090.2026-09-04` notes)",
    ("granite", "calibexp_r12epi"): "measured, not licensed -- calibrated int4 experts FAIL as registered on the second text: c4val1 +0.387 ppl (10x the 0.0033-nat floor), wikitext +0.014 (bo5; `e4b.serve.buildout.bo5.granite.b1.5090.2026-09-04` notes: refused)",
    ("olmoe", "nf4"): "baseline (reference); no paged-vs-own-attention parity receipt on this family",
    ("olmoe", "folds"): "no quality verdict on record -- the folds + epilogue were never scored on this family (the tp lane scored nf4 / int4exp / calib only, `e4b.serve.tp.olmoe.b1.5090.2026-09-04` notes); exact arithmetic (moves no weight); not licensed as a position",
    ("olmoe", "calattn"): "measured, not licensed -- calibrated int4 attention on OLMoE is refused on quality: +0.60 ppl on C4-val (private receipt; `e4b.serve.b1.qwen3-30b.int4attn-calib.5090` notes: Qwen3-specific, not a general lever); the tp lane's one wikitext reading (int4exp + calib 1.9295 vs NF4 1.9380) does not license a calibrated pack (the rule needs two texts)",
    ("olmoe", "int4all"): "measured, not licensed under the rule as written -- `e4b.serve.tp.olmoe.b1.5090.2026-09-04` calls this stack 'best licensed' on ONE text (wikitext, 2048 steps: int4exp -0.029 ppl, + calib -0.058 vs NF4); its calibrated attention pack needs two texts and carries the +0.60 C4-val FAIL above, and int4-b32 experts at <=1B active are the class STATUS records at ~1.2-1.8% ppl (Granite's retraction, #381); no second text on record",
    ("gptoss", "nf4_r12"): "the register's quoted best -- `e4b.serve.buildout.gptoss.b1.5090.2026-09-04` ('licensed configuration' in its claim text; its notes: no raw-text ppl instrument on this family, the folds' +0.049 nats cannot be read against the budget); exact arithmetic (norm/rotary folds); this lane's reference arm, ratio x1.000 by construction",
    ("gptoss", "store_r12"): "measured, not licensed -- quality gate OPEN on this family (no instrument; the MXFP4 store is exact against the checkpoint's own bytes, bo3o/bo3q): `e4b.serve.buildout.bo5.gptoss.b1.5090.2026-09-04` / `.b16` quote it as speed with the gate open",
    ("qwen3", "nf4"): "baseline (reference); paged decode at parity with the model's own attention, 0.00173 nats (`e4b.parity.qwen3.paged-vs-own-attention`)",
    ("qwen3", "folds"): "measured, not licensed as a position -- exact arithmetic (moves no weight); the verdict on record is bo5's: FAIL as registered on c4val1 by improving (-0.073 ppl, sub-floor; `e4b.serve.buildout.bo5.qwen3.b1.5090.2026-09-04` notes)",
    ("qwen3", "calattn"): "no quality verdict on record for calibrated attention on NF4 experts alone (the register scores it inside int4-expert stacks: bo5 `calib` = int4 experts + calibrated attention, c4val1 +0.048, one text); a calibrated pack needs two texts; not licensed",
    ("qwen3", "int4all"): "measured, not licensed -- bo5's `all` (RTN experts): FAIL as registered on c4val1 +0.063 ppl (noise-attributed, not retuned; `e4b.serve.buildout.bo5.qwen3.b1.5090.2026-09-04` / `.b16`)",
    ("qwen3", "calibexp_all"): "measured; the LICENSED configuration is the streamed 64k pack (bo6c, `e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`) -- this arm ran the hook's 16k default, and the 16k streamed full stack has no two-text verdict (bo6's two-text pass is the all-at-once 16k pack, `e4b.serve.buildout.bo6.qwen3.all-calibexp-allatonce.k8.2026-09-04`; the streamed 16k experts-only pack passed c4val1 only, `e4b.serve.buildout.bo6.qwen3.calibexp-streamed-16k.c4val1.2026-09-04`); the licensed stack's speed is therefore NOT yet measured (a follow-up arm)",
    ("qwen3", "calibexp_folds"): "measured, not licensed -- streamed 16k calibrated experts without calibrated attention: the 16k experts-only pack passed c4val1 (-0.0505) with wikitext unscored at 16k (`e4b.serve.buildout.bo6.qwen3.calibexp-streamed-16k.c4val1.2026-09-04`: pass on 1 text, needs 2); the 64k experts-only pack passes both texts (`e4b.serve.buildout.bo6c.qwen3.calibexp-streamed-64k.k8.2026-09-05`) but is not this arm's pack",
    ("gemma4", "nf4"): "baseline (reference); no reference exists for this family at 0.1-nat / 512-token resolution (`e4b.parity.gemma4.no-reference`, #359)",
    ("gemma4", "r1epi"): "exact arithmetic on NF4 experts (round-1 norm fold + router epilogue; the fold's kernel matches the module within 1 ULP per SERVING-THROUGHPUT) -- no K8 instrument exists for Gemma-4, so the register carries NO verdict for any Gemma-4 arm; quoted as the position that quantises nothing beyond NF4, with the register's caveat",
    ("gemma4", "int4_r1epi"): "measured, no quality verdict (no instrument) -- the register's quoted best for this family (bo3 `stack`; `e4b.serve.buildout.gemma4.b1.5090.2026-09-04` / `.b16`: 'licensed configuration' in the claim text, 'NO quality instrument on this family' in its notes and in SERVING-THROUGHPUT's gate column); not licensed by a verdict",
    ("gemma4", "calattn_r1epi"): "measured, no quality verdict (no instrument) -- a calibrated pack with an unreadable K8 (bo3 `best`, +0.235 nats vs `stack` inside the family's +-0.1-0.27 band; `e4b.serve.buildout.gemma4.b1.5090.2026-09-04` notes)",
    ("mixtral", "nf4"): "baseline (reference); no paged-vs-own-attention parity receipt on this family (the bf16 reference does not fit 32 GB)",
    ("mixtral", "folds"): "no quality verdict on record for the combined arm -- bo3 scored the pieces on wikitext, one text each: r12 +0.0009 ppl, epilogue -0.0087 ppl vs NF4 (bo3 receipts `mixtral_ppl_r12.json`, `mixtral_ppl_epi.json`; each within the uncalibrated 0.05 budget); exact arithmetic; not licensed as a position",
    ("mixtral", "lic"): "measured, not licensed -- RTN int4 experts + folds + epilogue: FAIL as registered on c4val1 +0.0575 ppl (by 0.008; floor unmeasured); the P30 'licensed stack' label is WITHDRAWN (`e4b.serve.buildout.bo5.mixtral.b1.5090.2026-09-04` / `.b16`)",
    ("mixtral", "all"): "measured, not licensed -- with calibrated int4 attention: FAIL as registered on c4val1 +0.116 ppl (`e4b.serve.buildout.mixtral.b1.5090.2026-09-04` / `.b16`, bo5 notes)",
}
# The licensed best per family, as the register stands at bundle time: (arm or None, basis). None = the licensed stack
# was not run on this lane (Qwen3: the 64k streamed pack). Where nothing above NF4 is licensed, the position is NF4.
LICENSED_BEST = {
    "granite": ("r12epi", "K8 verdict: +0.019 ppl wikitext, pass (`e4b.serve.buildout.granite.b1.5090.2026-09-04`)"),
    "olmoe": ("nf4", "nothing above NF4 is licensed on the register: the tp row's calibrated pack has one text (needs 2) and a +0.60 C4-val FAIL on record; the folds alone have no receipt"),
    "gptoss": ("nf4_r12", "the register's quoted best (`e4b.serve.buildout.gptoss.b1.5090.2026-09-04`); exact folds on NF4; no instrument on this family -- and it is this lane's reference arm"),
    "qwen3": (None, "the licensed stack is bo6c's streamed 64k pack (`e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`); bo7 ran the 16k default -- its speed is NOT measured on this box (follow-up arm)"),
    "gemma4": ("r1epi", "exact arithmetic (round-1 fold + epilogue) on NF4 experts; no K8 instrument on this family -- no verdict exists, the register's caveat applies"),
    "mixtral": ("nf4", "nothing above NF4 is licensed on the register: `lic` and `all` FAIL as registered (bo5), the calibrated stack FAILS (bo6b), the combined folds arm has no receipt"),
}
NEAREST = {"qwen3": "calibexp_all"}  # the measured arm nearest the unmeasured licensed stack (same stack, 16k pack)
PAT = re.compile(r"^(?P<fam>[a-z0-9]+)_(?P<kind>b1|b16)_(?P<arm>.+)\.json$")
ARMLINE = re.compile(r"^\[(?P<at>[^\]]+)\] arm (?P<fam>\w+)/(?P<arm>\S+) \(B=(?P<b>\d+) (?P<rest>.*)\)\s*$")
TS = re.compile(r"^\[[^\]]+\] ")


def fmt(v, p):
    return "—" if v is None else f"{v:.{p}f}"


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        fam, kind, arm = m.group("fam"), m.group("kind"), m.group("arm")
        j = json.load(open(f))
        r = rows.setdefault((fam, arm), {})
        if kind == "b1":
            r["b1_ms"] = round(j["step_ms_clean"], 2)
            r["b1_ms_raw"] = j["step_ms_clean"]
            r["b1_steps"] = j["n_steps"]
            r["mech"] = j.get("mech") or {}
        else:
            r["b16"] = j["aggregate_tok_s"]
            r["b16_ms"] = j["step_ms_clean"]
            r["b16_steps"] = j["n_steps"]
            r["b16_batch"] = j["batch"]
    return rows


def console(d):
    """(fam, arm, kind) -> {'at', 'alarm', 'fate'} from logs/outer.log; families whose bake failed."""
    fates, baked_fail = {}, set()
    p = os.path.join(d, "logs", "outer.log")
    if not os.path.exists(p):
        return fates, baked_fail
    cur = None
    for ln in open(p, errors="replace"):
        m = ARMLINE.match(ln)
        if m:
            am = re.search(r"alarm=(\d+)", m.group("rest"))
            cur = (m.group("fam"), m.group("arm"), "b" + m.group("b"))
            fates[cur] = {"at": m.group("at"), "alarm": int(am.group(1)) if am else None, "fate": None}
            continue
        if TS.match(ln) or re.match(r"^[A-Z0-9]+ DONE", ln):
            cur = None
        bf = re.match(r"^(\w+): BAKE FAILED", ln)
        if bf:
            baked_fail.add(bf.group(1))
        if cur and fates[cur]["fate"] is None:
            if "Alarm clock" in ln:
                fates[cur]["fate"] = "alarm"
            elif "Killed" in ln:
                fates[cur]["fate"] = "killed"
            elif re.search(r"REFUSED|Traceback|RuntimeError|Error\b", ln):
                fates[cur]["fate"] = "refused: " + ln.strip()[:160]
    return fates, baked_fail


def engaged(d, fam, arm, kind):
    """What the arm's run log says the hook engaged (INT4EXP / ATTNINT4 banners); '—' for arms with no quantised pack."""
    p = os.path.join(d, "logs", f"run_{fam}_{kind}_{arm}.log")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="replace").read()
    parts = []
    s = re.search(r"INT4EXP calibrating \(streamed\): (\d+) batches of \w+ budget GB (\S+)", txt)
    if s:
        tok = int(s.group(1)) * 4 * 512 // 1024
        g = r_ = 0
        for gq, rt in re.findall(r"INT4EXP calibrated experts: (\d+) gptq / (\d+) rtn", txt):
            g += int(gq)
            r_ += int(rt)
        ps = re.search(r"INT4EXP calibrated streaming: (\d+) layers in (\d+) passes", txt)
        parts.append(f"streamed GPTQ {tok}k tok, {s.group(2)} GiB budget" + (f", {ps.group(2)} passes" if ps else "")
                     + f", {g} gptq / {r_} rtn")
    e = re.search(r"INT4EXP enabled: (\d+) layers \(model_type=(\w+)\)", txt)
    if e:
        parts.append(f"INT4EXP {e.group(1)} layers ({e.group(2)})")
    a = re.search(r"ATTNINT4 calibrated: (\d+) projections", txt)
    if a:
        parts.append(f"ATTNINT4 {a.group(1)} projections")
    if "REFUSED" in txt:
        parts.append("REFUSED")
    return " · ".join(parts) if parts else "—"


def status_cell(fates, baked_fail, fam, arm, kind, have):
    if have:
        return "ok"
    if fam in baked_fail:
        return "bake failed"
    f = fates.get((fam, arm, kind))
    if not f:
        return "pending"
    if f["fate"] == "alarm":
        return f"alarm ({f['alarm']} s, started {f['at']})"
    if f["fate"] == "killed":
        return "killed"
    if f["fate"]:
        return f["fate"]
    return f"running (started {f['at']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    a = ap.parse_args()
    rows = load(a.dir)
    fates, baked_fail = console(a.dir)
    legend = set()
    best = {}
    for fam in FAMS:
        fams = {arm: r for (f, arm), r in rows.items() if f == fam}
        nf = NF4[fam]
        base = fams.get(nf, {})
        print(f"\n### {NAME[fam]}")
        print(f"Ratios are to `{nf}` on this box ({DESC[(fam, nf)]}). B=1 tok/s = 1000 / the timed step (ms to 0.01, the "
              "logs' resolution); B=16 = aggregate tok/s over 70 graph steps at batch 16. The licence column is the register's, "
              "not this lane's.")
        print("| arm | configuration | licence (register) | engaged (run log) | B=1 ms | B=1 tok/s | ×NF4 (B=1) | B=16 tok/s | ×NF4 (B=16) | status |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for arm in ORDER[fam]:
            r = fams.get(arm, {})
            b1 = r.get("b1_ms")
            b16 = r.get("b16")
            tps = 1000 / b1 if b1 else None
            x1 = (base["b1_ms"] / b1) if (b1 and base.get("b1_ms")) else None
            x16 = (b16 / base["b16"]) if (b16 and base.get("b16")) else None
            s1 = status_cell(fates, baked_fail, fam, arm, "b1", b1 is not None)
            s16 = status_cell(fates, baked_fail, fam, arm, "b16", b16 is not None)
            st = "ok" if (s1 == "ok" and s16 == "ok") else f"B=1 {s1}; B=16 {s16}"
            for s in (s1, s16):
                if s.startswith("pending"):
                    legend.add("pending")
                elif s.startswith("alarm"):
                    legend.add("alarm")
                elif s.startswith("refused"):
                    legend.add("refused")
            eng = engaged(a.dir, fam, arm, "b1") or engaged(a.dir, fam, arm, "b16") or "—"
            print(f"| `{arm}` | {DESC[(fam, arm)]} | {LICENCE[(fam, arm)]} | {eng} | {fmt(b1, 2)} | {fmt(tps, 1)} | "
                  f"{fmt(x1, 3)} | {fmt(b16, 1)} | {fmt(x16, 3)} | {st} |")
            r["x1"], r["x16"], r["tps"] = x1, x16, tps
        for arm, why in DROPPED.get(fam, []):
            print(f"| `{arm}` | — | — | — | — | — | — | — | — | {why} |")
        # instrument line: the kernel's own tallies from the B=1 receipts
        mech = {arm: r.get("mech") for arm, r in fams.items() if r.get("mech")}
        if mech:
            comp = sorted({json.dumps(m.get("compute"), sort_keys=True) for m in mech.values()})
            disp = {arm: {k: v for k, v in (m.get("dispatch") or {}).items() if v} for arm, m in mech.items()}
            print(f"\nInstrument ({NAME[fam]}, from the B=1 receipts' `mech` tallies, counted at warmup + graph capture): decode-attention "
                  f"compute path {' / '.join(comp)}"
                  + ("" if len(comp) == 1 else " -- DIFFERS BETWEEN ARMS")
                  + "; NF4 GEMV dispatch " + ", ".join(f"`{arm}` {d or 'none (int4 / MXFP4 path)'}" for arm, d in sorted(disp.items())) + ".")
        # licensed best
        lb, basis = LICENSED_BEST[fam]
        if lb is None:
            near = fams.get(NEAREST.get(fam, ""), {})
            best[fam] = {"arm": None, "basis": basis, "near": NEAREST.get(fam), "near_r": near}
        else:
            best[fam] = {"arm": lb, "basis": basis, "r": fams.get(lb, {})}
    # ---- licensed best per family, three axes
    print("\n### Licensed best per family -- the three axes (rule 3)")
    print("Ratio = the family's own NF4 arm on this box; tok/s = rental-measured on this box (one RTX 5090, driver 595.84, on an AMD EPYC "
          "7Q83 host, Vast.ai instance 49916675); the anchor-class projection is `159.2 tok/s x the B=1 ratio` and exists ONLY for "
          "Qwen3-30B-A3B at B=1 (the anchor class was never certified: 12 refusals) -- every other cell says so.")
    print("| family | licensed configuration | licence basis (register) | ×NF4 (B=1) | B=1 tok/s (this box) | anchor-class projection (B=1) | ×NF4 (B=16) | B=16 tok/s (this box) | anchor (B=16) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for fam in FAMS:
        b = best[fam]
        no_anchor = "no anchor projection (no anchor-class measurement exists for this family)"
        if b["arm"] is None:
            nr = b["near_r"]
            print(f"| {NAME[fam]} | NOT MEASURED on this lane -- {b['basis']} | see the nearest measured arm below | pending a follow-up arm | pending | "
                  f"not computed (rule: 159.2 x the licensed ratio; the licensed ratio is not measured on this box) | pending | pending | no anchor projection (B=16 has no anchor-class measurement) |")
            if nr:
                print(f"| {NAME[fam]} (nearest measured arm, `{b['near']}` -- the same stack with the 16k pack: measured, NOT the licensed pack) | "
                      f"{DESC[(fam, b['near'])]} | measured, not the licensed configuration | {fmt(nr.get('x1'), 3)} | {fmt(nr.get('tps'), 1)} | "
                      f"not quoted (an unlicensed arm gets no projection) | {fmt(nr.get('x16'), 3)} | {fmt(nr.get('b16'), 1)} | no anchor projection (B=16) |")
            continue
        r = b["r"]
        arm = b["arm"]
        pend = r.get("b1_ms") is None or r.get("b16") is None
        x1 = "1.000 (reference arm)" if arm == NF4[fam] and r.get("b1_ms") else fmt(r.get("x1"), 3)
        x16 = "1.000 (reference arm)" if arm == NF4[fam] and r.get("b16") else fmt(r.get("x16"), 3)
        anc1 = no_anchor
        if fam == "qwen3" and r.get("x1"):
            anc1 = f"≈ {ANCHOR_NF4 * r['x1']:.0f} tok/s (PROJECTION: 159.2 x {r['x1']:.3f}; uncertified)"
        cfg = f"`{arm}` -- {DESC[(fam, arm)]}"
        if pend:
            legend.add("pending")
        print(f"| {NAME[fam]} | {cfg} | {b['basis']} | {x1 if r.get('b1_ms') else 'pending'} | {fmt(r.get('tps'), 1) if r.get('b1_ms') else 'pending'} | "
              f"{anc1 if r.get('b1_ms') else 'pending'} | {x16 if r.get('b16') else 'pending'} | {fmt(r.get('b16'), 1) if r.get('b16') else 'pending'} | "
              f"no anchor projection (B=16 has no anchor-class measurement) |")
    # ---- cross-family summary of ratios (never absolutes across families)
    print("\n### Cross-family summary -- ratios only (families differ in size; no absolute is compared across families)")
    print("| family | licensed best: ×NF4 B=1 / B=16 | fastest measured arm on this box: ×NF4 B=1 (arm) | ×NF4 B=16 (arm) | that arm's label |")
    print("|---|---|---|---|---|")
    for fam in FAMS:
        fams = {arm: r for (f, arm), r in rows.items() if f == fam}
        b = best[fam]
        if b["arm"] is None:
            lic = "not measured (licensed stack not on this lane)"
        else:
            r = b["r"]
            lic = (f"{'1.000' if b['arm'] == NF4[fam] else fmt(r.get('x1'), 3)} / "
                   f"{'1.000' if b['arm'] == NF4[fam] else fmt(r.get('x16'), 3)} (`{b['arm']}`)") if r.get("b1_ms") and r.get("b16") else "pending"
        # fastest arm by the receipt's unrounded step (ties in the rounded ms are broken by the raw value); the ratio
        # printed is the rounded-convention one from the family table
        have1 = [(r["b1_ms_raw"], arm) for arm, r in fams.items() if r.get("b1_ms_raw")]
        have16 = [(-r["b16"], arm) for arm, r in fams.items() if r.get("b16")]
        m1 = min(have1)[1] if have1 else None
        m16 = min(have16)[1] if have16 else None
        c1 = f"{fmt(fams[m1]['x1'], 3)} (`{m1}`)" if m1 else "pending"
        c16 = f"{fmt(fams[m16]['x16'], 3)} (`{m16}`)" if m16 else "pending"
        lab = LICENCE[(fam, m1)].split(" -- ")[0] if m1 else "pending"
        print(f"| {NAME[fam]} | {lic} | {c1} | {c16} | {lab} |")
    print("\nEvery ratio above is to the family's own NF4 arm on this box in this session (rule 1). Labels are the register's (rule 2); "
          "`measured, not licensed` rows are speed of configurations the register does not license (rule 4) and are never quoted as a "
          "position. No number here is divided into a bo3 / bo5 / bo6 number (rule 5).")
    if "pending" in legend:
        print("`pending` = an arm the lane script runs whose receipt is not in this snapshot yet (arrives before merge).")
    if "alarm" in legend:
        print("`alarm (N s, started T)` = an arm killed by its own N-second alarm before it produced a receipt -- a harness limit, not a "
              "model result; no number is quoted for it.")
    if "refused" in legend:
        print("`refused: ...` = an arm the harness refused with a sentence (printed) -- a row, never a zero.")


if __name__ == "__main__":
    main()
