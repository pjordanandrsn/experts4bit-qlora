"""Reduce lane bo6's JSON receipts (<fam>_ppl_<arm>.json / <fam>_b1_<arm>.json / <fam>_b16_<arm>.json) into
one per-family table with the registered K8 gate applied in ITS OWN units (perplexity) on every text the arm
was scored on, nats quoted beside the verdict (they never change it), the calibration METHOD of every arm read
from its own run log, a repeat-controls table, the calibration-size / damping sweep, and the speed rows.

Usage: buildout_reduce.py <dir>

  <dir>   this bundle. A `_c4val` suffix on a ppl arm marks the second text (the JSON's own `ppl_source`
          field is what is read; the suffix is only stripped to name the configuration). Every K8 delta is
          against the NF4 arm re-scored on THIS lane (same box, same cut family, same window sha) -- there is
          no `--ref`: nothing on this page is compared across lanes. There is no NF4 speed arm on this lane
          either, so no ratio is printed; bo7 measures the same-box ratio.

Method, per arm, is read from `logs/run_<fam>_ppl_<arm>.log` (the hook prints one of two banners):
  `INT4EXP hessians: L layers, P (layer, expert) pairs from N batches of c4`   -> ALL-AT-ONCE: every layer's
      Hessian accumulated against the unquantised (NF4) prefix, then every layer packed (hook v4/v5 path,
      e4b integration-8 @db2a070);
  `INT4EXP calibrating (streamed): N batches of c4 budget GB G`                -> STREAMED: layers packed in
      chunks, each chunk's Hessians accumulated with the earlier chunks already on int4 -- GPTQ's sequential
      convention (`enable_serve_experts_int4_calibrated`, e4b @ae9dc122, hook v6).
Calibration tokens = N batches x 4 sequences x 512 tokens (the hook's `_calib_batches`); damping and the
`E4B_CALIB_NSEQ` override come from the arm's line in `logs/outer.log` (default damping = the kernel's 0.01).

An arm the lane script runs that has no receipt is read from `logs/outer.log`: a `... Alarm clock ...` line after the
arm's own line means the arm was killed by the lane's per-arm alarm (`perl -e 'alarm N'`) and prints as an `alarm`
row -- a harness limit, not a model result, and no number is quoted for it; a `... Killed ...` line is the container's
OOM kill (attempt 3). Neither is a verdict.

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

FLOOR = {"qwen3": 0.0095, "mixtral": None}
NAME = {"qwen3": "Qwen3-30B-A3B", "mixtral": "Mixtral-8x7B-Instruct"}
FAMS = ["qwen3", "mixtral"]
TEXTS = ["wikitext", "c4val1"]
BUDGET = 0.05
# The e4b cut each arm ran on (all with grouped-nf4-gemm main @0b25d13): attempt 2 @f42924d (the NF4
# references), attempt 3 @db2a070 (pack returns to the stack's device; all-at-once calibration), bo6b @ae9dc122
# (E4B_INT4_GPTQ_DAMP + streamed calibration). NF4 numerics do not touch #384 and do not depend on the cut.
CUT = {("qwen3", "nf4"): "f42924d", ("qwen3", "nf4_rep1"): "ae9dc12", ("qwen3", "nf4_c4val_rep1"): "ae9dc12",
       ("qwen3", "nf4_c4val_rep2"): "ae9dc12", ("qwen3", "calibexp"): "db2a070", ("qwen3", "all_calibexp"): "db2a070",
       ("qwen3", "calibexp_c4val_rep1"): "ae9dc12", ("qwen3", "calibexp_d01"): "ae9dc12",
       ("qwen3", "calibexp_n128"): "ae9dc12", ("qwen3", "calibexp_n512"): "ae9dc12",
       ("mixtral", "nf4"): "db2a070", ("mixtral", "lic_calibexp"): "ae9dc12", ("mixtral", "lic_calibexp_n128"): "ae9dc12"}
K8_BASE = {"qwen3": "nf4", "mixtral": "nf4"}
# Every arm below carries a GPTQ-calibrated int4 expert pack (and all_calibexp the C4-calibrated int4 attention
# pack as well): the calibrated rule applies.
CALIBRATED = {("qwen3", "calibexp"), ("qwen3", "all_calibexp"), ("qwen3", "calibexp_c4val_rep1"),
              ("qwen3", "calibexp_d01"), ("qwen3", "calibexp_n128"), ("qwen3", "calibexp_n512"),
              ("mixtral", "lic_calibexp"), ("mixtral", "lic_calibexp_n128")}
ORDER = {"qwen3": ["nf4", "calibexp", "all_calibexp", "calibexp_c4val_rep1", "calibexp_d01", "calibexp_n128", "calibexp_n512"],
         "mixtral": ["nf4", "lic_calibexp", "lic_calibexp_n128"]}
# The texts (and speed arms) the lane scripts run for each configuration; a missing receipt prints as pending.
EXPECTED = {("qwen3", "nf4"): {"wikitext", "c4val1"}, ("qwen3", "calibexp"): {"c4val1"},
            ("qwen3", "all_calibexp"): {"wikitext", "c4val1", "b1", "b16"}, ("qwen3", "calibexp_c4val_rep1"): {"c4val1"},
            ("qwen3", "calibexp_d01"): {"c4val1"}, ("qwen3", "calibexp_n128"): {"c4val1"}, ("qwen3", "calibexp_n512"): {"c4val1"},
            ("mixtral", "nf4"): {"wikitext", "c4val1"}, ("mixtral", "lic_calibexp"): {"wikitext", "c4val1", "b1", "b16"},
            ("mixtral", "lic_calibexp_n128"): {"c4val1"}}
REPEATS = {"qwen3": [("wikitext", "nf4", ["nf4_rep1"]), ("c4val1", "nf4", ["nf4_c4val_rep1", "nf4_c4val_rep2"])]}
SWEEP = {"qwen3": ("c4val1", ["calibexp", "calibexp_c4val_rep1", "calibexp_d01", "calibexp_n128", "calibexp_n512"])}
DESC = {("qwen3", "nf4"): "NF4 experts, bf16 attention -- re-scored on this box and cut (baseline, both texts)",
        ("qwen3", "calibexp"): "GPTQ-calibrated int4 experts alone (attempt 3; the text that failed with RTN experts on bo5)",
        ("qwen3", "all_calibexp"): "calibrated int4 experts + C4-calibrated int4 attention + r1 + r2 (rope-only fold) + router epilogue + #385 glue (attempt 3)",
        ("qwen3", "calibexp_c4val_rep1"): "calibrated int4 experts alone, queued as a repeat of `calibexp` -- the cut had changed method (bo6b)",
        ("qwen3", "calibexp_d01"): "calibrated int4 experts alone, damping 0.1 (E4B_INT4_GPTQ_DAMP=0.1; bo6b sweep)",
        ("qwen3", "calibexp_n128"): "calibrated int4 experts alone, 4x the calibration set (E4B_CALIB_NSEQ=128; bo6b sweep)",
        ("qwen3", "calibexp_n512"): "calibrated int4 experts alone, 16x the calibration set (E4B_CALIB_NSEQ=512; bo6b sweep)",
        ("mixtral", "nf4"): "NF4 experts, bf16 attention -- re-scored on this box and cut (baseline, both texts)",
        ("mixtral", "lic_calibexp"): "GPTQ-calibrated int4 experts + r1 + r2 (rope-only fold) + router epilogue, no calibrated attention -- bo5's `lic` with calibrated experts (bo6b, 8 GiB Hessian budget)",
        ("mixtral", "lic_calibexp_n128"): "lic_calibexp with 4x the calibration set (E4B_CALIB_NSEQ=128; bo6b)"}
PAT = re.compile(r"^(?P<fam>[a-z0-9]+)_(?P<kind>ppl|b1|b16)_(?P<arm>.+)\.json$")
OUTER = re.compile(r"^\[(?P<at>[^\]]+)\] (?:bo6[bc]: )?(?P<kind>K8|arm) (?P<fam>\w+)/(?P<arm>\S+) \((?P<env>.*)\)\s*$")
FATE = re.compile(r"^\./\S+\.sh: line \d+: +\d+ (?P<fate>Alarm clock|Killed)\b.*?alarm (?P<secs>\d+); exec")


def fmt(v, p, sign=False):
    if v is None:
        return "—"
    return f"{v:+.{p}f}" if sign else f"{v:.{p}f}"


def cfg_of(arm):
    return arm[:-len("_c4val")] if arm.endswith("_c4val") else arm


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        fam, kind, arm = m.group("fam"), m.group("kind"), m.group("arm")
        j = json.load(open(f))
        if kind == "ppl":
            r = rows.setdefault((fam, cfg_of(arm)), {"ppl": {}})
            r["ppl"][j["ppl_source"]] = {"nll": j["mean_nll"], "ppl": j["ppl"], "sha": j["text_sha"][:12],
                                         "steps": j["steps"], "arm": arm, "file": os.path.basename(f)}
        else:
            r = rows.setdefault((fam, arm), {"ppl": {}})
            if kind == "b1":
                r["b1_ms"] = round(j["step_ms_clean"], 2)
                r["b1_steps"] = j["n_steps"]
            else:
                r["b16"] = j["aggregate_tok_s"]
                r["b16_ms"] = j["step_ms_clean"]
                r["b16_steps"] = j["n_steps"]
    return rows


def outer_env(d):
    """(fam, arm-as-named-in-the-script) -> the env tokens on its `K8` / `arm` line in outer.log (last wins), and
    (fam, arm, kind) -> the fate of an arm that died (`Alarm clock` / `Killed`, the alarm seconds, when it started)."""
    env, fates, last = {}, {}, None
    p = os.path.join(d, "logs", "outer.log")
    if not os.path.exists(p):
        return env, fates
    for ln in open(p, errors="replace"):
        m = OUTER.match(ln)
        if m:
            e = dict(t.split("=", 1) for t in m.group("env").split() if "=" in t)
            env[(m.group("fam"), m.group("arm"))] = e
            kind = "ppl" if m.group("kind") == "K8" else "b" + e.get("B", "?")
            last = (m.group("fam"), m.group("arm"), kind, m.group("at"))
            fates.pop(last[:3], None)  # a re-run supersedes an earlier fate; a receipt supersedes both
            continue
        f = FATE.match(ln)
        if f and last:
            fates[last[:3]] = {"fate": f.group("fate"), "secs": int(f.group("secs")), "at": last[3]}
    return env, fates


def fate_cell(d, fam, arm, kind, fates, long=False):
    """The cell for an expected arm with no receipt: `alarm ...` / `killed ...` from outer.log, else pending."""
    f = fates.get((fam, arm, kind))
    if not f:
        return "pending (arrives before merge)"
    p = os.path.join(d, "logs", f"run_{fam}_{kind}_{arm}.log")
    txt = open(p, errors="replace").read() if os.path.exists(p) else ""
    passes = len(re.findall(r"INT4EXP calibrated experts:", txt))
    ml = re.search(r"quantized experts on (\d+)/(\d+) MoE layers", txt)
    layers = ml.group(2) if ml else "?"
    if f["fate"] == "Killed":
        return "killed (container OOM)"
    at = f["at"]
    if long:
        return (f"NOT MEASURED -- killed by the arm's own {f['secs']}-s alarm (started {at}) at streamed-calibration pass "
                f"{passes}/{layers}: a harness limit, not a model result; no number")
    return f"alarm ({f['secs']} s, calibration pass {passes}/{layers})"


def method(d, fam, arm, env):
    """Read the calibration method of one K8 arm from its run log; None when the arm has no calibrated experts."""
    p = os.path.join(d, "logs", f"run_{fam}_ppl_{arm}.log")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="replace").read()
    m = {"gptq": 0, "rtn": 0}
    a = re.search(r"INT4EXP hessians: (\d+) layers, (\d+) \(layer, expert\) pairs from (\d+) batches", txt)
    s = re.search(r"INT4EXP calibrating \(streamed\): (\d+) batches of \w+ budget GB (\S+)", txt)
    if a:
        m.update(kind="all-at-once", batches=int(a.group(3)), pairs=int(a.group(2)))
    elif s:
        m.update(kind="streamed", batches=int(s.group(1)), budget=s.group(2))
        p2 = re.search(r"INT4EXP calibrated streaming: (\d+) layers in (\d+) passes", txt)
        if p2:
            m["passes"] = int(p2.group(2))
    else:
        return None
    for g, r in re.findall(r"INT4EXP calibrated experts: (\d+) gptq / (\d+) rtn", txt):
        m["gptq"] += int(g)
        m["rtn"] += int(r)
    e = env.get((fam, arm), {})
    m["damp"] = e.get("E4B_INT4_GPTQ_DAMP", "0.01")
    m["nseq"] = int(e.get("E4B_CALIB_NSEQ", "32"))
    m["tokens"] = m["batches"] * 4 * 512
    m["complete"] = "INT4EXP enabled:" in txt
    return m


def method_cell(m):
    if m is None:
        return "—"
    s = f"{m['kind']} · {m['tokens'] // 1024}k tok · damp {m['damp']}"
    if m["kind"] == "streamed":
        s += f" · {m['budget']} GiB budget"
        if m.get("passes"):
            s += f", {m['passes']} passes"
    if m["gptq"] or m["rtn"]:
        s += f" · {m['gptq']} gptq / {m['rtn']} rtn"
    if not m.get("complete", True):
        s += " (INCOMPLETE: the arm alarmed mid-calibration; the pack counts are the passes that finished)"
    return s


def fails_in(ver):
    return ver.startswith("FAIL")


def verdict(fam, cfg, deltas, is_base, pending):
    """deltas: {text: dppl}. Returns (per-text cells, registered verdict) in ppl units."""
    if is_base:
        return {t: "baseline" for t in deltas}, "baseline"
    cal = (fam, cfg) in CALIBRATED
    cells, fails = {}, []
    for t, dp in deltas.items():
        bad = (dp > BUDGET) if cal else (abs(dp) > BUDGET)
        cells[t] = "FAIL" if bad else "pass"
        if bad:
            fails.append(f"{t} {dp:+.4f}" + ("" if cal or dp > 0 else " by improving"))
    if fails:
        return cells, "FAIL as registered (" + "; ".join(fails) + ")"
    if pending:
        return cells, f"pending ({', '.join(sorted(pending))} arrives before merge)"
    n = len(deltas)
    if cal:
        if n >= 2 and all(dp < 0 for dp in deltas.values()):
            return cells, f"pass — licensed (improves on {n} texts)"
        if n >= 2:
            return cells, f"pass on {n} texts (within +{BUDGET}; no same-sign improvement to claim)"
        if all(dp < 0 for dp in deltas.values()):
            return cells, "pass (1 text; the improvement is not claimable until wikitext agrees)"
        return cells, "pass (1 text; needs 2)"
    return cells, f"pass ({n} text{'s' if n > 1 else ''})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    a = ap.parse_args()
    rows = load(a.dir)
    env, fates = outer_env(a.dir)
    legend = set()
    for fam in FAMS:
        fams = {c: r for (f, c), r in rows.items() if f == fam}
        if not fams:
            continue
        floor = FLOOR[fam]
        base = fams.get(K8_BASE[fam], {"ppl": {}})
        print(f"\n### {NAME[fam]}")
        hdr = (f"arithmetic-order floor {floor} nats" if floor else "arithmetic-order floor unmeasured")
        fl = []
        for t in TEXTS:
            bp = base["ppl"].get(t)
            if bp and floor:
                fl.append(f"{t}: base ppl {bp['ppl']:.3f} → ±{bp['ppl'] * (math.exp(floor) - 1):.3f} ppl")
        if fl:
            hdr += " = " + "; ".join(fl)
        print(hdr + ". Every delta is against `nf4` re-scored on this lane (same box, same window sha). "
              "Speed ratio: not measured on this lane (no NF4 speed arm; bo7 measures it).")
        print("| arm | configuration | method (from the run log) | e4b cut | K8 wikitext nll | Δppl (Δnats) | gate | K8 c4val1 nll | Δppl (Δnats) | gate | registered verdict (ppl) | B=1 ms | B=1 tok/s | B=16 tok/s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = ORDER[fam] + sorted(c for c in fams if c not in ORDER[fam] and not any(c in reps for _, _, reps in REPEATS.get(fam, [])))
        for cfg in order:
            r = fams.get(cfg, {"ppl": {}})
            exp = EXPECTED.get((fam, cfg), set())
            if not r["ppl"] and not r.get("b1_ms") and not r.get("b16") and not exp:
                continue
            is_base = cfg == K8_BASE[fam]
            nll_s, d_s, deltas, pending, dead, meth = {}, {}, {}, set(), set(), None
            for t in TEXTS:
                p = r["ppl"].get(t)
                if not p:
                    d_s[t] = "—"
                    if t not in exp:
                        nll_s[t] = "—"
                        continue
                    arm_t = cfg if t == "wikitext" else f"{cfg}_c4val"
                    nll_s[t] = fate_cell(a.dir, fam, arm_t, "ppl", fates)
                    (dead if (fam, arm_t, "ppl") in fates else pending).add(t)
                    legend.add("alarm" if (fam, arm_t, "ppl") in fates else "pending")
                    continue
                nll_s[t] = f"{p['nll']:.5f}"
                mm = method(a.dir, fam, p["arm"], env)
                meth = meth or mm
                if is_base:
                    d_s[t] = "—"
                    deltas[t] = 0.0
                    continue
                bp = base["ppl"].get(t)
                if not bp or bp["sha"] != p["sha"] or bp["steps"] != p["steps"]:
                    d_s[t] = "— (no matching baseline)"
                    continue
                dn = p["nll"] - bp["nll"]
                dp = p["ppl"] - bp["ppl"]
                deltas[t] = dp
                tag = ""
                if floor is not None:
                    tag = " sub-floor" if abs(dn) <= floor else f" {abs(dn) / floor:.1f}× floor"
                d_s[t] = f"{dp:+.4f} ({dn:+.4f}{tag})"
            if meth is None and (fam, cfg) in CALIBRATED:
                for t in TEXTS:  # a pending arm whose log has started
                    meth = meth or method(a.dir, fam, cfg if t == "wikitext" else f"{cfg}_c4val", env)
            cells, ver = verdict(fam, cfg, deltas, is_base, pending)
            if dead and deltas and not fails_in(ver):
                ver += f" ({', '.join(sorted(dead))} not measured: alarm)"
            if not deltas and not is_base:
                ver = "pending (arrives before merge)" if pending else ("not measured (alarm)" if dead else "—")
            b1 = r.get("b1_ms")
            tps = 1000 / b1 if b1 else None
            b16 = r.get("b16")

            def speed_cell(kind, val, p):
                if val is not None:
                    return fmt(val, p)
                if kind not in exp:
                    return "—"
                c = fate_cell(a.dir, fam, cfg, kind, fates)
                legend.add("alarm" if (fam, cfg, kind) in fates else "pending")
                return "pending" if c.startswith("pending") else c
            b1_s = speed_cell("b1", b1, 2)
            tps_s = speed_cell("b1", tps, 1)
            b16_s = speed_cell("b16", b16, 1)
            print(f"| `{cfg}` | {DESC.get((fam, cfg), cfg)} | {method_cell(meth)} | {CUT.get((fam, cfg), '—')} | "
                  f"{nll_s['wikitext']} | {d_s['wikitext']} | {cells.get('wikitext', '—')} | {nll_s['c4val1']} | "
                  f"{d_s['c4val1']} | {cells.get('c4val1', '—')} | {ver} | {b1_s} | {tps_s} | {b16_s} |")
        # ---- repeat controls
        reps = REPEATS.get(fam)
        if reps:
            print(f"\nRepeat controls ({NAME[fam]}; the released checkpoint re-baked to NF4 by bo6b after attempt 3 freed the arena, then re-scored on the same box and window; `mean_nll` compared at full float precision):")
            print("| text | original arm | nll | repeats | nll | spread (nats) | reading |")
            print("|---|---|---|---|---|---|---|")
            for t, orig, rl in reps:
                o = fams.get(orig, {}).get("ppl", {}).get(t)
                got = [(c, fams[c]["ppl"][t]) for c in rl if c in fams and t in fams[c]["ppl"]]
                if not o or not got:
                    continue
                vals = [o["nll"]] + [g["nll"] for _, g in got]
                spread = max(vals) - min(vals)
                same = all(g["sha"] == o["sha"] for _, g in got)
                reading = ("bit-identical → the K8 instrument is run-to-run deterministic on one box + cut" if spread == 0 and same
                           else f"spread {spread:.2e} nats" + ("" if same else " (WINDOW SHA DIFFERS)"))
                print(f"| {t} | `{o['arm']}` | {o['nll']:.16g} | " + ", ".join(f"`{g['arm']}`" for _, g in got) + " | "
                      + " / ".join(f"{g['nll']:.16g}" for _, g in got) + f" | {spread:.1e} | {reading} |")
        # ---- sweep
        sw = SWEEP.get(fam)
        if sw:
            t, arms = sw
            bp = base["ppl"].get(t)
            have = [c for c in arms if c in fams and t in fams[c]["ppl"]]
            if bp and have:
                print(f"\nCalibration method / size / damping sweep ({NAME[fam]}, {t}, calibrated int4 experts alone; NF4 {bp['nll']:.5f} / ppl {bp['ppl']:.5f} on this lane):")
                print("| arm | method | calibration tokens | damping | gptq / rtn packs | nll | ppl | Δppl (Δnats) | gate |")
                print("|---|---|---|---|---|---|---|---|---|")
                by_key = {}
                for c in have:
                    p = fams[c]["ppl"][t]
                    m = method(a.dir, fam, p["arm"], env) or {}
                    dn, dp = p["nll"] - bp["nll"], p["ppl"] - bp["ppl"]
                    tag = "" if floor is None else (" sub-floor" if abs(dn) <= floor else f" {abs(dn) / floor:.1f}× floor")
                    cell = "FAIL" if dp > BUDGET else "pass"
                    print(f"| `{c}` | {m.get('kind', '—')} | {m['tokens'] // 1024}k | {m.get('damp', '—')} | {m['gptq']} / {m['rtn']} | "
                          f"{p['nll']:.5f} | {p['ppl']:.5f} | {dp:+.4f} ({dn:+.4f}{tag}) | {cell} |")
                    by_key[(m.get("kind"), m.get("tokens"), m.get("damp"))] = p
                a1, s1 = by_key.get(("all-at-once", 16384, "0.01")), by_key.get(("streamed", 16384, "0.01"))
                if a1 and s1:
                    dn, dp = s1["nll"] - a1["nll"], s1["ppl"] - a1["ppl"]
                    tag = "" if floor is None else f" ({abs(dn) / floor:.1f}× the {floor}-nat floor)"
                    print(f"\nMethod effect at 16k tokens, damping 0.01, the same 8 calibration batches, the same box: all-at-once "
                          f"{a1['ppl']:.5f} → streamed {s1['ppl']:.5f} = {dp:+.4f} ppl ({dn:+.4f} nats{tag}); registered verdict "
                          f"{'FAIL' if a1['ppl'] - bp['ppl'] > BUDGET else 'pass'} → {'FAIL' if s1['ppl'] - bp['ppl'] > BUDGET else 'pass'}.")
        # ---- speed
        sp = [(c, r) for c, r in fams.items() if r.get("b1_ms") or r.get("b16")]
        sp += [(c, {}) for c in ORDER[fam] if c not in dict(sp) and {"b1", "b16"} & EXPECTED.get((fam, c), set())]
        if sp:
            print(f"\nSpeed on this lane ({NAME[fam]}; rental-measured tok/s on this box, one RTX 5090 on a Threadripper PRO 7975WX host; "
                  "B=1 = 1000 / timed graph step, B=16 = aggregate over 70 graph steps). Ratio: not measured on this lane (bo7 measures it).")
            for c, r in sp:
                b1, b16 = r.get("b1_ms"), r.get("b16")
                s = f"- `{c}`: B=1 " + (f"{b1:.2f} ms = {1000 / b1:.1f} tok/s ({r.get('b1_steps')} timed steps)" if b1
                                        else fate_cell(a.dir, fam, c, "b1", fates, long=True))
                s += "; B=16 " + (f"{b16:.1f} tok/s ({r.get('b16_ms'):.2f} ms/step, {r.get('b16_steps')} steps)" if b16
                                  else fate_cell(a.dir, fam, c, "b16", fates, long=True))
                print(s)
    print("\nGate cells and the registered verdict are in perplexity, the registered unit; the nats beside them are read against the"
          " family's arithmetic-order floor and never change the verdict. `pass (1 text; ...)` = a calibrated pack within +0.05 ppl on"
          " the one text scored; an improvement is claimable only with the same sign on wikitext (outside the calibration domain).")
    if "alarm" in legend:
        print("`alarm (N s, calibration pass k/L)` = an arm the lane script runs that was killed by its own N-second alarm (`Alarm clock` in"
              " `logs/outer.log`) after k of the L streamed-calibration passes, before it produced a receipt -- a harness limit, not a model"
              " result; no number is quoted for it.")
    if "pending" in legend:
        print("`pending (arrives before merge)` = an arm the lane script runs whose receipt is not in this snapshot yet.")


if __name__ == "__main__":
    main()
