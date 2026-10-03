#!/usr/bin/env python3
"""Lane P106's reducer (bench/p106/PREREG-p106.md): the Gated DeltaNet kernels' quality against transformers' torch path,
and their prefill TTFT, on Qwen3.6-35B-A3B through e4b's hybrid paged path, both paths in ONE process (``p106_box.py``).

Inputs in the run directory:
- ``box.json``: the box's record (``p106_box.py``);
- ``kernels_fc.json``: what transformers resolved at import after both installs (p106_run.sh's engagement probe);
- ``toggle_check.json``: the cross-process check that ``GdnToggle("torch")`` in the kernel process reproduces a
  kernel-free process's logits bit for bit (``toggle_probe.py``, tiny dense and MoE hybrids);
- ``summary.txt``: the premise line (``premise fc ok`` or ``premise fc failed``).

The verdict:
- NO_READING: no premise line, a failed premise, or no box record (nothing is read);
- VOID, any of:
  - the toggle check did not pass;
  - the engagement is not as registered: kernels_fc.json and the box's resolved record name fla's chunk and fused
    recurrent rules and causal-conv1d's interface; the box's torch record names transformers' modules only; the
    model's Gated DeltaNet modules are among the toggled modules;
  - the loaded commit is not the pinned revision;
  - the kernels' logits are bit-identical to the torch path's on every compared position (the switch was inert);
  - the null pair (the torch path against itself) reads a mean KL above NULL_MAX on either phase (the instrument's
    floor is within a tenth of the bar);
  - the mutant passes the bar on both phases (the gate cannot fail);
- NEUTRAL: on BOTH phases (prefill positions, decode steps) mean KL(torch || kernels) <= KL_MAX, argmax agreement
  >= AGREE_MIN and mean d_nll (kernels minus torch) <= DNLL_MAX;
- COST: otherwise.

TTFT is reported, never gated: the median torch / kernels ratio per prompt length.

    python p106_reduce.py --dir RUN_DIR --out verdict.json
    python p106_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"
KL_MAX, AGREE_MIN, DNLL_MAX = 0.05, 0.85, 0.01        # P97's registered faithfulness bar (G2), plus a d_nll ceiling
NULL_MAX = KL_MAX / 10
PHASES = ("prefill", "decode")
RULES = {"torch_chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk",
         "torch_recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent",
         "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface",
         "causal_conv1d_update": "causal_conv1d.causal_conv1d_interface"}
PINS = {"flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0", "kernels": None}
TOGGLED = ("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")


def passes(s):
    return s["mean_kl"] <= KL_MAX and s["argmax_agree"] >= AGREE_MIN and s["mean_d_nll"] <= DNLL_MAX


def engagement_errors(kern, box):
    errs = []
    for name, mod in RULES.items():
        if kern.get(name) != mod:
            errs.append(f"kernels_fc.json {name} = {kern.get(name)!r}, registered {mod!r}")
    for dist, ver in PINS.items():
        if kern.get(dist) != ver:
            errs.append(f"kernels_fc.json {dist} = {kern.get(dist)!r}, registered {ver!r}")
    if kern.get("torch_held") is not True:
        errs.append("torch not held")
    want = {f"{m}.{n}": mod for m in TOGGLED for n, mod in RULES.items()}
    if box.get("toggle_resolved") != want:
        errs.append(f"box resolved record {box.get('toggle_resolved')} is not the registered {want}")
    tr = box.get("toggle_torch") or {}
    if set(tr) != set(want) or not all(v.startswith("transformers.models.") for v in tr.values()):
        errs.append(f"box torch record {tr} is not transformers' own functions")
    if box.get("toggle_final") != want:
        errs.append("the box did not end on the resolved kernels")
    gm = box.get("gdn_modules") or []
    if not gm or not set(gm) <= set(TOGGLED):
        errs.append(f"the model's Gated DeltaNet modules {gm} are not among the toggled modules")
    return errs


def reduce(run: Path) -> dict:
    summ = (run / "summary.txt").read_text() if (run / "summary.txt").is_file() else ""
    premise = "ok" if "premise fc ok" in summ else ("failed" if "premise fc failed" in summ else None)
    out = {"lane": "P106", "premise": premise}
    if premise != "ok" or not (run / "box.json").is_file():
        out.update(verdict="NO_READING", reasons=[f"premise {premise}" if premise != "ok" else "no box record"])
        return out
    box = json.loads((run / "box.json").read_text())
    kern = json.loads((run / "kernels_fc.json").read_text()) if (run / "kernels_fc.json").is_file() else {}
    tog = json.loads((run / "toggle_check.json").read_text()) if (run / "toggle_check.json").is_file() else {}
    q, m = box["quality"], box["mutant"]["result"]
    void = []
    if tog.get("ok") is not True:
        void.append(f"toggle check {tog.get('ok')!r}")
    void += engagement_errors(kern, box)
    if box.get("loaded_commit") != REV:
        void.append(f"loaded commit {box.get('loaded_commit')} is not {REV}")
    k = q["kernels"]
    if all(k[ph]["identical_rows"] == k[ph]["n"] for ph in PHASES):
        void.append("the kernels' logits equal the torch path's on every position: the switch was inert")
    for ph in PHASES:
        nl = q["null"][ph]
        if not nl.get("n"):
            void.append(f"no null pair on {ph}")
        elif nl["mean_kl"] > NULL_MAX:
            void.append(f"null pair {ph} mean KL {nl['mean_kl']:.3e} > {NULL_MAX}")
    if all(passes(m[ph]) for ph in PHASES):
        void.append("the mutant passes the bar on both phases: the gate cannot fail")
    out["quality"] = {ph: {key: k[ph][key] for key in ("n", "mean_kl", "median_kl", "max_kl", "argmax_agree", "mean_d_nll",
                                                         "mean_nll_ref", "mean_nll_alt", "identical_rows")} for ph in PHASES}
    out["null"] = {ph: q["null"][ph] for ph in PHASES}
    out["mutant"] = {ph: {key: m[ph][key] for key in ("n", "mean_kl", "argmax_agree", "mean_d_nll")} for ph in PHASES}
    out["ttft"] = {n: {"median_ms": v["median_ms"], "speedup": v["speedup"]} for n, v in box.get("ttft", {}).items()}
    out["bar"] = {"kl_max": KL_MAX, "agree_min": AGREE_MIN, "dnll_max": DNLL_MAX, "null_max": NULL_MAX}
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    bad = [f"{ph}: KL {k[ph]['mean_kl']:.3e} agree {k[ph]['argmax_agree']:.4f} d_nll {k[ph]['mean_d_nll']:+.4f}"
           for ph in PHASES if not passes(k[ph])]
    out.update(verdict="COST" if bad else "NEUTRAL", reasons=bad)
    return out


# ------------------------------------------------------------------------------------------------ self-test --
def _stats(kl=1e-3, agree=0.97, d=0.0, n=100, ident=3):
    return {"n": n, "mean_kl": kl, "median_kl": kl, "max_kl": kl * 10, "argmax_agree": agree, "mean_d_nll": d,
            "mean_nll_ref": 2.0, "mean_nll_alt": 2.0 + d, "identical_rows": ident}


def _good():
    want = {f"{m}.{n}": mod for m in TOGGLED for n, mod in RULES.items()}
    box = {"toggle_resolved": dict(want), "toggle_final": dict(want),
           "toggle_torch": {key: key.rsplit(".", 1)[0] for key in want},
           "gdn_modules": ["transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"], "loaded_commit": REV,
           "quality": {"kernels": {ph: _stats() for ph in PHASES},
                       "null": {ph: _stats(kl=0.0, agree=1.0, ident=100) for ph in PHASES}},
           "mutant": {"result": {ph: _stats(kl=3.0, agree=0.2, d=2.0) for ph in PHASES}},
           "ttft": {"4096": {"median_ms": {"torch": 900.0, "resolved": 600.0}, "speedup": 1.5}}}
    kern = dict(RULES, **PINS, torch_held=True)
    return box, kern, {"ok": True}


def _run(tmp: Path, box, kern, tog, premise="premise fc ok"):
    for f in tmp.iterdir():
        f.unlink()
    (tmp / "summary.txt").write_text(premise + "\n")
    if box is not None:
        (tmp / "box.json").write_text(json.dumps(box))
    if kern is not None:
        (tmp / "kernels_fc.json").write_text(json.dumps(kern))
    if tog is not None:
        (tmp / "toggle_check.json").write_text(json.dumps(tog))
    return reduce(tmp)


def self_test() -> int:
    import tempfile
    cases = []

    def case(name, want, mutate=None, premise="premise fc ok", drop=None):
        box, kern, tog = _good()
        if mutate:
            mutate(box, kern, tog)
        parts = {"box": box, "kern": kern, "tog": tog}
        if drop:
            parts[drop] = None
        cases.append((name, want, parts, premise))

    case("all good", "NEUTRAL")
    case("premise failed", "NO_READING", premise="premise fc failed")
    case("no premise line", "NO_READING", premise="")
    case("no box record", "NO_READING", drop="box")
    case("toggle check failed", "VOID", lambda b, k, t: t.update(ok=False))
    case("no toggle check", "VOID", drop="tog")
    case("no engagement record", "VOID", drop="kern")
    case("fla not engaged", "VOID", lambda b, k, t: k.update(torch_chunk_gated_delta_rule="transformers.x"))
    case("conv not engaged", "VOID", lambda b, k, t: k.update(causal_conv1d_fn="transformers.x"))
    case("kernels hub present", "VOID", lambda b, k, t: k.update(kernels="0.1"))
    case("wrong fla pin", "VOID", lambda b, k, t: k.update(**{"flash-linear-attention": "0.5.3"}))
    case("box torch record names fla", "VOID",
         lambda b, k, t: b["toggle_torch"].update({next(iter(b["toggle_torch"])): "fla.ops.x"}))
    case("box did not end resolved", "VOID", lambda b, k, t: b.update(toggle_final=b["toggle_torch"]))
    case("model GDN module not toggled", "VOID", lambda b, k, t: b.update(gdn_modules=["transformers.models.qwen3_next.m"]))
    case("wrong commit", "VOID", lambda b, k, t: b.update(loaded_commit="0" * 40))
    case("inert switch", "VOID",
         lambda b, k, t: [s.update(identical_rows=s["n"]) for s in b["quality"]["kernels"].values()])
    case("identical on prefill only is not inert", "NEUTRAL",
         lambda b, k, t: b["quality"]["kernels"]["prefill"].update(identical_rows=100))
    case("noisy null", "VOID", lambda b, k, t: b["quality"]["null"]["decode"].update(mean_kl=NULL_MAX * 1.01))
    case("empty null", "VOID", lambda b, k, t: b["quality"]["null"]["prefill"].update(n=0))
    case("mutant passes", "VOID", lambda b, k, t: b["mutant"].update(result={ph: _stats() for ph in PHASES}))
    case("mutant fails on decode only still live", "NEUTRAL",
         lambda b, k, t: b["mutant"]["result"].update(prefill=_stats()))
    case("prefill KL over", "COST", lambda b, k, t: b["quality"]["kernels"]["prefill"].update(mean_kl=KL_MAX * 1.01))
    case("decode KL at the bar", "NEUTRAL", lambda b, k, t: b["quality"]["kernels"]["decode"].update(mean_kl=KL_MAX))
    case("agreement under", "COST", lambda b, k, t: b["quality"]["kernels"]["decode"].update(argmax_agree=0.849))
    case("d_nll over", "COST", lambda b, k, t: b["quality"]["kernels"]["prefill"].update(mean_d_nll=0.0101))
    case("kernels better in nats", "NEUTRAL", lambda b, k, t: b["quality"]["kernels"]["decode"].update(mean_d_nll=-0.2))
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, want, parts, premise in cases:
            got = _run(tmp, copy.deepcopy(parts["box"]), parts["kern"], parts["tog"], premise)["verdict"]
            if got != want:
                fails += 1
                print(f"SELF-TEST FAIL {name}: got {got}, want {want}")
    if fails:
        print(f"self-test FAILED ({fails} of {len(cases)} cases)")
        return 1
    print(f"self-test OK ({len(cases)} cases)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    out = reduce(Path(a.dir))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("VERDICT", out["verdict"], "|", "; ".join(out.get("reasons", [])) or "-")
    print("QUALITY", json.dumps(out.get("quality")))
    print("TTFT", json.dumps(out.get("ttft")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
