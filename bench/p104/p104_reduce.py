#!/usr/bin/env python3
"""Lane P104's reducer (bench/p104/PREREG-p104.md; e4b#928): the Gated DeltaNet kernels under the hybrid paged path, with
a chunk-matched premise (lane P103's reducer, its rule unchanged; only the box's premise set differs).

The box runs P98's three arms (g graphs, e the padded-eager oracle, p plain eager; ``p98_box.py``) in up to three
PHASES on one box: ``t`` transformers' torch path, ``f`` with flash-linear-attention, ``fc`` with causal-conv1d as
well. Each phase's three records are judged by P98's registered rule, unchanged (``p98_reduce.reduce``, staged beside
this file at its registered bytes). This file adds what the phases need:

- ENGAGEMENT (``kernels_<phase>.json``, what transformers resolves at import): phase t resolves all four Gated
  DeltaNet functions to transformers' own module and has no kernel distribution installed; phase f resolves the chunk
  and fused-recurrent rules to ``fla.*`` (flash-linear-attention 0.5.2) and the conv functions to transformers'; phase
  fc resolves the rules to ``fla.*`` and the conv functions to ``causal_conv1d.*`` (causal-conv1d 1.7.0). In every
  phase torch is the image's (held by a constraint) and the ``kernels`` hub package is absent.
- PHASE STATES (``summary.txt``): ``phase <p> ok`` / ``install_failed`` / ``premise_failed``, ``phase fc skipped``. The
  premise a kernel phase must pass is the chunk-matched set (p104_run.sh), not lane P103's single-call check.

Per phase:
- t: SUPPORTED / NOT_SUPPORTED / VOID by P98's rule; a wrong engagement record is VOID.
- f, fc: UNAVAILABLE when its install (or probe) failed; NOT_RUN when skipped; NOT_SUPPORTED when its premise failed
  with the kernels engaged (the hybrid GPU tests failed on the card); otherwise P98's rule, with a wrong engagement
  record VOID.

The LANE verdict:
- VOID when phase t is not SUPPORTED (the torch-path baseline, P101's reading, did not hold on this box);
- otherwise phase fc's state when fc ran to a P98 verdict or failed its premise, else phase f's, else UNAVAILABLE.

Reported, not gated: decode tok/s per arm and phase; each kernel phase's speed ratio to phase t per arm (g / e / p) and
workload; ms per step by rows; each kernel phase's graph-arm token agreement with phase t's graph arm. ``recommend``
(the registered consequence's mechanical half): true iff phase fc is SUPPORTED and its graph arm decodes at least 1.05x
phase t's graph arm on W16 or W1.

    python p104_reduce.py --dir RUN_DIR --out verdict.json
    python p104_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# beside this file on the box (both staged into /root/p104); bench/p98/ in the repository
_P98 = HERE / "p98_reduce.py" if (HERE / "p98_reduce.py").is_file() else HERE.parent / "p98" / "p98_reduce.py"
_spec = importlib.util.spec_from_file_location("p98_reduce", _P98)
p98 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p98)

PHASES = ("t", "f", "fc")
TF_MOD = "transformers.models.qwen3_5.modeling_qwen3_5"
RULES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
CONVS = ("causal_conv1d_fn", "causal_conv1d_update")
PINS = {"flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0"}
RECOMMEND_MIN = 1.05


def _engaged(phase, k):
    """Reasons a phase's engagement record is not the registered one ([] when it is)."""
    if not isinstance(k, dict):
        return [f"phase {phase}: no engagement record"]
    why = []
    if k.get("phase") != phase:
        why.append(f"phase {phase}: engagement record is phase {k.get('phase')!r}")
    if k.get("torch_held") is not True:
        why.append(f"phase {phase}: torch moved off the image's version ({k.get('torch')})")
    if k.get("kernels") is not None:
        why.append(f"phase {phase}: the kernels hub package is installed ({k.get('kernels')})")
    rule_mod = [k.get(r) or "" for r in RULES]
    conv_mod = [k.get(c) or "" for c in CONVS]
    if phase == "t":
        if any(m != TF_MOD for m in rule_mod + conv_mod):
            why.append(f"phase t: not transformers' torch path ({rule_mod + conv_mod})")
        if k.get("flash-linear-attention") or k.get("causal-conv1d"):
            why.append("phase t: a kernel distribution is installed")
    else:
        if not all(m.startswith("fla.") for m in rule_mod):
            why.append(f"phase {phase}: the gated delta rules are not fla's ({rule_mod})")
        if k.get("flash-linear-attention") != PINS["flash-linear-attention"]:
            why.append(f"phase {phase}: flash-linear-attention {k.get('flash-linear-attention')} != {PINS['flash-linear-attention']}")
        if phase == "f":
            if any(m != TF_MOD for m in conv_mod):
                why.append(f"phase f: the conv functions are not transformers' ({conv_mod})")
            if k.get("causal-conv1d"):
                why.append("phase f: causal-conv1d is installed")
        else:
            if not all(m.startswith("causal_conv1d") for m in conv_mod):
                why.append(f"phase fc: the conv functions are not causal_conv1d's ({conv_mod})")
            if k.get("causal-conv1d") != PINS["causal-conv1d"]:
                why.append(f"phase fc: causal-conv1d {k.get('causal-conv1d')} != {PINS['causal-conv1d']}")
    return why


def _states(summary_text):
    """Phase states as the runner records them in summary.txt."""
    st = {"t": "ok"}
    for p, s in re.findall(r"(?m)^phase (fc|f) (ok|install_failed|premise_failed)\b", summary_text):
        st[p] = s
    if re.search(r"(?m)^phase fc skipped", summary_text):
        st["fc"] = "skipped"
    return st


def reduce(phase_recs, kernels, states):
    out = {"phases": {}, "why": [], "engagement": {}}
    res = {}
    for p in PHASES:
        st = states.get(p)
        eng = _engaged(p, kernels.get(p)) if st not in (None, "install_failed", "skipped") else []
        out["engagement"][p] = eng
        if st is None or st == "skipped":
            res[p] = {"verdict": "NOT_RUN"}
        elif st == "install_failed":
            res[p] = {"verdict": "UNAVAILABLE"}
        elif st == "premise_failed":
            res[p] = {"verdict": "VOID" if eng else "NOT_SUPPORTED", "why": eng,
                      "not_supported": [] if eng else [f"phase {p}: the hybrid GPU premise failed with the kernels engaged"]}
        else:
            r = p98.reduce(phase_recs.get(p) or {})
            if eng:
                r = dict(r, verdict="VOID", why=list(r.get("why", [])) + eng)
            res[p] = r
        out["phases"][p] = {k: res[p].get(k) for k in ("verdict", "why", "not_supported") if k in res[p]}
    if res["t"]["verdict"] != "SUPPORTED":
        out["why"].append(f"phase t (the torch-path baseline) is {res['t']['verdict']}")
        out["verdict"] = "VOID"
    elif res["fc"]["verdict"] in ("SUPPORTED", "NOT_SUPPORTED", "VOID"):
        out["verdict"] = res["fc"]["verdict"]
    elif res["f"]["verdict"] in ("SUPPORTED", "NOT_SUPPORTED", "VOID"):
        out["verdict"] = res["f"]["verdict"]
    else:
        out["verdict"] = "UNAVAILABLE"

    def tps(p, arm, w):
        return ((res[p].get("decode_tok_per_s") or {}).get(w) or {}).get(arm)

    out["decode_tok_per_s"] = {p: res[p].get("decode_tok_per_s") for p in PHASES if res[p].get("decode_tok_per_s")}
    out["ms_per_step_by_rows"] = {p: res[p].get("ms_per_step_by_rows") for p in PHASES if res[p].get("ms_per_step_by_rows")}
    ratio = {}
    for p in ("f", "fc"):
        if not res[p].get("decode_tok_per_s"):
            continue
        ratio[p] = {arm: {w: (tps(p, arm, w) / tps("t", arm, w) if tps(p, arm, w) and tps("t", arm, w) else None)
                          for w in ("w16", "w1")} for arm in ("g", "e", "p")}
    out["over_torch_path"] = ratio

    def agree(a, b):
        pairs = [(x, y) for ta, tb in zip(a, b) for x, y in zip(ta, tb)]
        return sum(x == y for x, y in pairs) / len(pairs) if pairs else None

    out["graph_token_agreement_with_t"] = {
        p: {w: agree(((phase_recs.get(p) or {}).get("g") or {}).get(w, {}).get("tokens") or [],
                     ((phase_recs.get("t") or {}).get("g") or {}).get(w, {}).get("tokens") or []) for w in ("w16", "w1")}
        for p in ("f", "fc") if (phase_recs.get(p) or {}).get("g")}
    g_fc = (ratio.get("fc") or {}).get("g") or {}
    out["recommend"] = bool(out["verdict"] == "SUPPORTED" and res["fc"]["verdict"] == "SUPPORTED"
                            and any((g_fc.get(w) or 0) >= RECOMMEND_MIN for w in ("w16", "w1")))
    return out


def _kern(phase):
    k = {"phase": phase, "torch": "2.8.0+cu128", "torch_held": True, "fla-core": None,
         "flash-linear-attention": None, "causal-conv1d": None, "kernels": None}
    for r in RULES:
        k[r] = TF_MOD if phase == "t" else "fla.ops.gated_delta_rule.x"
    for c in CONVS:
        k[c] = "causal_conv1d.causal_conv1d_interface" if phase == "fc" else TF_MOD
    if phase != "t":
        k["flash-linear-attention"] = k["fla-core"] = "0.5.2"
    if phase == "fc":
        k["causal-conv1d"] = "1.7.0"
    return k


def _fixture():
    recs = {p: p98._fixture() for p in PHASES}
    speed = {"t": 1.0, "f": 1.1, "fc": 1.2}
    for p in PHASES:
        for arm in ("g", "e", "p"):
            for w in ("w16", "w1"):
                recs[p][arm][w]["decode"]["tok_per_s"] *= speed[p]
    return recs, {p: _kern(p) for p in PHASES}, {"t": "ok", "f": "ok", "fc": "ok"}


def self_test():
    cases = []

    def case(name, want, edit=None, want_recommend=None):
        recs, kern, st = _fixture()
        if edit:
            edit(recs, kern, st)
        got = reduce(recs, kern, st)
        ok = got["verdict"] == want and (want_recommend is None or got["recommend"] == want_recommend)
        cases.append((name, ok, want, got["verdict"], got["recommend"], got["why"], got["phases"]))

    case("every phase holds; fc 1.2x the torch path", "SUPPORTED", want_recommend=True)
    case("fc only 1.03x faster: SUPPORTED, not recommended", "SUPPORTED",
         lambda r, k, s: [r["fc"][a][w]["decode"].__setitem__("tok_per_s", r["t"][a][w]["decode"]["tok_per_s"] * 1.03)
                          for a in ("g", "e", "p") for w in ("w16", "w1")], want_recommend=False)
    case("the baseline's graph tokens differ (phase t NOT_SUPPORTED)", "VOID",
         lambda r, k, s: r["t"]["g"]["w16"]["tokens"][0].__setitem__(0, 9))
    case("the baseline resolved fla", "VOID", lambda r, k, s: k["t"].__setitem__(RULES[0], "fla.ops.x"))
    case("the baseline has a kernel distribution installed", "VOID",
         lambda r, k, s: k["t"].__setitem__("causal-conv1d", "1.7.0"))
    case("fc graph tokens differ from its oracle", "NOT_SUPPORTED", lambda r, k, s: r["fc"]["g"]["w1"]["tokens"][0].__setitem__(3, 8))
    case("fc premise failed with the kernels engaged", "NOT_SUPPORTED", lambda r, k, s: s.__setitem__("fc", "premise_failed"),
         want_recommend=False)
    case("fc's conv still transformers' (not engaged)", "VOID", lambda r, k, s: k["fc"].__setitem__(CONVS[1], TF_MOD))
    case("fc pulled another causal-conv1d", "VOID", lambda r, k, s: k["fc"].__setitem__("causal-conv1d", "1.6.0"))
    case("torch moved in fc", "VOID", lambda r, k, s: k["fc"].__setitem__("torch_held", False))
    case("the kernels hub package appeared in f (and so in fc)", "VOID",
         lambda r, k, s: (k["f"].__setitem__("kernels", "0.9"), k["fc"].__setitem__("kernels", "0.9")))
    case("only f's record is off: f VOID, the headline follows fc", "SUPPORTED",
         lambda r, k, s: k["f"].__setitem__(RULES[1], TF_MOD), want_recommend=True)
    case("fc install failed: headline is f", "SUPPORTED", lambda r, k, s: s.__setitem__("fc", "install_failed"),
         want_recommend=False)
    case("f premise failed, fc skipped", "NOT_SUPPORTED",
         lambda r, k, s: (s.__setitem__("f", "premise_failed"), s.__setitem__("fc", "skipped")))
    case("f install failed, fc skipped: nothing engaged", "UNAVAILABLE",
         lambda r, k, s: (s.__setitem__("f", "install_failed"), s.__setitem__("fc", "skipped")))
    case("an fc arm record missing", "VOID", lambda r, k, s: r["fc"].__setitem__("e", None))
    case("fc ran an eager step under graphs", "NOT_SUPPORTED",
         lambda r, k, s: r["fc"]["g"]["graph_stats"]["4"].__setitem__("eager_steps", 1))
    case("fc's plain arm disagrees with its graph arm (reported only)", "SUPPORTED",
         lambda r, k, s: r["fc"]["p"]["w16"]["tokens"][2].__setitem__(1, 5))
    case("f's graph tokens differ from t's (reported only)", "SUPPORTED",
         lambda r, k, s: [r["f"][a]["w16"]["tokens"][1].__setitem__(0, 4) for a in ("g", "e")])
    case("summary states parsed", "SUPPORTED",
         lambda r, k, s: s.update(_states("premise t rc=0: 9 passed\nphase f ok\nphase fc ok\n")))
    bad = [c for c in cases if not c[1]]
    for n, _ok, w, g_, rcm, why, ph in bad:
        print(f"SELF-TEST FAIL {n}: want {w}, got {g_} (recommend {rcm}; {why}; {ph})")
    st = _states("phase f ok\nphase fc skipped: phase f did not hold\n")
    if st != {"t": "ok", "f": "ok", "fc": "skipped"}:
        print(f"SELF-TEST FAIL state parse: {st}")
        return 1
    if bad:
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
    d = Path(a.dir)

    def load(name):
        f = d / name
        return json.loads(f.read_text()) if f.is_file() else None

    recs = {p: {arm: load(f"arm_{arm}_{p}.json") for arm in ("g", "e", "p")} for p in PHASES}
    kern = {p: load(f"kernels_{p}.json") for p in PHASES}
    st = _states((d / "summary.txt").read_text() if (d / "summary.txt").is_file() else "")
    v = reduce(recs, kern, st)
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P104 VERDICT {v['verdict']} (recommend {v['recommend']})")
    for p in PHASES:
        print(f"  phase {p}: {v['phases'][p].get('verdict')} {v['phases'][p].get('why') or ''} {v['phases'][p].get('not_supported') or ''}")
    for w in v["why"]:
        print(f"  void: {w}")
    print(f"  decode tok/s {v['decode_tok_per_s']}")
    print(f"  over the torch path {v['over_torch_path']} | graph token agreement with t {v['graph_token_agreement_with_t']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
