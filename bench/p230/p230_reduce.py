#!/usr/bin/env python3
"""p230_reduce.py -- lane P230's registered rule (bench/p230/PREREG-p230.md), from the four arm receipts, and the
mechanical classification of every fetched box. P113's reducer (bench/p113/p113_reduce.py) by named edits.

Receipts: arm_OFF1, arm_DEF1, arm_DEF2, arm_OFF2 (.json; run in that order, a palindrome), each from p230_box.py. OFF
is ``GNF4_PDL=0``; DEF is the shipped default (the switch and its cap unset: on, capped at 8 rows). All four are the
default serve_paged server (graphs on, every serving lever unset) on Qwen3-30B-A3B's NF4 store. Every rate is the
decode-only slope (each pass's wall minus its largest ttft).

The verdict is the first of these that applies:
  VOID          an arm is missing or not ok; a receipt names another e4b / grouped-nf4-gemm commit or model revision;
                the arms read different prompts, ran different lengths or resolved different server configurations; a
                decode-only slope is void; or engagement is wrong:
                - every arm reports "graph" for every one of its buckets, and its buckets include 1 and 16;
                - ``pdl_active`` is False in OFF and True in DEF, and ``pdl_cap`` is 8 in every arm;
                - every build launches the same number of switched kernels, more than zero, in P112 Amendment 1's
                  accounting form, with no more compile launches than variants; K33 (``_gemv_nf4_bw``),
                  ``_swiglu_rows`` and ``_combine_rows`` each launch in every build;
                - in OFF nothing carries PDL; in DEF the cap splits the build (at least one handled launch carries PDL
                  and at least one does not), and at least one handled K33 launch carries PDL;
                - per bucket, read from the capture (each launch tagged with the bucket being captured): in DEF, bucket 1
                  has PDL launches and bucket 16 has none; in OFF, no bucket has any.
  FUNCTION_FAIL an arm emits a token different from OFF1's on any row of either workload at either length, or an arm's
                timed reps do not all digest the same. Read BEFORE the timing rungs: token identity does not depend on
                timing, so noise never hides a broken bitwise contract.
  NOISY         a self-pair (OFF2/OFF1 or DEF2/DEF1, decode tok/s, either workload) falls outside [0.995, 1.005], or
                the W16 control does: either DEF/OFF pair on W16, where the cap launches nothing with PDL.
  KEEP          g1 = min(DEF1/OFF1, DEF2/OFF2) on W1 >= 1.005.
  SLOWER        h1 = max(DEF1/OFF1, DEF2/OFF2) on W1 <= 0.995.
  NULL          otherwise.

KEEP, NULL, SLOWER and FUNCTION_FAIL are answers (the runner exits 0); VOID and NOISY are not (exit 27).

**The classification** (:func:`classify`, PREREG "Outcomes and the stop"): from the runner's exit code and whether it
wrote ``STARTED`` (once every pre-install check has passed, before the first install command). Before the marker, a
host fault (13, 15, 16, 18) is HOST_REFUSED (cap 4 a trip, 8 in all); the guard fit (41) is FIT_REFUSED (no start, a
retry id; two consecutive are one strike); a kit or launch fault (9, 10, 28, 78, 130, any other code) is
PRE_INSTALL_HALT (the lane halts until it is fixed). After it, 0 is an ANSWER and any other code is a box-started run
WITHOUT an answer (a strike). ``--classify --work W`` reads the box's work directory as the driver fetched it and
FAILS CLOSED (exit 2, no class) unless W holds exactly one ``P230_EXIT_CODE.<nonce>`` and ``RUN_ID``. A box the
launcher tore down before the runner ran (NOT_RUN) is decided from the launcher's receipt and never classified here.

    python p230_reduce.py --dir W --out verdict.json --e4b-sha SHA
    python p230_reduce.py --classify --work W
    python p230_reduce.py --self-test
"""
import argparse
import glob
import json
import os
import sys
import tempfile

TAGS = ("OFF1", "DEF1", "DEF2", "OFF2")
WORKLOADS = ("W16", "W1")
SELF_LO, SELF_HI = 0.995, 1.005      # the self-pairs and the W16 control (P113's 0.9988-1.0018; P111's 0.997-1.002)
KEEP_BAR = 1.005                     # the smaller W1 pair must clear the widest tolerated noise
SLOWER_BAR = 0.995                   # the larger W1 pair must sit below it
CAP_ROWS = 8
GNF4_SHA = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"
REVS = {"Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"}
MUST_LAUNCH = ("_gemv_nf4_bw", "_swiglu_rows", "_combine_rows")
ANSWERS = ("KEEP", "NULL", "SLOWER", "FUNCTION_FAIL")
HOST_CODES = {13: "free disk", 15: "card, compute capability or VRAM", 16: "host RAM", 18: "CUDA unusable"}
FIT_CODES = {41: "the guard does not fit the deadline (a slow boot)"}
KIT_CODES = {9: "staging, install or hash pins", 10: "torch does not import (the image)",
             28: "the image's torch is not the pinned build", 78: "run id, environment or frozen knobs", 130: "a signal"}


def _setting(t):
    return t[:3]


def _tok(r, w, n):
    return r["workloads"][w]["tokens"][str(n)]


def _rate(r, w):
    return r["workloads"][w]["decode_tok_s"]


def _launches(r):
    acct = r.get("pdl_launches_build") or {}
    return sum(v[0] for v in acct.values())


def _engagement(t, r):
    out, s = [], _setting(t)
    cfg = r.get("config") or {}
    buckets = [str(b) for b in cfg.get("buckets") or []]
    g = r.get("graph_status") or {}
    if not {"1", "16"} <= set(buckets) or [g.get(b) for b in buckets] != ["graph"] * len(buckets):
        out.append(f"{t}: buckets {buckets}, graph_status {g}")
    if bool(r.get("pdl_active")) != (s == "DEF"):
        out.append(f"{t}: pdl_active={r.get('pdl_active')}")
    if r.get("pdl_cap") != CAP_ROWS:
        out.append(f"{t}: pdl_cap={r.get('pdl_cap')}")
    acct = r.get("pdl_launches_build") or {}
    if not acct or _launches(r) == 0:
        out.append(f"{t}: the build launched no switched gnf4 kernel")
    for name in MUST_LAUNCH:
        if not (acct.get(name) or [0])[0]:
            out.append(f"{t}: {name} never launched in the build (the default route did not reach it)")
    handled = with_pdl = 0
    for name, row in sorted(acct.items()):
        if len(row) != 5:
            out.append(f"{t}: {name} accounting row {row} is not P112 Amendment 1's form")
            continue
        n, w, compiles, variants, variants_pdl = row
        handled += n - compiles
        with_pdl += w
        if compiles > variants:
            out.append(f"{t}: {name} has {compiles} compile launches but {variants} compiled variants")
        if s == "OFF" and (w or variants_pdl):
            out.append(f"{t}: {name} carried PDL with the switch off ({w} launches, {variants_pdl} variants)")
    byb = r.get("pdl_by_bucket")
    if not isinstance(byb, dict) or not all(isinstance(v, list) and len(v) == 2 for v in byb.values()):
        out.append(f"{t}: no per-bucket accounting ({byb!r})")
    else:
        if s == "DEF" and not (byb.get("1") or [0, 0])[1]:
            out.append(f"{t}: bucket 1 carried no PDL launch ({byb.get('1')})")
        if s == "DEF" and (byb.get("16") or [0, 0])[1]:
            out.append(f"{t}: bucket 16 carried {byb['16'][1]} PDL launch(es): the W16 control is not PDL-free")
        if s == "OFF" and any(v[1] for v in byb.values()):
            out.append(f"{t}: PDL launches with the switch off, by bucket {byb}")
    if s == "DEF":
        if not 0 < with_pdl < handled:
            out.append(f"{t}: the cap did not split the build ({with_pdl} of {handled} handled launches carried PDL)")
        k33 = acct.get("_gemv_nf4_bw") or [0, 0, 0, 0, 0]
        if len(k33) == 5 and not k33[1]:
            out.append(f"{t}: no handled K33 launch carried PDL")
    return out


def reduce(arms: dict, e4b_sha: str) -> dict:
    out = {"verdict": None, "reasons": [], "bars": {"self_pair_and_w16_control": [SELF_LO, SELF_HI],
                                                    "keep": KEEP_BAR, "slower": SLOWER_BAR}}
    missing = [t for t in TAGS if t not in arms or arms[t].get("status") != "ok"]
    if missing:
        out.update(verdict="VOID", reasons=[f"arm(s) missing or not ok: {missing}"])
        return out
    void = []
    for t, r in arms.items():
        if r.get("e4b_sha") != e4b_sha:
            void.append(f"{t}: e4b {r.get('e4b_sha')} != {e4b_sha}")
        if r.get("gnf4_sha") != GNF4_SHA:
            void.append(f"{t}: grouped-nf4-gemm {r.get('gnf4_sha')} != {GNF4_SHA}")
        if REVS.get(r.get("model")) != r.get("revision"):
            void.append(f"{t}: model {r.get('model')}@{r.get('revision')} is not a registered revision")
        void += _engagement(t, r)
    if len({json.dumps(r.get("config"), sort_keys=True) for r in arms.values()}) != 1:
        void.append("the arms resolved different server configurations")
    if len({_launches(r) for r in arms.values()}) != 1:
        void.append(f"the builds launched different numbers of switched kernels: {[_launches(arms[t]) for t in TAGS]}")
    if len({r["prompts_sha256"] for r in arms.values()}) != 1:
        void.append("the arms read different prompts")
    if len({(r["short"], r["long"], r["reps"]) for r in arms.values()}) != 1:
        void.append("the arms ran different lengths")
    rates = {t: {w: _rate(arms[t], w) for w in WORKLOADS} for t in TAGS}
    if any(v is None or v <= 0 for d in rates.values() for v in d.values()):
        void.append("a decode-only slope is void")
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    OFF1 = arms["OFF1"]
    short, long_ = OFF1["short"], OFF1["long"]
    out["decode_tok_s"] = rates
    out["ms_per_step"] = {t: {w: arms[t]["workloads"][w].get("decode_ms_per_step") for w in WORKLOADS} for t in TAGS}
    out["whole_pass_tok_s"] = {t: {w: (arms[t]["workloads"][w].get("whole_pass") or {}).get("decode_tok_s")
                                   for w in WORKLOADS} for t in TAGS}
    out["switched_launches_build"] = {t: [_launches(arms[t]),
                                          sum(v[1] for v in arms[t]["pdl_launches_build"].values())] for t in TAGS}
    selfp = {f"{s}2/{s}1 {w}": rates[f"{s}2"][w] / rates[f"{s}1"][w] for s in ("OFF", "DEF") for w in WORKLOADS}
    out["self_pairs"] = {k: round(v, 5) for k, v in selfp.items()}
    pairs = {w: [rates["DEF1"][w] / rates["OFF1"][w], rates["DEF2"][w] / rates["OFF2"][w]] for w in WORKLOADS}
    out["pair_ratios"] = {w: [round(x, 5) for x in v] for w, v in pairs.items()}
    g1, h1 = min(pairs["W1"]), max(pairs["W1"])
    out["g1"], out["h1"] = round(g1, 5), round(h1, 5)
    fn = []
    for w in WORKLOADS:
        for n in (short, long_):
            for t in TAGS[1:]:
                if _tok(arms[t], w, n) != _tok(OFF1, w, n):
                    rows = [i for i, (x, y) in enumerate(zip(_tok(arms[t], w, n), _tok(OFF1, w, n))) if x != y]
                    fn.append(f"{t} != OFF1 on {w} at {n} tokens, rows {rows}")
    for t in TAGS:
        for w in WORKLOADS:
            for n, ds in arms[t]["workloads"][w]["rep_digests"].items():
                if len(set(ds)) != 1:
                    fn.append(f"{t} {w} at {n} tokens: timed reps differ")
    noisy = [f"{k} = {v:.5f}" for k, v in selfp.items() if not SELF_LO <= v <= SELF_HI]
    noisy += [f"W16 control DEF/OFF pair {i + 1} = {v:.5f}" for i, v in enumerate(pairs["W16"])
              if not SELF_LO <= v <= SELF_HI]
    txt = f"W1 pairs {out['pair_ratios']['W1']} (g1 {out['g1']}, h1 {out['h1']}); W16 control {out['pair_ratios']['W16']}"
    if fn:
        out.update(verdict="FUNCTION_FAIL", reasons=fn)
    elif noisy:
        out.update(verdict="NOISY", reasons=noisy)
    elif g1 >= KEEP_BAR:
        out.update(verdict="KEEP", reasons=[f"tokens identical on every row; {txt}"])
    elif h1 <= SLOWER_BAR:
        out.update(verdict="SLOWER", reasons=[f"tokens identical on every row; {txt}"])
    else:
        out.update(verdict="NULL", reasons=[f"tokens identical on every row; {txt}"])
    return out


# ---------------------------------------------------------------- classification --

def classify(rc: int, started: bool) -> dict:
    """The registered class of one fetched box: ``{"event", "started", "host_refused_cap", "halt", "why"}``."""
    if not started:
        if rc in HOST_CODES:
            return {"event": "HOST_REFUSED", "started": False, "host_refused_cap": True, "halt": False,
                    "why": HOST_CODES[rc]}
        if rc in FIT_CODES:
            return {"event": "FIT_REFUSED", "started": False, "host_refused_cap": False, "halt": False,
                    "why": FIT_CODES[rc] + "; two consecutive are one strike"}
        return {"event": "PRE_INSTALL_HALT", "started": False, "host_refused_cap": False, "halt": True,
                "why": KIT_CODES.get(rc, f"unexpected pre-install exit {rc}")}
    if rc == 0:
        return {"event": "ANSWER", "started": True, "host_refused_cap": False, "halt": False, "why": "the verdict"}
    why = "VOID or NOISY (the verdict): a strike" if rc == 27 else f"exit {rc} after the start: a strike"
    return {"event": "NO_ANSWER", "started": True, "host_refused_cap": False, "halt": False, "why": why}


def read_box(work: str) -> dict:
    """The box's own files under ``work``; raises ValueError unless it is the runner's work directory."""
    codes = sorted(glob.glob(os.path.join(work, "P230_EXIT_CODE.*")))
    if len(codes) != 1 or not os.path.isfile(os.path.join(work, "RUN_ID")):
        state = ("STARTED PRESENT: by the table this box is a box-started run without an answer unless the hand reading "
                 "shows otherwise" if os.path.exists(os.path.join(work, "STARTED")) else
                 "STARTED absent here (if the box started and nothing could be fetched, it still counts as started)")
        raise ValueError(f"{work!r} is not a P230 box work directory: it needs exactly one P230_EXIT_CODE.<nonce> "
                         f"(found {len(codes)}) and RUN_ID. {state}. The lane HALTS until the box's files are read by hand.")
    rc = int(open(codes[0], encoding="utf-8").read().strip())
    return {"run_id": open(os.path.join(work, "RUN_ID"), encoding="utf-8").read().strip(), "rc": rc,
            "started": os.path.exists(os.path.join(work, "STARTED"))}


# ------------------------------------------------------------------ self-test --

def _fake(tag, rate16, rate1, toks, *, e4b="a" * 40, graphs=None, pdl=None, cap=CAP_ROWS, acct=None, config=None,
          digests=None, short=4, long_=8, byb=None):
    s = _setting(tag)
    config = config or {"max_seqs": 16, "buckets": [1, 2, 4, 8, 16], "placement": "all-vram", "graphs": True}
    if graphs is None:
        graphs = {str(b): "graph" for b in config["buckets"]}
    if acct is None:
        acct = {"OFF": {"_gemv_nf4_bw": [960, 0, 4, 4, 0], "_swiglu_rows": [480, 0, 4, 4, 0],
                        "_combine_rows": [480, 0, 4, 4, 0]},
                "DEF": {"_gemv_nf4_bw": [960, 192, 4, 4, 2], "_swiglu_rows": [480, 96, 4, 4, 2],
                        "_combine_rows": [480, 96, 4, 4, 2]}}[s]

    if byb is None:
        byb = {"OFF": {"1": [4, 0], "2": [4, 0], "16": [4, 0], "none": [10, 0]},
               "DEF": {"1": [4, 4], "2": [4, 0], "16": [4, 0], "none": [10, 0]}}[s]

    def wl(rate):
        return {"decode_tok_s": rate, "decode_ms_per_step": round(1000 / rate, 3) if rate else None,
                "tokens": {str(short): [t[:short] for t in toks], str(long_): toks},
                "rep_digests": digests or {str(short): ["x"] * 3, str(long_): ["y"] * 3},
                "whole_pass": {"decode_tok_s": rate * 0.98 if rate else None}}
    return {"tag": tag, "status": "ok", "e4b_sha": e4b, "gnf4_sha": GNF4_SHA, "model": "Qwen/Qwen3-30B-A3B",
            "revision": REVS["Qwen/Qwen3-30B-A3B"], "graph_status": graphs, "config": config,
            "pdl_active": (s == "DEF") if pdl is None else pdl, "pdl_cap": cap, "pdl_launches_build": acct,
            "pdl_by_bucket": byb,
            "prompts_sha256": "p", "short": short, "long": long_, "reps": 3,
            "workloads": {"W16": wl(rate16), "W1": wl(rate1)}}


def _classify_cases() -> list:
    out = []
    with tempfile.TemporaryDirectory() as d:
        def box(rc, started, rid="p230-5090-1", codes=1):
            w = os.path.join(d, f"w{rc}{int(started)}{codes}")
            os.makedirs(w)
            for i in range(codes):
                open(os.path.join(w, f"P230_EXIT_CODE.n{i}"), "w").write(f"{rc}\n")
            open(os.path.join(w, "RUN_ID"), "w").write(rid + "\n")
            if started:
                open(os.path.join(w, "STARTED"), "w").close()
            return w
        b = read_box(box(13, False))
        out.append(("disk before the start is HOST_REFUSED", classify(b["rc"], b["started"])["event"] == "HOST_REFUSED"))
        b = read_box(box(41, False))
        out.append(("the fit is FIT_REFUSED, never a halt", classify(b["rc"], b["started"])["event"] == "FIT_REFUSED"
                    and not classify(b["rc"], b["started"])["halt"]))
        out.append(("a kit fault halts", classify(9, False)["event"] == "PRE_INSTALL_HALT" and classify(9, False)["halt"]))
        b = read_box(box(0, True))
        out.append(("0 after the start is an ANSWER", classify(b["rc"], b["started"])["event"] == "ANSWER"))
        b = read_box(box(27, True))
        out.append(("27 after the start is NO_ANSWER", classify(b["rc"], b["started"])["event"] == "NO_ANSWER"))
        out.append(("an unknown pre-start code halts", classify(5, False)["event"] == "PRE_INSTALL_HALT"))
        for name, w in (("two exit files", box(0, True, codes=2)), ("no exit file", box(0, True, codes=0))):
            try:
                read_box(w)
                out.append((f"fails closed: {name}", False))
            except ValueError:
                out.append((f"fails closed: {name}", True))
    return out


def self_test() -> int:
    base = [[r * 100 + i for i in range(8)] for r in range(16)]
    RATES = {"OFF": (1700.0, 140.0), "DEF": (1700.0 * 1.001, 140.0 * 1.008)}

    def arms(rates=None, **over):
        rt = dict(RATES, **(rates or {}))
        a = {t: _fake(t, *rt[_setting(t)], base) for t in TAGS}
        a.update(over)
        return a

    def v(a):
        return reduce(a, "a" * 40)["verdict"]
    cases = [
        ("keep", v(arms()) == "KEEP"),
        ("keep just above the bar", v(arms(rates={"DEF": (1700.0, 140.0 * 1.0051)})) == "KEEP"),
        ("null just under the bar", v(arms(rates={"DEF": (1700.0, 140.0 * 1.0049)})) == "NULL"),
        ("null", v(arms(rates={"DEF": (1700.0, 140.0)})) == "NULL"),
        ("slower", v(arms(rates={"DEF": (1700.0, 140.0 * 0.99)})) == "SLOWER"),
        ("missing", v({k: x for k, x in arms().items() if k != "OFF2"}) == "VOID"),
        ("wrong e4b", v(arms(DEF1=_fake("DEF1", *RATES["DEF"], base, e4b="b" * 40))) == "VOID"),
        ("switch not read", v(arms(DEF2=_fake("DEF2", *RATES["DEF"], base, pdl=False))) == "VOID"),
        ("cap not 8", v(arms(DEF1=_fake("DEF1", *RATES["DEF"], base, cap=0))) == "VOID"),
        ("pdl with the switch off", v(arms(OFF2=_fake("OFF2", *RATES["OFF"], base,
                                                       acct={"_gemv_nf4_bw": [960, 1, 4, 4, 0], "_swiglu_rows": [480, 0, 4, 4, 0],
                                                             "_combine_rows": [480, 0, 4, 4, 0]}))) == "VOID"),
        ("the cap did not split", v(arms(DEF1=_fake("DEF1", *RATES["DEF"], base,
                                                     acct={"_gemv_nf4_bw": [960, 956, 4, 4, 4], "_swiglu_rows": [480, 476, 4, 4, 4],
                                                           "_combine_rows": [480, 476, 4, 4, 4]}))) == "VOID"),
        ("K33 never carried PDL", v(arms(DEF2=_fake("DEF2", *RATES["DEF"], base,
                                                     acct={"_gemv_nf4_bw": [960, 0, 4, 4, 0], "_swiglu_rows": [480, 96, 4, 4, 2],
                                                           "_combine_rows": [480, 96, 4, 4, 2]}))) == "VOID"),
        ("K33 never launched", v(arms(OFF1=_fake("OFF1", *RATES["OFF"], base,
                                                  acct={"_swiglu_rows": [480, 0, 4, 4, 0], "_combine_rows": [480, 0, 4, 4, 0]}))) == "VOID"),
        ("old accounting form", v(arms(OFF1=_fake("OFF1", *RATES["OFF"], base, acct={"_gemv_nf4_bw": [960, 0]}))) == "VOID"),
        ("launch counts differ", v(arms(DEF1=_fake("DEF1", *RATES["DEF"], base,
                                                    acct={"_gemv_nf4_bw": [950, 190, 4, 4, 2], "_swiglu_rows": [480, 96, 4, 4, 2],
                                                          "_combine_rows": [480, 96, 4, 4, 2]}))) == "VOID"),
        ("a bucket eager", v(arms(DEF2=_fake("DEF2", *RATES["DEF"], base,
                                             graphs={"1": "graph", "2": "graph", "4": "graph", "8": "graph", "16": "eager: x"}))) == "VOID"),
        ("configs differ", v(arms(OFF2=_fake("OFF2", *RATES["OFF"], base,
                                             config={"max_seqs": 32, "buckets": [1, 2, 4, 8, 16, 32], "placement": "all-vram",
                                                     "graphs": True}))) == "VOID"),
        ("void slope", v(arms(DEF1=_fake("DEF1", None, 141.0, base))) == "VOID"),
        ("noisy self-pair", v(arms(OFF2=_fake("OFF2", 1700.0, 140.0 * 1.006, base))) == "NOISY"),
        ("noisy W16 control", v(arms(rates={"DEF": (1700.0 * 1.006, 140.0 * 1.008)})) == "NOISY"),
    ]
    off = [list(r) for r in base]
    off[3][5] += 1
    cases.append(("a token moves", v(arms(DEF2=_fake("DEF2", *RATES["DEF"], off))) == "FUNCTION_FAIL"))
    cases.append(("a token moves on a noisy box: FUNCTION_FAIL first",
                  v(arms(DEF2=_fake("DEF2", *RATES["DEF"], off), OFF2=_fake("OFF2", 1700.0, 140.0 * 1.006, base)))
                  == "FUNCTION_FAIL"))
    cases.append(("bucket 16 carried PDL", v(arms(DEF1=_fake("DEF1", *RATES["DEF"], base,
                                                            byb={"1": [4, 4], "16": [4, 1]}))) == "VOID"))
    cases.append(("bucket 1 carried none", v(arms(DEF2=_fake("DEF2", *RATES["DEF"], base,
                                                            byb={"1": [4, 0], "16": [4, 0]}))) == "VOID"))
    cases.append(("off carried PDL in a bucket", v(arms(OFF1=_fake("OFF1", *RATES["OFF"], base,
                                                                  byb={"1": [4, 1], "16": [4, 0]}))) == "VOID"))
    cases.append(("no per-bucket record", v(arms(OFF2=_fake("OFF2", *RATES["OFF"], base, byb={"1": "x"}))) == "VOID"))
    cases.append(("reps differ", v(arms(DEF1=_fake("DEF1", *RATES["DEF"], base,
                                                   digests={"4": ["x"] * 3, "8": ["y", "z", "y"]}))) == "FUNCTION_FAIL"))
    r = reduce(arms(), "a" * 40)
    cases.append(("report", r["g1"] == round(1.008, 5) and r["switched_launches_build"]["DEF1"] == [1920, 384]
                  and r["whole_pass_tok_s"]["OFF1"]["W1"] == round(140.0 * 0.98, 4)))
    cases += _classify_cases()
    bad = [n for n, ok in cases if not ok]
    print(f"p230_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--classify", action="store_true")
    p.add_argument("--work")
    p.add_argument("--dir")
    p.add_argument("--out")
    p.add_argument("--e4b-sha", default=os.environ.get("E4B_SHA", ""))
    p.add_argument("--check-args", action="store_true", help="parse and validate the arguments, then exit (tests)")
    a = p.parse_args(argv)
    if a.check_args:
        ok = bool(a.work) if a.classify else bool(a.dir and a.out and a.e4b_sha)
        print("ARGS OK" if ok else "ARGS BAD")
        return 0 if ok else 2
    if a.self_test:
        return self_test()
    if a.classify:
        try:
            box = read_box(a.work or "")
        except ValueError as e:
            print(f"P230_CLASSIFY_REFUSED {e}")
            return 2
        print("P230_CLASS " + json.dumps(dict(classify(box["rc"], box["started"]), run_id=box["run_id"], rc=box["rc"])))
        return 0
    arms = {}
    for t in TAGS:
        f = os.path.join(a.dir, f"arm_{t}.json")
        if os.path.exists(f):
            arms[t] = json.load(open(f))
    v = reduce(arms, a.e4b_sha)
    json.dump(v, open(a.out, "w"), indent=1)
    print(f"P230_VERDICT {v['verdict']} {json.dumps(v['reasons'])}")
    for k in ("g1", "h1", "pair_ratios", "self_pairs", "decode_tok_s", "ms_per_step", "whole_pass_tok_s",
              "switched_launches_build"):
        if k in v:
            print(f"  {k}: {json.dumps(v[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
