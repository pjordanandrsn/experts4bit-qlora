#!/usr/bin/env python3
"""sc1g_reduce.py -- lane SC1g (#846): the registered rule (bench/sc2/SC1g-PREREG.md, amendment A1).

Windows: conv1, conv2 are GRADED (ultrachat_200k conversations in gpt-oss's chat template); wikitext is a DESCRIPTIVE control
(the chat-framed wikitext the proof showed is out of distribution). Every row is SC1's quality row (sc1_reduce.quality_row:
the window's text_sha, scored range [513, 2560], 2,048 steps, the scorer's own verdict). An e4b row is VALID only if its ROUTE
RECORD passes (sc1g_k8.py writes routes/<arm>.<pid>.json at exit; atexit does not run on a signal, so a missing, empty or null
record FAILS the row):
  T == 1 rows (served, pdl0, nofold, chunk1) on the MXFP4 store: `mxfp4_gemv|le256` at least steps x 24 layers, no
                 `mxfp4_*|gt256`, no route outside {mxfp4_gemv|*, nf4_mtile_host|*} (the 512-token prompt runs the kept NF4)
  prefill-shaped at least one `nf4_mtile_host|*` and no `mxfp4_*|gt256` (KEEP_NF4 observed)
  e4b_nf4 rows   a non-empty record with no `mxfp4_*` route at all
The floor (registered): per graded window, |NLL(e4b_serve prefill, chunk 64) - NLL(e4b_serve prefill, chunk 128)|; F = the max
over conv1 and conv2. A delta is "within the floor" iff |delta| <= F on BOTH graded windows. G1-G5 are read on conv1/conv2 only;
G6 is the kernel check on the 5090 (sc1g_gemv_check.py: the kernel against the bf16-rounded exact reference, with a mutation arm
that must disagree). The control and the diagnostics are reported, never graded.

  sc1g_reduce.py --dir SC1G_DIR --out verdict_sc1g.json [--prove]
  sc1g_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRCS = ("conv1", "conv2")          # graded
CTRL = ("wikitext",)               # descriptive
REP = ("conv3", "conv4")           # A3's replication windows (box J only): MXFP4 served, NF4 served and GEMV=0 rows
SHAPES = ("prefill", "served")
LAYERS = 24                        # gpt-oss-20b's MoE layers
SAME = 1e-9                        # two NLLs this close are the same arithmetic (the q8 switch did not engage)
# arm -> shape -> receipt stem (format with the window); e4b's quoted prefill row is chunk 128
ARMS = {
    "e4b_serve": {"served": "e4b_serve_served_{}", "prefill": "e4b_serve_prefill128_{}"},
    "e4b_serve_p64": {"prefill": "e4b_serve_prefill64_{}"},
    "e4b_nf4": {"served": "e4b_nf4_served_{}", "prefill": "e4b_nf4_prefill128_{}"},
    "vllm": {"served": "nll_vllm_served_{}", "prefill": "nll_vllm_prefill_{}"},
    "sglang_native": {"served": "nll_sglang_native_served_{}", "prefill": "nll_sglang_native_prefill_{}"},
    "sglang_marlin": {"served": "nll_sglang_marlin_served_{}", "prefill": "nll_sglang_marlin_prefill_{}"},
    "llamacpp": {"served": "nll_llamacpp_decode_{}", "prefill": "nll_llamacpp_prefill_{}"},
    "llamacpp_q8": {"served": "nll_llamacpp_q8_decode_{}", "prefill": "nll_llamacpp_q8_prefill_{}"},
}
# the diagnostics (graded windows only; descriptive)
DIAG = {"e4b_serve_pdl0": "e4b_serve_pdl0_{}", "e4b_serve_nofold": "e4b_serve_nofold_{}",
        "e4b_serve_chunk1": "e4b_serve_chunk1_{}", "e4b_nf4_chunk1": "e4b_nf4_chunk1_{}",
        # A2 (box J): the chunk-free full forward (the truth anchor), the fp8 paged kernel's K/V roundings modelled inside it,
        # the finer key groups, and the served shape on the real kernel at 16 key groups
        "e4b_nf4_full": "e4b_nf4_full_{}", "e4b_nf4_fqkv": "e4b_nf4_fqkv_{}", "e4b_nf4_fqk": "e4b_nf4_fqk_{}",
        "e4b_nf4_fqv": "e4b_nf4_fqv_{}", "e4b_nf4_fqkv16": "e4b_nf4_fqkv16_{}", "e4b_serve_kvg16": "e4b_serve_kvg16_{}",
        # A3 (box J again): the decode rows on bf16 activations (E4B_MXFP4_GEMV=0), a true MXFP4-weights prefill (KEEP_NF4=0), and
        # the explicit --kv-groups 4 served row (== auto at head_dim 64: a determinism control)
        "e4b_serve_v1": "e4b_serve_v1_{}", "e4b_mxpre_prefill128": "e4b_mxpre_prefill128_{}", "e4b_serve_kvg4": "e4b_serve_kvg4_{}"}
BOXJ_SERVED_CONV1 = 0.904969107589033   # sc1g-diag-1 (adertha-receipts a6a16350): MXFP4 served on conv1, A3's K4 repeat target
K_DET = 1e-4            # K4: the repeat must land this close (the eager K8 loop is bit-reproducible across boxes)
K_PRE = 0.05            # K1: MXFP4-weights prefill within this of NF4 prefill
K_GAP = 0.10            # K5: MXFP4 - NF4 served at least this -- ~2x box J's largest NF4 path-to-path spread (0.051, conv2)
K_MIN_WINDOWS = 3       # the across-window reads need at least this many windows with every row VALID
J_TOL_PDL = 0.005       # J6: PDL=0 within this of the served row is no bug (P113 read PDL value-identical)
J_TOL_FOLD = 0.035      # J6: folds-off within ~2x gpt-oss's arithmetic-order floor (0.0176) is no bug: the folds reorder arithmetic
J_GAIN = 0.035          # J4: ~2x the floor -- one floor is noise, and a J4 HOLDS feeds a registered default read
MEANING = {"pdl0": "moving vs served => an ordering (PDL) bug in the served T == 1 path",
           "nofold": "moving vs served => a fold (E4B_FUSE_*) bug in the served T == 1 path",
           "chunk1_vs_served": "the T == 1 expert route held, attention switched from paged fp8-KV decode to transformers' eager "
                               "attention with a bf16 cache: the share of the served-vs-prefill gap the paged path carries",
           "chunk1_vs_prefill": "attention held (transformers' eager), the expert route switched from T == 1 to the prefill M-tile"}
ARITH = {"e4b_serve": "served: MXFP4 GEMV on int8 activations (W4A8), paged fp8 decode attention; prefill: kept-NF4 host M-tile, transformers' eager attention",
         "e4b_serve_p64": "the floor's second order (chunk 64)",
         "e4b_nf4": "NF4 experts everywhere (the control)",
         "vllm": "Marlin W4A16; TRITON_ATTN",
         "sglang_native": "flashinfer_mxfp4 (cutlass_sm120, W4A8 with MXFP8 activations); triton attention",
         "sglang_marlin": "Marlin W4A16; triton attention",
         "llamacpp": "published GGUF (attention Q8_0); MMVQ W4A8 decode, MMQ W4A4 prefill",
         "llamacpp_q8": "published GGUF (attention Q8_0); GGML_CUDA_MMQ_PREC=q8: W4A8 prefill"}
# the proof's subset (conv1): A1's new paths plus one scoring per comparator engine
PROVE = ("e4b_serve_served_conv1", "e4b_serve_prefill64_conv1", "e4b_serve_chunk1_conv1", "nll_vllm_prefill_conv1",
         "nll_sglang_native_prefill_conv1", "nll_llamacpp_q8_prefill_conv1")


# ---- A4: the fidelity instrument's reading (KL65 to box R's bf16-dequant reference; sc1g_kl.py) ----------------------
A4_SRCS = ("conv1", "conv2", "conv3", "conv4")   # graded
A4_CTRL = ("wikitext",)                          # descriptive
KL_ARMS = {"e4b_mxfp4": "e4b_serve_served_{}", "e4b_nf4": "e4b_nf4_served_{}", "vllm": "nll_vllm_served_{}",
           "sglang_native": "nll_sglang_native_served_{}", "sglang_marlin": "nll_sglang_marlin_served_{}",
           "llamacpp": "nll_llamacpp_decode_{}", "llamacpp_q8": "nll_llamacpp_q8_decode_{}"}
COMPARATORS = ("vllm", "sglang_native", "sglang_marlin", "llamacpp", "llamacpp_q8")
NATIVE_MXFP4 = ("e4b_mxfp4",) + COMPARATORS      # every engine serving the checkpoint's own MXFP4 expert weights
KA_RATIO = 3.0          # K-A: e4b NF4 KL65 >= 3x e4b MXFP4 KL65 on every graded window (P44 read 11.6x on its prompts)
L1_RATIO = 2.0          # L1: e4b MXFP4 pooled KL65 <= 2x the best comparator's
A4_MIN_WINDOWS = 3      # an engine's pooled read needs this many graded windows VALID, else UNREAD
ALIGN_TOL = 1e-9        # the named rows' target log-probs must reproduce the arm's own mean NLL to this
PROVE_A4 = ("e4b_serve_served_conv1", "nll_vllm_served_conv1", "nll_sglang_native_served_conv1", "nll_llamacpp_q8_decode_conv1")


def _kl():
    sys.path.insert(0, HERE)
    import sc1g_kl
    return sc1g_kl


def a4_refs(d: str) -> tuple:
    """Box R's artifacts as staged into the receipt (`<d>/ref`): ({src: arrays}, R's calibration, why-not). R must have
    read R_OK and every artifact must hash to its registered sha, else the whole A4 reading is UNREAD."""
    K = _kl()
    rd = os.path.join(d, "ref")
    try:
        shas = json.load(open(os.path.join(rd, "ref_shas.json")))
        rv = json.load(open(os.path.join(rd, "r_verdict.json")))
        rc = json.load(open(os.path.join(rd, "r_calib.json")))
    except (OSError, ValueError) as e:
        return {}, None, f"box R's receipt is not staged ({e.__class__.__name__})"
    if rv.get("verdict") != "R_OK":
        return {}, rc, f"box R read {rv.get('verdict')}: {rv.get('checks')}"
    refs = {}
    for s in A4_SRCS + A4_CTRL:
        pth = os.path.join(rd, f"ref_{s}.npz")
        if s in shas and os.path.exists(pth):
            try:
                refs[s] = K.load_artifact(pth, shas[s])
            except SystemExit as e:
                return {}, rc, str(e)
    return refs, rc, None


def kl_row(d: str, label: str, src: str, R, refs: dict) -> dict:
    """One engine x window KL65 row, VALID only when the arm itself is VALID (route gates included), the named record is
    complete, and its target log-probs reproduce the arm's own mean NLL."""
    import numpy as np
    K = _kl()
    stem = KL_ARMS[label].format(src)
    q = row(d, stem, src, R)
    if q["verdict"] != "VALID":
        return {"verdict": "UNREAD", "why": f"arm {q['verdict']}: {q.get('why')}", "stem": stem}
    if src not in refs:
        return {"verdict": "UNREAD", "why": "no registered reference artifact", "stem": stem}
    pth = os.path.join(d, f"named_{stem}.npz")
    if not os.path.exists(pth):
        return {"verdict": "UNREAD", "why": "no named record", "stem": stem}
    z = np.load(pth)
    lp, tlp = z["eng_lp"], z["eng_target_lp"]
    rec = _load(d, stem) or {}
    nm = rec.get("named") or {}
    meta = (json.load(open(pth + ".json")) if os.path.exists(pth + ".json") else {})
    if nm.get("void_positions"):
        return {"verdict": "VOID", "why": f"{nm['void_positions']} positions lacked a named log-prob", "stem": stem}
    if label.startswith("e4b") and not meta:
        return {"verdict": "VOID", "why": "no capture meta record (sc1g_k8's proxy did not write its .json)", "stem": stem}
    if meta and meta.get("calls") != meta.get("positions"):
        return {"verdict": "VOID", "why": f"{meta.get('calls')} log_softmax rows for {meta.get('positions')} positions", "stem": stem}
    if lp.shape != refs[src]["lp"].shape or not np.all(np.isfinite(lp)) or not np.all(np.isfinite(tlp)):
        return {"verdict": "VOID", "why": "named record incomplete or misshapen", "stem": stem}
    gap = abs(float(-np.mean(tlp)) - float(q["mean_nll"]))
    if gap > ALIGN_TOL:
        return {"verdict": "VOID", "why": f"named target log-probs give NLL {-np.mean(tlp):.9f} vs the arm's {q['mean_nll']:.9f}", "stem": stem}
    w = K.window_read(refs[src], lp, tlp)
    return dict(w, stem=stem, alignment_gap=gap, arm_mean_nll=q["mean_nll"])


def a4(d: str) -> dict:
    """A4's reading: the instrument gate (box R), every KL row, the predictions K-A / L1 / L2, and the descriptive table."""
    R = _sc1()
    refs, rc, why = a4_refs(d)
    rows = {lab: {s: kl_row(d, lab, s, R, refs) for s in A4_SRCS + A4_CTRL} for lab in KL_ARMS}
    unread = {"verdict": "UNREAD", "why": why}

    def km(lab, s):
        x = rows[lab][s]
        return x["kl65_mean"] if x["verdict"] == "VALID" else None

    def pooled(lab):
        v = [km(lab, s) for s in A4_SRCS if km(lab, s) is not None]
        return (sum(v) / len(v), len(v)) if len(v) >= A4_MIN_WINDOWS else (None, len(v))

    pred = {}
    if why:
        pred = {"K-A": unread, "L1": unread, "L2": unread}
    else:
        both = [s for s in A4_SRCS if km("e4b_mxfp4", s) is not None and km("e4b_nf4", s) is not None]
        ratios = {s: round(km("e4b_nf4", s) / km("e4b_mxfp4", s), 3) if km("e4b_mxfp4", s) > 0 else None for s in both}
        pred["K-A"] = ({"verdict": "UNREAD", "why": f"{len(both)} windows < {A4_MIN_WINDOWS}"} if len(both) < A4_MIN_WINDOWS else
                       {"verdict": "HOLDS" if all(r is not None and r >= KA_RATIO for r in ratios.values()) else "REFUTED",
                        "nf4_over_mxfp4": ratios})
        e4b, ne = pooled("e4b_mxfp4")
        comp = {c: pooled(c)[0] for c in COMPARATORS if pooled(c)[0] is not None}
        pred["L1"] = ({"verdict": "UNREAD", "why": "e4b or every comparator lacks enough graded windows"} if e4b is None or not comp else
                      {"verdict": "HOLDS" if e4b <= L1_RATIO * min(comp.values()) else "REFUTED", "e4b_pooled": e4b,
                       "best_comparator": min(comp, key=comp.get), "best_pooled": min(comp.values()),
                       "ratio": (e4b / min(comp.values()) if min(comp.values()) > 0 else None)})
        nf4_scale = [rc["windows"][s]["calib_nf4"]["kl_full_mean"] for s in A4_SRCS
                     if s in (rc or {}).get("windows", {}) and "calib_nf4" in rc["windows"][s]]
        scale = sum(nf4_scale) / len(nf4_scale) if len(nf4_scale) >= A4_MIN_WINDOWS else None
        eng = {lab: pooled(lab)[0] for lab in NATIVE_MXFP4 if pooled(lab)[0] is not None}
        pred["L2"] = ({"verdict": "UNREAD", "why": "no R NF4-pair scale or no engine read"} if scale is None or not eng else
                      {"verdict": "HOLDS" if all(v < scale for v in eng.values()) else "REFUTED", "nf4_requant_scale": scale,
                       "at_or_above": sorted(k for k, v in eng.items() if v >= scale)})
    floor = {s: (rc or {}).get("windows", {}).get(s, {}).get("floor_F") for s in A4_SRCS + A4_CTRL}
    table = {lab: {s: (None if rows[lab][s]["verdict"] != "VALID" else
                       {"kl65": rows[lab][s]["kl65_mean"], "nll": rows[lab][s].get("engine_nll"),
                        "within_floor_F": (floor[s] is not None and rows[lab][s]["kl65_mean"] <= floor[s])})
                   for s in A4_SRCS + A4_CTRL} for lab in KL_ARMS}
    rank = sorted(((lab, pooled(lab)[0]) for lab in KL_ARMS if pooled(lab)[0] is not None), key=lambda x: x[1])
    return {"instrument": {"box_r": "R_OK" if not why else "NOT_OK", "why": why}, "rows": rows, "predictions": pred,
            "descriptive": {"per_window": table, "pooled_rank": rank, "floor_F": floor,
                            "reference_nll": {s: (rc or {}).get("windows", {}).get(s, {}).get("reference_nll_decode") for s in A4_SRCS + A4_CTRL},
                            "note": "NLL is descriptive under A4; the KL65 rank is what A4 reads"}}


def prove_a4(d: str) -> dict:
    R = _sc1()
    refs, _rc, why = a4_refs(d)
    lab_of = {v.format("conv1"): k for k, v in KL_ARMS.items()}
    out = {stem: (kl_row(d, lab_of[stem], "conv1", R, refs) if not why else {"verdict": "UNREAD", "why": why}) for stem in PROVE_A4}
    bad = {k: f"{v['verdict']}: {v.get('why')}" for k, v in out.items() if v["verdict"] != "VALID"}
    return {"rows": out, "bad": bad, "proved": not bad, "box_r": why or "R_OK"}


# ---- A5: the full-vocabulary KL reading ---------------------------------------------------------------------------------
KL5_ARMS = {"e4b_mxfp4": "e4b_serve_served_{}", "e4b_nf4": "e4b_nf4_served_{}", "vllm": "nll_vllm_served_{}",
            "llamacpp": "nll_llamacpp_decode_{}", "llamacpp_q8": "nll_llamacpp_q8_decode_{}"}
KL5_UNREAD = {"sglang_native": "no full-distribution path (A5, registered before data)",
              "sglang_marlin": "no full-distribution path (A5, registered before data)"}
COMPARATORS5 = ("vllm", "llamacpp", "llamacpp_q8")
NATIVE_MXFP4_5 = ("e4b_mxfp4",) + COMPARATORS5
PROVE_A5 = ("e4b_serve_served_conv1", "nll_vllm_served_conv1", "nll_llamacpp_q8_decode_conv1")
SUPPORT_KEYS = ("eng_kl_common", "eng_masked_mass", "eng_n_masked")   # every engine's per-position support record (#1223 review)


def _fin(fn, a):
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(fn(a)) if a.size else None


def a5_refs(d: str) -> tuple:
    """Box R's A5 receipt as staged into `<d>/ref`: (registered full shas, R's calibration, R's verdict, why-not). The whole
    A5 reading is UNREAD unless R read R_OK under rule A5."""
    rd = os.path.join(d, "ref")
    try:
        shas = json.load(open(os.path.join(rd, "ref_full_shas.json")))
        rv = json.load(open(os.path.join(rd, "r_verdict.json")))
        rc = json.load(open(os.path.join(rd, "r_calib.json")))
    except (OSError, ValueError) as e:
        return {}, None, None, f"box R's A5 receipt is not staged ({e.__class__.__name__})"
    if rv.get("rule") != "A5" or rv.get("verdict") != "R_OK":
        return shas, rc, rv, f"box R read {rv.get('verdict')} under rule {rv.get('rule', 'A4')}"
    return shas, rc, rv, None


def kl_row_full(d: str, label: str, src: str, R, shas: dict) -> dict:
    """One engine x window full-vocabulary KL row; VALID only when the arm is VALID, its record is complete and finite,
    e4b's proxy meta exists with one row per step, vLLM's full vocabulary verified, the recorded reference sha is the
    registered one, no position is void, the engine masks (-inf) no token the reference gives mass, and the target
    log-probs reproduce the arm's own mean NLL to ALIGN_TOL. Every row read carries `support` (#1223 review): the reference
    mass on the engine's masked tokens, the masked count, the common-support KL -- what a common-support rule would grade,
    registered on the proof's numbers if the proof finds masked mass, never on the reading's."""
    import numpy as np
    stem = KL5_ARMS[label].format(src)
    q = row(d, stem, src, R)
    if q["verdict"] != "VALID":
        return {"verdict": "UNREAD", "why": f"arm {q['verdict']}: {q.get('why')}", "stem": stem}
    pth = os.path.join(d, f"kl_{stem}.npz")
    rec = _load(d, stem) or {}
    if (rec.get("kl_full") or {}).get("verdict") == "VOID":
        return {"verdict": "VOID", "why": rec["kl_full"]["why"], "stem": stem}
    if not os.path.exists(pth):
        return {"verdict": "UNREAD", "why": "no KL record", "stem": stem}
    meta = json.load(open(pth + ".json")) if os.path.exists(pth + ".json") else {}
    if label.startswith("e4b"):
        if not meta:
            return {"verdict": "VOID", "why": "no capture meta record (sc1g_k8's proxy did not write its .json)", "stem": stem}
        if meta.get("calls") != meta.get("positions"):
            return {"verdict": "VOID", "why": f"{meta.get('calls')} rows for {meta.get('positions')} positions", "stem": stem}
    ref_sha = meta.get("ref_sha") or (rec.get("kl_full") or {}).get("ref_sha")
    if ref_sha and shas.get(src) and ref_sha != shas[src]:
        return {"verdict": "VOID", "why": f"KL read against reference {ref_sha[:12]}, registered {shas[src][:12]}", "stem": stem}
    z = np.load(pth)
    if not all(k in z.files for k in SUPPORT_KEYS):
        return {"verdict": "VOID", "why": "no support record (eng_kl_common / eng_masked_mass / eng_n_masked)", "stem": stem}
    kl, tlp = z["eng_kl"], z["eng_target_lp"]
    mm, klc, nm = z["eng_masked_mass"], z["eng_kl_common"], z["eng_n_masked"]
    masked = np.isfinite(mm) & (mm > 0)
    void_n = int(np.isnan(kl).sum())
    sup = {"positions_masked": int(masked.sum()), "masked_mass_max": _fin(np.max, mm), "masked_mass_mean": _fin(np.mean, mm),
           "n_masked_max": _fin(np.max, nm), "kl_common_mean": _fin(np.mean, klc), "void_positions": void_n}
    if void_n:
        first = ((meta.get("void_first") or (rec.get("kl_full") or {}).get("void_first") or [{}]) + [{}])[0]
        return {"verdict": "VOID", "why": f"{void_n} void positions (first: t={first.get('t')} {first.get('why')})", "stem": stem,
                "support": sup}
    if masked.any():
        return {"verdict": "VOID", "why": f"the engine masks tokens carrying reference mass at {sup['positions_masked']} positions "
                                          f"(max {sup['masked_mass_max']:.3e}): the full KL is infinite and no common-support "
                                          f"rule is registered", "stem": stem, "support": sup}
    if not (np.all(np.isfinite(kl)) and np.all(np.isfinite(tlp))):
        return {"verdict": "VOID", "why": "KL record incomplete (non-finite entries)", "stem": stem}
    gap = abs(float(-np.mean(tlp)) - float(q["mean_nll"]))
    if gap > ALIGN_TOL:
        return {"verdict": "VOID", "why": f"target log-probs give NLL {-np.mean(tlp):.9f} vs the arm's {q['mean_nll']:.9f}", "stem": stem}
    return {"verdict": "VALID", "stem": stem, "kl_mean": float(np.mean(kl)), "kl_p50": float(np.median(kl)),
            "kl_p95": float(np.quantile(kl, 0.95)), "positions": int(kl.size), "engine_nll": float(q["mean_nll"]), "alignment_gap": gap,
            "support": sup}


def a5(d: str) -> dict:
    """A5's reading: the R gate, every full-KL row, per-window gradability (graded only if R's floor F < 1e-2; any engine
    KL within its window's F is unresolved), and K-A / L1 / L2 with the registered floor rules."""
    R = _sc1()
    shas, rc, rv, why = a5_refs(d)
    rows = {lab: {s: kl_row_full(d, lab, s, R, shas) for s in A4_SRCS + A4_CTRL} for lab in KL5_ARMS}
    for lab, reason in KL5_UNREAD.items():
        rows[lab] = {s: {"verdict": "UNREAD", "why": reason} for s in A4_SRCS + A4_CTRL}
    F = {s: (rc or {}).get("windows", {}).get(s, {}).get("floor_F") for s in A4_SRCS + A4_CTRL}
    nf4 = {s: (rc or {}).get("windows", {}).get(s, {}).get("calib_nf4", {}).get("kl_full_mean") for s in A4_SRCS + A4_CTRL}
    gradable = [s for s in A4_SRCS if F.get(s) is not None and F[s] < 1e-2]

    def k(lab, s):
        x = rows[lab][s]
        return x["kl_mean"] if x["verdict"] == "VALID" else None

    def resolved(v, s):
        return v is not None and v > F[s]

    pred = {}
    if why:
        pred = {g: {"verdict": "UNREAD", "why": why} for g in ("K-A", "L1", "L2")}
    else:
        # K-A: NF4 >= 3x MXFP4 per window; NF4 unresolved -> drop; MXFP4 unresolved (NF4 resolved) -> F as its upper bound
        ka = {}
        for s in gradable:
            m, n = k("e4b_mxfp4", s), k("e4b_nf4", s)
            if m is None or n is None or not resolved(n, s):
                continue
            den = m if resolved(m, s) else F[s]
            ka[s] = {"nf4": n, "mxfp4_or_bound": den, "bound_used": not resolved(m, s), "holds": n >= KA_RATIO * den}
        pred["K-A"] = ({"verdict": "UNREAD", "why": f"{len(ka)} windows < {A4_MIN_WINDOWS}", "windows": ka} if len(ka) < A4_MIN_WINDOWS else
                       {"verdict": "HOLDS" if all(x["holds"] for x in ka.values()) else "REFUTED", "windows": ka})
        # L1: windows whose best comparator is unresolved are dropped; an unresolved e4b takes F as its upper bound
        kept = []
        for s in gradable:
            comps = [k(c, s) for c in COMPARATORS5 if k(c, s) is not None]
            if k("e4b_mxfp4", s) is None or not comps or not resolved(min(comps), s):
                continue
            kept.append(s)
        if len(kept) < A4_MIN_WINDOWS:
            pred["L1"] = {"verdict": "UNREAD", "why": f"{len(kept)} windows < {A4_MIN_WINDOWS} with e4b and a resolved comparator", "windows": kept}
        else:
            e4b_eff = sum(max(k("e4b_mxfp4", s), F[s]) if not resolved(k("e4b_mxfp4", s), s) else k("e4b_mxfp4", s) for s in kept) / len(kept)
            comp = {c: sum(k(c, s) for s in kept) / len(kept) for c in COMPARATORS5 if all(k(c, s) is not None for s in kept)}
            best = min(comp, key=comp.get) if comp else None
            pred["L1"] = ({"verdict": "UNREAD", "why": "no comparator VALID on every kept window"} if best is None else
                          {"verdict": "HOLDS" if e4b_eff <= L1_RATIO * comp[best] else "REFUTED", "e4b_pooled": e4b_eff,
                           "best_comparator": best, "best_pooled": comp[best], "windows": kept})
        # L2: each native-MXFP4 engine's pooled KL < R's NF4-requant pooled KL, over windows where both are resolved
        l2 = {}
        for lab in NATIVE_MXFP4_5:
            ws = [s for s in gradable if resolved(k(lab, s), s) and resolved(nf4.get(s), s)]
            if len(ws) >= A4_MIN_WINDOWS:
                e, sc = sum(k(lab, s) for s in ws) / len(ws), sum(nf4[s] for s in ws) / len(ws)
                l2[lab] = {"pooled": e, "nf4_scale": sc, "below": e < sc, "windows": ws}
        pred["L2"] = ({"verdict": "UNREAD", "why": "no engine resolved on >= 3 windows"} if not l2 else
                      {"verdict": "HOLDS" if all(x["below"] for x in l2.values()) else "REFUTED", "engines": l2,
                       "at_or_above": sorted(lab for lab, x in l2.items() if not x["below"])})
    table = {lab: {s: (None if rows[lab][s]["verdict"] != "VALID" else
                       {"kl": rows[lab][s]["kl_mean"], "nll": rows[lab][s].get("engine_nll"),
                        "within_F": (F[s] is not None and rows[lab][s]["kl_mean"] <= F[s])}) for s in A4_SRCS + A4_CTRL}
             for lab in KL5_ARMS}
    return {"instrument": {"box_r": (rv or {}).get("verdict"), "rule": (rv or {}).get("rule"), "why": why}, "rows": rows,
            "gradable_windows": gradable, "floor_F": F, "nf4_requant_kl": nf4, "predictions": pred,
            "descriptive": {"per_window": table, "note": "NLL descriptive; wikitext is the out-of-distribution control, never graded"}}


def prove_a5(d: str) -> dict:
    R = _sc1()
    shas, _rc, _rv, why = a5_refs(d)
    lab_of = {v.format("conv1"): lab for lab, v in KL5_ARMS.items()}
    out = {stem: (kl_row_full(d, lab_of[stem], "conv1", R, shas) if not why else {"verdict": "UNREAD", "why": why}) for stem in PROVE_A5}
    bad = {k: f"{v['verdict']}: {v.get('why')}" for k, v in out.items() if v["verdict"] != "VALID"}
    # per engine, the reference mass on the tokens it masks (#1223 review): non-zero anywhere -> a common-support rule with a
    # mass bound is registered on THESE numbers before the reading box (each row's common-support KL is already recorded)
    support = {k: v.get("support") for k, v in out.items()}
    need = sorted(k for k, s in support.items() if s and s.get("positions_masked"))
    return {"rows": out, "bad": bad, "proved": not bad, "box_r": why or "R_OK", "support": support, "support_rule_needed": need}


def _sc1():
    for p in (os.path.join(HERE, "sc1_reduce.py"), os.path.join(HERE, "..", "sc1", "sc1_reduce.py")):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("sc1_reduce", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["sc1_reduce"] = m          # its dataclasses resolve their module through sys.modules
            spec.loader.exec_module(m)
            return m
    raise SystemExit("sc1_reduce.py not found beside this file or in ../sc1")


def _load(d, stem):
    p = os.path.join(d, stem + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def _wsha(d, src):
    p = os.path.join(d, f"k8_window_{src}.json")
    return json.load(open(p)).get("text_sha") if os.path.exists(p) else None


def route_gate(d, stem, steps=2048):
    """-> (ok, why, merged route_seen). Reads every routes/<stem>.<pid>.json."""
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "routes", stem + ".*.json")))]
    if not recs:
        return False, "no route record (atexit did not run, or the process never loaded e4b)", None
    seen = {}
    for r in recs:
        if r.get("route_seen") is None:
            return False, "route record is null (the e4b build predates e4b#1129's counters)", None
        for k, v in r["route_seen"].items():
            seen[k] = seen.get(k, 0) + int(v)
    if not seen:
        return False, "route record is empty", seen
    if stem.startswith("e4b_nf4_"):
        bad = [k for k in seen if k.startswith("mxfp4_")]
        return (not bad), (f"MXFP4 route(s) {bad} on the NF4 control" if bad else ""), seen
    if stem.startswith("e4b_mxpre_"):                 # A3: NF4 emptied -> every row on the MXFP4 store's grouped v1 kernel
        other = [k for k in seen if not k.startswith("mxfp4_grouped_v1|")]
        if other or "mxfp4_grouped_v1|gt256" not in seen:
            return False, f"routes {sorted(seen)}: a KEEP_NF4=0 prefill must run mxfp4_grouped_v1 only, incl. rows > 256", seen
        return True, "", seen
    if "_v1_" in stem:                                # A3: E4B_MXFP4_GEMV=0 -> T == 1 rows on the grouped v1 kernel, bf16
        other = [k for k in seen if not (k.startswith("mxfp4_grouped_v1|le256") or k.startswith("nf4_mtile_host|"))]
        n = seen.get("mxfp4_grouped_v1|le256", 0)
        if other or n < steps * LAYERS:
            return False, f"routes {sorted(seen)}: GEMV=0 must run mxfp4_grouped_v1|le256 for every step x layer", seen
        return True, "", seen
    gt = [k for k in seen if k.startswith("mxfp4_") and k.endswith("|gt256")]
    if gt:
        return False, f"{gt}: rows above 256 on the MXFP4 store (the NF4 stacks were not kept)", seen
    if any(f"_{x}_" in stem for x in ("served", "pdl0", "nofold", "chunk1", "kvg16", "kvg4")):
        other = [k for k in seen if not (k.startswith("mxfp4_gemv|") or k.startswith("nf4_mtile_host|"))]
        n = seen.get("mxfp4_gemv|le256", 0)
        if other:
            return False, f"route(s) {other} outside the T == 1 GEMV + the prompt's NF4", seen
        if n < steps * LAYERS:
            return False, f"mxfp4_gemv|le256 {n} < {steps} steps x {LAYERS} layers", seen
        return True, "", seen
    if not any(k.startswith("nf4_mtile_host|") for k in seen):
        return False, "no nf4_mtile_host route on the prefill shape", seen
    return True, "", seen


def row(d, stem, src, R):
    q = R.quality_row(_load(d, stem), _wsha(d, src))
    q["stem"] = stem
    if stem.startswith("e4b_") and q["verdict"] == "VALID":
        ok, why, seen = route_gate(d, stem)
        q["route_seen"] = seen
        if not ok:
            q.update(verdict="VOID", why=f"route gate: {why}")
    if stem.startswith("nll_llamacpp"):
        q["mmq_prec_env"] = (_load(d, stem) or {}).get("mmq_prec_env")
    return q


def gemv(d):
    """The kernel check receipts in the box's dir: -> ({name: summary}, G6)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "gemv_*.json"))):
        r = json.load(open(p))
        out[os.path.basename(p)[:-5]] = {"verdict": r.get("verdict"), "kernel_vs_exact_worst": r.get("kernel_vs_exact_worst"),
                                        "metric": r.get("metric"), "device": r.get("device"), "mutate": r.get("mutate"),
                                        "summary": r.get("summary")}
    real = {k: v for k, v in out.items() if not v["mutate"]}
    mut = [v for v in out.values() if v["mutate"]]
    if not real or not mut:
        g6 = {"verdict": "UNREAD", "why": "no kernel check receipt" if not real else "no mutation arm"}
    elif any(v["verdict"] != "KERNEL_DISAGREES" for v in mut):
        g6 = {"verdict": "UNREAD", "why": "the mutation arm AGREED: the check could not see a wrong expert (inert)"}
    else:
        bad = [k for k, v in real.items() if v["verdict"] != "KERNEL_AGREES"]
        g6 = {"verdict": "REFUTED" if bad else "HOLDS", "disagreeing": bad,
              "worst": max((v["kernel_vs_exact_worst"] or 0.0) for v in real.values())}
    return out, g6


def reduce(d: str) -> dict:
    R = _sc1()
    wins = SRCS + CTRL + REP
    rows = {a: {sh: {s: row(d, stem.format(s), s, R) for s in wins} for sh, stem in shp.items()} for a, shp in ARMS.items()}
    diag = {a: {s: row(d, stem.format(s), s, R) for s in SRCS + REP} for a, stem in DIAG.items()}

    def nll(arm, shape, src):
        x = rows.get(arm, {}).get(shape, {}).get(src)
        return x["mean_nll"] if x and x["verdict"] == "VALID" else None

    def dnll(arm, src):
        x = diag[arm][src]
        return x["mean_nll"] if x["verdict"] == "VALID" else None

    def floor(srcs):
        f = {}
        for s in srcs:
            a, b = nll("e4b_serve_p64", "prefill", s), nll("e4b_serve", "prefill", s)
            f[s] = None if None in (a, b) else abs(a - b)
        return f, (None if None in f.values() else max(f.values()))
    fl, F = floor(SRCS)
    flc, Fc = floor(CTRL)

    def delta(a1, a2, shape, srcs=SRCS):
        out = {}
        for s in srcs:
            x, y = nll(a1, shape, s), nll(a2, shape, s)
            out[s] = None if None in (x, y) else round(x - y, 6)
        return out

    def within(dl):
        if F is None or None in dl.values():
            return None
        return all(abs(v) <= F for v in dl.values())

    def sub(a, b):
        return None if None in (a, b) else round(a - b, 6)

    p = {}
    d1 = delta("e4b_serve", "vllm", "served")
    w = within(d1)
    p["G1"] = {"verdict": "UNREAD" if w is None else ("HOLDS" if w else "REFUTED"), "delta": d1, "floor": F}
    pre = {s: nll("e4b_serve", "prefill", s) for s in SRCS}
    srv = {s: nll("e4b_serve", "served", s) for s in SRCS}
    if None in pre.values() or None in srv.values():
        p["G2"] = {"verdict": "UNREAD"}
    else:
        gap = {s: round(pre[s] - srv[s], 6) for s in SRCS}
        p["G2"] = {"verdict": "HOLDS" if all(v >= 0 for v in gap.values()) else "REFUTED", "prefill_minus_served": gap}
    d3 = delta("llamacpp", "llamacpp_q8", "prefill")
    if None not in d3.values() and all(abs(v) < SAME for v in d3.values()):
        p["G3"] = {"verdict": "UNREAD", "why": "NOT_ENGAGED: default and q8 prefill NLL identical (the switch did not take)", "delta": d3}
    else:
        w = within(d3)
        p["G3"] = {"verdict": "UNREAD" if w is None else ("HOLDS" if not w else "REFUTED"), "delta": d3, "floor": F}
    d4 = {sh: delta("vllm", "sglang_marlin", sh) for sh in SHAPES}
    w4 = [within(d4[sh]) for sh in SHAPES]
    p["G4"] = {"verdict": "UNREAD" if None in w4 else ("HOLDS" if all(w4) else "REFUTED"), "matched_arm_disagreement": d4, "floor": F}
    st = {f"{a}/{sh}/{s}": rows[a][sh][s]["verdict"] for a, shp in ARMS.items() for sh in shp for s in SRCS}
    p["G5"] = {"verdict": "HOLDS" if all(v == "VALID" for v in st.values()) else "REFUTED",
               "not_valid": {k: v for k, v in st.items() if v != "VALID"}}
    kern, p["G6"] = gemv(d)
    jp = jpredict(nll, dnll)
    kp = kpredict(nll, dnll)

    # ---- descriptive: the diagnostics, the served-vs-prefill gap per arm, the control
    dg = {}
    for s in SRCS:
        srv_m, pre_m = nll("e4b_serve", "served", s), nll("e4b_serve", "prefill", s)
        srv_n, pre_n = nll("e4b_nf4", "served", s), nll("e4b_nf4", "prefill", s)
        c1m, c1n = dnll("e4b_serve_chunk1", s), dnll("e4b_nf4_chunk1", s)
        frac = {}
        for k, (sv, c1, pr) in {"mxfp4": (srv_m, c1m, pre_m), "nf4": (srv_n, c1n, pre_n)}.items():
            frac[k] = None if None in (sv, c1, pr) or abs(sv - pr) < 1e-12 else round((sv - c1) / (sv - pr), 4)
        dg[s] = {"pdl0_minus_served": sub(dnll("e4b_serve_pdl0", s), srv_m),
                 "nofold_minus_served": sub(dnll("e4b_serve_nofold", s), srv_m),
                 "mxfp4": {"served_minus_prefill": sub(srv_m, pre_m), "served_minus_chunk1": sub(srv_m, c1m),
                           "chunk1_minus_prefill": sub(c1m, pre_m)},
                 "nf4": {"served_minus_prefill": sub(srv_n, pre_n), "served_minus_chunk1": sub(srv_n, c1n),
                         "chunk1_minus_prefill": sub(c1n, pre_n)},
                 "share_of_served_gap_closed_by_chunk1": frac}
    gaps = {a: {s: sub(nll(a, "served", s), nll(a, "prefill", s)) for s in SRCS + CTRL}
            for a, shp in ARMS.items() if "served" in shp and "prefill" in shp}
    arms_cmp = ("e4b_serve", "e4b_nf4", "sglang_native", "sglang_marlin", "llamacpp", "llamacpp_q8")
    rep = {a: {sh: delta(a, "vllm", sh) for sh in SHAPES} for a in arms_cmp}
    ctl = {a: {sh: delta(a, "vllm", sh, CTRL) for sh in SHAPES} for a in arms_cmp}
    roles = {}
    for s in SRCS:
        p_ = os.path.join(d, f"k8_window_{s}.json")
        roles[s] = json.load(open(p_)).get("scored_target_roles") if os.path.exists(p_) else None
    return {"rows": rows, "diagnostic_rows": diag, "arith": ARITH, "floor": {"per_window": fl, "F": F},
            "predictions": p, "delta_vs_vllm": rep,
            "within_floor_vs_vllm": {a: {sh: within(v) for sh, v in by.items()} for a, by in rep.items()},
            "diagnostics": {"meaning": MEANING, "per_window": dg}, "served_minus_prefill": gaps, "kernel_check": kern,
            "scored_target_roles": roles, "diag_predictions": jp, "a3_predictions": kp, "a4": a4(d), "a5": a5(d),
            "attn_check_1175": (json.load(open(os.path.join(d, "attn_check_5090.json")))
                                if os.path.exists(os.path.join(d, "attn_check_5090.json")) else None),
            "control_wikitext": {"floor": {"per_window": flc, "F": Fc}, "delta_vs_vllm": ctl,
                                 "note": "descriptive: out of distribution for gpt-oss (the proof read ppl ~560)"}}


def jpredict(nll, dnll):
    """A2's registered box-J predictions, read on conv1 (conv2 reported as replication where its rows exist)."""
    def g(arm, s, shape=None):
        return nll(arm, shape, s) if shape else dnll(arm, s)

    def j(s):
        srv_n, full, c1n = g("e4b_nf4", s, "served"), g("e4b_nf4_full", s), g("e4b_nf4_chunk1", s)
        fqkv, fqk, fqv, fqkv16 = g("e4b_nf4_fqkv", s), g("e4b_nf4_fqk", s), g("e4b_nf4_fqv", s), g("e4b_nf4_fqkv16", s)
        srv_m, kvg16, c1m = g("e4b_serve", s, "served"), g("e4b_serve_kvg16", s), g("e4b_serve_chunk1", s)
        pdl0, nofold = g("e4b_serve_pdl0", s), g("e4b_serve_nofold", s)
        out = {}
        gap = None if None in (srv_n, full) else srv_n - full
        out["J1"] = ({"verdict": "UNREAD"} if gap is None or c1n is None or abs(gap) < 1e-9 else
                     {"verdict": "HOLDS" if (srv_n - c1n) / gap >= 0.5 else "REFUTED", "share": round((srv_n - c1n) / gap, 4),
                      "nf4_served_minus_full": round(gap, 6)})
        out["J2"] = ({"verdict": "UNREAD"} if gap is None or fqkv is None else
                     {"verdict": "HOLDS" if fqkv - full >= 0.5 * gap else "REFUTED", "fqkv_minus_full": round(fqkv - full, 6),
                      "half_gap": round(0.5 * gap, 6)})
        out["J3"] = ({"verdict": "UNREAD"} if None in (full, fqk, fqv) else
                     {"verdict": "HOLDS" if fqk - full > fqv - full else "REFUTED", "k": round(fqk - full, 6), "v": round(fqv - full, 6)})
        out["J4"] = ({"verdict": "UNREAD"} if None in (fqkv, fqkv16, srv_m, kvg16) else
                     {"verdict": "HOLDS" if (fqkv - fqkv16 >= J_GAIN and srv_m - kvg16 >= J_GAIN) else "REFUTED",
                      "modelled_gain": round(fqkv - fqkv16, 6), "served_gain": round(srv_m - kvg16, 6)})
        out["J5"] = ({"verdict": "UNREAD"} if None in (c1m, c1n) else
                     {"verdict": "HOLDS" if c1m - c1n >= 0 else "REFUTED", "mxfp4_minus_nf4_chunk1": round(c1m - c1n, 6)})
        out["J6"] = ({"verdict": "UNREAD"} if None in (srv_m, pdl0, nofold) else
                     {"verdict": "HOLDS" if abs(pdl0 - srv_m) <= J_TOL_PDL and abs(nofold - srv_m) <= J_TOL_FOLD else "REFUTED",
                      "pdl0": round(pdl0 - srv_m, 6), "nofold": round(nofold - srv_m, 6)})
        return out
    return {"conv1": j("conv1"), "conv2_replication": j("conv2")}


def kpredict(nll, dnll):
    """A3's registered predictions (box J again), read on conv1 (conv2 reported as replication where its rows exist)."""
    def k(s):
        srv_m, srv_n = nll("e4b_serve", "served", s), nll("e4b_nf4", "served", s)
        v1, mxpre, pre_n = dnll("e4b_serve_v1", s), dnll("e4b_mxpre_prefill128", s), nll("e4b_nf4", "prefill", s)
        pdl0, nofold, kvg4 = dnll("e4b_serve_pdl0", s), dnll("e4b_serve_nofold", s), dnll("e4b_serve_kvg4", s)
        out = {}
        out["K1"] = ({"verdict": "UNREAD"} if None in (mxpre, pre_n) else
                     {"verdict": "HOLDS" if mxpre - pre_n <= K_PRE else "REFUTED", "mxfp4_prefill_minus_nf4_prefill": round(mxpre - pre_n, 6)})
        gap = None if None in (srv_m, srv_n) else srv_m - srv_n
        out["K2"] = ({"verdict": "UNREAD"} if gap is None or v1 is None or abs(gap) < 1e-9 else
                     {"verdict": "HOLDS" if (srv_m - v1) / gap >= 0.5 else "REFUTED", "share_carried_by_int8": round((srv_m - v1) / gap, 4),
                      "mxfp4_minus_nf4_served": round(gap, 6), "v1_minus_nf4_served": round(v1 - srv_n, 6)})
        out["K3"] = ({"verdict": "UNREAD"} if None in (srv_m, pdl0, nofold) else
                     {"verdict": "HOLDS" if abs(pdl0 - srv_m) <= J_TOL_PDL and abs(nofold - srv_m) <= J_TOL_FOLD else "REFUTED",
                      "pdl0": round(pdl0 - srv_m, 6), "nofold": round(nofold - srv_m, 6)})
        if s == "conv1":
            out["K4"] = ({"verdict": "UNREAD"} if None in (srv_m, kvg4) else
                         {"verdict": "HOLDS" if abs(srv_m - BOXJ_SERVED_CONV1) <= K_DET and abs(kvg4 - srv_m) <= 1e-9 else "REFUTED",
                          "repeat_minus_boxj": round(srv_m - BOXJ_SERVED_CONV1, 9), "kvg4_minus_auto": round(kvg4 - srv_m, 12)})
        out["K5"] = ({"verdict": "UNREAD"} if gap is None else
                     {"verdict": "HOLDS" if gap >= K_GAP else "REFUTED", "mxfp4_minus_nf4_served": round(gap, 6)})
        return out

    # across every graded + replication window with the rows VALID: K5 on each window, K2 pooled (sum over windows)
    per = {}
    for s in SRCS + REP:
        m, n, v = nll("e4b_serve", "served", s), nll("e4b_nf4", "served", s), dnll("e4b_serve_v1", s)
        if None not in (m, n):
            per[s] = {"mxfp4_minus_nf4_served": round(m - n, 6), "v1_minus_nf4_served": None if v is None else round(v - n, 6),
                      "_m": m, "_n": n, "_v": v}
    trip = [s for s, x in per.items() if x["_v"] is not None]
    den = sum(per[s]["_m"] - per[s]["_n"] for s in trip)
    share = None if not trip or abs(den) < 1e-9 else sum(per[s]["_m"] - per[s]["_v"] for s in trip) / den
    aw = {"windows_read": sorted(per), "per_window": {s: {k: v for k, v in x.items() if not k.startswith("_")} for s, x in per.items()},
          "K5_every_window": ({"verdict": "UNREAD", "why": f"{len(per)} windows < {K_MIN_WINDOWS}"} if len(per) < K_MIN_WINDOWS else
                              {"verdict": "HOLDS" if all(x["_m"] - x["_n"] >= K_GAP for x in per.values()) else "REFUTED",
                               "min_gap": round(min(x["_m"] - x["_n"] for x in per.values()), 6),
                               "mean_gap": round(sum(x["_m"] - x["_n"] for x in per.values()) / len(per), 6)}),
          "K2_pooled": ({"verdict": "UNREAD", "why": f"{len(trip)} windows < {K_MIN_WINDOWS}"} if len(trip) < K_MIN_WINDOWS or share is None else
                        {"verdict": "HOLDS" if share >= 0.5 else "REFUTED", "pooled_share_carried_by_int8": round(share, 4),
                         "windows": trip})}
    return {"conv1": k("conv1"), "conv2_replication": k("conv2"), "across_windows": aw}


def prove(d: str) -> dict:
    R = _sc1()
    out = {stem: row(d, stem, "conv1", R) for stem in PROVE}
    bad = {k: f"{v['verdict']}: {v['why']}" for k, v in out.items() if v["verdict"] != "VALID"}
    kern, g6 = gemv(d)
    if g6["verdict"] != "HOLDS":
        bad["kernel_check"] = f"G6 {g6['verdict']}: {g6.get('why') or g6.get('disagreeing')}"
    if not any(k.startswith("gemv_capture_conv1") for k in kern):
        bad["kernel_check_capture"] = "no kernel check on captured conv1 activations"
    return {"rows": out, "kernel_check": kern, "G6": g6, "bad": bad, "proved": not bad}


# ------------------------------------------------------------------ self-test --

def _fixture(d, nll, routes, sha="s" * 64, steps=2048, gemv_v=("KERNEL_AGREES", "KERNEL_DISAGREES"), rep=False):
    for s in SRCS + CTRL + (REP if rep else ()):
        json.dump({"ids": [0] * 2561, "text_sha": sha, "source": s, "prompt_len": 512, "steps": 2048,
                   "scored_target_roles": {"assistant": 0.95}}, open(os.path.join(d, f"k8_window_{s}.json"), "w"))
    os.makedirs(os.path.join(d, "routes"), exist_ok=True)
    reps = REP if rep else ()
    stems = [(a, sh, stem.format(s)) for a, shp in ARMS.items() for sh, stem in shp.items() for s in SRCS + CTRL + reps]
    stems += [(a, "diag", stem.format(s)) for a, stem in DIAG.items() for s in SRCS + reps]
    for a, sh, st in stems:
        s = st.rsplit("_", 1)[1]
        v = nll(a, sh, s)
        if v is None:
            continue
        if st.startswith("nll_llamacpp"):
            rec = {"mean_nll": v, "text_sha256": sha, "targets": {"first_index": 513, "last_index": 2560}, "steps": steps}
        else:
            rec = {"mean_nll": v, "text_sha": sha, "prompt_len": 512, "steps": steps}
        json.dump(rec, open(os.path.join(d, st + ".json"), "w"))
        if st.startswith("e4b_"):
            r = routes(st)
            if r is not False:
                json.dump({"pid": 1, "route_seen": r, "attn_seen": None}, open(os.path.join(d, "routes", st + ".1.json"), "w"))
    for name, v, mut in (("gemv_capture_conv1_5090", gemv_v[0], False), ("gemv_mutate_5090", gemv_v[1], True)):
        json.dump({"verdict": v, "kernel_vs_exact_worst": 1e-6 if v == "KERNEL_AGREES" else 0.9, "metric": "kernel_vs_exact_bf16",
                   "device": {"cc": [12, 0]}, "mutate": mut, "summary": {}}, open(os.path.join(d, name + ".json"), "w"))


def _good_routes(st):
    if st.startswith("e4b_nf4_"):
        return {"nf4_singleton|le256": 49152, "nf4_mtile_host|gt256": 24}
    if st.startswith("e4b_mxpre_"):
        return {"mxfp4_grouped_v1|gt256": 400}
    if "_v1_" in st:
        return {"mxfp4_grouped_v1|le256": 2048 * 24 + 16 * 24, "nf4_mtile_host|gt256": 24}
    if any(f"_{x}_" in st for x in ("served", "pdl0", "nofold", "chunk1", "kvg16", "kvg4")):
        return {"mxfp4_gemv|le256": 2048 * 24 + 16 * 24, "nf4_mtile_host|gt256": 24}
    return {"nf4_mtile_host|gt256": 400}


def _base(a, sh, s):
    b = {"conv1": 1.2, "conv2": 1.4, "conv3": 1.3, "conv4": 1.1, "wikitext": 6.3}[s]
    off = {"e4b_serve": 0.004 if sh == "served" else 0.012, "e4b_serve_p64": 0.012 + (0.006 if s == "conv1" else -0.004),
           "e4b_nf4": 0.02, "vllm": 0.0, "sglang_native": 0.003, "sglang_marlin": 0.002,
           "llamacpp": 0.002 if sh == "served" else 0.04, "llamacpp_q8": 0.002 if sh == "served" else 0.005,
           "e4b_serve_pdl0": 0.004, "e4b_serve_nofold": 0.004, "e4b_serve_chunk1": 0.008, "e4b_nf4_chunk1": 0.02,
           "e4b_nf4_full": 0.0, "e4b_nf4_fqkv": 0.004, "e4b_nf4_fqk": 0.003, "e4b_nf4_fqv": 0.001, "e4b_nf4_fqkv16": 0.002,
           "e4b_serve_kvg16": 0.004, "e4b_serve_v1": 0.004, "e4b_mxpre_prefill128": 0.012, "e4b_serve_kvg4": 0.004}[a]
    return b + off


def self_test() -> int:
    import tempfile
    cases = []
    with tempfile.TemporaryDirectory() as d:          # every prediction holds; F = 0.006 (conv1's chunk pair)
        _fixture(d, _base, _good_routes)
        v = reduce(d)
        cases.append(("all hold", all(v["predictions"][g]["verdict"] == "HOLDS" for g in ("G1", "G2", "G3", "G4", "G5", "G6"))
                      and abs(v["floor"]["F"] - 0.006) < 1e-9 and v["diagnostics"]["per_window"]["conv1"]["pdl0_minus_served"] == 0.0))
    with tempfile.TemporaryDirectory() as d:          # an e4b arm killed by alarm: no route record -> VOID, never a pass
        _fixture(d, _base, lambda st: False if st == "e4b_serve_served_conv2" else _good_routes(st))
        v = reduce(d)
        cases.append(("missing record fails", v["rows"]["e4b_serve"]["served"]["conv2"]["verdict"] == "VOID"
                      and v["predictions"]["G1"]["verdict"] == "UNREAD" and v["predictions"]["G5"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:          # KEEP_NF4 not engaged: prefill rows on the MXFP4 store above 256 rows
        _fixture(d, _base, lambda st: {"mxfp4_gemv|gt256": 400} if "prefill" in st and st.startswith("e4b_serve") else _good_routes(st))
        v = reduce(d)
        cases.append(("mxfp4 gt256 fails", v["rows"]["e4b_serve"]["prefill"]["conv1"]["verdict"] == "VOID" and v["floor"]["F"] is None))
    with tempfile.TemporaryDirectory() as d:          # a pre-#1129 build: null record
        _fixture(d, _base, lambda st: None if st == "e4b_nf4_served_conv1" else _good_routes(st))
        cases.append(("null record fails", reduce(d)["rows"]["e4b_nf4"]["served"]["conv1"]["verdict"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:          # q8 did not engage: identical prefill -> G3 UNREAD
        _fixture(d, lambda a, sh, s: _base("llamacpp", sh, s) if a == "llamacpp_q8" else _base(a, sh, s), _good_routes)
        cases.append(("q8 not engaged", reduce(d)["predictions"]["G3"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:          # e4b served 0.03 off vLLM on the graded windows: outside a 0.006 floor
        _fixture(d, lambda a, sh, s: _base(a, sh, s) + (0.026 if (a, sh) == ("e4b_serve", "served") else 0), _good_routes)
        cases.append(("G1 refuted", reduce(d)["predictions"]["G1"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:          # the control is never graded: a large wikitext gap leaves G1 holding
        _fixture(d, lambda a, sh, s: _base(a, sh, s) + (0.5 if (a, sh, s) == ("e4b_serve", "served", "wikitext") else 0), _good_routes)
        v = reduce(d)
        cases.append(("control ungraded", v["predictions"]["G1"]["verdict"] == "HOLDS"
                      and v["control_wikitext"]["delta_vs_vllm"]["e4b_serve"]["served"]["wikitext"] > 0.4))
    with tempfile.TemporaryDirectory() as d:          # a row on other text: sha mismatch VOIDs it
        _fixture(d, _base, _good_routes)
        json.dump({"mean_nll": 2.0, "text_sha": "x" * 64, "prompt_len": 512, "steps": 2048, "scored_index_range": [513, 2560]},
                  open(os.path.join(d, "nll_vllm_served_conv1.json"), "w"))
        cases.append(("sha VOID", reduce(d)["rows"]["vllm"]["served"]["conv1"]["verdict"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:          # the proof's subset alone proves
        _fixture(d, _base, _good_routes)
        for p in glob.glob(os.path.join(d, "*_conv2.json")) + glob.glob(os.path.join(d, "*_wikitext.json")):
            os.remove(p)
        cases.append(("proof subset", prove(d)["proved"]))
    with tempfile.TemporaryDirectory() as d:          # a served record short of the steps x layers GEMV count
        _fixture(d, _base, lambda st: {"mxfp4_gemv|le256": 100, "nf4_mtile_host|gt256": 24} if st == "e4b_serve_served_conv1" else _good_routes(st))
        cases.append(("gemv count", reduce(d)["rows"]["e4b_serve"]["served"]["conv1"]["verdict"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:          # the mutation agreed: the check is inert -> G6 UNREAD, the proof fails
        _fixture(d, _base, _good_routes, gemv_v=("KERNEL_AGREES", "KERNEL_AGREES"))
        cases.append(("inert check", reduce(d)["predictions"]["G6"]["verdict"] == "UNREAD" and not prove(d)["proved"]))
    with tempfile.TemporaryDirectory() as d:          # the kernel disagrees with its exact reference on the 5090
        _fixture(d, _base, _good_routes, gemv_v=("KERNEL_DISAGREES", "KERNEL_DISAGREES"))
        cases.append(("kernel disagrees", reduce(d)["predictions"]["G6"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:          # chunk 1 closes the served gap: share ~1 (the paged path carries it)
        _fixture(d, lambda a, sh, s: _base(a, sh, s) + (0.25 if (a, sh) == ("e4b_serve", "served") else 0), _good_routes)
        share = reduce(d)["diagnostics"]["per_window"]["conv1"]["share_of_served_gap_closed_by_chunk1"]["mxfp4"]
        cases.append(("chunk1 share", share is not None and share > 0.9))
    with tempfile.TemporaryDirectory() as d:          # A2: the fp8 K/V rounding carries the NF4 served gap; finer keys recover it
        def fp8(a, sh, s):
            v = {"e4b_nf4": 0.30 if sh == "served" else 0.02, "e4b_nf4_full": 0.10, "e4b_nf4_chunk1": 0.12,
                 "e4b_nf4_fqkv": 0.28, "e4b_nf4_fqk": 0.25, "e4b_nf4_fqv": 0.12, "e4b_nf4_fqkv16": 0.14,
                 "e4b_serve": 0.50 if sh == "served" else 0.02, "e4b_serve_kvg16": 0.35, "e4b_serve_chunk1": 0.30,
                 "e4b_serve_pdl0": 0.501, "e4b_serve_nofold": 0.499}.get(a)
            return _base(a, sh, s) if v is None else 1.0 + v
        _fixture(d, fp8, _good_routes)
        jp = reduce(d)["diag_predictions"]["conv1"]
        cases.append(("J predictions", all(jp[k]["verdict"] == "HOLDS" for k in ("J1", "J2", "J3", "J4", "J5", "J6"))
                      and abs(jp["J1"]["share"] - 0.9) < 1e-9))
    with tempfile.TemporaryDirectory() as d:          # A2: nothing closes the gap -> J1/J2 REFUTED; a missing arm -> UNREAD
        def flat(a, sh, s):
            v = {"e4b_nf4": 0.30 if sh == "served" else 0.02, "e4b_nf4_full": 0.10, "e4b_nf4_chunk1": 0.29, "e4b_nf4_fqkv": 0.11}.get(a)
            return None if a == "e4b_nf4_fqk" else (_base(a, sh, s) if v is None else 1.0 + v)
        _fixture(d, flat, _good_routes)
        jp = reduce(d)["diag_predictions"]["conv1"]
        cases.append(("J refuted", jp["J1"]["verdict"] == "REFUTED" and jp["J2"]["verdict"] == "REFUTED" and jp["J3"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:          # the bands: folds-off +0.02 is inside ~2x floor (HOLDS); a 0.03 gain is not J4
        def bands(a, sh, s):
            v = {"e4b_serve": 0.50 if sh == "served" else 0.02, "e4b_serve_pdl0": 0.501, "e4b_serve_nofold": 0.52,
                 "e4b_serve_kvg16": 0.47, "e4b_nf4_fqkv": 0.28, "e4b_nf4_fqkv16": 0.25}.get(a)
            return _base(a, sh, s) if v is None else 1.0 + v
        _fixture(d, bands, _good_routes)
        jp = reduce(d)["diag_predictions"]["conv1"]
        cases.append(("J bands", jp["J6"]["verdict"] == "HOLDS" and jp["J4"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:          # A3: the int8 activations carry the MXFP4 cost; the weights cost nothing
        tab = {("e4b_serve", "served", "conv1"): BOXJ_SERVED_CONV1, ("e4b_serve_kvg4", "diag", "conv1"): BOXJ_SERVED_CONV1,
               ("e4b_nf4", "served", "conv1"): 0.736, ("e4b_serve_v1", "diag", "conv1"): 0.745,
               ("e4b_nf4", "prefill", "conv1"): 0.731, ("e4b_mxpre_prefill128", "diag", "conv1"): 0.735,
               ("e4b_serve_pdl0", "diag", "conv1"): BOXJ_SERVED_CONV1, ("e4b_serve_nofold", "diag", "conv1"): BOXJ_SERVED_CONV1 + 0.02,
               ("e4b_serve", "served", "conv2"): 1.85, ("e4b_nf4", "served", "conv2"): 1.65,
               ("e4b_serve_v1", "diag", "conv2"): 1.66}
        for s, (m, n, v) in {"conv3": (1.40, 1.25, 1.26), "conv4": (1.10, 0.95, 0.97)}.items():
            tab.update({("e4b_serve", "served", s): m, ("e4b_nf4", "served", s): n, ("e4b_serve_v1", "diag", s): v})

        def int8(a, sh, s):
            return tab.get((a, sh, s), tab.get((a, "diag", s), _base(a, sh, s)))
        _fixture(d, int8, _good_routes, rep=True)
        kp = reduce(d)["a3_predictions"]
        c1, aw = kp["conv1"], kp["across_windows"]
        cases.append(("A3 int8", all(c1[k]["verdict"] == "HOLDS" for k in ("K1", "K2", "K3", "K4", "K5"))
                      and c1["K2"]["share_carried_by_int8"] > 0.9 and kp["conv2_replication"]["K5"]["verdict"] == "HOLDS"
                      and aw["windows_read"] == ["conv1", "conv2", "conv3", "conv4"] and aw["K5_every_window"]["verdict"] == "HOLDS"
                      and aw["K2_pooled"]["verdict"] == "HOLDS"))
        # one replication window where MXFP4 costs nothing: K5 across windows REFUTED although conv1's K5 HOLDS
        tab[("e4b_serve", "served", "conv4")] = 0.96
        _fixture(d, int8, _good_routes, rep=True)
        kp = reduce(d)["a3_predictions"]
        cases.append(("A3 replication", kp["conv1"]["K5"]["verdict"] == "HOLDS"
                      and kp["across_windows"]["K5_every_window"]["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:          # A3: only two windows read -> the across-window reads are UNREAD, not HOLDS
        _fixture(d, _base, _good_routes)
        aw = reduce(d)["a3_predictions"]["across_windows"]
        cases.append(("A3 too few windows", aw["K5_every_window"]["verdict"] == "UNREAD" and aw["K2_pooled"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:          # A3 gates: a KEEP_NF4=0 prefill that still ran NF4, or GEMV=0 that ran the GEMV
        _fixture(d, _base, lambda st: {"nf4_mtile_host|gt256": 400} if st.startswith("e4b_mxpre_")
                 else ({"mxfp4_gemv|le256": 49152} if "_v1_" in st else _good_routes(st)))
        v = reduce(d)["diagnostic_rows"]
        cases.append(("A3 gates", v["e4b_mxpre_prefill128"]["conv1"]["verdict"] == "VOID" and v["e4b_serve_v1"]["conv1"]["verdict"] == "VOID"))
    cases += _a4_self_test(tempfile)
    cases += _a5_self_test(tempfile)
    bad = [n for n, ok in cases if not ok]
    print(f"sc1g_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def _a4_fixture(d, sigma, r_ok=True, nf4_scale=0.5, misalign=None, V=200, P=2048):
    """Box R's staged receipt + every KL arm's named record: engine logits = the reference's + N(0, sigma[label]) noise."""
    import numpy as np
    import torch
    K = _kl()
    _fixture(d, _base, _good_routes, rep=True)
    rd = os.path.join(d, "ref")
    os.makedirs(rd, exist_ok=True)
    g = torch.Generator().manual_seed(7)
    shas, calib = {}, {"windows": {}}
    for s in A4_SRCS + A4_CTRL:
        ref_logits = torch.randn(P, V, generator=g) * 5.0      # peaked enough that the top-64 mass clears 0.99
        rows = K.reference_rows(ref_logits, torch.randint(0, V, (P,), generator=g).numpy())
        shas[s] = K.save_artifact(os.path.join(rd, f"ref_{s}.npz"), rows, {"source": s})
        calib["windows"][s] = {"floor_F": 1e-4, "calib_nf4": {"kl_full_mean": nf4_scale}, "reference_nll_decode": 1.0}
        for lab, stem in KL_ARMS.items():
            st = stem.format(s)
            eng = torch.log_softmax((ref_logits + sigma[lab] * torch.randn(P, V, generator=g)).double(), -1)
            lp = eng.gather(1, torch.as_tensor(rows["ids"]).long()).numpy()
            rec = _load(d, st)
            tlp = np.full(P, -rec["mean_nll"] + (1e-3 if (lab, s) == misalign else 0.0))
            np.savez(os.path.join(d, f"named_{st}.npz"), eng_lp=lp, eng_target_lp=tlp)
            if lab.startswith("e4b"):
                json.dump({"calls": P, "positions": P}, open(os.path.join(d, f"named_{st}.npz.json"), "w"))
    json.dump(shas, open(os.path.join(rd, "ref_shas.json"), "w"))
    json.dump({"verdict": "R_OK" if r_ok else "R_NOT_OK", "checks": {}}, open(os.path.join(rd, "r_verdict.json"), "w"))
    json.dump(calib, open(os.path.join(rd, "r_calib.json"), "w"))


def _a4_self_test(tempfile) -> list:
    cases = []
    good = {"e4b_mxfp4": 0.05, "e4b_nf4": 0.2, "vllm": 0.04, "sglang_native": 0.045, "sglang_marlin": 0.045,
            "llamacpp": 0.06, "llamacpp_q8": 0.05}
    with tempfile.TemporaryDirectory() as d:
        _a4_fixture(d, good)
        r = a4(d)
        pr = r["predictions"]
        cases.append(("A4 reads", r["instrument"]["box_r"] == "R_OK" and pr["K-A"]["verdict"] == "HOLDS"
                      and pr["L1"]["verdict"] == "HOLDS" and pr["L1"]["best_comparator"] == "vllm" and pr["L2"]["verdict"] == "HOLDS"
                      and r["descriptive"]["pooled_rank"][0][0] == "vllm"))
        cases.append(("A4 proof", prove_a4(d)["proved"]))
    with tempfile.TemporaryDirectory() as d:          # box R not OK -> the whole reading UNREAD, whatever the rows say
        _a4_fixture(d, good, r_ok=False)
        cases.append(("A4 needs R_OK", all(x["verdict"] == "UNREAD" for x in a4(d)["predictions"].values())))
    with tempfile.TemporaryDirectory() as d:          # a named record whose target log-probs miss the arm's NLL -> VOID
        _a4_fixture(d, good, misalign=("vllm", "conv2"))
        cases.append(("A4 alignment", a4(d)["rows"]["vllm"]["conv2"]["verdict"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:          # an e4b named record without the proxy's meta (a broken proxy) -> VOID
        _a4_fixture(d, good)
        os.remove(os.path.join(d, "named_e4b_nf4_served_conv1.npz.json"))
        cases.append(("A4 e4b meta required", a4(d)["rows"]["e4b_nf4"]["conv1"]["verdict"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:          # a tampered reference artifact -> refused, everything UNREAD
        _a4_fixture(d, good)
        with open(os.path.join(d, "ref", "ref_conv3.npz"), "ab") as f:
            f.write(b"x")
        r = a4(d)
        cases.append(("A4 sha refusal", r["instrument"]["box_r"] == "NOT_OK" and r["predictions"]["L1"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:          # NF4 barely worse than MXFP4 -> K-A REFUTED; e4b far from vLLM -> L1 REFUTED;
        bad = dict(good, e4b_nf4=0.06, e4b_mxfp4=0.09)  # an NF4 scale under the engines -> L2 REFUTED
        _a4_fixture(d, bad, nf4_scale=1e-5)
        pr = a4(d)["predictions"]
        cases.append(("A4 refutations", pr["K-A"]["verdict"] == "REFUTED" and pr["L1"]["verdict"] == "REFUTED"
                      and pr["L2"]["verdict"] == "REFUTED" and "e4b_mxfp4" in pr["L2"]["at_or_above"]))
    return cases


def _a5_fixture(d, kl, F=1e-3, nf4=0.05, verdict="R_OK", rule="A5", vllm_void=False, drop_e4b_meta=False, P=2048, mask=None,
                void=None, n_masked=0.0, legacy=False):
    """Box R's A5 receipt + every full-KL arm's record: eng_kl = kl[label] (a constant per window), aligned target lps, and
    the support arrays. mask {(label, src): (mass, positions)}: the engine masks reference mass there (kl = inf); void
    {(label, src): positions}: those positions' reads failed (kl NaN, a void_first reason); legacy: no support arrays."""
    import numpy as np
    _fixture(d, _base, _good_routes, rep=True)
    rd = os.path.join(d, "ref")
    os.makedirs(rd, exist_ok=True)
    srcs = A4_SRCS + A4_CTRL
    json.dump({s: f"{i:064x}" for i, s in enumerate(srcs)}, open(os.path.join(rd, "ref_full_shas.json"), "w"))
    json.dump({"rule": rule, "verdict": verdict, "checks": {}}, open(os.path.join(rd, "r_verdict.json"), "w"))
    json.dump({"rule": "A5", "windows": {s: {"floor_F": F, "calib_nf4": {"kl_full_mean": nf4}} for s in srcs}},
              open(os.path.join(rd, "r_calib.json"), "w"))
    for lab, stem in KL5_ARMS.items():
        for i, s in enumerate(srcs):
            st = stem.format(s)
            rec = _load(d, st)
            k_, mm = np.full(P, kl[lab]), np.zeros(P)
            if mask and (lab, s) in mask:
                m, n = mask[(lab, s)]
                mm[:n], k_[:n] = m, np.inf
            nv = (void or {}).get((lab, s), 0)
            k_[P - nv:] = np.nan
            arrs = dict(eng_kl=k_, eng_target_lp=np.full(P, -rec["mean_nll"]))
            if not legacy:
                arrs.update(eng_kl_common=np.full(P, kl[lab]), eng_masked_mass=mm, eng_n_masked=np.full(P, n_masked))
            np.savez(os.path.join(d, f"kl_{st}.npz"), **arrs)
            vf = [{"t": P - nv, "why": "FloatingPointError: NaN or +inf in the engine's row"}] if nv else []
            if lab.startswith("e4b") and not (drop_e4b_meta and lab == "e4b_nf4" and s == "conv1"):
                json.dump({"calls": P, "positions": P, "ref_sha": f"{i:064x}", "void_positions": nv, "void_first": vf},
                          open(os.path.join(d, f"kl_{st}.npz.json"), "w"))
            if lab == "vllm" and vllm_void and s == "conv2":
                rec["kl_full"] = {"verdict": "VOID", "why": "logprobs=-1 returned 20 entries, not the vocabulary"}
                json.dump(rec, open(os.path.join(d, st + ".json"), "w"))


def _a5_self_test(tempfile) -> list:
    cases = []
    good = {"e4b_mxfp4": 2e-3, "e4b_nf4": 2e-2, "vllm": 1.5e-3, "llamacpp": 3e-3, "llamacpp_q8": 2.5e-3}
    with tempfile.TemporaryDirectory() as d:
        _a5_fixture(d, good)
        r = a5(d)
        pr = r["predictions"]
        cases.append(("A5 reads", r["gradable_windows"] == list(A4_SRCS) and pr["K-A"]["verdict"] == "HOLDS"
                      and pr["L1"]["verdict"] == "HOLDS" and pr["L1"]["best_comparator"] == "vllm" and pr["L2"]["verdict"] == "HOLDS"
                      and r["rows"]["sglang_native"]["conv1"]["verdict"] == "UNREAD"))
        cases.append(("A5 proof", prove_a5(d)["proved"]))
    with tempfile.TemporaryDirectory() as d:          # MXFP4 within F: F stands in as its upper bound, NF4 must clear 3F
        _a5_fixture(d, dict(good, e4b_mxfp4=2e-3, e4b_nf4=2e-2), F=3e-3)
        ka = a5(d)["predictions"]["K-A"]
        _a5_fixture(d, dict(good, e4b_mxfp4=2e-3, e4b_nf4=8e-3), F=3e-3)
        ka2 = a5(d)["predictions"]["K-A"]
        cases.append(("A5 K-A bound", ka["verdict"] == "HOLDS" and all(x["bound_used"] for x in ka["windows"].values())
                      and ka2["verdict"] == "REFUTED"))
    with tempfile.TemporaryDirectory() as d:          # the best comparator within F on every window: L1 has nothing to read
        _a5_fixture(d, dict(good, vllm=2e-3), F=3e-3)
        cases.append(("A5 L1 drops unresolved comparators", a5(d)["predictions"]["L1"]["verdict"] == "UNREAD"))
    with tempfile.TemporaryDirectory() as d:          # box R not OK, or not an A5 receipt: the whole reading is UNREAD
        _a5_fixture(d, good, verdict="R_NO_GRADABLE")
        un1 = all(x["verdict"] == "UNREAD" for x in a5(d)["predictions"].values())
        _a5_fixture(d, good, rule="A4")
        un2 = all(x["verdict"] == "UNREAD" for x in a5(d)["predictions"].values())
        cases.append(("A5 needs R_OK under A5", un1 and un2))
    with tempfile.TemporaryDirectory() as d:          # vLLM's full vocabulary did not verify -> VOID; e4b without its meta -> VOID
        _a5_fixture(d, good, vllm_void=True, drop_e4b_meta=True)
        rows = a5(d)["rows"]
        cases.append(("A5 VOID rows", rows["vllm"]["conv2"]["verdict"] == "VOID" and rows["e4b_nf4"]["conv1"]["verdict"] == "VOID"))
    with tempfile.TemporaryDirectory() as d:          # vLLM masks reference mass on conv1 (1e-12 at 3 positions): VOID, mass reported
        _a5_fixture(d, good, mask={("vllm", "conv1"): (1e-12, 3)})
        row, pv = a5(d)["rows"]["vllm"]["conv1"], prove_a5(d)
        cases.append(("A5 masked reference mass -> VOID with the mass; the proof names the support rule needed",
                      row["verdict"] == "VOID" and row["support"]["positions_masked"] == 3 and row["support"]["masked_mass_max"] == 1e-12
                      and not pv["proved"] and pv["support_rule_needed"] == ["nll_vllm_served_conv1"]))
    with tempfile.TemporaryDirectory() as d:          # two of e4b's positions failed their read: VOID with the reason, run kept
        _a5_fixture(d, good, void={("e4b_mxfp4", "conv1"): 2})
        row = a5(d)["rows"]["e4b_mxfp4"]["conv1"]
        cases.append(("A5 void positions -> VOID with the first reason", row["verdict"] == "VOID" and row["why"].startswith("2 void positions")
                      and "NaN or +inf" in row["why"]))
    with tempfile.TemporaryDirectory() as d:          # masking only ids the reference also gives no mass: VALID, the count reported
        _a5_fixture(d, good, n_masked=5.0)
        row, pv = a5(d)["rows"]["llamacpp_q8"]["conv1"], prove_a5(d)
        cases.append(("A5 masking only reference -inf ids is VALID; the count is reported", row["verdict"] == "VALID"
                      and row["support"]["n_masked_max"] == 5.0 and pv["proved"] and pv["support_rule_needed"] == []))
    with tempfile.TemporaryDirectory() as d:          # a record without the support arrays is not read
        _a5_fixture(d, good, legacy=True)
        cases.append(("A5 a record without support arrays -> VOID", a5(d)["rows"]["vllm"]["conv1"]["verdict"] == "VOID"))
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--prove", action="store_true")
    ap.add_argument("--prove-a4", action="store_true", help="A4's proof: one named KL row per engine path on conv1")
    ap.add_argument("--prove-a5", action="store_true", help="A5's proof: one full-KL row per KL-capable engine path on conv1")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.dir:
        ap.error("--dir is required")
    if a.prove_a5:
        v = prove_a5(a.dir)
        if a.out:
            json.dump(v, open(a.out, "w"), indent=1, sort_keys=True, default=float)
        for k, why in v["bad"].items():
            print(f"SC1G_PROVE_A5_BAD {k}: {why}")
        for k, s in v["support"].items():
            print(f"SC1G_PROVE_A5_SUPPORT {k} " + (json.dumps(s, default=float) if s else "unread"))
        if v["support_rule_needed"]:
            print(f"SC1G_PROVE_A5_SUPPORT_RULE_NEEDED {' '.join(v['support_rule_needed'])}: these engines mask tokens carrying "
                  f"reference mass -- register a common-support rule with a mass bound on this proof's numbers before the reading box")
        print(f"SC1G_PROVE_A5 {'OK' if v['proved'] else 'FAILED'} (box R {v['box_r']}; "
              + " ".join(f"{k}={x.get('kl_mean')}" for k, x in v["rows"].items()) + ")")
        return 0 if v["proved"] else 1
    if a.prove_a4:
        v = prove_a4(a.dir)
        if a.out:
            json.dump(v, open(a.out, "w"), indent=1, sort_keys=True, default=float)
        for k, why in v["bad"].items():
            print(f"SC1G_PROVE_A4_BAD {k}: {why}")
        print(f"SC1G_PROVE_A4 {'OK' if v['proved'] else 'FAILED'} (box R {v['box_r']}; "
              + " ".join(f"{k}={x.get('kl65_mean')}" for k, x in v["rows"].items()) + ")")
        return 0 if v["proved"] else 1
    v = prove(a.dir) if a.prove else reduce(a.dir)
    if a.out:
        json.dump(v, open(a.out, "w"), indent=1, sort_keys=True, default=float)
    if a.prove:
        for k, why in v["bad"].items():
            print(f"SC1G_PROVE_BAD {k}: {why}")
        print(f"SC1G_PROVE {'OK' if v['proved'] else 'FAILED'} ({len(v['rows'])} rows; G6 {v['G6']['verdict']})")
        return 0 if v["proved"] else 1
    for g, x in v["predictions"].items():
        print(f"SC1G_{g} {x['verdict']} {json.dumps({k: w for k, w in x.items() if k not in ('verdict', 'not_valid')})[:300]}")
    print(f"SC1G_FLOOR {json.dumps(v['floor'])}")
    print(f"SC1G_DIAG {json.dumps(v['diagnostics']['per_window'])[:600]}")
    for w, ks in v["a3_predictions"].items():           # A3's K1-K5 per window, and the across-window reads
        print(f"SC1G_A3 {w} {json.dumps(ks)[:900]}")
    a5r = v.get("a5") or {}
    print(f"SC1G_A5_INSTRUMENT {json.dumps(a5r.get('instrument'))} gradable={a5r.get('gradable_windows')}")
    for g, x in (a5r.get("predictions") or {}).items():
        print(f"SC1G_A5_{g} {x['verdict']} {json.dumps({k: w for k, w in x.items() if k != 'verdict'}, default=float)[:400]}")
    a4r = v.get("a4") or {}
    print(f"SC1G_A4_INSTRUMENT {json.dumps(a4r.get('instrument'))}")
    for g, x in (a4r.get("predictions") or {}).items():
        print(f"SC1G_A4_{g} {x['verdict']} {json.dumps({k: w for k, w in x.items() if k != 'verdict'}, default=float)[:400]}")
    print(f"SC1G_A4_RANK {json.dumps((a4r.get('descriptive') or {}).get('pooled_rank'), default=float)[:400]}")
    if v.get("attn_check_1175"):
        ac = v["attn_check_1175"]
        print(f"SC1G_ATTN {ac.get('verdict')} {json.dumps(ac.get('per_k_groups'))[:600]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
