#!/usr/bin/env python3
"""sc1_reduce.py -- the reducer of lane SC1 (bench/sc1/SC1-PREREG.md; experts4bit-qlora#846): one box's fetched receipts
-> the per-arm speed table, the draws, the licence block, the comparability block, the positions, the controls, TTFT /
resources / energy, the predictions P1-P14, the frontier table, and `--cross-box` for the anchor-ratio comparison.
It licenses nothing beyond the K8 gate it imports, moves no default, and quotes no number a VOID arm touched. stdlib only
(it runs on the box's image python and on the Mac). Shape: bench/p58/p58_reduce.py (e4b_row / vllm_row / pair / valid /
batch_section, VOID reasons as pre-registered engagement rules) extended; vocabulary and `--selftest` discipline from
bench/tc1/tc1_reduce.py; the K8 rows as bench/p88/p88_reduce.py read them.

RECEIPT CONVENTIONS ASSUMED (no orchestration report had landed when this was written; these are the names the lane's
brief gave, and the drivers' own receipt shapes as read from bench/sc1/{vllm,sglang,llamacpp}/ on campaign/sc and
bench/sc1/{exl3,lmdeploy}/ on sc1/exl3-lmdeploy -- if the orchestration lands with other names, change `FILES` below):
  <run>/sc1/ (or <run>/ when it holds the receipts directly)
    prompts_b1.json, prompts_b16.json, prompts_b16_same.json   {prompts, prompts_sha256, rows_sha256, batch}
    k8_window_<src>.json                                        {ids, text_sha, source}         src in wikitext, c4val1
    e4b_<arm>_b<B>_r<n>.json + logs/run_e4b_<arm>_b<B>_r<n>.log  step_decomp --b1d-timed / BV3 JSON: step_ms_clean,
        n_steps, tokens, recompiles_in_window, aggregate_tok_s (B=16); arms lic | rtn | nf4_ctrl | int4 | lic_degraded
    e4bsched_<arm>_b<B>_r<n>.json (+ logs/run_e4bsched_...)     p37-shaped, engine "e4b-sched": decode_tok_s,
        decode_ms_per_step, prompts_sha256, prompt_tokens, tokens, census {fuse_qkv_n, fuse_t1_glue_n,
        fuse_t1_glue_r2_n, fuse_router_epilogue_n, int4_expert_layers, int4_attn_projections (AFTER q/k/v fusion: 192 - 2 x
        fuse_qkv_n), attn_int4_calib_projections | attn_int4_rtn_projections (BEFORE fusion: 192, the arm's lever), graph_status,
        levers_env, fingerprint?}, graph_stats; arms lic_sched | int4_sched | lic_sched_sameprompt
        (another agent writes that driver concurrently: this file codes to the shape named here)
    <engine>_<arm>_b<B>_r<n>.json (+ logs/run_<engine>_<arm>_b<B>_r<n>.log)   engines vllm | sglang | llamacpp |
        exl3 | lmdeploy, p37-shaped (decode_tok_s, decode_ms_per_step, prompts_sha256, prompt_tokens, tokens, status
        or verdict, engagement, resolved / server_args / props)
    k8_<name>.json          step_decomp K8 rows: mean_nll, ppl, text_sha, steps, tokens_scored, ppl_source,
        attn_compute (+ census or logs/census_k8_<name>.txt for the =1 rows); names lic_auto_<src>, lic_1_<src>,
        nf4_<src> (box A); int4_<src> / rtn_<src> (boxes B/C, optional)
    oracle_<src>.json        step_decomp --ppl-oracle upstream: mean_nll, ppl, text_sha, steps, ppl_source
    nll_<engine>[_<variant>]_<mode>_<src>.json   the quality scorers (mode prefill | served | decode | decode_prefix |
        served_tail2; variant fp8kv | awq | iq4xs for the arms whose checkpoint or KV differs); e4b's prefill shape is
        nll_e4b_prefill_<src>.json (--ppl-oracle eager on LICENV); e4b's served shape is its K8 `auto` row
    ttft_<engine>_<len>.json  len 512 | 4096 (engine e4b = the sched engine)
    energy_<tag>.csv + energy_<tag>.json   nvidia-smi --query-gpu=power.draw.instant,... --format=csv -lms 50 and
        {start_epoch, stop_epoch, tokens, engine?, batch?, interval_ms?, sampler_start_epoch?}
    summary.txt (a `PACK <fp>` line), forensics.txt, versions.txt, box.json ({"SC1_BOX": "A"}, host facts)
  Output: RESULTS-sc1-<box>-generated.md + verdict.json beside the receipts (or --out-dir).

EVERY RULE'S THRESHOLD, AS IMPLEMENTED (the registration's numbers; nothing else):
  draws         spread = (max - min) / min over the arm's non-VOID draws. 2 draws, spread <= 0.03: STABLE. 2 draws,
                0.03 < spread <= 0.05: THIRD_DRAW_REQUIRED (no position until the third lands). 3 draws: point = median,
                spread over 3 <= 0.05 STABLE else UNSTABLE. spread > 0.05 at any count: UNSTABLE (flag, no position).
                1 draw: SINGLE (stability unmeasured; no position; controls and labelled rows may still be read).
  position      median(comparator tok/s) / median(anchor tok/s); interval = (min, max) over every cross-draw ratio.
                Quoted iff both arms VALID on every draw, neither UNSTABLE / THIRD_DRAW_REQUIRED / SINGLE, and the
                comparator's served-shape quality is COMPARABLE vs the anchor on BOTH texts. Anchor: e4bsched/lic_sched
                (box A) or e4bsched/int4_sched (boxes B, C). Window ratio beside it = comparator / e4b window arm.
  licence       k8_gate.verdict(pairs, calibrated=True, budget=0.05, calibration_domain="c4val1"); pairs = (nf4_<src>,
                lic_auto_<src>) -> LICENSED-B1, (nf4_<src>, lic_1_<src>) -> LICENSED-B16, both texts; a row is VOID when
                text_sha != k8_window_<src>.json's or steps != 2048; the =1 rows need K19 `_gemm_int4_b32_grouped_smallm`
                AND K23 `_tile_table_r1` in their census. Either gate failing -> QUALITY_FAIL, and no position is quoted
                from the RTN rows either.
  comparability |delta_pair| <= 0.0095 CLOSE; <= 0.02 COMPARABLE; else NOT_COMPARABLE (inclusive bounds, nats per
                text, per shape). delta_bf16 = mean_nll - oracle_<src>.mean_nll (UNREAD when the oracle is missing or
                its sha differs). A quality row is VOID when its text_sha != the window file's or its scored range !=
                [513, 2560] (scored_index_range, targets{first,last}, or prompt_len+1..prompt_len+steps) or steps != 2048
                or the scorer itself said VOID / void_rows > 0. OOD-flattery flag: delta_bf16 < -0.02.
  substitution  |median(e4b/lic_b<B>) / median(e4b/rtn_b<B>) - 1| <= 0.02 at B=1 AND B=16 on box A (window arms).
  SAMEPROMPT    ratio = median(sameprompt tok/s) / median(distinct tok/s) at B=16; >= 1.3 "fired" (P7), > 1.0 "weak",
                <= 1.0 "did not fire" -- a LABEL, never a VOID.
  degraded      median(lic_b16 tok/s) / median(lic_degraded_b16 tok/s) >= 1.05 AND K19 absent from the degraded census
                (and present in lic_b16's): PASS; else FAIL -> e4b's B=16 rows (window and sched) VOID.
  method pair   window tok/s / sched tok/s - 1 per B on the same stack (expected 0.03..0.15 at B=1; reported).
  cross-box     anchor ratios (vllm/gptq_graph over the e4b sched anchor) per B: max/min - 1 <= 0.05 agrees (P13).
  energy        J/token = trapezoid(power.draw.instant [W], dt) over samples inside [start_epoch, stop_epoch] / tokens.
  predictions   P1 B=1 vllm/lic_sched in [0.90, 1.15]; P2 B=16 in [1.00, 1.30]; P3 sglang within 10 % (B=1) / 15 % (B=16)
                of the box's vllm anchor, or UNSUPPORTED; P4 llamacpp q4km B=1 [0.8, 1.15], B=16 [0.35, 0.7]; P5 exl3
                B=1 [0.9, 1.4], B=16 [0.5, 1.1]; P5b lmdeploy B=1 [0.9, 1.3], B=16 [0.9, 1.4] or UNSUPPORTED; P6
                |served - prefill| > 0.005 on e4b-lic and < 0.002 on vllm (both texts); P7 SAMEPROMPT >= 1.3 on vllm and
                e4b-sched; P8 fp8kv/auto within 5 % at B=1, > 1 at B=16, delta_bf16 gap < 0.01 (served, both texts);
                P9 LICENSED-B1 and LICENSED-B16; P10 TTFT-4096 vllm/e4b in [0.5, 1.0]; P11 J/token B=1 within 15 %,
                B=16 the faster engine has the lower J/token; P12 the substitution licence; P13 cross-box within 5 %;
                P14 nodetok/matched within 3 % (B=1) and 1 % (B=16). HOLD / REFUTED / UNREAD, mechanically.

Usage: sc1_reduce.py <run_dir> [--box A|B|C] [--out-dir D] [--box-a <boxA run_dir>]
       sc1_reduce.py --cross-box <run_dir> <run_dir> [<run_dir>]
       sc1_reduce.py --selftest
"""
from __future__ import annotations

import argparse
import calendar
import csv
import glob
import importlib.util
import io
import json
import math
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass

# ----------------------------------------------------------------------------------------------------- registration
PREREG = "bench/sc1/SC1-PREREG.md"
TEXTS = ("wikitext", "c4val1")
CALIB_DOMAIN = "c4val1"
BATCHES = (1, 16)
P, S = 512, 2048
SCORED_RANGE = [P + 1, P + S]                 # [513, 2560]
LONG = 128                                    # generated tokens per row on every timed arm
NEED_STEPS = {1: 127, 16: 70}                 # the e4b window arms' step counts (P58)
PROMPT_TOKENS = 512
CLOSE, COMPARABLE = 0.0095, 0.02
K8_BUDGET = 0.05
DRAW_THIRD, DRAW_UNSTABLE = 0.03, 0.05
SUBSTITUTION = 0.02
SAMEPROMPT_FIRED = 1.3
DEGRADED_MIN = 1.05
CROSS_BOX = 0.05
OOD_FLATTERY = 0.02
K19_KERNEL = "_gemm_int4_b32_grouped_smallm"
K23_KERNEL = "_tile_table_r1"
ANCHOR_SCHED = {"A": "lic_sched", "B": "int4_sched", "C": "int4_sched"}
ANCHOR_WINDOW = {"A": "lic", "B": "int4", "C": "int4"}
COMPARATOR_ENGINES = ("vllm", "sglang", "llamacpp", "exl3", "lmdeploy")
MATCHED_ARMS = {"vllm": "gptq_graph", "sglang": "gptq_matched", "llamacpp": "q4km", "exl3": "4bpw", "lmdeploy": "w4a16"}
NATIVE_ARMS = {"sglang": {"gptq_native"}, "exl3": {"4bpw_cu13"}, "vllm": set(), "llamacpp": set(), "lmdeploy": set()}
QUALITY_VARIANT = {("vllm", "gptq_fp8kv"): "fp8kv", ("vllm", "awq_graph"): "awq", ("llamacpp", "iq4xs"): "iq4xs"}
SERVED_MODES = ("served", "decode", "decode_prefix")
VERDICTS = ("VALID", "VOID", "OOM", "UNSUPPORTED", "UNSTABLE", "HARNESS_ERROR", "ALARM", "NOT_RUN")
STATUS_MAP = {"ok": "VALID", "void": "VOID", "oom": "OOM", "refused": "UNSUPPORTED", "unsupported": "UNSUPPORTED",
              "install_failed": "UNSUPPORTED", "import_failed": "UNSUPPORTED", "load_fault": "UNSUPPORTED",
              "harness_error": "HARNESS_ERROR", "alarm": "ALARM", "phase_alarm": "ALARM", "not_run": "NOT_RUN",
              "host-limited": "NOT_RUN", "unreadable": "HARNESS_ERROR"}
# bits per weight: computed from the receipt (expert_store_bytes / expert_params) when it carries them; else the
# registration's nominal figure, tagged "(nominal)"
BPW_NOMINAL = {("e4b", "lic"): 4.50, ("e4b", "rtn"): 4.50, ("e4b", "int4"): 4.50, ("e4b", "lic_sched"): 4.50,
               ("e4b", "int4_sched"): 4.50, ("e4b", "nf4_ctrl"): 4.50, ("e4b", "lic_degraded"): 4.50,
               ("vllm", "gptq_graph"): 4.15, ("vllm", "gptq_fp8kv"): 4.15, ("vllm", "awq_graph"): 4.15,
               ("vllm", "gptq_graph_nodetok"): 4.15, ("sglang", "gptq_matched"): 4.15, ("sglang", "gptq_native"): 4.15,
               ("llamacpp", "q4km"): 4.86, ("llamacpp", "iq4xs"): 4.29, ("exl3", "4bpw"): 4.0, ("exl3", "4bpw_cu13"): 4.0,
               ("lmdeploy", "w4a16"): 4.15}
PRED_BANDS = {"P1": (0.90, 1.15), "P2": (1.00, 1.30), "P4_b1": (0.8, 1.15), "P4_b16": (0.35, 0.7), "P5_b1": (0.9, 1.4),
              "P5_b16": (0.5, 1.1), "P5b_b1": (0.9, 1.3), "P5b_b16": (0.9, 1.4), "P10": (0.5, 1.0)}
P3_TOL = {1: 0.10, 16: 0.15}
P6_E4B_MIN, P6_VLLM_MAX = 0.005, 0.002
P8_B1_TOL, P8_DBF16_MAX = 0.05, 0.01
P11_B1_TOL = 0.15
P14_TOL = {1: 0.03, 16: 0.01}


# ------------------------------------------------------------------------------------------------------- io helpers
def jload(path):
    """A receipt, or {"status": "unreadable", ...} (never None for a file that exists); None when the file is absent."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"status": "unreadable", "reason": f"{type(e).__name__}: {e}"[:160]}


def grep(path, pat):
    try:
        with open(path, errors="replace") as f:
            return [line.rstrip() for line in f if re.search(pat, line)]
    except OSError:
        return []


def read_text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x, nd=1):
    return "—" if x is None else f"{100 * x:+.{nd}f} %"


def ratio(num, den):
    return (num / den) if (num is not None and den not in (None, 0)) else None


def run_dir_of(d):
    """<run>/sc1 when it exists (the orchestration's layout), else <run> itself."""
    sub = os.path.join(d, "sc1")
    return sub if os.path.isdir(sub) else d


def box_of(d, explicit=None):
    if explicit:
        return explicit.upper()
    b = jload(os.path.join(d, "box.json")) or {}
    for k in ("SC1_BOX", "box", "sc1_box"):
        if isinstance(b, dict) and b.get(k):
            return str(b[k]).upper()
    m = re.search(r"sc1([abc])-", os.path.basename(os.path.abspath(d)) + os.path.basename(os.path.dirname(os.path.abspath(d))))
    return m.group(1).upper() if m else "A"


def pack_fingerprint(d):
    """The Phase-0 pack fingerprint: the `PACK <fp>` line of summary.txt (the first one), else None."""
    for line in grep(os.path.join(d, "summary.txt"), r"^PACK\s+\S+"):
        return line.split()[1]
    return None


def window_sha(d, src):
    w = jload(os.path.join(d, f"k8_window_{src}.json"))
    return (w or {}).get("text_sha") if isinstance(w, dict) else None


def prompts_file(d, B, same=False):
    return jload(os.path.join(d, "prompts_b16_same.json" if same else f"prompts_b{B}.json")) or {}


# ---------------------------------------------------------------------------------------------- the K8 gate (import)
def _load_k8_gate():
    """experts4bit_qlora.k8_gate: the package when importable, else the module FILE beside this repo's package (the
    package __init__ imports bitsandbytes, which the box's reducer venv and the Mac need not have), else the local copy
    below. Returns (module_or_namespace, which)."""
    try:
        from experts4bit_qlora import k8_gate  # type: ignore
        return k8_gate, "package"
    except Exception:  # noqa: BLE001
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "..", "..", "experts4bit_qlora", "k8_gate.py")
    if os.path.isfile(cand):
        try:
            spec = importlib.util.spec_from_file_location("k8_gate_file", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod          # dataclasses resolve the module by name on 3.14
            spec.loader.exec_module(mod)
            return mod, "file"
        except Exception:  # noqa: BLE001
            pass
    return sys.modules[__name__], "local-copy"


# --- byte-identical local copy of experts4bit_qlora/k8_gate.py's rule (Arm, _paired, verdict); the test asserts the
# --- function sources equal the package's. Used only when neither the package nor its file is importable.
BUDGET = 0.05


@dataclass(frozen=True)
class Arm:
    ppl: float
    text_sha: str
    steps: int
    ppl_source: str = "wikitext"


def _paired(base: Arm, cand: Arm) -> float:
    if base.text_sha != cand.text_sha:
        raise ValueError(f"arms scored different text ({base.ppl_source}: "
                         f"{base.text_sha[:12]} vs {cand.text_sha[:12]})")
    if base.steps != cand.steps:
        raise ValueError(f"arms scored different step counts "
                         f"({base.steps} vs {cand.steps})")
    return cand.ppl - base.ppl


def verdict(pairs: list[tuple[Arm, Arm]], calibrated: bool,
            budget: float = BUDGET, calibration_domain: str | None = None
            ) -> tuple[bool, list[str]]:
    """``pairs`` = ``[(base, candidate), ...]``, one per scoring text.
    Returns ``(passed, lines)``; the lines are the receipt."""
    if not pairs:
        raise ValueError("no arms to gate")
    deltas = [_paired(b, c) for b, c in pairs]
    lines = [f"K8 {c.ppl_source}: base={b.ppl:.5f} cand={c.ppl:.5f} "
             f"delta={d:+.5f} sha={b.text_sha[:12]} steps={b.steps}"
             for (b, c), d in zip(pairs, deltas)]
    if not calibrated:
        ok = all(abs(d) <= budget for d in deltas)
        lines.append(f"K8 VERDICT {'PASS' if ok else 'FAIL'} "
                     f"(uncalibrated: |delta| <= {budget} on every text)")
        return ok, lines
    within = all(d <= budget for d in deltas)
    improving = [d < 0 for d in deltas]
    if any(improving):
        # an improvement must be corroborated: same sign on >= 2 texts,
        # one outside the calibration domain
        srcs = {c.ppl_source for (_b, c), d in zip(pairs, deltas) if d < 0}
        outside = (calibration_domain is None
                   or any(s != calibration_domain for s in srcs))
        corroborated = all(improving) and len(pairs) >= 2 and outside
        if not corroborated:
            lines.append("K8 VERDICT FAIL (calibrated: an improvement needs "
                         "the same sign on >= 2 scoring texts, one outside "
                         "the calibration domain -- an improvement that "
                         "moves with the calibration text is fitting it)")
            return False, lines
    lines.append(f"K8 VERDICT {'PASS' if within else 'FAIL'} "
                 f"(calibrated: delta <= +{budget}"
                 + ("; improvement corroborated on "
                    f"{len(pairs)} texts" if any(improving) else "") + ")")
    return within, lines


# ---------------------------------------------------------------------------------------------- generic arithmetic
BAND_EPS = 1e-12   # inclusive bounds up to floating round-off: a delta formed as log(a) - log(b) sits ~1e-17 off an exact boundary


def band(delta):
    """The comparability band on a signed nats delta: CLOSE |d| <= 0.0095, COMPARABLE <= 0.02, else NOT_COMPARABLE
    (bounds inclusive to within BAND_EPS)."""
    if delta is None:
        return "UNREAD"
    a = abs(delta)
    if a <= CLOSE + BAND_EPS:
        return "CLOSE"
    if a <= COMPARABLE + BAND_EPS:
        return "COMPARABLE"
    return "NOT_COMPARABLE"


def spread_of(vals):
    vals = [v for v in vals if v]
    return ((max(vals) - min(vals)) / min(vals)) if len(vals) >= 2 else None


def draws_summary(vals, registered_single=False):
    """vals = the tok/s of the arm's non-VOID draws in draw order. -> {n, point, spread, status, usable, why}.
    status: STABLE | THIRD_DRAW_REQUIRED | UNSTABLE | SINGLE | NONE. `usable` = may enter a position."""
    vals = [v for v in vals if v]
    out = {"n": len(vals), "values": vals, "point": None, "spread": None, "status": "NONE", "usable": False, "why": ""}
    if not vals:
        out["why"] = "no VALID draw"
        return out
    out["point"] = statistics.median(vals)
    if len(vals) == 1:
        out["status"] = "SINGLE"
        out["why"] = "single draw: stability unmeasured" + ("" if registered_single else "; no position")
        return out
    sp = spread_of(vals)
    out["spread"] = sp
    if sp > DRAW_UNSTABLE:
        out["status"] = "UNSTABLE"
        out["why"] = f"draws {', '.join(f'{v:.1f}' for v in vals)} tok/s spread {100 * sp:.1f} % > {100 * DRAW_UNSTABLE:.0f} %: UNSTABLE, no position"
        return out
    if len(vals) == 2 and sp > DRAW_THIRD:
        out["status"] = "THIRD_DRAW_REQUIRED"
        out["why"] = f"two draws spread {100 * sp:.1f} % > {100 * DRAW_THIRD:.0f} %: the registration asks for a third draw (point = median of three); none landed"
        return out
    out["status"] = "STABLE"
    out["usable"] = True
    out["why"] = f"{len(vals)} draws, spread {100 * sp:.1f} %"
    return out


def interval(num_vals, den_vals):
    """Point = median/median; interval = (min, max) over every cross-draw ratio."""
    if not num_vals or not den_vals:
        return None
    cross = [n / d for n in num_vals for d in den_vals if d]
    return {"point": statistics.median(num_vals) / statistics.median(den_vals), "min": min(cross), "max": max(cross), "n_cross": len(cross)}


def energy_integral(csv_text, start, stop, interval_ms=50, sampler_start=None, fields=None):
    """Trapezoid of power.draw.instant [W] over the samples inside [start, stop] (epoch s). Timestamps come from a
    `timestamp` column when the sampler wrote one, else from sampler_start + i * interval_ms, else (stated) the rows are
    assumed to span [start, stop] uniformly. -> {joules, seconds, n_samples, mean_w, basis}."""
    rows = list(csv.reader(io.StringIO(csv_text.strip())))
    if not rows:
        return {"joules": None, "seconds": None, "n_samples": 0, "mean_w": None, "basis": "empty csv"}
    header = [h.strip() for h in rows[0]]
    data = rows[1:]
    if fields and not any(h.startswith("power.draw") for h in header):   # the sampler writes noheader; its fields ride the receipt (A5)
        header = [f.strip() for f in (fields.split(",") if isinstance(fields, str) else fields)]
        data = rows
    pcol = next((i for i, h in enumerate(header) if h.startswith("power.draw")), None)
    tcol = next((i for i, h in enumerate(header) if h in ("timestamp", "epoch")), None)
    body = [r for r in data if len(r) > (pcol or 0)]
    if pcol is None:
        return {"joules": None, "seconds": None, "n_samples": 0, "mean_w": None, "basis": "no power.draw column"}
    pts = []
    for i, r in enumerate(body):
        try:
            w = float(r[pcol].strip().split()[0])
        except (ValueError, IndexError):
            continue
        if tcol is not None:
            t = _parse_ts(r[tcol].strip())
            basis = "csv timestamps"
        elif sampler_start is not None:
            t = sampler_start + i * interval_ms / 1000.0
            basis = f"sampler_start_epoch + i x {interval_ms} ms"
        else:
            t = start + (stop - start) * (i / max(len(body) - 1, 1))
            basis = "rows assumed to span [start, stop] uniformly (no timestamps, no sampler_start_epoch)"
        if t is not None and start <= t <= stop:
            pts.append((t, w))
    if len(pts) < 2:
        return {"joules": None, "seconds": None, "n_samples": len(pts), "mean_w": None, "basis": "fewer than 2 samples inside the window"}
    pts.sort()
    joules = sum((t1 - t0) * (w0 + w1) / 2 for (t0, w0), (t1, w1) in zip(pts, pts[1:]))
    secs = pts[-1][0] - pts[0][0]
    return {"joules": joules, "seconds": secs, "n_samples": len(pts), "mean_w": (joules / secs) if secs else None, "basis": basis}


def _parse_ts(s):
    try:
        return float(s)
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            import time
            tt = time.strptime(s.split(".")[0] if ".%f" not in fmt else s, fmt.replace(".%f", "") if ".%f" not in fmt else fmt)
            frac = float("0." + s.split(".")[1]) if "." in s and ".%f" in fmt else 0.0
            return calendar.timegm(tt) + frac
        except ValueError:
            continue
    return None


# ------------------------------------------------------------------------------------------------- census readers
def parse_census_names(text):
    """Kernel names from a torch-profiler census table (bench/p42's shape): the first column of every row that carries
    a `# of Calls` figure. -> {name: calls}."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 10 or parts[0].startswith("-") or parts[0] == "Name":
            continue
        try:
            calls = int(parts[-1])
        except ValueError:
            continue
        name = line.strip().split("  ")[0].strip()
        out[name] = out.get(name, 0) + calls
    return out


def census_kernels(rec, d, stem):
    """The kernels an arm's census names: the receipt's `census` (a dict name->calls, a dict with `kernels`, or a list)
    or logs/census_<stem>.txt (profiler table); None when there is no census at all (never an empty set)."""
    c = (rec or {}).get("census")
    if isinstance(c, dict):
        if isinstance(c.get("kernels"), (list, dict)):
            k = c["kernels"]
            return set(k) if isinstance(k, list) else {n for n, v in k.items() if v}
        names = {n for n, v in c.items() if isinstance(v, (int, float)) and v and ("gemm" in n or "kernel" in n or n.startswith("_"))}
        if names:
            return names
        if any(isinstance(v, bool) for v in c.values()):
            names = set()
            if c.get("k19_engaged") or c.get("k19"):
                names.add(K19_KERNEL)
            if c.get("k23_engaged") or c.get("k23") or c.get("lean_glue_engaged"):
                names.add(K23_KERNEL)
            return names
    elif isinstance(c, list):
        return set(c)
    path = os.path.join(d, "logs", f"census_{stem}.txt")
    if os.path.exists(path):
        return set(parse_census_names(read_text(path)))
    return None


# --------------------------------------------------------------------------------------------------- arm readers
def status_of(rec):
    """-> (verdict, reason) from the receipt's own status/verdict; VALID when it says ok or says nothing."""
    if rec is None:
        return "NOT_RUN", "no receipt"
    st = str(rec.get("status") or "").lower()
    if st in ("", "ok") and str(rec.get("verdict") or "").upper() == "VOID":
        return "VOID", str(rec.get("void_reason") or "scorer/driver said VOID")[:200]
    if st in ("", "ok"):
        return "VALID", ""
    v = STATUS_MAP.get(st, "HARNESS_ERROR")
    return v, str(rec.get("void_reason") or rec.get("reason") or rec.get("error") or st)[:200]


def tok_s_of(rec, B, window=False):
    """The arm's speed as tok/s: window arms from step_ms_clean (B=1) / aggregate_tok_s (B=16); p37-shaped arms from
    decode_tok_s. ms/step beside it."""
    if window:
        ms = rec.get("step_ms_clean")
        if ms is None:
            return None, None
        tok = (1000.0 / ms) if B == 1 else rec.get("aggregate_tok_s")
        return tok, ms
    return rec.get("decode_tok_s"), rec.get("decode_ms_per_step")


def check_tokens(rec, B, n=LONG):
    """Every row generated exactly n tokens (the registered workload): tokens is a dict row->ids or a list."""
    toks = rec.get("tokens")
    if toks is None:
        return "no `tokens` in the receipt (generated-token count unverifiable)"
    rows = list(toks.values()) if isinstance(toks, dict) else (toks if (toks and isinstance(toks[0], list)) else [toks])
    if len(rows) != B:
        return f"{len(rows)} token rows != B={B}"
    bad = [i for i, r in enumerate(rows) if len(r) != n]
    return f"rows {bad[:6]} generated != {n} tokens" if bad else ""


def check_prompts(rec, pf, B, same=False):
    why = []
    psha = pf.get("prompts_sha256")
    rsha = rec.get("prompts_sha256")
    if same:
        sp = rec.get("sameprompt") or {}
        rsha = sp.get("effective_prompts_sha256") or rsha    # the rows the arm ran = the same-file's (A5)
        if not (sp.get("rows_identical") or rec.get("rows_identical")):
            why.append("SAMEPROMPT arm does not say rows_identical")
    if psha and rsha and rsha != psha:
        why.append("prompts_sha differs from the prompt file's")
    elif not psha:
        why.append("prompt file missing: prompts_sha unverified")
    pt = rec.get("prompt_tokens_engine")
    if pt is None:
        pt = rec.get("prompt_tokens")
    if isinstance(pt, list):
        if len(pt) != B or any(x != PROMPT_TOKENS for x in pt):
            why.append(f"prompt tokens per row {sorted(set(pt))} != {PROMPT_TOKENS} x {B}")
    elif isinstance(pt, (int, float)):
        if int(pt) != PROMPT_TOKENS:
            why.append(f"engine counted {pt} prompt tokens != {PROMPT_TOKENS}")
    else:
        why.append("no prompt-token count in the receipt")
    return why


def e4b_window_row(rec, d, arm, B, log, fp):
    """The e4b window arms (step_decomp --b1d-timed / BV3): P58's rules plus SC1's lic banners and the K19 census."""
    st, why0 = status_of(rec)
    if st != "VALID":
        return {"tok": None, "ms": None, "verdict": st, "why": why0}
    tok, ms = tok_s_of(rec, B, window=True)
    if tok is None:
        return {"tok": None, "ms": None, "verdict": "VOID", "why": "no step_ms_clean / aggregate_tok_s in receipt"}
    why = []
    if rec.get("recompiles_in_window", 0):
        why.append(f"recompiles_in_window={rec.get('recompiles_in_window')}")
    if rec.get("n_steps") != NEED_STEPS[B]:
        why.append(f"n_steps {rec.get('n_steps')} != {NEED_STEPS[B]}")
    base = arm.split("_")[0]
    text = read_text(log)
    stem = f"e4b_{arm}_b{B}"
    kern = census_kernels(rec, d, stem + "_r1") or census_kernels(rec, d, stem)
    if base in ("lic", "rtn", "int4"):
        if not re.search(r"INT4EXP.*48 layers", text):
            why.append("no INT4EXP 48-layer banner")
        if not re.search(r"ATTNINT4.*192", text):
            why.append("no ATTNINT4 192 banner")
        if not re.search(r"fused q/k/v projections on 48 attention modules", text):
            why.append("no `fused q/k/v projections on 48 attention modules` banner (the registered fused set)")
        if not re.search(r"B1D_TIMED_GRAPH|BV3_GRAPH|graph", text) and str(rec.get("b1d_loop", "")) != "graph":
            why.append("no graph banner (the arm's own single-step graph)")
    if base == "lic":
        m = re.search(r"INT4EXP licensed artifact (\S+)", text)
        if not m:
            why.append("no `INT4EXP licensed artifact <fp>` line")
        elif fp and m.group(1).rstrip(".,;:") != fp:
            why.append(f"artifact fingerprint {m.group(1)[:12]} != PACK {fp[:12]}")
        elif not fp:
            why.append("no `PACK <fp>` line in summary.txt: fingerprint unverified")
        if B == 16 and arm == "lic":
            if kern is None:
                why.append("no census for the B=16 lic arm (K19 engagement unverifiable)")
            elif not any(K19_KERNEL.lstrip("_") in k for k in kern):
                why.append("K19 kernel absent from the B=16 lic census (not engaged)")
        if arm == "lic_degraded":
            if kern is None:
                why.append("no census for the degraded arm (K19 absence unverifiable)")
            elif any(K19_KERNEL.lstrip("_") in k for k in kern):
                why.append("K19 kernel PRESENT in the degraded census (the lever did not disengage)")
    if base == "nf4" and re.search(r"INT4EXP|ATTNINT4|fused q/k/v", text):
        why.append("control arm shows int4/fusion banners")
    return {"tok": tok, "ms": ms, "verdict": "VOID" if why else "VALID", "why": "; ".join(why), "k19": (None if kern is None else any(K19_KERNEL.lstrip("_") in k for k in kern))}


def e4b_sched_row(rec, d, arm, B, log, fp, pf):
    """The e4b scheduler-slope arms (serve_paged.build_engine, p37-shaped, engine 'e4b-sched'): census predicates."""
    st, why0 = status_of(rec)
    if st != "VALID":
        return {"tok": None, "ms": None, "verdict": st, "why": why0}
    tok, ms = tok_s_of(rec, B)
    if tok is None:
        return {"tok": None, "ms": None, "verdict": "VOID", "why": "no decode_tok_s in receipt"}
    why = check_prompts(rec, pf, B, same=arm.endswith("sameprompt"))
    t = check_tokens(rec, B)
    if t:
        why.append(t)
    if str(rec.get("engine", "e4b-sched")) not in ("e4b-sched", "e4b"):
        why.append(f"engine {rec.get('engine')!r} is not e4b-sched")
    c = rec.get("census")
    if not isinstance(c, dict):
        why.append("no `census` in the sched receipt (build_engine's info)")
    else:
        if c.get("fuse_qkv_n") != 48:
            why.append(f"census fuse_qkv_n {c.get('fuse_qkv_n')} != 48")
        if c.get("int4_expert_layers") != 48:
            why.append(f"census int4_expert_layers {c.get('int4_expert_layers')} != 48")
        # A10: two counts, each in its own units. BEFORE fusion the attention lever converts 192 projections (48 layers x
        # q/k/v/o): the licensed arms through the calibrated lever (LICENV: E4B_SERVE_ATTN_INT4_CALIB=1), the RTN arms through
        # the rtn lever (SPEEDENV: E4B_SERVE_ATTN_INT4=1). serve_paged.build_engine then counts Int4Linear modules AFTER
        # _apply_fusions merges q/k/v, so a fully converted fused model reads 192 - 2 x fuse_qkv_n = 96. The registered rule
        # compared that post-fusion count with 192 and VOIDed every engaged sched arm (sc1b-5090-1: 96, rtn 192, calib 0).
        calib, rtn = c.get("attn_int4_calib_projections"), c.get("attn_int4_rtn_projections")
        if arm.startswith("lic"):
            levers = [("attn_int4_calib_projections", calib, "attn_int4_rtn_projections", rtn)]
        elif arm.startswith("int4"):
            levers = [("attn_int4_rtn_projections", rtn, "attn_int4_calib_projections", calib)]
        else:
            levers = [("attn_int4_calib_projections", calib, "attn_int4_rtn_projections", rtn),
                      ("attn_int4_rtn_projections", rtn, "attn_int4_calib_projections", calib)]
        if not any(on == 192 and (off or 0) == 0 for _, on, _, off in levers):
            why.append("census attention lever: " + " or ".join(f"{n} {on} (want 192) with {m} {off} (want 0)" for n, on, m, off in levers))
        fq = c.get("fuse_qkv_n") if isinstance(c.get("fuse_qkv_n"), int) else 0
        want_post = 192 - 2 * fq
        if c.get("int4_attn_projections") != want_post:
            why.append(f"census int4_attn_projections {c.get('int4_attn_projections')} != {want_post} "
                       f"(192 - 2 x fuse_qkv_n {fq}: Int4Linear modules counted after q/k/v fusion)")
        r2 = c.get("fuse_t1_glue_r2_n")
        folds_ok = bool(c.get("fuse_t1_glue_n")) and bool(c.get("fuse_router_epilogue_n")) and (bool(r2) and all(r2) if isinstance(r2, list) else bool(r2))
        if not folds_ok:
            why.append("census fold counts not all > 0 (fuse_t1_glue_n / fuse_t1_glue_r2_n / fuse_router_epilogue_n: the folds fuse_qkv applies)")
        gs = c.get("graph_status") or {}
        captured = (isinstance(gs, dict) and str(gs.get(str(B), gs.get(B, ""))).startswith("graph")) or bool(re.search(rf"DECODE_GRAPH bucket={B} captured", read_text(log)))
        if not captured:
            why.append(f"no `DECODE_GRAPH bucket={B} captured` (graph_status / log)")
        if arm.startswith("lic"):
            cfp = c.get("fingerprint") or (c.get("levers_env") or {}).get("E4B_INT4_EXPECTED_FINGERPRINT")
            if not cfp:
                why.append("census carries no pack fingerprint")
            elif fp and cfp != fp:
                why.append(f"census fingerprint {str(cfp)[:12]} != PACK {fp[:12]}")
    return {"tok": tok, "ms": ms, "verdict": "VOID" if why else "VALID", "why": "; ".join(why)}


def vllm_row(rec, arm, B, log, pf):
    st, why0 = status_of(rec)
    if st != "VALID":
        return {"tok": None, "ms": None, "verdict": st, "why": why0}
    tok, ms = tok_s_of(rec, B)
    if tok is None:
        return {"tok": None, "ms": None, "verdict": "VOID", "why": "no decode_tok_s in receipt"}
    same = "sameprompt" in arm
    why = check_prompts(rec, pf, B, same=same)
    t = check_tokens(rec, B)
    if t:
        why.append(t)
    eng = rec.get("engagement") or {}
    res = rec.get("resolved") or {}
    native = arm.endswith("native")
    if eng.get("status") != "grepped":
        why.append("engagement.status != grepped (no log was grepped: Marlin / graphs unverifiable)")
    else:
        if not eng.get("marlin_moe"):
            why.append("no `Using 'MARLIN' WNA16 MoE backend.` line")
        if arm.startswith("awq"):
            if res.get("quantization") != "auto_awq":
                why.append(f"resolved.quantization {res.get('quantization')!r} != auto_awq")
        elif not eng.get("marlin_linear"):
            why.append("no `Using MarlinLinearKernel` line")
        if "eager" in arm:
            if eng.get("cudagraphs_captured"):
                why.append("eager arm captured graphs")
        elif not eng.get("cudagraphs_captured"):
            why.append("graph arm without CUDA-graph capture")
    ab = str(res.get("attention_backend") or "")
    if "fp8" in arm:
        if not ab:
            why.append("fp8-KV arm did not record its attention_backend")
    elif not native and ab and "FLASH_ATTN" not in ab.upper():
        why.append(f"attention_backend {ab} != FLASH_ATTN on a bf16-KV arm")
    if not native:
        if res.get("enable_prefix_caching") not in (False, None) or res.get("enable_prefix_caching") is None and rec.get("llm_kwargs", {}).get("enable_prefix_caching") not in (False, None):
            why.append("prefix caching not OFF on a timed arm")
        if res.get("max_model_len") not in (None, 2048):
            why.append(f"max_model_len {res.get('max_model_len')} != 2048")
        if res.get("max_num_seqs") not in (None, B, 16):
            why.append(f"max_num_seqs {res.get('max_num_seqs')} not in ({B}, 16)")
    return {"tok": tok, "ms": ms, "verdict": "VOID" if why else "VALID", "why": "; ".join(why),
            "note": f"attn {ab or '?'}; capture {res.get('cudagraph_capture_sizes')}; detok {(rec.get('generation') or {}).get('detokenize', '?')}"}


def sglang_row(rec, arm, B, log, pf):
    st, why0 = status_of(rec)
    if st != "VALID":
        return {"tok": None, "ms": None, "verdict": st, "why": why0}
    tok, ms = tok_s_of(rec, B)
    if tok is None:
        return {"tok": None, "ms": None, "verdict": "VOID", "why": "no decode_tok_s in receipt"}
    why = check_prompts(rec, pf, B)
    t = check_tokens(rec, B)
    if t:
        why.append(t)
    eng = rec.get("engagement") or {}
    sa = rec.get("server_args") or {}
    native = arm.endswith("native")
    if not eng:
        why.append("no engagement record (server.sh's <log>.engagement.json)")
    else:
        if not eng.get("gptq_marlin_banner"):
            why.append("no `Using gptq_marlin kernel.` banner")
        if str(eng.get("attention_backend") or sa.get("attention_backend")) != "flashinfer":
            why.append(f"attention_backend {eng.get('attention_backend')!r} != flashinfer")
        gc = str(eng.get("gencode") or "")
        if not re.search(r"sm_120[fa]", gc):
            why.append(f"Marlin MoE JIT gencode {gc!r} is not sm_120f/sm_120a (the on-disk JIT leaf)")
    cached = rec.get("cached_tokens_long")
    if isinstance(cached, list) and any(c for c in cached if c):
        why.append("cached_tokens > 0 on a radix-off arm")
    if not native:
        if (sa.get("disable_radix_cache") if sa else eng.get("disable_radix_cache")) is False:
            why.append("radix cache ON on a matched arm")
        mrr = sa.get("max_running_requests") if sa else eng.get("max_running_requests")
        if mrr not in (None, 16):
            why.append(f"max_running_requests {mrr} != 16")
    return {"tok": tok, "ms": ms, "verdict": "VOID" if why else "VALID", "why": "; ".join(why), "note": f"mode {rec.get('server_mode')}; gencode {eng.get('gencode')}"}


def llamacpp_row(rec, arm, B, log, pf):
    st, why0 = status_of(rec)
    if st != "VALID":
        return {"tok": None, "ms": None, "verdict": st, "why": why0}
    tok, ms = tok_s_of(rec, B)
    if tok is None:
        return {"tok": None, "ms": None, "verdict": "VOID", "why": "no decode_tok_s in receipt"}
    why = check_prompts(rec, pf, B)
    t = check_tokens(rec, B)
    if t:
        why.append(t)
    text = read_text(log)
    m = re.search(r"offloaded (\d+)/(\d+) layers", text)
    if not m:
        why.append("no `offloaded N/N layers to GPU` line in the server log")
    elif m.group(1) != m.group(2):
        why.append(f"offloaded {m.group(1)}/{m.group(2)} layers (not all)")
    if not re.search(r"flash_attn\s*=\s*enabled|flash attention enabled", text, re.I):
        why.append("no `flash_attn = enabled` line")
    if B == 16 and not re.search(r"n_slots\s*=\s*16|n_parallel\s*=\s*16", text):
        why.append("no `n_slots = 16` line on the B=16 arm")
    env = rec.get("env") or {}
    if env.get("GGML_CUDA_MMQ_PREC"):
        why.append(f"GGML_CUDA_MMQ_PREC={env['GGML_CUDA_MMQ_PREC']} set (registered unset)")
    return {"tok": tok, "ms": ms, "verdict": "VOID" if why else "VALID", "why": "; ".join(why), "note": "MMVQ (dp4a) at B=1 / MMQ (int8 mma) at B=16 by design"}


def exl3_lmd_row(rec, engine, arm, B, log, pf):
    st, why0 = status_of(rec)
    if st != "VALID":
        return {"tok": None, "ms": None, "verdict": st, "why": why0}
    tok, ms = tok_s_of(rec, B)
    if tok is None:
        return {"tok": None, "ms": None, "verdict": "VOID", "why": "no decode_tok_s in receipt"}
    why = check_prompts(rec, pf, B)
    t = check_tokens(rec, B)
    if t:
        why.append(t)
    meta = rec.get("meta") or rec.get("job_meta") or {}
    cached = meta.get("cached_tokens") or rec.get("cached_tokens")
    if isinstance(cached, list) and any(c for c in cached if c):
        why.append("cached_tokens > 0 (a prompt page was reused)")
    res = rec.get("resolved") or {}
    if engine == "exl3":
        er = meta.get("eos_reason")
        if isinstance(er, list) and any(x not in (None, "max_new_tokens") for x in er):
            why.append(f"a job ended on {sorted(set(er))} not max_new_tokens")
    if engine == "lmdeploy":
        if res.get("prefix_caching") is True:
            why.append("prefix caching ON")
        if res.get("cuda_graphs") is True:
            why.append("cuda_graphs True (TurboMind has none at 0.18.0: a different build)")
    return {"tok": tok, "ms": ms, "verdict": "VOID" if why else "VALID", "why": "; ".join(why), "note": str(res.get("kernel_family") or res.get("bpw_layer") or "")[:90]}


def arm_row(engine, rec, d, arm, B, log, fp, pf):
    if engine == "e4b":
        return e4b_window_row(rec, d, arm, B, log, fp)
    if engine == "e4bsched":
        return e4b_sched_row(rec, d, arm, B, log, fp, pf)
    if engine == "vllm":
        return vllm_row(rec, arm, B, log, pf)
    if engine == "sglang":
        return sglang_row(rec, arm, B, log, pf)
    if engine == "llamacpp":
        return llamacpp_row(rec, arm, B, log, pf)
    return exl3_lmd_row(rec, engine, arm, B, log, pf)


# ------------------------------------------------------------------------------------------------ the arm table
ARM_RE = re.compile(r"^(e4bsched|e4b|vllm|sglang|llamacpp|exl3|lmdeploy)_(.+)_b(1|16)_r(\d+)\.json$")


def scan_arms(d, fp):
    """Every timed receipt in d -> {(engine, arm, B): {"draws": [row...], "files": [...]}}; rows carry verdict/why."""
    arms = {}
    pfs = {B: prompts_file(d, B) for B in BATCHES}
    pf_same = prompts_file(d, 16, same=True)
    for path in sorted(glob.glob(os.path.join(d, "*_b*_r*.json"))):
        m = ARM_RE.match(os.path.basename(path))
        if not m:
            continue
        engine, arm, B, n = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        rec = jload(path)
        log = os.path.join(d, "logs", f"run_{engine}_{arm}_b{B}_r{n}.log")
        pf = pf_same if ("sameprompt" in arm and pf_same) else pfs[B]
        row = arm_row(engine, rec, d, arm, B, log, fp, pf)
        row.update({"draw": n, "file": os.path.basename(path), "rec": rec})
        arms.setdefault((engine, arm, B), {"draws": []})["draws"].append(row)
    for k, a in arms.items():
        a["draws"].sort(key=lambda r: r["draw"])
        vals = [r["tok"] for r in a["draws"] if r["verdict"] == "VALID"]
        single = k[1] in ("nf4_ctrl", "iq4xs", "lic_degraded") or "sameprompt" in k[1]
        a["summary"] = draws_summary(vals, registered_single=single)
        a["all_valid"] = bool(a["draws"]) and all(r["verdict"] == "VALID" for r in a["draws"])
        a["verdict"] = ("VALID" if a["all_valid"] else next((r["verdict"] for r in a["draws"] if r["verdict"] != "VALID"), "NOT_RUN"))
        if a["all_valid"] and a["summary"]["status"] == "UNSTABLE":
            a["verdict"] = "UNSTABLE"
    return arms


def void_arm(arms, key, why):
    """A rule outside the arm's own receipt VOIDs it (the degraded control failing VOIDs e4b's B=16 rows)."""
    a = arms.get(key)
    if a:
        a["verdict"] = "VOID"
        a["all_valid"] = False
        a["summary"] = draws_summary([])
        a["summary"]["why"] = why
        for r in a["draws"]:
            if r["verdict"] == "VALID":
                r["verdict"], r["why"] = "VOID", why


# -------------------------------------------------------------------------------------------- licence (box A)
def k8_row(d, name, wsha):
    """One K8 row -> (Arm | None, why, rec). VOID when the sha is off the window file or steps != 2048."""
    rec = jload(os.path.join(d, f"k8_{name}.json"))
    if rec is None:
        return None, "missing", None
    st, why0 = status_of(rec)
    if st != "VALID":
        return None, f"{st}: {why0}", rec
    if "mean_nll" not in rec and "ppl" not in rec:
        return None, "no mean_nll / ppl", rec
    sha = str(rec.get("text_sha") or rec.get("text_sha256") or "")
    steps = int(rec.get("steps") or rec.get("tokens_scored") or -1)
    if wsha is None:
        return None, "k8_window file missing: sha unpinned", rec
    if sha != wsha:
        return None, f"text_sha {sha[:12]} != window {wsha[:12]}", rec
    if steps != S:
        return None, f"steps {steps} != {S}", rec
    ppl = rec.get("ppl")
    if ppl is None:
        ppl = math.exp(float(rec["mean_nll"]))
    src = rec.get("ppl_source") or name.rsplit("_", 1)[-1]
    return Arm(ppl=float(ppl), text_sha=sha, steps=steps, ppl_source=src), "", rec


def licence_block(d, box):
    """LICENSED-B1 / LICENSED-B16 / QUALITY_FAIL through k8_gate.verdict(calibrated=True, calibration_domain='c4val1')."""
    gate, which = _load_k8_gate()
    out = {"box": box, "gate_impl": which, "rows": {}, "b1": None, "b16": None, "lines": [], "verdict": "UNREAD", "why": []}
    if box != "A":
        out["why"].append(f"box {box} carries no licence rows (box A's reading applies through the substitution licence)")
        return out
    rows = {}
    for src in TEXTS:
        wsha = window_sha(d, src)
        for tag in ("nf4", "lic_auto", "lic_1"):
            name = f"{tag}_{src}"
            arm, why, rec = k8_row(d, name, wsha)
            info = {"arm": arm, "why": why, "mean_nll": (rec or {}).get("mean_nll"), "attn_compute": (rec or {}).get("attn_compute"),
                    "knobs": {k: v for k, v in (((rec or {}).get("levers_env") or (rec or {}).get("env") or {}).items())
                              if k in ("E4B_INT4_GROUPED_SMALLM", "E4B_INT4_LEAN_GLUE", "E4B_NF4_GROUPED_SMALLM", "E4B_MXFP4_GROUPED_SMALLM")}}
            if tag == "lic_1" and arm is not None:
                kern = census_kernels(rec, d, f"k8_{name}")
                if kern is None:
                    info["why"] = "no K19/K23 census for the =1 row"
                    info["arm"] = None
                else:
                    k19 = any(K19_KERNEL.lstrip("_") in k for k in kern)
                    k23 = any(K23_KERNEL.lstrip("_") in k for k in kern)
                    info["k19"], info["k23"] = k19, k23
                    if not (k19 and k23):
                        info["why"] = f"census: K19 {'present' if k19 else 'ABSENT'}, K23 {'present' if k23 else 'ABSENT'} (both required on the =1 rows)"
                        info["arm"] = None
            rows[name] = info
    out["rows"] = rows

    def gate_pairs(cand_tag):
        pairs, miss = [], []
        for src in TEXTS:
            b, c = rows[f"nf4_{src}"], rows[f"{cand_tag}_{src}"]
            if b["arm"] is None or c["arm"] is None:
                miss.append(f"{src}: nf4 {b['why'] or 'ok'} / {cand_tag} {c['why'] or 'ok'}")
            else:
                pairs.append((b["arm"], c["arm"]))
        if miss or len(pairs) != len(TEXTS):
            return None, [f"not read: {'; '.join(miss) or 'a text is missing'}"]
        try:
            ok, lines = gate.verdict(pairs, calibrated=True, budget=K8_BUDGET, calibration_domain=CALIB_DOMAIN)
        except ValueError as e:
            return None, [f"not read: {e}"]
        return ok, lines

    b1, l1 = gate_pairs("lic_auto")
    b16, l16 = gate_pairs("lic_1")
    out["b1"], out["b16"] = b1, b16
    out["lines"] = [f"auto (B=1 arithmetic): {ln}" for ln in l1] + [f"=1 (K19+K23, B=16 arithmetic): {ln}" for ln in l16]
    labels = []
    if b1 is True:
        labels.append("LICENSED-B1")
    if b16 is True:
        labels.append("LICENSED-B16")
    if b1 is False or b16 is False:
        labels.append("QUALITY_FAIL")
    out["verdict"] = " ".join(labels) if labels else "UNREAD"
    out["quality_fail"] = (b1 is False or b16 is False)
    if b1 is None:
        out["why"].append("LICENSED-B1 unread")
    if b16 is None:
        out["why"].append("LICENSED-B16 unread")
    return out


# ------------------------------------------------------------------------------------------- comparability
NLL_RE = re.compile(r"^nll_(vllm|sglang|llamacpp|exl3|lmdeploy|e4b)(?:_(fp8kv|awq|iq4xs|cu13))?_(prefill|served|decode|decode_prefix|served_tail2)_(wikitext|c4val1)\.json$")


def quality_row(rec, wsha):
    """-> {mean_nll, ppl, verdict, why, top1, label, hist, shape}. VOID per the registration (sha, range, steps, scorer)."""
    q = {"mean_nll": None, "ppl": None, "verdict": "NOT_RUN", "why": "", "top1": None, "label": "", "hist": None}
    if rec is None:
        q["why"] = "missing"
        return q
    st, why0 = status_of(rec)
    if st != "VALID":
        q.update(verdict=st, why=why0)
        return q
    why = []
    sha = str(rec.get("text_sha") or rec.get("text_sha256") or "")
    if wsha is None:
        why.append("k8_window file missing: sha unpinned")
    elif sha != wsha:
        why.append(f"text_sha {sha[:12]} != window {str(wsha)[:12]}")
    rng = rec.get("scored_index_range")
    if rng is None and isinstance(rec.get("targets"), dict):
        rng = [rec["targets"].get("first_index"), rec["targets"].get("last_index")]
    if rng is None and rec.get("prompt_len") is not None and rec.get("steps") is not None:
        rng = [int(rec["prompt_len"]) + 1, int(rec["prompt_len"]) + int(rec["steps"])]
    if rng is None:
        why.append("scored range unstated")
    elif [int(x) for x in rng] != SCORED_RANGE:
        why.append(f"scored_index_range {rng} != {SCORED_RANGE}")
    steps = rec.get("steps") or rec.get("tokens_scored")
    if steps is not None and int(steps) != S:
        why.append(f"steps {steps} != {S}")
    if rec.get("void_rows"):
        why.append(f"void_rows {rec['void_rows']}")
    if rec.get("mean_nll") is None:
        why.append("no mean_nll")
    q["mean_nll"] = rec.get("mean_nll")
    q["ppl"] = rec.get("ppl") if rec.get("ppl") is not None else (math.exp(rec["mean_nll"]) if rec.get("mean_nll") is not None else None)
    t1 = rec.get("top1_agreement")
    if t1 is None:
        t1 = rec.get("top1_agree_frac")
    q["top1"] = t1
    q["hist"] = rec.get("suffix_histogram") or rec.get("recomputed_suffix_histogram") or (rec.get("cache") or {}).get("mean_recomputed_suffix")
    q["label"] = str(rec.get("mode_label") or rec.get("mode") or "")[:110]
    if rec.get("cache_predicate") == "FAIL":
        q["label"] += " [cache_predicate FAIL: relabelled by the scorer]"
    q.update(verdict="VOID" if why else "VALID", why="; ".join(why))
    return q


def comparability_block(d, box):
    """Per engine x text x shape: mean NLL, ppl, delta_bf16, delta_pair, band, top-1, label, histogram."""
    out = {"oracle": {}, "rows": [], "e4b_served": {}, "e4b_prefill": {}, "served_vs_prefill": {}}
    wsha = {src: window_sha(d, src) for src in TEXTS}
    for src in TEXTS:
        o = jload(os.path.join(d, f"oracle_{src}.json"))
        q = quality_row(o, wsha[src]) if o is not None else {"mean_nll": None, "verdict": "NOT_RUN", "why": "missing"}
        out["oracle"][src] = q
    # e4b's shapes: served = the K8 auto row (box A) / the RTN K8 row (B, C); prefill = nll_e4b_prefill_<src>
    for src in TEXTS:
        names = ["lic_auto"] if box == "A" else ["int4", "rtn", "lic_auto"]
        served = None
        for nm in names:
            rec = jload(os.path.join(d, f"k8_{nm}_{src}.json"))
            if rec is not None:
                served = quality_row(rec, wsha[src])
                served["source"] = f"k8_{nm}_{src}"
                break
        out["e4b_served"][src] = served or {"mean_nll": None, "verdict": "NOT_RUN", "why": "no e4b K8 row for this box", "source": None}
        pre = jload(os.path.join(d, f"nll_e4b_prefill_{src}.json"))
        out["e4b_prefill"][src] = quality_row(pre, wsha[src]) if pre is not None else {"mean_nll": None, "verdict": "NOT_RUN", "why": "missing"}
    rows = []
    for path in sorted(glob.glob(os.path.join(d, "nll_*.json"))):
        m = NLL_RE.match(os.path.basename(path))
        if not m or m.group(1) == "e4b":
            continue
        engine, variant, mode, src = m.group(1), m.group(2), m.group(3), m.group(4)
        shape = "prefill" if mode == "prefill" else ("served" if mode in SERVED_MODES else "cross-check")
        q = quality_row(jload(path), wsha[src])
        q.update(engine=engine, variant=variant, mode=mode, src=src, shape=shape, file=os.path.basename(path))
        _deltas(q, out, src, shape)
        rows.append(q)
    for src in TEXTS:
        for shape, q in (("served", out["e4b_served"][src]), ("prefill", out["e4b_prefill"][src])):
            q.update(engine="e4b", variant=None, mode=shape, src=src, shape=shape)
            _deltas(q, out, src, shape, is_e4b=True)
    out["rows"] = rows
    # served-vs-prefill per engine (variant) per text: the engagement reading
    keyed = {}
    for q in rows + [out["e4b_served"][s] for s in TEXTS] + [out["e4b_prefill"][s] for s in TEXTS]:
        if q.get("shape") in ("served", "prefill") and q.get("verdict") == "VALID":
            keyed.setdefault((q["engine"], q.get("variant"), q["src"]), {})[q["shape"]] = q["mean_nll"]
    for (engine, variant, src), v in keyed.items():
        if "served" in v and "prefill" in v:
            out["served_vs_prefill"][(engine, variant, src)] = v["served"] - v["prefill"]
    return out


def _deltas(q, out, src, shape, is_e4b=False):
    o = out["oracle"][src]
    q["delta_bf16"] = (q["mean_nll"] - o["mean_nll"]) if (q.get("verdict") == "VALID" and o.get("verdict") == "VALID" and q.get("mean_nll") is not None) else None
    q["bf16_note"] = "" if q["delta_bf16"] is not None else ("oracle " + (o.get("why") or o.get("verdict", "missing")))
    q["ood_flag"] = bool(q["delta_bf16"] is not None and q["delta_bf16"] < -OOD_FLATTERY)
    if is_e4b:
        q["delta_pair"], q["band"] = 0.0 if q.get("verdict") == "VALID" else None, "anchor"
        return
    e = out["e4b_served"][src] if shape == "served" else out["e4b_prefill"][src] if shape == "prefill" else {"verdict": "N-A"}
    q["delta_pair"] = (q["mean_nll"] - e["mean_nll"]) if (q.get("verdict") == "VALID" and e.get("verdict") == "VALID" and q.get("mean_nll") is not None) else None
    q["band"] = band(q["delta_pair"]) if q["delta_pair"] is not None else ("UNREAD" if q.get("verdict") == "VALID" else q.get("verdict"))


def served_quality_for(comp, engine, arm):
    """The served-shape band per text for a comparator arm (its variant's rows, else the engine's primary rows)."""
    variant = QUALITY_VARIANT.get((engine, arm))
    bands, deltas = {}, {}
    for src in TEXTS:
        cands = [q for q in comp["rows"] if q["engine"] == engine and q["shape"] == "served" and q["src"] == src and q.get("variant") == variant]
        bands[src] = cands[0]["band"] if cands else "UNREAD"
        deltas[src] = cands[0].get("delta_pair") if cands else None
    ok = all(b in ("CLOSE", "COMPARABLE") for b in bands.values())
    close = all(b == "CLOSE" for b in bands.values())
    return {"bands": bands, "deltas": deltas, "comparable": ok, "close": close}


# ------------------------------------------------------------------------------------------------- positions
def positions_block(arms, comp, box, licence, substitution=None):
    anchor_arm, window_arm = ANCHOR_SCHED[box], ANCHOR_WINDOW[box]
    out = {"anchor": f"e4bsched/{anchor_arm}", "window_anchor": f"e4b/{window_arm}", "rows": []}
    rtn_blocked = bool(licence and licence.get("quality_fail")) and box == "A"
    for (engine, arm, B), a in sorted(arms.items()):
        if engine not in COMPARATOR_ENGINES:
            continue
        anc = arms.get(("e4bsched", anchor_arm, B))
        win = arms.get(("e4b", window_arm, B))
        q = served_quality_for(comp, engine, arm)
        pos = {"engine": engine, "arm": arm, "B": B, "quoted": False, "why": [], "native": arm in NATIVE_ARMS.get(engine, set()),
               "quality": q, "contract": contract_of(engine, arm, a), "window_ratio": None}
        if a["verdict"] != "VALID":
            pos["why"].append(f"{engine}/{arm} {a['verdict']}")
        elif not a["summary"]["usable"]:
            pos["why"].append(f"{engine}/{arm} {a['summary']['status']}: {a['summary']['why']}")
        if anc is None:
            pos["why"].append(f"anchor {out['anchor']}_b{B} missing")
        elif anc["verdict"] != "VALID":
            pos["why"].append(f"anchor {anc['verdict']}: {anc['summary']['why'] or (anc['draws'][0]['why'] if anc['draws'] else '')}")
        elif not anc["summary"]["usable"]:
            pos["why"].append(f"anchor {anc['summary']['status']}: {anc['summary']['why']}")
        if not q["comparable"]:
            pos["why"].append("served-shape quality not COMPARABLE on both texts: " + ", ".join(f"{s} {b}" for s, b in q["bands"].items()))
        if licence and box == "A" and B == 16 and licence.get("b16") is not True:
            pos["why"].append("B=16 position needs LICENSED-B16")
        if licence and box == "A" and B == 1 and licence.get("b1") is not True:
            pos["why"].append("B=1 position needs LICENSED-B1")
        if rtn_blocked:
            pos["why"].append("QUALITY_FAIL: no position from the RTN rows either")
        if anc and a["all_valid"] and a["summary"]["usable"] and anc["summary"]["usable"]:
            iv = interval([r["tok"] for r in a["draws"]], [r["tok"] for r in anc["draws"]])
            pos.update(iv)
        if win and win["all_valid"] and win["summary"]["point"] and a["summary"]["point"]:
            pos["window_ratio"] = a["summary"]["point"] / win["summary"]["point"]
        pos["quoted"] = not pos["why"] and pos.get("point") is not None
        if substitution is not None and box != "A":
            pos["substitution"] = substitution
        out["rows"].append(pos)
    return out


def contract_of(engine, arm, a):
    rec = (a["draws"][0]["rec"] if a["draws"] else {}) or {}
    res = rec.get("resolved") or rec.get("server_args") or {}
    bits = bpw_of(engine, arm, rec)
    kv = res.get("kv_cache_dtype") or res.get("quant_policy") or "—"
    graphs = res.get("cudagraph_capture_sizes") or res.get("cuda_graphs") or rec.get("cuda_graph_config_json") or "—"
    pc = res.get("enable_prefix_caching", res.get("prefix_caching", res.get("disable_radix_cache")))
    return {"bpw": bits[0], "bpw_basis": bits[1], "kv": str(kv), "graphs": str(graphs)[:40], "prefix_cache": str(pc), "model": str(rec.get("model") or "")[:60]}


def bpw_of(engine, arm, rec):
    """bits/weight: expert_store_bytes x 8 / expert_params from the receipt when present, else the nominal."""
    for src in (rec, rec.get("resolved") or {}, rec.get("contract") or {}):
        if isinstance(src, dict):
            if src.get("bpw") is not None and isinstance(src["bpw"], (int, float)):
                return float(src["bpw"]), "receipt"
            if src.get("expert_store_bytes") and src.get("expert_params"):
                return 8.0 * float(src["expert_store_bytes"]) / float(src["expert_params"]), "computed"
            if isinstance(src.get("bpw_layer"), (int, float)):
                return float(src["bpw_layer"]), "receipt (bpw_layer)"
    nom = BPW_NOMINAL.get((engine, arm))
    return nom, "nominal" if nom is not None else "unknown"


# -------------------------------------------------------------------------------------------------- controls
def substitution_block(arms):
    """Box A: |median(lic_b<B>) / median(rtn_b<B>) - 1| <= 2 % at both B (window arms)."""
    out = {"gap": {}, "holds": None, "why": []}
    for B in BATCHES:
        lic, rtn = arms.get(("e4b", "lic", B)), arms.get(("e4b", "rtn", B))
        if not (lic and rtn and lic["all_valid"] and rtn["all_valid"] and lic["summary"]["point"] and rtn["summary"]["point"]):
            out["gap"][B] = None
            out["why"].append(f"B={B}: lic or rtn not VALID/present")
            continue
        out["gap"][B] = lic["summary"]["point"] / rtn["summary"]["point"] - 1
    gaps = [g for g in out["gap"].values() if g is not None]
    if len(gaps) == len(BATCHES):
        out["holds"] = all(abs(g) <= SUBSTITUTION for g in gaps)
        if not out["holds"]:
            out["why"].append("|lic - rtn| > 2 % at " + ", ".join(f"B={B} ({pct(g)})" for B, g in out["gap"].items() if g is not None and abs(g) > SUBSTITUTION))
    return out


def controls_block(arms, box):
    out = {"sameprompt": {}, "degraded": None, "method_pair": {}}
    for engine, distinct, same in (("vllm", "gptq_graph", "gptq_graph_sameprompt"), ("e4bsched", ANCHOR_SCHED[box], f"{ANCHOR_SCHED[box]}_sameprompt")):
        a, s = arms.get((engine, distinct, 16)), arms.get((engine, same, 16))
        r = {"ratio": None, "label": "UNREAD", "why": ""}
        if not s:
            r["why"] = f"no {engine}/{same} receipt"
        elif s["verdict"] != "VALID" or not a or a["verdict"] not in ("VALID",) or not a["summary"]["point"]:
            r["why"] = f"{engine}: sameprompt {s['verdict']} / distinct {a['verdict'] if a else 'missing'}"
        else:
            r["ratio"] = s["summary"]["point"] / a["summary"]["point"]
            if r["ratio"] >= SAMEPROMPT_FIRED:
                r["label"] = f"detector fired (x{r['ratio']:.2f}): the distinct rows are confirmed distinct"
            elif r["ratio"] > 1.0:
                r["label"] = f"detector fired weakly (x{r['ratio']:.2f} < {SAMEPROMPT_FIRED}): distinct rows labelled, not VOID"
            else:
                r["label"] = f"detector did not fire (x{r['ratio']:.2f}): stated, no VOID"
        out["sameprompt"][engine] = r
    lic, deg = arms.get(("e4b", "lic", 16)), arms.get(("e4b", "lic_degraded", 16))
    dg = {"ratio": None, "verdict": "UNREAD", "why": "", "k19_lic": None, "k19_degraded": None}
    if box == "A":
        if not deg:
            dg["why"] = "no e4b/lic_degraded_b16 receipt"
        elif not lic or lic["verdict"] != "VALID":
            dg["why"] = f"lic_b16 {lic['verdict'] if lic else 'missing'}"
        else:
            drow = deg["draws"][0]
            dg["k19_lic"] = lic["draws"][0].get("k19")
            dg["k19_degraded"] = drow.get("k19")
            if drow["tok"]:
                dg["ratio"] = lic["summary"]["point"] / drow["tok"]
            slower = dg["ratio"] is not None and dg["ratio"] >= DEGRADED_MIN
            k19_absent = drow.get("k19") is False
            if slower and k19_absent and drow["verdict"] == "VALID":
                dg["verdict"] = "PASS"
                dg["why"] = f"degraded x{dg['ratio']:.3f} slower (>= {DEGRADED_MIN}), K19 absent in its census, present in lic_b16's"
            else:
                dg["verdict"] = "FAIL"
                dg["why"] = ("; ".join(x for x in [
                    (f"degraded only x{dg['ratio']:.3f} (< {DEGRADED_MIN})" if (dg["ratio"] is not None and not slower) else ""),
                    ("K19 not shown absent in the degraded census" if not k19_absent else ""),
                    (drow["why"] if drow["verdict"] != "VALID" else "")] if x) or "unreadable")
                why = f"degraded control FAIL ({dg['why']}): the instrument cannot see the kernel it times"
                void_arm(arms, ("e4b", "lic", 16), why)
                void_arm(arms, ("e4bsched", "lic_sched", 16), why)
    else:
        dg["why"] = f"box {box}: the degraded control runs on box A"
    out["degraded"] = dg
    for B in BATCHES:
        w, s = arms.get(("e4b", ANCHOR_WINDOW[box], B)), arms.get(("e4bsched", ANCHOR_SCHED[box], B))
        if w and s and w["all_valid"] and s["all_valid"] and w["summary"]["point"] and s["summary"]["point"]:
            out["method_pair"][B] = w["summary"]["point"] / s["summary"]["point"] - 1
        else:
            out["method_pair"][B] = None
    return out


# ------------------------------------------------------------------------------ TTFT, resources, energy
def ttft_block(d):
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "ttft_*_*.json"))):
        m = re.match(r"^ttft_([a-z0-9]+)_(512|4096)\.json$", os.path.basename(path))
        if not m:
            continue
        rec = jload(path)
        st, why = status_of(rec)
        med = rec.get("ttft_s_median", rec.get("ttft_median_s"))
        if med is None and rec.get("ttft_ms_median") is not None:
            med = rec["ttft_ms_median"] / 1000.0
        mn = rec.get("ttft_s_min", rec.get("ttft_min_s", rec.get("ttft_s") if isinstance(rec.get("ttft_s"), (int, float)) else None))
        if mn is None and rec.get("ttft_ms_min") is not None:
            mn = rec["ttft_ms_min"] / 1000.0
        out[(m.group(1), int(m.group(2)))] = {"median_s": med, "min_s": mn, "verdict": st, "why": why,
                                              "own_timer": rec.get("prompt_ms_server") or (rec.get("timings") or {}).get("prompt_ms") or rec.get("engine_prefill_ms")}
    return out


def resources_block(arms, d):
    """Three VRAM columns per arm (reserved / peak allocated / demand or policy reservation) + host RSS / CPU / PCIe
    where the receipt carries them; the nvidia-smi max from vram_<stem>.txt as the generic fallback."""
    rows = []
    for (engine, arm, B), a in sorted(arms.items()):
        rec = (a["draws"][0]["rec"] if a["draws"] else {}) or {}
        mem = rec.get("mem_after_runs") or rec.get("mem") or {}
        w = mem.get("worker") or mem.get("driver_process") or {}
        vp = rec.get("vram_peak_bytes") or {}
        gb = lambda b: (float(b) / 2 ** 30) if isinstance(b, (int, float)) else None  # noqa: E731
        reserved = gb(w.get("memory_reserved") or w.get("max_memory_reserved") or rec.get("memory_reserved_bytes"))
        peak = gb(w.get("max_memory_allocated") or rec.get("max_memory_allocated_bytes") or vp.get("mem_get_info_drop_bytes") or vp.get("nvidia_smi_used_max_bytes"))
        if peak is None and rec.get("peak_vram_gb") is not None:
            peak = float(rec["peak_vram_gb"])
        demand = gb((mem.get("reservation") or {}).get("requested_bytes") or rec.get("demand_bytes"))
        smi = None
        stem = f"{engine}_{arm}_b{B}"
        for cand in (os.path.join(d, f"vram_{stem}_r1.txt"), os.path.join(d, f"vram_{stem}.txt")):
            if os.path.exists(cand):
                try:
                    vals = [float(line.split()[1].rstrip(",")) for line in open(cand) if len(line.split()) > 1]
                    smi = max(vals) / 1024 if vals else None
                except (OSError, ValueError):
                    smi = None
                break
        rows.append({"engine": engine, "arm": arm, "B": B, "verdict": a["verdict"], "reserved_gb": reserved, "peak_gb": peak, "demand_gb": demand,
                     "smi_max_gb": smi, "host_rss_gb": gb(rec.get("host_rss_bytes") or (rec.get("host") or {}).get("rss_bytes")),
                     "cpu_share": rec.get("cpu_share") or (rec.get("host") or {}).get("cpu_share"),
                     "pcie": rec.get("pcie") or (rec.get("host") or {}).get("pcie"), "loadavg": rec.get("loadavg") or (rec.get("host") or {}).get("loadavg")})
    return rows


def energy_block(d):
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "energy_*.json"))):
        tag = os.path.basename(path)[len("energy_"):-5]
        meta = jload(path) or {}
        csv_path = os.path.join(d, f"energy_{tag}.csv")
        r = {"tag": tag, "engine": meta.get("engine") or tag.split("_")[0], "batch": meta.get("batch") or meta.get("B"), "tokens": meta.get("tokens"),
             "joules": None, "j_per_token": None, "mean_w": None, "seconds": None, "n_samples": 0, "basis": "", "why": ""}
        try:
            start, stop = float(meta["start_epoch"]), float(meta["stop_epoch"])
        except (KeyError, TypeError, ValueError):
            r["why"] = "no start_epoch/stop_epoch"
            out[tag] = r
            continue
        if not os.path.exists(csv_path):
            r["why"] = "csv missing"
            out[tag] = r
            continue
        e = energy_integral(read_text(csv_path), start, stop, int(meta.get("interval_ms") or 50), meta.get("sampler_start_epoch"),
                            fields=meta.get("sampler_fields") or meta.get("fields"))
        r.update({k: e[k] for k in ("joules", "seconds", "n_samples", "mean_w", "basis")})
        if e["joules"] is not None and meta.get("tokens"):
            r["j_per_token"] = e["joules"] / float(meta["tokens"])
        elif e["joules"] is None:
            r["why"] = e["basis"]
        else:
            r["why"] = "no tokens in the json"
        out[tag] = r
    return out


# -------------------------------------------------------------------------------------------- predictions
def _in(x, lo, hi):
    return x is not None and lo <= x <= hi


def _pos(positions, engine, arm, B):
    return next((p for p in positions["rows"] if (p["engine"], p["arm"], p["B"]) == (engine, arm, B) and p["quoted"]), None)


def predictions_block(R):
    """P1-P14 -> (verdict HOLD | REFUTED | UNREAD, evidence); only quoted positions and VALID rows are evidence."""
    box, arms, pos, comp, ctl, lic, sub, ttft, en = R["box"], R["arms"], R["positions"], R["comparability"], R["controls"], R["licence"], R["substitution"], R["ttft"], R["energy"]
    P_ = {}

    def band_pred(name, engine, arm, B, lo, hi):
        p = _pos(pos, engine, arm, B)
        if p is None:
            a = arms.get((engine, arm, B))
            P_[name] = ("UNREAD", f"no quoted position for {engine}/{arm} B={B}" + (f" ({a['verdict']})" if a else " (no receipt)"))
        else:
            P_[name] = ("HOLD" if _in(p["point"], lo, hi) else "REFUTED", f"{p['point']:.3f} [{p['min']:.3f}, {p['max']:.3f}] vs [{lo}, {hi}]")

    if box == "A":
        band_pred("P1", "vllm", "gptq_graph", 1, *PRED_BANDS["P1"])
        band_pred("P2", "vllm", "gptq_graph", 16, *PRED_BANDS["P2"])
    else:
        P_["P1"] = P_["P2"] = ("UNREAD", f"box {box}: P1/P2 are box A's")
    # P3: sglang within 10 % / 15 % of the box's vllm anchor ratio, or UNSUPPORTED
    if box == "C" or any(k[0] == "sglang" for k in arms):
        legs = []
        for B in BATCHES:
            s, v = _pos(pos, "sglang", "gptq_matched", B), _pos(pos, "vllm", "gptq_graph", B)
            sa = arms.get(("sglang", "gptq_matched", B))
            if sa and sa["verdict"] == "UNSUPPORTED":
                legs.append(("HOLD", f"B={B} sglang UNSUPPORTED (the prediction's stated alternative)"))
            elif s is None or v is None:
                legs.append(("UNREAD", f"B={B} sglang or vllm position not quoted"))
            else:
                g = abs(s["point"] / v["point"] - 1)
                legs.append(("HOLD" if g <= P3_TOL[B] else "REFUTED", f"B={B} sglang/vllm {s['point'] / v['point']:.3f} ({100 * g:.1f} % vs {100 * P3_TOL[B]:.0f} %)"))
        P_["P3"] = _combine(legs)
    else:
        P_["P3"] = ("UNREAD", f"box {box}: SGLang is box C's")
    for name, engine, arm, k in (("P4", "llamacpp", "q4km", "P4"), ("P5", "exl3", "4bpw", "P5"), ("P5b", "lmdeploy", "w4a16", "P5b")):
        if any(key[0] == engine for key in arms):
            legs = []
            for B in BATCHES:
                a = arms.get((engine, arm, B))
                if name == "P5b" and a and a["verdict"] == "UNSUPPORTED":
                    legs.append(("HOLD", f"B={B} UNSUPPORTED (the prediction's stated alternative)"))
                    continue
                p = _pos(pos, engine, arm, B)
                lo, hi = PRED_BANDS[f"{k}_b{B}"]
                legs.append(("UNREAD", f"B={B} no quoted position") if p is None else ("HOLD" if _in(p["point"], lo, hi) else "REFUTED", f"B={B} {p['point']:.3f} vs [{lo}, {hi}]"))
            P_[name] = _combine(legs)
        else:
            P_[name] = ("UNREAD", f"no {engine} arms on box {box}")
    # P6: |served - prefill| > 0.005 on e4b-lic (box A), < 0.002 on vllm, both texts
    legs = []
    for src in TEXTS:
        de = comp["served_vs_prefill"].get(("e4b", None, src))
        dv = comp["served_vs_prefill"].get(("vllm", None, src))
        if box == "A":
            legs.append(("UNREAD", f"{src}: e4b served/prefill pair unread") if de is None else ("HOLD" if abs(de) > P6_E4B_MIN else "REFUTED", f"{src}: e4b {de:+.4f}"))
        legs.append(("UNREAD", f"{src}: vllm served/prefill pair unread") if dv is None else ("HOLD" if abs(dv) < P6_VLLM_MAX else "REFUTED", f"{src}: vllm {dv:+.4f}"))
    P_["P6"] = _combine(legs)
    legs = []
    for engine in ("vllm", "e4bsched"):
        r = ctl["sameprompt"].get(engine) or {}
        legs.append(("UNREAD", f"{engine}: {r.get('why') or 'unread'}") if r.get("ratio") is None else ("HOLD" if r["ratio"] >= SAMEPROMPT_FIRED else "REFUTED", f"{engine} x{r['ratio']:.2f}"))
    P_["P7"] = _combine(legs)
    # P8: fp8kv vs kv auto
    legs = []
    for B, rule in ((1, "within"), (16, "faster")):
        a, fp8 = arms.get(("vllm", "gptq_graph", B)), arms.get(("vllm", "gptq_fp8kv", B))
        if not (a and fp8 and a["all_valid"] and fp8["all_valid"] and a["summary"]["usable"] and fp8["summary"]["usable"]):
            legs.append(("UNREAD", f"B={B} fp8kv or kv-auto arm not VALID/stable"))
            continue
        r = fp8["summary"]["point"] / a["summary"]["point"]
        ok = abs(r - 1) <= P8_B1_TOL if rule == "within" else r > 1.0
        legs.append(("HOLD" if ok else "REFUTED", f"B={B} fp8kv/auto x{r:.3f}"))
    for src in TEXTS:
        qa = next((q for q in comp["rows"] if (q["engine"], q.get("variant"), q["shape"], q["src"]) == ("vllm", None, "served", src)), None)
        qf = next((q for q in comp["rows"] if (q["engine"], q.get("variant"), q["shape"], q["src"]) == ("vllm", "fp8kv", "served", src)), None)
        if not (qa and qf and qa.get("delta_bf16") is not None and qf.get("delta_bf16") is not None):
            legs.append(("UNREAD", f"{src}: fp8kv/auto delta_bf16 unread"))
        else:
            g = qf["delta_bf16"] - qa["delta_bf16"]
            legs.append(("HOLD" if g < P8_DBF16_MAX else "REFUTED", f"{src}: delta_bf16 gap {g:+.4f}"))
    P_["P8"] = _combine(legs)
    if box == "A":
        if lic["b1"] is None or lic["b16"] is None:
            P_["P9"] = ("UNREAD", "; ".join(lic["why"]) or lic["verdict"])
        else:
            P_["P9"] = ("HOLD" if (lic["b1"] and lic["b16"]) else "REFUTED", lic["verdict"])
    else:
        P_["P9"] = ("UNREAD", f"box {box}: the licence is box A's")
    tv, te = ttft.get(("vllm", 4096)), ttft.get(("e4b", 4096))
    if tv and te and tv["verdict"] == "VALID" and te["verdict"] == "VALID" and tv["median_s"] and te["median_s"]:
        r = tv["median_s"] / te["median_s"]
        P_["P10"] = ("HOLD" if _in(r, *PRED_BANDS["P10"]) else "REFUTED", f"TTFT-4096 vllm/e4b {r:.3f}")
    else:
        P_["P10"] = ("UNREAD", "TTFT-4096 for vllm and e4b not both VALID")
    legs = []
    jt = {(r["engine"], int(r["batch"])): r["j_per_token"] for r in en.values() if r.get("j_per_token") and r.get("batch") is not None}
    for B in BATCHES:
        jv, je = jt.get(("vllm", B)), jt.get(("e4b", B), jt.get(("e4bsched", B)))
        if jv is None or je is None:
            legs.append(("UNREAD", f"B={B} J/token for vllm and e4b not both read"))
        elif B == 1:
            legs.append(("HOLD" if abs(jv / je - 1) <= P11_B1_TOL else "REFUTED", f"B=1 vllm/e4b J/token {jv / je:.3f}"))
        else:
            p = _pos(pos, "vllm", "gptq_graph", 16)
            if p is None:
                legs.append(("UNREAD", "B=16 position not quoted (who is faster is unread)"))
            else:
                faster_vllm = p["point"] > 1.0
                more_eff_vllm = jv < je
                legs.append(("HOLD" if faster_vllm == more_eff_vllm else "REFUTED", f"B=16 vllm faster={faster_vllm}, vllm J/token lower={more_eff_vllm}"))
    P_["P11"] = _combine(legs)
    if box == "A":
        P_["P12"] = ("UNREAD", "; ".join(sub["why"])) if sub["holds"] is None else ("HOLD" if sub["holds"] else "REFUTED", ", ".join(f"B={B} {pct(g)}" for B, g in sub["gap"].items()))
    else:
        P_["P12"] = ("UNREAD", f"box {box}: read on box A (see --cross-box / --box-a)")
    P_["P13"] = R.get("cross_box_p13") or ("UNREAD", "single box: --cross-box reads P13")
    legs = []
    for B in BATCHES:
        a, nd = arms.get(("vllm", "gptq_graph", B)), arms.get(("vllm", "gptq_graph_nodetok", B))
        if not (a and nd and a["all_valid"] and nd["all_valid"] and a["summary"]["point"] and nd["summary"]["point"]):
            legs.append(("UNREAD", f"B={B} nodetok pair not both VALID"))
        else:
            g = abs(nd["summary"]["point"] / a["summary"]["point"] - 1)
            legs.append(("HOLD" if g <= P14_TOL[B] else "REFUTED", f"B={B} nodetok/matched {nd['summary']['point'] / a['summary']['point']:.3f} ({100 * g:.1f} % vs {100 * P14_TOL[B]:.0f} %)"))
    P_["P14"] = _combine(legs)
    return P_


def _combine(legs):
    if not legs:
        return ("UNREAD", "no legs")
    ev = "; ".join(e for _, e in legs)
    if any(v == "REFUTED" for v, _ in legs):
        return ("REFUTED", ev)
    if any(v == "UNREAD" for v, _ in legs):
        return ("UNREAD", ev)
    return ("HOLD", ev)


# ---------------------------------------------------------------------------------------------- frontier
def frontier_rows(R):
    """Per B, every VALID arm: tok/s on the sched axis (e4b: its sched arm; comparators: decode_tok_s), delta_bf16 served
    per text, bpw, resident GB."""
    rows = []
    res = {(r["engine"], r["arm"], r["B"]): r for r in R["resources"]}
    comp = R["comparability"]
    for (engine, arm, B), a in sorted(R["arms"].items()):
        if a["verdict"] != "VALID" or not a["summary"]["point"]:
            continue
        if engine == "e4b" or "sameprompt" in arm or arm == "lic_degraded":
            continue                                           # window arms = the kernel ceiling, not the sched axis; controls are labels
        if engine == "e4bsched":
            d16 = {src: comp["e4b_served"][src].get("delta_bf16") for src in TEXTS}
            bpw = bpw_of("e4b", arm, (a["draws"][0]["rec"] or {}))
        else:
            variant = QUALITY_VARIANT.get((engine, arm))
            d16 = {src: next((q.get("delta_bf16") for q in comp["rows"] if (q["engine"], q.get("variant"), q["shape"], q["src"]) == (engine, variant, "served", src)), None) for src in TEXTS}
            bpw = bpw_of(engine, arm, (a["draws"][0]["rec"] or {}))
        r = res.get((engine, arm, B), {})
        rows.append({"B": B, "engine": engine, "arm": arm, "tok_s": a["summary"]["point"], "delta_bf16_wikitext": d16["wikitext"], "delta_bf16_c4val1": d16["c4val1"],
                     "bpw": bpw[0], "bpw_basis": bpw[1], "resident_gb": r.get("peak_gb") if r.get("peak_gb") is not None else r.get("smi_max_gb"),
                     "stability": a["summary"]["status"], "native": arm in NATIVE_ARMS.get(engine, set())})
    return rows


def frontier_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["B", "engine", "arm", "tok_s_sched_axis", "delta_bf16_wikitext_served", "delta_bf16_c4val1_served", "bpw", "bpw_basis", "resident_gb", "stability", "native"])
    for r in rows:
        w.writerow([r["B"], r["engine"], r["arm"], f"{r['tok_s']:.1f}", f(r["delta_bf16_wikitext"], 4), f(r["delta_bf16_c4val1"], 4), f(r["bpw"], 2), r["bpw_basis"], f(r["resident_gb"], 2), r["stability"], r["native"]])
    return buf.getvalue()


# --------------------------------------------------------------------------------------------- reduce + render
def reduce_dir(run_dir, box=None, box_a_dir=None):
    d = run_dir_of(run_dir)
    box = box_of(d, box)
    fp = pack_fingerprint(d)
    arms = scan_arms(d, fp)
    licence = licence_block(d, box)
    comp = comparability_block(d, box)
    sub = substitution_block(arms) if box == "A" else None
    if box != "A":
        if box_a_dir:
            da = run_dir_of(box_a_dir)
            sub = substitution_block(scan_arms(da, pack_fingerprint(da)))
            sub["source"] = os.path.abspath(da)
        else:
            sub = {"gap": {B: None for B in BATCHES}, "holds": None, "why": [f"box {box}: the substitution licence is box A's (pass --box-a <boxA run_dir> or use --cross-box)"]}
    ctl = controls_block(arms, box)                       # may VOID e4b's B=16 rows: before positions
    pos = positions_block(arms, comp, box, licence, substitution=sub if box != "A" else None)
    R = {"box": box, "dir": os.path.abspath(d), "pack_fp": fp, "arms": arms, "licence": licence, "comparability": comp, "substitution": sub,
         "controls": ctl, "positions": pos, "ttft": ttft_block(d), "resources": resources_block(arms, d), "energy": energy_block(d),
         "versions": read_text(os.path.join(d, "versions.txt")).strip(), "forensics": read_text(os.path.join(d, "forensics.txt")).strip(),
         "box_json": jload(os.path.join(d, "box.json")) or {}}
    R["frontier"] = frontier_rows(R)
    R["predictions"] = predictions_block(R)
    return R


def anchor_ratios(R):
    """The box's vllm/gptq_graph over the e4b sched anchor per B (sched) and over the window anchor (window), quoted
    positions only for the sched ratio; the window ratio is informational."""
    out = {}
    for B in BATCHES:
        p = _pos(R["positions"], "vllm", "gptq_graph", B)
        out[B] = {"sched": p["point"] if p else None, "interval": (p["min"], p["max"]) if p else None, "window": (p or {}).get("window_ratio")}
    return out


def cross_box(run_dirs, boxes=None):
    Rs = []
    for i, rd in enumerate(run_dirs):
        Rs.append(reduce_dir(rd, (boxes[i] if boxes and i < len(boxes) else None)))
    A = next((R for R in Rs if R["box"] == "A"), None)
    out = {"boxes": {R["box"]: anchor_ratios(R) for R in Rs}, "agree": {}, "substitution": A["substitution"] if A else None, "p13": None, "lines": []}
    legs = []
    for B in BATCHES:
        vals = {R["box"]: anchor_ratios(R)[B]["sched"] for R in Rs}
        have = {b: v for b, v in vals.items() if v}
        if len(have) >= 2:
            g = max(have.values()) / min(have.values()) - 1
            out["agree"][B] = {"ratios": have, "gap": g, "within": g <= CROSS_BOX}
            legs.append(("HOLD" if g <= CROSS_BOX else "REFUTED", f"B={B} " + ", ".join(f"{b} {v:.3f}" for b, v in sorted(have.items())) + f" (gap {100 * g:.1f} % vs {100 * CROSS_BOX:.0f} %)"))
        else:
            out["agree"][B] = {"ratios": have, "gap": None, "within": None}
            legs.append(("UNREAD", f"B={B} fewer than two boxes with a quoted vLLM anchor ratio"))
    out["p13"] = _combine(legs) if len(Rs) >= 2 else ("UNREAD", "cross-box needs two or three run dirs")
    for R in Rs:
        R["cross_box_p13"] = out["p13"]
        if R["box"] != "A" and A:
            R["substitution"] = A["substitution"]
            for p in R["positions"]["rows"]:
                p["substitution"] = A["substitution"]
        R["predictions"] = predictions_block(R)
    out["reductions"] = Rs
    out["lines"].append("| B | " + " | ".join(f"box {R['box']} vllm/e4b-sched (interval)" for R in Rs) + " | gap | within 5 % |")
    out["lines"].append("|---|" + "---|" * len(Rs) + "---|---|")
    for B in BATCHES:
        cells = []
        for R in Rs:
            a = anchor_ratios(R)[B]
            cells.append(f"{a['sched']:.3f} [{a['interval'][0]:.3f}, {a['interval'][1]:.3f}]" if a["sched"] else "not quoted")
        ag = out["agree"][B]
        out["lines"].append(f"| {B} | " + " | ".join(cells) + f" | {pct(ag['gap'])} | {ag['within'] if ag['within'] is not None else '—'} |")
    if A and A["substitution"]:
        s = A["substitution"]
        out["lines"].append(f"\nSubstitution licence (box A, |lic/rtn - 1| <= {100 * SUBSTITUTION:.0f} % at both B): "
                            + (f"{'HOLDS' if s['holds'] else 'FAILS'} -- " if s["holds"] is not None else "UNREAD -- ")
                            + ", ".join(f"B={B} {pct(g)}" for B, g in s["gap"].items()) + ("; " + "; ".join(s["why"]) if s["why"] else "")
                            + ". Boxes B/C anchors are RTN: " + ("licensed as the e4b stand-in." if s["holds"] else "labelled 'RTN speed, <gap> from the licensed pack'; their ratios carry it."))
    return out


def _subst_label(sub):
    if not sub or sub.get("holds") is None:
        return "substitution licence UNREAD"
    g = ", ".join(f"B={B} {pct(x)}" for B, x in sub["gap"].items() if x is not None)
    return f"RTN anchor licensed by box A (|lic - rtn| {g})" if sub["holds"] else f"RTN speed, {g} from the licensed pack (substitution FAILED)"


def render(R):
    box, arms = R["box"], R["arms"]
    o = [f"# SC1 box {box} -- generated by sc1_reduce.py ({R['dir']})",
         f"Registration: {PREREG}. Pack fingerprint (summary.txt PACK): {R['pack_fp'] or 'MISSING'}. "
         f"Anchor: {R['positions']['anchor']} (sched axis); window anchor {R['positions']['window_anchor']} (kernel ceiling). "
         "VOID never enters a ratio or a quality reading; a position is quoted only when both arms are VALID and stable and the comparator's "
         "served-shape quality is COMPARABLE on both texts.",
         "```\n" + (R["versions"] or "(versions.txt missing)") + "\n```"]
    if R["forensics"]:
        o.append("box: " + " | ".join(line.strip() for line in R["forensics"].splitlines() if line.strip())[:600])
    o.append("\n## 1. Arms (speed verdict per draw, VOID reason = the pre-registered engagement rule that failed)")
    o.append("| engine | arm | B | draw | verdict | tok/s | ms/step | why / note |")
    o.append("|---|---|---|---|---|---|---|---|")
    for (engine, arm, B), a in sorted(arms.items()):
        for r in a["draws"]:
            o.append(f"| {engine} | {arm} | {B} | r{r['draw']} | {r['verdict']} | {f(r['tok'], 1)} | {f(r['ms'])} | {(r['why'] or r.get('note') or '')[:200]} |")
    o.append("\n## 2. Draws (point = median; spread = (max-min)/min; third draw at > 3 %; UNSTABLE at > 5 %)")
    o.append("| engine | arm | B | n | values tok/s | spread | status | arm verdict | why |")
    o.append("|---|---|---|---|---|---|---|---|---|")
    for (engine, arm, B), a in sorted(arms.items()):
        s = a["summary"]
        o.append(f"| {engine} | {arm} | {B} | {s['n']} | {', '.join(f'{v:.1f}' for v in s['values'])} | {pct(s['spread']) if s['spread'] is not None else '—'} | {s['status']} | {a['verdict']} | {s['why'][:160]} |")
    L = R["licence"]
    o.append(f"\n## 3. Licence (box A; k8_gate.verdict calibrated=True budget={K8_BUDGET} calibration_domain={CALIB_DOMAIN!r}; gate ran from: {L['gate_impl']})")
    o.append(f"**{L['verdict']}**" + ("  -- " + "; ".join(L["why"]) if L["why"] else ""))
    for name, info in L["rows"].items():
        a = info["arm"]
        o.append(f"- k8_{name}: " + (f"ppl {a.ppl:.5f} nll {f(info['mean_nll'], 5)} sha {a.text_sha[:12]} steps {a.steps}" if a else f"NOT USED ({info['why']})")
                 + f"; attn_compute {info.get('attn_compute')}; knobs {info.get('knobs') or '—'}" + (f"; census K19 {info.get('k19')} K23 {info.get('k23')}" if "k19" in info else ""))
    for ln in L["lines"]:
        o.append(f"    {ln}")
    C = R["comparability"]
    o.append(f"\n## 4. Comparability (nats per text, per shape; CLOSE |d_pair| <= {CLOSE}, COMPARABLE <= {COMPARABLE}; d_bf16 vs oracle_<src>.json)")
    for src in TEXTS:
        oc = C["oracle"][src]
        o.append(f"- oracle {src}: " + (f"nll {oc['mean_nll']:.5f}" if oc.get("verdict") == "VALID" else f"UNREAD ({oc.get('why') or oc.get('verdict')}) -> every d_bf16 on {src} is UNREAD, never 0"))
    o.append("| engine | variant | shape | mode | text | mean NLL | ppl | d_bf16 | d_pair | band | top-1 | cache label / suffix histogram | verdict / why |")
    o.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    e4b_rows = [C["e4b_served"][s] for s in TEXTS] + [C["e4b_prefill"][s] for s in TEXTS]
    for q in e4b_rows + C["rows"]:
        hist = q.get("hist")
        hist_s = (json.dumps(hist)[:60] if hist is not None else "")
        o.append(f"| {q.get('engine', 'e4b')} | {q.get('variant') or ''} | {q.get('shape')} | {q.get('mode')} | {q.get('src')} | {f(q.get('mean_nll'), 5)} | {f(q.get('ppl'), 4)} | "
                 f"{f(q.get('delta_bf16'), 4) if q.get('delta_bf16') is not None else 'UNREAD'} | {f(q.get('delta_pair'), 4) if q.get('delta_pair') is not None else '—'} | {q.get('band', '')}"
                 f"{' OOD-FLATTERY?' if q.get('ood_flag') else ''} | {f(q.get('top1'), 3)} | {q.get('label', '')[:70]} {hist_s} | {q.get('verdict')} {q.get('why', '')[:90]} {q.get('bf16_note', '')[:60]} |")
    if C["served_vs_prefill"]:
        o.append("- served - prefill (the engagement reading): " + "; ".join(f"{e}{'/' + v if v else ''} {s} {d:+.4f}" for (e, v, s), d in sorted(C["served_vs_prefill"].items(), key=str)))
    o.append("\n## 5. Positions (point = median/median over draws; interval = min..max of the cross-draw ratios; sched axis)")
    o.append("| engine | arm | B | quoted | ratio vs anchor | interval | window ratio (kernel ceiling) | quality served (wikitext, c4val1) | contract | why not |")
    o.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in R["positions"]["rows"]:
        q = p["quality"]
        qs = ", ".join(f"{s} {b}" for s, b in q["bands"].items()) + (" (CLOSE on both)" if q["close"] else "")
        c = p["contract"]
        cs = f"{c['model']}; bpw {f(c['bpw'], 2)} ({c['bpw_basis']}); kv {c['kv']}; graphs {c['graphs']}; prefix-cache {c['prefix_cache']}" + ("; NATIVE (labelled)" if p["native"] else "")
        if p.get("substitution") is not None:
            cs += "; " + _subst_label(p["substitution"])
        o.append(f"| {p['engine']} | {p['arm']} | {p['B']} | {'YES' if p['quoted'] else 'no'} | {f(p.get('point'))} | "
                 f"{('[' + f(p.get('min')) + ', ' + f(p.get('max')) + ']') if p.get('point') is not None else '—'} | {f(p.get('window_ratio'))} | {qs} | {cs} | {'; '.join(p['why'])[:220]} |")
    S_ = R["substitution"]
    if S_:
        o.append(f"- substitution licence (|lic/rtn - 1| <= {100 * SUBSTITUTION:.0f} % at both B, box A window arms): "
                 + ("HOLDS" if S_["holds"] else "FAILS" if S_["holds"] is False else "UNREAD") + " -- " + ", ".join(f"B={B} {pct(g)}" for B, g in S_["gap"].items())
                 + ("; " + "; ".join(S_["why"]) if S_["why"] else "") + (f" (from {S_['source']})" if S_.get("source") else ""))
    T = R["controls"]
    o.append("\n## 6. Controls")
    for engine, r in T["sameprompt"].items():
        o.append(f"- SAMEPROMPT {engine}: ratio {f(r['ratio'])} -> {r['label']}" + (f" ({r['why']})" if r["why"] else ""))
    dg = T["degraded"]
    o.append(f"- degraded (lic_b16 / lic_degraded_b16 >= {DEGRADED_MIN} AND K19 absent in the degraded census): **{dg['verdict']}** -- ratio {f(dg['ratio'])}; K19 lic {dg['k19_lic']} / degraded {dg['k19_degraded']}; {dg['why']}")
    o.append("- method pair (window / sched - 1, same stack): " + ", ".join(f"B={B} {pct(g)}" for B, g in T["method_pair"].items()) + " (expected +3..+15 % at B=1)")
    o.append("\n## 7. TTFT (median of 3 warm uncached max_tokens=1 requests; engine's own timer beside where present)")
    o.append("| engine | prompt | median s | min s | own timer | verdict |")
    o.append("|---|---|---|---|---|---|")
    for (engine, ln), t in sorted(R["ttft"].items()):
        o.append(f"| {engine} | {ln} | {f(t['median_s'], 4)} | {f(t['min_s'], 4)} | {t['own_timer'] or '—'} | {t['verdict']} {t['why'][:80]} |")
    tv, te = R["ttft"].get(("vllm", 4096)), R["ttft"].get(("e4b", 4096))
    if tv and te and tv["median_s"] and te["median_s"]:
        o.append(f"- TTFT-4096 vllm / e4b = {tv['median_s'] / te['median_s']:.3f}")
    o.append("\n## 8. Resources (GB; reserved / peak allocated / demand or policy reservation; nvidia-smi max as fallback)")
    o.append("| engine | arm | B | verdict | reserved | peak alloc | demand | smi max | host RSS | cpu share | pcie | loadavg |")
    o.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in R["resources"]:
        o.append(f"| {r['engine']} | {r['arm']} | {r['B']} | {r['verdict']} | {f(r['reserved_gb'], 2)} | {f(r['peak_gb'], 2)} | {f(r['demand_gb'], 2)} | {f(r['smi_max_gb'], 2)} | {f(r['host_rss_gb'], 2)} | {f(r['cpu_share'])} | {r['pcie'] or '—'} | {r['loadavg'] or '—'} |")
    o.append("\n## 9. Energy (J/token = trapezoid of power.draw.instant over [start, stop] / tokens; board power)")
    o.append("| tag | engine | B | tokens | samples | seconds | mean W | J | J/token | basis / why |")
    o.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag, r in sorted(R["energy"].items()):
        o.append(f"| {tag} | {r['engine']} | {r['batch']} | {r['tokens']} | {r['n_samples']} | {f(r['seconds'], 2)} | {f(r['mean_w'], 1)} | {f(r['joules'], 1)} | {f(r['j_per_token'], 4)} | {r['basis'] or r['why']} |")
    o.append("\n## 10. Predictions (HOLD / REFUTED / UNREAD; only quoted positions and VALID rows are evidence)")
    for k in ("P1", "P2", "P3", "P4", "P5", "P5b", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14"):
        v, ev = R["predictions"].get(k, ("UNREAD", "not evaluated"))
        o.append(f"- {k}: **{v}** -- {ev}")
    o.append("\n## 11. Frontier (per B, every VALID arm on the sched axis: tok/s, d_bf16 served per text, bpw, resident GB)")
    o.append("| B | engine | arm | tok/s | d_bf16 wikitext | d_bf16 c4val1 | bpw | resident GB | stability | native |")
    o.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in R["frontier"]:
        o.append(f"| {r['B']} | {r['engine']} | {r['arm']} | {r['tok_s']:.1f} | {f(r['delta_bf16_wikitext'], 4) if r['delta_bf16_wikitext'] is not None else 'UNREAD'} | "
                 f"{f(r['delta_bf16_c4val1'], 4) if r['delta_bf16_c4val1'] is not None else 'UNREAD'} | {f(r['bpw'], 2)} ({r['bpw_basis']}) | {f(r['resident_gb'], 2)} | {r['stability']} | {r['native']} |")
    o.append("\nCSV: frontier-sc1-" + box + ".csv beside this file.")
    o.append("\nStated limits: one card class, one prompt distribution, one request shape, n = 2-3 draws, in-process loops on both sides of "
             "every engine-level ratio; the sched-vs-window method pair is reported, never averaged; nothing here moves a gate, a licence, "
             "a kernel default or a claim id.")
    return "\n".join(o)


def verdict_json(R):
    """A JSON-serialisable summary (tuple keys -> strings; receipts dropped)."""
    def key(k):
        return "/".join(str(x) for x in k) if isinstance(k, tuple) else str(k)
    arms = {key(k): {"verdict": a["verdict"], "summary": a["summary"], "draws": [{kk: vv for kk, vv in r.items() if kk != "rec"} for r in a["draws"]]} for k, a in R["arms"].items()}
    lic = dict(R["licence"])
    lic["rows"] = {n: {"used": i["arm"] is not None, "ppl": (i["arm"].ppl if i["arm"] else None), "why": i["why"], "mean_nll": i["mean_nll"], "k19": i.get("k19"), "k23": i.get("k23")} for n, i in lic["rows"].items()}
    comp = dict(R["comparability"])
    comp["served_vs_prefill"] = {key(k): v for k, v in comp["served_vs_prefill"].items()}
    out = {"lane": "SC1", "box": R["box"], "dir": R["dir"], "pack_fp": R["pack_fp"], "arms": arms, "licence": lic, "comparability": comp,
           "substitution": R["substitution"], "controls": R["controls"], "positions": R["positions"], "ttft": {key(k): v for k, v in R["ttft"].items()},
           "resources": R["resources"], "energy": R["energy"], "predictions": {k: {"verdict": v, "evidence": e} for k, (v, e) in R["predictions"].items()},
           "frontier": R["frontier"], "thresholds": {"close": CLOSE, "comparable": COMPARABLE, "k8_budget": K8_BUDGET, "draw_third": DRAW_THIRD,
                                                     "draw_unstable": DRAW_UNSTABLE, "substitution": SUBSTITUTION, "sameprompt_fired": SAMEPROMPT_FIRED,
                                                     "degraded_min": DEGRADED_MIN, "cross_box": CROSS_BOX}}
    return json.loads(json.dumps(out, default=str))


def write_outputs(R, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    md = os.path.join(out_dir, f"RESULTS-sc1-{R['box']}-generated.md")
    with open(md, "w") as fh:
        fh.write(render(R) + "\n")
    with open(os.path.join(out_dir, "verdict.json"), "w") as fh:
        json.dump(verdict_json(R), fh, indent=1)
    with open(os.path.join(out_dir, f"frontier-sc1-{R['box']}.csv"), "w") as fh:
        fh.write(frontier_csv(R["frontier"]))
    return md


# ------------------------------------------------------------------------------------------------- selftest
_W = {"wikitext": "w" * 64, "c4val1": "c" * 64}
_PSHA = {1: "1" * 64, 16: "6" * 64, "same": "5" * 64}
_FP = "fp123abc"
_NF4_PPL = {"wikitext": 6.3576, "c4val1": 12.0}
_LIC_PPL = {"wikitext": 6.37, "c4val1": 12.03}


def _tokens(B, n=LONG):
    return {str(i): list(range(n)) for i in range(B)}


def _k8(ppl, src, census=None, sha=None, steps=S):
    rec = {"k8": "ppl", "steps": steps, "tokens_scored": steps, "mean_nll": math.log(ppl), "ppl": ppl, "prompt_len": P, "ppl_source": src,
           "text_sha": sha or _W[src], "attn_compute": "fp8", "levers_env": {"E4B_INT4_GROUPED_SMALLM": "auto", "E4B_INT4_LEAN_GLUE": "auto"}}
    if census is not None:
        rec["census"] = {"kernels": census}
    return rec


def _nll(engine, src, nll, mode="served", variant=None, sha=None, rng=None):
    rec = {"engine": engine, "mode": mode, "mean_nll": nll, "ppl": math.exp(nll), "steps": S, "tokens_scored": S, "prompt_len": P, "status": "ok",
           "ppl_source": src, "wall_s": 1.0}
    sha = sha or _W[src]
    if engine == "llamacpp":
        rec.update(text_sha256=sha, targets={"first_index": (rng or SCORED_RANGE)[0], "last_index": (rng or SCORED_RANGE)[1]}, top1_agree_frac=0.61)
    else:
        rec.update(text_sha=sha)
        if engine in ("vllm", "sglang"):
            rec.update(scored_index_range=list(rng or SCORED_RANGE), top1_agreement=0.62)
            if mode == "served":
                rec.update(mode_label="served (partial-block, M<=16)" if engine == "vllm" else "served (T==1)", cache_predicate="PASS",
                           **({"suffix_histogram": {"513": 1, "8": 2047}} if engine == "vllm" else {"recomputed_suffix_histogram": {"513": 1, "1": 2047}}))
        elif rng:
            rec.update(scored_index_range=list(rng))
        if engine == "lmdeploy" and mode == "decode_prefix":
            rec["cache"] = {"cache_granularity_tokens": 64, "mean_recomputed_suffix": 32.5, "label": "partial-block (M <= 64), not T == 1"}
    return rec


def _e4b_window(B, ms, tok16=None):
    rec = {"b1d_timed": True, "b1d_loop": "graph", "n_steps": NEED_STEPS[B], "step_ms_clean": ms, "recompiles_in_window": 0,
           "tokens": (list(range(NEED_STEPS[1])) if B == 1 else {str(i): list(range(70 + i)) for i in range(16)}), "fuse_qkv": True}
    if B == 16:
        rec.update(batch=16, bv3=True, aggregate_tok_s=tok16)
    return rec


def _e4b_log(arm, B):
    base = arm.split("_")[0]
    lines = []
    if base in ("lic", "rtn", "int4"):
        lines += [(f"INT4EXP licensed artifact {_FP}: " if base == "lic" else "INT4EXP RTN: ") + "int4 expert stores on 48 layers",
                  "ATTNINT4 calibrated int4 attention: 192 projections", "fused q/k/v projections on 48 attention modules"]
    lines.append("B1D_TIMED_GRAPH steps=127" if B == 1 else "BV3_GRAPH batch=16 steps=70")
    return "\n".join(lines) + "\n"


def _census(B, fp=_FP, arm="lic_sched"):
    # the shape serve_paged.build_engine writes (sc1b-5090-1): Int4Linear counted after q/k/v fusion = 192 - 2 x 48 = 96, and the
    # pre-fusion lever count 192 on the arm's lever (licensed: calibrated; RTN/int4: rtn) with the other lever at 0 (A10)
    lic = arm.startswith("lic")
    return {"fuse_qkv_n": 48, "fuse_t1_glue_n": 48, "fuse_t1_glue_r2_n": [48, 48], "fuse_router_epilogue_n": 48, "int4_expert_layers": 48,
            "int4_attn_projections": 96, "attn_int4_calib_projections": 192 if lic else 0, "attn_int4_rtn_projections": 0 if lic else 192, "graph_status": {"1": "graph", "16": "graph"}, "levers_env": {"E4B_INT4_EXPECTED_FINGERPRINT": fp,
            "E4B_INT4_GROUPED_SMALLM": "auto"}, "fingerprint": fp, "moe_layers": 48}


def _p37(engine, arm, B, tok, same=False):
    rec = {"engine": engine, "arm": arm, "batch": B, "status": "ok", "decode_tok_s": tok, "decode_ms_per_step": round(1000.0 * B / tok, 3),
           "decode_tok_s_median": tok, "prompts_sha256": _PSHA["same" if same else B], "prompt_tokens": [PROMPT_TOKENS] * B, "tokens": _tokens(B),
           "model": {"vllm": "Qwen/Qwen3-30B-A3B-GPTQ-Int4", "sglang": "Qwen/Qwen3-30B-A3B-GPTQ-Int4", "llamacpp": "unsloth/Qwen3-30B-A3B-GGUF Q4_K_M",
                     "exl3": "turboderp/Qwen3-30B-A3B-exl3 4.0bpw", "lmdeploy": "Qwen/Qwen3-30B-A3B-GPTQ-Int4", "e4bsched": "Qwen/Qwen3-30B-A3B"}[engine]}
    if same:
        rec["sameprompt"] = {"rows_identical": True, "file_prompts_sha256": _PSHA["same"]}
    if engine == "e4bsched":
        rec.update(engine="e4b-sched", census=_census(B, arm=arm), graph_stats={"replays": 100})
    elif engine == "vllm":
        rec.update(engagement={"status": "grepped", "marlin_moe": True, "marlin_linear": True, "cudagraphs_captured": True},
                   resolved={"attention_backend": "FLASHINFER" if "fp8" in arm else "FLASH_ATTN", "enable_prefix_caching": False, "max_model_len": 2048,
                             "max_num_seqs": B, "quantization": "auto_awq" if arm.startswith("awq") else "auto_gptq", "kv_cache_dtype": "fp8" if "fp8" in arm else "auto",
                             "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32][: (7 if B == 16 else 2)]},
                   generation={"detokenize": "nodetok" not in arm},
                   mem_after_runs={"worker": {"memory_reserved": 29 * 2 ** 30, "max_memory_allocated": 26 * 2 ** 30}, "reservation": {"requested_bytes": int(0.9 * 32 * 2 ** 30)}})
    elif engine == "sglang":
        rec.update(engagement={"gptq_marlin_banner": True, "attention_backend": "flashinfer", "gencode": "-gencode=arch=compute_120f,code=sm_120f", "jit_target_tag": "sm120f"},
                   server_args={"disable_radix_cache": True, "max_running_requests": 16, "attention_backend": "flashinfer", "kv_cache_dtype": "auto"},
                   cached_tokens_long=[0] * B, server_mode="native" if arm.endswith("native") else "matched")
    elif engine == "llamacpp":
        rec.update(verdict="VALID", env={"GGML_CUDA_MMQ_PREC": None}, mode="slope")
    elif engine == "exl3":
        rec.update(meta={"eos_reason": ["max_new_tokens"] * B, "cached_tokens": [0] * B}, resolved={"bpw_layer": 4.02, "bpw_head": 6.0},
                   vram_peak_bytes={"mem_get_info_drop_bytes": 17 * 2 ** 30})
    elif engine == "lmdeploy":
        rec.update(meta={"status": ["FINISH"] * B, "cached_tokens": [0] * B}, resolved={"prefix_caching": False, "cuda_graphs": False, "kernel_family": "sm80 s16816 u4"},
                   vram_peak_bytes={"mem_get_info_drop_bytes": 19 * 2 ** 30, "nvidia_smi_used_max_bytes": 20 * 2 ** 30})
    return rec


def _llamacpp_log(B):
    return "load_tensors: offloaded 49/49 layers to GPU\nllama_context: flash_attn = enabled\n" + ("srv init: n_slots = 16\n" if B == 16 else "")


def _energy(tag, engine, B, watts, seconds, tokens=1024, t0=1_700_000_000.0):
    n = int(seconds / 0.05) + 1
    csv_text = "power.draw.instant [W], memory.used [MiB], utilization.gpu [%], clocks.sm [MHz], pcie.link.gen.current\n" + "".join(f"{watts:.2f} W, 20000 MiB, 99 %, 2800 MHz, 5\n" for _ in range(n))
    meta = {"tag": tag, "engine": engine, "batch": B, "tokens": tokens, "start_epoch": t0, "stop_epoch": t0 + seconds, "interval_ms": 50, "sampler_start_epoch": t0}
    return csv_text, meta


def _box_set(box="A"):
    """A complete, clean receipt set for one box -> (files, logs, texts)."""
    F, L, T = {}, {}, {}
    F["box.json"] = {"SC1_BOX": box, "host": {"cpu": "AMD EPYC 9655", "nproc": 16}}
    T["summary.txt"] = f"PACK {_FP}\n"
    T["versions.txt"] = "vllm 0.30.0\ne4b main\n"
    T["forensics.txt"] = "NVIDIA GeForce RTX 5090, 32607 MiB, 580.119.02\n"
    for B in BATCHES:
        F[f"prompts_b{B}.json"] = {"batch": B, "prompts_sha256": _PSHA[B], "rows_sha256": ["r"] * B, "row_step": 512}
    F["prompts_b16_same.json"] = {"batch": 16, "prompts_sha256": _PSHA["same"], "rows_sha256": ["r"] * 16}
    for src in TEXTS:
        F[f"k8_window_{src}.json"] = {"ids": [1, 2, 3], "text_sha": _W[src], "source": src}
        o = math.log(_NF4_PPL[src]) - 0.05
        F[f"oracle_{src}.json"] = {"k8": "ppl", "mean_nll": o, "ppl": math.exp(o), "text_sha": _W[src], "steps": S, "tokens_scored": S, "prompt_len": P,
                                   "ppl_source": src, "attn_path": "upstream-oracle"}
    win, sched = ANCHOR_WINDOW[box], ANCHOR_SCHED[box]
    wtok = {1: (4.0, None), 16: (16 / 1400 * 1000, 1400.0)} if box == "A" else {1: (3.96, None), 16: (16 / 1414 * 1000, 1414.0)}
    stok = {1: 230.0, 16: 1300.0} if box == "A" else {1: 232.0, 16: 1310.0}
    vtok = {1: 240.0, 16: 1500.0} if box == "A" else {1: 242.0, 16: 1512.0}
    for B in BATCHES:
        for r in (1, 2):
            F[f"e4b_{win}_b{B}_r{r}.json"] = _e4b_window(B, wtok[B][0] * (1 + 0.004 * (r - 1)), wtok[B][1] and wtok[B][1] * (1 - 0.004 * (r - 1)))
            L[f"run_e4b_{win}_b{B}_r{r}.log"] = _e4b_log(win, B)
            F[f"e4bsched_{sched}_b{B}_r{r}.json"] = _p37("e4bsched", sched, B, stok[B] * (1 + 0.005 * (r - 1)))
            F[f"vllm_gptq_graph_b{B}_r{r}.json"] = _p37("vllm", "gptq_graph", B, vtok[B] * (1 + 0.006 * (r - 1)))
    if box == "A":
        F["e4b_lic_b16_r1.json"]["census"] = {"kernels": [K19_KERNEL, "_gemv_int4_b32"]}
        F["e4b_lic_b16_r2.json"]["census"] = {"kernels": [K19_KERNEL, "_gemv_int4_b32"]}
        for B in BATCHES:
            for r in (1, 2):
                F[f"e4b_rtn_b{B}_r{r}.json"] = _e4b_window(B, 3.96 if B == 1 else 16 / 1414 * 1000, None if B == 1 else 1414.0)
                L[f"run_e4b_rtn_b{B}_r{r}.log"] = _e4b_log("rtn", B)
                F[f"vllm_gptq_fp8kv_b{B}_r{r}.json"] = _p37("vllm", "gptq_fp8kv", B, (238.0 if B == 1 else 1540.0) * (1 + 0.004 * (r - 1)))
            F[f"e4b_nf4_ctrl_b{B}_r1.json"] = _e4b_window(B, 10.0 if B == 1 else 16 / 480 * 1000, None if B == 1 else 480.0)
            L[f"run_e4b_nf4_ctrl_b{B}_r1.log"] = _e4b_log("nf4_ctrl", B)
            F[f"vllm_gptq_graph_nodetok_b{B}_r1.json"] = _p37("vllm", "gptq_graph_nodetok", B, 243.0 if B == 1 else 1508.0)
        F["e4b_lic_degraded_b16_r1.json"] = _e4b_window(16, 16 / 1272.7 * 1000, 1272.7)
        F["e4b_lic_degraded_b16_r1.json"]["census"] = {"kernels": ["_gemm_int4_b32_smallm", "_gemv_int4_b32"]}
        L["run_e4b_lic_degraded_b16_r1.log"] = _e4b_log("lic_degraded", 16)
        F["vllm_gptq_graph_sameprompt_b16_r1.json"] = _p37("vllm", "gptq_graph_sameprompt", 16, 2850.0, same=True)
        F["e4bsched_lic_sched_sameprompt_b16_r1.json"] = _p37("e4bsched", "lic_sched_sameprompt", 16, 2470.0, same=True)
        for src in TEXTS:
            F[f"k8_nf4_{src}.json"] = _k8(_NF4_PPL[src], src)
            F[f"k8_lic_auto_{src}.json"] = _k8(_LIC_PPL[src], src)
            F[f"k8_lic_1_{src}.json"] = _k8(_LIC_PPL[src] + 0.005, src, census=[K19_KERNEL, K23_KERNEL, "_gemv_int4_b32"])
            e_served = math.log(_LIC_PPL[src])
            F[f"nll_e4b_prefill_{src}.json"] = _nll("e4b", src, e_served - 0.0055, mode="prefill")
            v_served = e_served + (0.008 if src == "wikitext" else 0.015)
            F[f"nll_vllm_served_{src}.json"] = _nll("vllm", src, v_served, "served")
            F[f"nll_vllm_prefill_{src}.json"] = _nll("vllm", src, v_served - 0.001, "prefill")
            F[f"nll_vllm_fp8kv_served_{src}.json"] = _nll("vllm", src, v_served + 0.004, "served", variant="fp8kv")
            F[f"nll_vllm_fp8kv_prefill_{src}.json"] = _nll("vllm", src, v_served + 0.003, "prefill", variant="fp8kv")
        for engine, med in (("vllm", 0.42), ("e4b", 0.60)):
            F[f"ttft_{engine}_4096.json"] = {"status": "ok", "ttft_s_median": med, "ttft_s_min": med - 0.01, "prompt_len": 4096}
            F[f"ttft_{engine}_512.json"] = {"status": "ok", "ttft_s_median": med / 6, "ttft_s_min": med / 6 - 0.002, "prompt_len": 512}
        for tag, engine, B, w, sec in (("e4b_b1", "e4b", 1, 300.0, 4.0), ("vllm_b1", "vllm", 1, 320.0, 4.0), ("e4b_b16", "e4b", 16, 400.0, 0.8), ("vllm_b16", "vllm", 16, 420.0, 0.7)):
            T[f"energy_{tag}.csv"], F[f"energy_{tag}.json"] = _energy(tag, engine, B, w, sec)
    else:
        for src in TEXTS:
            F[f"k8_int4_{src}.json"] = _k8(_LIC_PPL[src] + 0.02, src)
            e_served = math.log(_LIC_PPL[src] + 0.02)
            F[f"nll_e4b_prefill_{src}.json"] = _nll("e4b", src, e_served - 0.01, mode="prefill")
            F[f"nll_vllm_served_{src}.json"] = _nll("vllm", src, e_served + 0.006, "served")
            F[f"nll_vllm_prefill_{src}.json"] = _nll("vllm", src, e_served + 0.005, "prefill")
        if box == "B":
            for engine, arm, t1, t16 in (("llamacpp", "q4km", 230.0, 700.0), ("exl3", "4bpw", 260.0, 1000.0), ("lmdeploy", "w4a16", 250.0, 1400.0)):
                for B, t in ((1, t1), (16, t16)):
                    for r in (1, 2):
                        F[f"{engine}_{arm}_b{B}_r{r}.json"] = _p37(engine, arm, B, t * (1 + 0.005 * (r - 1)))
                        if engine == "llamacpp":
                            L[f"run_llamacpp_{arm}_b{B}_r{r}.log"] = _llamacpp_log(B)
            F["llamacpp_iq4xs_b1_r1.json"] = _p37("llamacpp", "iq4xs", 1, 245.0)
            L["run_llamacpp_iq4xs_b1_r1.log"] = _llamacpp_log(1)
            for src in TEXTS:
                e_served = math.log(_LIC_PPL[src] + 0.02)
                F[f"nll_llamacpp_decode_{src}.json"] = _nll("llamacpp", src, e_served + 0.012, "decode")
                F[f"nll_llamacpp_prefill_{src}.json"] = _nll("llamacpp", src, e_served + 0.011, "prefill")
                F[f"nll_exl3_decode_{src}.json"] = _nll("exl3", src, e_served - 0.004, "decode")
                F[f"nll_exl3_prefill_{src}.json"] = _nll("exl3", src, e_served - 0.005, "prefill")
                F[f"nll_lmdeploy_decode_prefix_{src}.json"] = _nll("lmdeploy", src, e_served + 0.007, "decode_prefix")
                F[f"nll_lmdeploy_prefill_{src}.json"] = _nll("lmdeploy", src, e_served + 0.006, "prefill")
        if box == "C":
            for B, t in ((1, 236.0), (16, 1450.0)):
                for r in (1, 2):
                    F[f"sglang_gptq_matched_b{B}_r{r}.json"] = _p37("sglang", "gptq_matched", B, t * (1 + 0.004 * (r - 1)))
                F[f"sglang_gptq_native_b{B}_r1.json"] = _p37("sglang", "gptq_native", B, t * 1.05)
            for src in TEXTS:
                e_served = math.log(_LIC_PPL[src] + 0.02)
                F[f"nll_sglang_served_{src}.json"] = _nll("sglang", src, e_served + 0.009, "served")
                F[f"nll_sglang_prefill_{src}.json"] = _nll("sglang", src, e_served + 0.008, "prefill")
    return F, L, T


def _write_set(root, F, L, T):
    d = os.path.join(root, "sc1")
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    for name, obj in F.items():
        with open(os.path.join(d, name), "w") as fh:
            json.dump(obj, fh)
    for name, text in L.items():
        with open(os.path.join(d, "logs", name), "w") as fh:
            fh.write(text)
    for name, text in T.items():
        with open(os.path.join(d, name), "w") as fh:
            fh.write(text)
    return root


def selftest():
    tmp = tempfile.mkdtemp(prefix="sc1_reduce_selftest_")
    n = [0]

    def case(name, mut=None, box="A", check=None):
        """Build a clean set for `box`, apply `mut(F, L, T)`, reduce, print the case, run `check(R)`."""
        n[0] += 1
        F, L, T = _box_set(box)
        if mut:
            mut(F, L, T)
        root = _write_set(os.path.join(tmp, f"case{n[0]:02d}"), F, L, T)
        R = reduce_dir(root)
        line = check(R) if check else ""
        print(f"CASE {n[0]:02d} [{name}] -> {line}")
        return R

    def arm(R, engine, a, B):
        return R["arms"][(engine, a, B)]

    def pos(R, engine, a, B):
        return next(p for p in R["positions"]["rows"] if (p["engine"], p["arm"], p["B"]) == (engine, a, B))

    def expect_void(engine, a, B, needle):
        def chk(R):
            x = arm(R, engine, a, B)
            assert x["verdict"] == "VOID", (engine, a, B, x["verdict"], [r["why"] for r in x["draws"]])
            whys = " | ".join(r["why"] for r in x["draws"])
            assert needle.lower() in whys.lower(), (needle, whys)
            if engine in COMPARATOR_ENGINES:
                assert not pos(R, engine, a, B)["quoted"]
            return f"VOID: {whys[:150]}"
        return chk

    # 1. clean VALID position (box A): every arm VALID, both vllm positions quoted with point + interval, licence both legs, predictions
    def chk_clean(R):
        for k, a in R["arms"].items():
            assert a["verdict"] == "VALID", (k, a["verdict"], [r["why"] for r in a["draws"]])
        p1, p16 = pos(R, "vllm", "gptq_graph", 1), pos(R, "vllm", "gptq_graph", 16)
        assert p1["quoted"] and p16["quoted"], (p1["why"], p16["why"])
        exp1 = statistics.median([240.0, 240.0 * 1.006]) / statistics.median([230.0, 230.0 * 1.005])
        assert abs(p1["point"] - exp1) < 1e-9 and p1["n_cross"] == 4 and p1["min"] <= p1["point"] <= p1["max"], p1
        assert abs(p1["min"] - 240.0 / (230.0 * 1.005)) < 1e-9 and abs(p1["max"] - 240.0 * 1.006 / 230.0) < 1e-9
        assert p1["quality"]["bands"] == {"wikitext": "CLOSE", "c4val1": "COMPARABLE"} and not p1["quality"]["close"]
        assert abs(p1["window_ratio"] - statistics.median([240.0, 240.0 * 1.006]) / statistics.median([250.0, 1000 / (4.0 * 1.004)])) < 1e-9
        assert R["licence"]["verdict"] == "LICENSED-B1 LICENSED-B16" and R["licence"]["gate_impl"] in ("package", "file"), R["licence"]
        assert R["controls"]["degraded"]["verdict"] == "PASS", R["controls"]["degraded"]
        assert R["substitution"]["holds"] is True
        assert "fired" in R["controls"]["sameprompt"]["vllm"]["label"] and "fired" in R["controls"]["sameprompt"]["e4bsched"]["label"]
        assert abs(R["controls"]["method_pair"][1] - (statistics.median([250.0, 1000 / (4.0 * 1.004)]) / statistics.median([230.0, 230.0 * 1.005]) - 1)) < 1e-9
        Pp = {k: v for k, (v, _) in R["predictions"].items()}
        for k in ("P1", "P2", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P14"):
            assert Pp[k] == "HOLD", (k, R["predictions"][k])
        for k in ("P3", "P4", "P5", "P5b", "P13"):
            assert Pp[k] == "UNREAD", (k, R["predictions"][k])
        e = R["energy"]["e4b_b1"]
        assert abs(e["joules"] - 1200.0) < 1e-6 and abs(e["j_per_token"] - 1200.0 / 1024) < 1e-9, e
        fr = {(r["engine"], r["arm"], r["B"]) for r in R["frontier"]}
        assert ("vllm", "gptq_graph", 1) in fr and ("e4bsched", "lic_sched", 16) in fr and not any(k[0] == "e4b" for k in fr)
        md = render(R)
        assert "LICENSED-B1 LICENSED-B16" in md and "| vllm | gptq_graph | 1 | YES |" in md
        out = write_outputs(R, os.path.join(R["dir"], "out"))
        assert os.path.exists(out) and os.path.exists(os.path.join(R["dir"], "out", "verdict.json"))
        json.load(open(os.path.join(R["dir"], "out", "verdict.json")))
        return f"P1 {p1['point']:.3f} [{p1['min']:.3f}, {p1['max']:.3f}], P2 {p16['point']:.3f}; licence {R['licence']['verdict']}; all 14 predictions read as expected"
    case("clean VALID position (box A)", check=chk_clean)

    # VOID reasons, each once
    def m_sha(F, L, T):
        F["vllm_gptq_graph_b1_r1.json"]["prompts_sha256"] = "9" * 64
    case("VOID vllm prompts_sha differs from the prompt file", m_sha, check=expect_void("vllm", "gptq_graph", 1, "prompts_sha differs"))

    def m_tokens(F, L, T):
        F["vllm_gptq_graph_b16_r2.json"]["tokens"]["3"] = list(range(127))
    case("VOID vllm a row generated != 128 tokens", m_tokens, check=expect_void("vllm", "gptq_graph", 16, "generated != 128"))

    def m_ptoks(F, L, T):
        F["vllm_gptq_graph_b1_r1.json"]["prompt_tokens"] = [513]
    case("VOID vllm engine counted 513 prompt tokens (a BOS slipped in)", m_ptoks, check=expect_void("vllm", "gptq_graph", 1, "prompt tokens per row"))

    def m_marlin(F, L, T):
        F["vllm_gptq_graph_b1_r1.json"]["engagement"]["marlin_moe"] = False
    case("VOID vllm no Marlin WNA16 MoE line", m_marlin, check=expect_void("vllm", "gptq_graph", 1, "MARLIN"))

    def m_graph(F, L, T):
        F["vllm_gptq_graph_b16_r1.json"]["engagement"]["cudagraphs_captured"] = False
    case("VOID vllm graph arm without CUDA-graph capture", m_graph, check=expect_void("vllm", "gptq_graph", 16, "without CUDA-graph capture"))

    def m_nolog(F, L, T):
        F["vllm_gptq_graph_b16_r1.json"]["engagement"] = {"status": "no_log", "marlin_moe": None, "marlin_linear": None, "cudagraphs_captured": None}
    case("VOID vllm engagement not grepped (no log)", m_nolog, check=expect_void("vllm", "gptq_graph", 16, "no log was grepped"))

    def m_pc(F, L, T):
        F["vllm_gptq_graph_b1_r1.json"]["resolved"]["enable_prefix_caching"] = True
    case("VOID vllm prefix caching ON on a timed arm", m_pc, check=expect_void("vllm", "gptq_graph", 1, "prefix caching not OFF"))

    def m_attn(F, L, T):
        F["vllm_gptq_graph_b1_r1.json"]["resolved"]["attention_backend"] = "TRITON_ATTN"
    case("VOID vllm bf16-KV arm not on FLASH_ATTN", m_attn, check=expect_void("vllm", "gptq_graph", 1, "FLASH_ATTN"))

    def m_recomp(F, L, T):
        F["e4b_lic_b1_r1.json"]["recompiles_in_window"] = 2
    case("VOID e4b window dynamo recompiles inside the window", m_recomp, check=expect_void("e4b", "lic", 1, "recompiles_in_window=2"))

    def m_steps(F, L, T):
        F["e4b_lic_b16_r1.json"]["n_steps"] = 69
    case("VOID e4b window n_steps != 70", m_steps, check=expect_void("e4b", "lic", 16, "n_steps 69"))

    def m_fp(F, L, T):
        L["run_e4b_lic_b1_r1.log"] = L["run_e4b_lic_b1_r1.log"].replace(_FP, "fpOTHER")
    case("VOID e4b lic artifact fingerprint != PACK", m_fp, check=expect_void("e4b", "lic", 1, "fingerprint"))

    def m_fuse(F, L, T):
        L["run_e4b_lic_b1_r1.log"] = L["run_e4b_lic_b1_r1.log"].replace("fused q/k/v projections on 48 attention modules", "")
    case("VOID e4b lic window without the 48-module q/k/v fusion banner", m_fuse, check=expect_void("e4b", "lic", 1, "fused q/k/v"))

    def m_k19(F, L, T):
        F["e4b_lic_b16_r1.json"]["census"] = {"kernels": ["_gemv_int4_b32"]}
    case("VOID e4b lic_b16 K19 kernel absent from the census", m_k19, check=expect_void("e4b", "lic", 16, "K19 kernel absent"))

    def m_nocensus(F, L, T):
        del F["e4b_lic_b16_r2.json"]["census"]
    case("VOID e4b lic_b16 without a census (K19 unverifiable, never assumed)", m_nocensus, check=expect_void("e4b", "lic", 16, "no census"))

    def m_ctrl(F, L, T):
        L["run_e4b_nf4_ctrl_b1_r1.log"] += "INT4EXP RTN: int4 expert stores on 48 layers\n"
    case("VOID nf4 control shows int4 banners", m_ctrl, check=expect_void("e4b", "nf4_ctrl", 1, "control arm shows"))

    def m_sched_fuse(F, L, T):
        F["e4bsched_lic_sched_b1_r1.json"]["census"]["fuse_qkv_n"] = 47
    case("VOID sched census fuse_qkv_n != 48", m_sched_fuse, check=expect_void("e4bsched", "lic_sched", 1, "fuse_qkv_n 47"))

    def m_sched_prefusion(F, L, T):
        F["e4bsched_lic_sched_b1_r1.json"]["census"]["int4_attn_projections"] = 192
    case("VOID sched census counts 192 Int4Linear after fusing q/k/v on 48 layers (want 96: the fused qkv is not int4, or fusion did not run)",
         m_sched_prefusion, check=expect_void("e4bsched", "lic_sched", 1, "int4_attn_projections 192 != 96"))

    def m_sched_lever_short(F, L, T):
        F["e4bsched_lic_sched_b16_r1.json"]["census"]["attn_int4_calib_projections"] = 191
    case("VOID sched census: the calibrated attention lever converted 191 of 192 projections", m_sched_lever_short,
         check=expect_void("e4bsched", "lic_sched", 16, "attn_int4_calib_projections 191"))

    def m_sched_lever_missing(F, L, T):
        del F["e4bsched_lic_sched_b1_r2.json"]["census"]["attn_int4_calib_projections"]
    case("VOID sched census without the pre-fusion lever count (never assumed)", m_sched_lever_missing,
         check=expect_void("e4bsched", "lic_sched", 1, "attn_int4_calib_projections None"))

    def m_sched_wrong_lever(F, L, T):
        c = F["e4bsched_lic_sched_b16_r2.json"]["census"]
        c["attn_int4_calib_projections"], c["attn_int4_rtn_projections"] = 0, 192
    case("VOID sched census: a licensed arm ran the RTN attention lever", m_sched_wrong_lever,
         check=expect_void("e4bsched", "lic_sched", 16, "attn_int4_calib_projections 0"))

    def m_sched_graph(F, L, T):
        F["e4bsched_lic_sched_b16_r1.json"]["census"]["graph_status"] = {"1": "graph", "16": "eager: capture failed"}
    case("VOID sched DECODE_GRAPH bucket=16 not captured", m_sched_graph, check=expect_void("e4bsched", "lic_sched", 16, "DECODE_GRAPH bucket=16"))

    def m_sched_fp(F, L, T):
        F["e4bsched_lic_sched_b1_r2.json"]["census"]["fingerprint"] = "fpOTHER"
        F["e4bsched_lic_sched_b1_r2.json"]["census"]["levers_env"]["E4B_INT4_EXPECTED_FINGERPRINT"] = "fpOTHER"
    case("VOID sched census fingerprint != PACK", m_sched_fp, check=expect_void("e4bsched", "lic_sched", 1, "fingerprint"))

    def m_sched_folds(F, L, T):
        F["e4bsched_lic_sched_b1_r1.json"]["census"]["fuse_t1_glue_r2_n"] = [48, 0]
    case("VOID sched census a fold count is 0", m_sched_folds, check=expect_void("e4bsched", "lic_sched", 1, "fold counts"))

    def m_llama(F, L, T):
        L["run_llamacpp_q4km_b1_r1.log"] = L["run_llamacpp_q4km_b1_r1.log"].replace("49/49", "48/49")
    case("VOID llama.cpp offloaded 48/49 layers", m_llama, box="B", check=expect_void("llamacpp", "q4km", 1, "offloaded 48/49"))

    def m_llama_prec(F, L, T):
        F["llamacpp_q4km_b16_r1.json"]["env"]["GGML_CUDA_MMQ_PREC"] = "1"
    case("VOID llama.cpp GGML_CUDA_MMQ_PREC set", m_llama_prec, box="B", check=expect_void("llamacpp", "q4km", 16, "GGML_CUDA_MMQ_PREC"))

    def m_llama_void(F, L, T):
        F["llamacpp_q4km_b1_r2.json"].update(verdict="VOID", void_reason="row 0: generated 7 / tokens_predicted 7 != 128 (stop_type='eos')")
    case("VOID llama.cpp driver's own VOID (a row stopped early)", m_llama_void, box="B", check=expect_void("llamacpp", "q4km", 1, "tokens_predicted 7 != 128"))

    def m_sgl(F, L, T):
        F["sglang_gptq_matched_b1_r1.json"]["engagement"]["gencode"] = "-gencode=arch=compute_90,code=sm_90"
    case("VOID sglang Marlin MoE JIT not built for sm_120", m_sgl, box="C", check=expect_void("sglang", "gptq_matched", 1, "sm_120f"))

    def m_sgl_cached(F, L, T):
        F["sglang_gptq_matched_b16_r1.json"]["cached_tokens_long"] = [0] * 15 + [512]
    case("VOID sglang cached_tokens > 0 on a radix-off arm", m_sgl_cached, box="C", check=expect_void("sglang", "gptq_matched", 16, "cached_tokens"))

    def m_exl3(F, L, T):
        F["exl3_4bpw_b16_r1.json"]["meta"]["cached_tokens"] = [0] * 15 + [256]
    case("VOID exl3 a prompt page was reused", m_exl3, box="B", check=expect_void("exl3", "4bpw", 16, "prompt page"))

    def m_lmd(F, L, T):
        F["lmdeploy_w4a16_b1_r1.json"]["resolved"]["prefix_caching"] = True
    case("VOID lmdeploy prefix caching ON", m_lmd, box="B", check=expect_void("lmdeploy", "w4a16", 1, "prefix caching ON"))

    # non-readings
    def m_statuses(F, L, T):
        F["lmdeploy_w4a16_b16_r1.json"] = {"status": "refused", "reason": "TurboMind: no kernel for sm_120"}
        F["lmdeploy_w4a16_b16_r2.json"] = {"status": "refused", "reason": "TurboMind: no kernel for sm_120"}
        F["exl3_4bpw_b1_r1.json"] = {"status": "oom", "reason": "CUDA out of memory"}
        F["llamacpp_q4km_b16_r2.json"] = {"status": "harness_error", "error": "Traceback"}
        del F["exl3_4bpw_b16_r2.json"]
    def chk_statuses(R):
        assert arm(R, "lmdeploy", "w4a16", 16)["verdict"] == "UNSUPPORTED"
        assert arm(R, "exl3", "4bpw", 1)["verdict"] == "OOM"
        assert arm(R, "llamacpp", "q4km", 16)["verdict"] == "HARNESS_ERROR"
        assert arm(R, "exl3", "4bpw", 16)["summary"]["status"] == "SINGLE" and not pos(R, "exl3", "4bpw", 16)["quoted"]
        v, ev = R["predictions"]["P5b"]
        assert v == "HOLD" and "UNSUPPORTED" in ev, (v, ev)                  # P5b's stated alternative
        assert R["predictions"]["P5"][0] == "UNREAD"
        return "UNSUPPORTED / OOM / HARNESS_ERROR / SINGLE as named; P5b HOLD by its UNSUPPORTED alternative"
    case("non-readings: UNSUPPORTED, OOM, HARNESS_ERROR, a missing second draw = SINGLE", m_statuses, box="B", check=chk_statuses)

    # UNSTABLE and the third-draw rule
    def m_unstable(F, L, T):
        F["vllm_gptq_graph_b1_r2.json"]["decode_tok_s"] = 240.0 * 1.06
    def chk_unstable(R):
        a = arm(R, "vllm", "gptq_graph", 1)
        assert a["verdict"] == "UNSTABLE" and a["summary"]["status"] == "UNSTABLE" and a["all_valid"]
        assert not pos(R, "vllm", "gptq_graph", 1)["quoted"] and R["predictions"]["P1"][0] == "UNREAD"
        return f"UNSTABLE spread {100 * a['summary']['spread']:.1f} % > 5 %; position not quoted; P1 UNREAD"
    case("UNSTABLE: two draws 6 % apart", m_unstable, check=chk_unstable)

    def m_third_missing(F, L, T):
        F["vllm_gptq_graph_b16_r2.json"]["decode_tok_s"] = 1500.0 * 1.04
    def chk_third_missing(R):
        a = arm(R, "vllm", "gptq_graph", 16)
        assert a["summary"]["status"] == "THIRD_DRAW_REQUIRED" and a["verdict"] == "VALID" and not a["summary"]["usable"]
        assert not pos(R, "vllm", "gptq_graph", 16)["quoted"]
        return "two draws 4 % apart -> THIRD_DRAW_REQUIRED, no position"
    case("third draw rule: 4 % apart without a third draw", m_third_missing, check=chk_third_missing)

    def m_third_landed(F, L, T):
        F["vllm_gptq_graph_b16_r2.json"]["decode_tok_s"] = 1500.0 * 1.04
        F["vllm_gptq_graph_b16_r3.json"] = _p37("vllm", "gptq_graph", 16, 1500.0 * 1.01)
    def chk_third_landed(R):
        a = arm(R, "vllm", "gptq_graph", 16)
        assert a["summary"]["status"] == "STABLE" and a["summary"]["n"] == 3 and abs(a["summary"]["point"] - 1500.0 * 1.01) < 1e-9
        p = pos(R, "vllm", "gptq_graph", 16)
        assert p["quoted"] and p["n_cross"] == 6
        return f"third draw landed: point = median of three = {a['summary']['point']:.1f}; interval over 6 cross-draw ratios"
    case("third draw rule: the third draw lands, point = median of three", m_third_landed, check=chk_third_landed)

    # licence
    def m_qfail(F, L, T):
        F["k8_lic_auto_wikitext.json"] = _k8(_NF4_PPL["wikitext"] + 0.06, "wikitext")
    def chk_qfail(R):
        assert R["licence"]["verdict"] == "LICENSED-B16 QUALITY_FAIL" and R["licence"]["quality_fail"]
        for B in BATCHES:
            assert not pos(R, "vllm", "gptq_graph", B)["quoted"]
        assert any("RTN rows either" in w for w in pos(R, "vllm", "gptq_graph", 16)["why"])
        assert R["predictions"]["P9"][0] == "REFUTED"
        return "QUALITY_FAIL on the auto rows (+0.06 ppl > 0.05): no position at either B, none from the RTN rows; P9 REFUTED"
    case("QUALITY_FAIL licence (auto row over budget)", m_qfail, check=chk_qfail)

    def m_improve(F, L, T):
        F["k8_lic_auto_c4val1.json"] = _k8(_NF4_PPL["c4val1"] - 0.01, "c4val1")
    def chk_improve(R):
        assert R["licence"]["b1"] is False and any("improvement needs" in ln for ln in R["licence"]["lines"])
        return "an improvement on c4val1 only (the calibration domain) is not corroborated -> FAIL by the clause"
    case("QUALITY_FAIL licence: improvement on the calibration text only", m_improve, check=chk_improve)

    def m_k23(F, L, T):
        F["k8_lic_1_wikitext.json"]["census"] = {"kernels": [K19_KERNEL]}
    def chk_k23(R):
        assert R["licence"]["b1"] is True and R["licence"]["b16"] is None and R["licence"]["verdict"] == "LICENSED-B1"
        assert not pos(R, "vllm", "gptq_graph", 16)["quoted"] and pos(R, "vllm", "gptq_graph", 1)["quoted"]
        return "=1 row without K23 in its census -> LICENSED-B16 unread; B=16 position not quoted, B=1 still quoted"
    case("licence: the =1 row's census lacks K23 (LICENSED-B16 unread)", m_k23, check=chk_k23)

    def m_k8sha(F, L, T):
        F["k8_nf4_c4val1.json"]["text_sha"] = "z" * 64
    def chk_k8sha(R):
        assert R["licence"]["verdict"] == "UNREAD" and "text_sha" in R["licence"]["rows"]["nf4_c4val1"]["why"]
        return "K8 row off its window file's sha -> not used; licence UNREAD"
    case("licence: a K8 row whose text_sha differs from the window file is VOID", m_k8sha, check=chk_k8sha)

    # comparability
    def m_notcomp(F, L, T):
        F["nll_vllm_served_wikitext.json"]["mean_nll"] += 0.03
    def chk_notcomp(R):
        p = pos(R, "vllm", "gptq_graph", 1)
        assert p["quality"]["bands"]["wikitext"] == "NOT_COMPARABLE" and not p["quoted"]
        row = next(q for q in R["comparability"]["rows"] if (q["engine"], q.get("variant"), q["shape"], q["src"]) == ("vllm", None, "served", "wikitext"))
        assert abs(row["delta_pair"] - 0.038) < 1e-9
        return f"delta_pair {row['delta_pair']:+.4f} > 0.02 -> NOT_COMPARABLE; position not quoted"
    case("NOT_COMPARABLE served-shape row blocks the position", m_notcomp, check=chk_notcomp)

    def m_bounds(F, L, T):
        e = math.log(_LIC_PPL["wikitext"])
        F["nll_vllm_served_wikitext.json"]["mean_nll"] = e + 0.0095
        F["nll_vllm_served_c4val1.json"]["mean_nll"] = math.log(_LIC_PPL["c4val1"]) + 0.02
    def chk_bounds(R):
        p = pos(R, "vllm", "gptq_graph", 1)
        assert p["quality"]["bands"] == {"wikitext": "CLOSE", "c4val1": "COMPARABLE"} and p["quoted"], p["quality"]
        return "delta_pair exactly 0.0095 -> CLOSE, exactly 0.02 -> COMPARABLE (inclusive); quoted"
    case("comparability bands at the boundaries", m_bounds, check=chk_bounds)

    def m_qsha(F, L, T):
        F["nll_vllm_served_c4val1.json"]["text_sha"] = "z" * 64
        F["nll_vllm_prefill_wikitext.json"]["scored_index_range"] = [512, 2559]
    def chk_qsha(R):
        rows = {(q["shape"], q["src"]): q for q in R["comparability"]["rows"] if q["engine"] == "vllm" and q.get("variant") is None}
        assert rows[("served", "c4val1")]["verdict"] == "VOID" and "text_sha" in rows[("served", "c4val1")]["why"]
        assert rows[("prefill", "wikitext")]["verdict"] == "VOID" and "scored_index_range" in rows[("prefill", "wikitext")]["why"]
        assert not pos(R, "vllm", "gptq_graph", 1)["quoted"] and R["predictions"]["P6"][0] == "UNREAD"
        return "quality rows VOID by sha / by scored range -> band UNREAD, position not quoted, P6 UNREAD"
    case("quality row VOID: text_sha off the window; scored range != [513, 2560]", m_qsha, check=chk_qsha)

    def m_oracle(F, L, T):
        del F["oracle_wikitext.json"]
    def chk_oracle(R):
        rows = [q for q in R["comparability"]["rows"] if q["src"] == "wikitext"]
        assert rows and all(q["delta_bf16"] is None for q in rows) and all("oracle" in q["bf16_note"] for q in rows)
        assert R["comparability"]["e4b_served"]["wikitext"]["delta_bf16"] is None
        assert pos(R, "vllm", "gptq_graph", 1)["quoted"]                       # d_bf16 is a column, not a gate
        fr = next(r for r in R["frontier"] if (r["engine"], r["arm"], r["B"]) == ("vllm", "gptq_graph", 1))
        assert fr["delta_bf16_wikitext"] is None and fr["delta_bf16_c4val1"] is not None
        md = render(R)
        assert "UNREAD" in md and "every d_bf16 on wikitext is UNREAD, never 0" in md
        assert R["predictions"]["P8"][0] == "UNREAD"
        return "d_bf16 UNREAD (not 0) on every wikitext row and in the frontier; the position still reads (d_bf16 is a column); P8 UNREAD"
    case("missing oracle: d_bf16 UNREAD, never 0", m_oracle, check=chk_oracle)

    def m_oracle_sha(F, L, T):
        F["oracle_c4val1.json"]["text_sha"] = "z" * 64
    def chk_oracle_sha(R):
        assert R["comparability"]["oracle"]["c4val1"]["verdict"] == "VOID"
        assert all(q["delta_bf16"] is None for q in R["comparability"]["rows"] if q["src"] == "c4val1")
        return "oracle off its window sha -> VOID -> d_bf16 UNREAD on c4val1"
    case("oracle whose text_sha differs from the window is VOID", m_oracle_sha, check=chk_oracle_sha)

    def m_ood(F, L, T):
        F["nll_vllm_served_wikitext.json"]["mean_nll"] = math.log(_NF4_PPL["wikitext"]) - 0.08
    def chk_ood(R):
        row = next(q for q in R["comparability"]["rows"] if (q["engine"], q.get("variant"), q["shape"], q["src"]) == ("vllm", None, "served", "wikitext"))
        assert row["ood_flag"] and row["delta_bf16"] < -OOD_FLATTERY
        return f"a 4-bit arm {row['delta_bf16']:+.3f} nats BELOW the bf16 oracle -> OOD-flattery flag (instrument fault, stated)"
    case("OOD-flattery flag when the oracle reads above a 4-bit arm by > 0.02", m_ood, check=chk_ood)

    # substitution licence
    def m_sub(F, L, T):
        for r in (1, 2):
            F[f"e4b_rtn_b16_r{r}.json"]["aggregate_tok_s"] = 1400.0 * 1.03
    def chk_sub(R):
        s = R["substitution"]
        assert s["holds"] is False and abs(s["gap"][16] - (1400.0 * (1 + (1 - 0.004) / 2 * 0 + 0) / (1400.0 * 1.03) - 1)) < 0.01, s
        assert R["predictions"]["P12"][0] == "REFUTED"
        return f"|lic - rtn| at B=16 = {pct(s['gap'][16])} > 2 % -> substitution FAILS; P12 REFUTED"
    case("substitution licence failing (rtn 3 % faster than lic at B=16)", m_sub, check=chk_sub)

    # SAMEPROMPT
    def m_same(F, L, T):
        F["vllm_gptq_graph_sameprompt_b16_r1.json"]["decode_tok_s"] = 1500.0 * 0.98
    def chk_same(R):
        r = R["controls"]["sameprompt"]["vllm"]
        assert "did not fire" in r["label"] and arm(R, "vllm", "gptq_graph", 16)["verdict"] == "VALID" and pos(R, "vllm", "gptq_graph", 16)["quoted"]
        assert R["predictions"]["P7"][0] == "REFUTED"
        return f"vllm SAMEPROMPT x{r['ratio']:.2f}: {r['label']} -- a label, the distinct rows stay VALID and quoted; P7 REFUTED"
    case("SAMEPROMPT label: detector did not fire (no VOID)", m_same, check=chk_same)

    def m_same_sha(F, L, T):
        F["vllm_gptq_graph_sameprompt_b16_r1.json"]["sameprompt"]["rows_identical"] = False
    case("VOID SAMEPROMPT arm that does not say rows_identical", m_same_sha, check=expect_void("vllm", "gptq_graph_sameprompt", 16, "rows_identical"))

    # degraded control
    def m_deg(F, L, T):
        F["e4b_lic_degraded_b16_r1.json"]["aggregate_tok_s"] = 1400.0 / 1.02
    def chk_deg(R):
        dg = R["controls"]["degraded"]
        assert dg["verdict"] == "FAIL" and "< 1.05" in dg["why"]
        assert arm(R, "e4b", "lic", 16)["verdict"] == "VOID" and arm(R, "e4bsched", "lic_sched", 16)["verdict"] == "VOID"
        assert not pos(R, "vllm", "gptq_graph", 16)["quoted"] and pos(R, "vllm", "gptq_graph", 1)["quoted"]
        assert R["predictions"]["P2"][0] == "UNREAD" and R["predictions"]["P1"][0] == "HOLD"
        return f"degraded only x{dg['ratio']:.3f} -> FAIL -> e4b lic_b16 + lic_sched_b16 VOID; B=16 position not quoted, B=1 unaffected"
    case("degraded FAIL (not 1.05x slower) VOIDs e4b's B=16 rows", m_deg, check=chk_deg)

    def m_deg_k19(F, L, T):
        F["e4b_lic_degraded_b16_r1.json"]["census"] = {"kernels": [K19_KERNEL]}
    def chk_deg_k19(R):
        dg = R["controls"]["degraded"]
        assert dg["verdict"] == "FAIL" and arm(R, "e4bsched", "lic_sched", 16)["verdict"] == "VOID"
        return "degraded census still shows K19 -> FAIL (the lever did not disengage) -> B=16 rows VOID"
    case("degraded FAIL (K19 present in the degraded census)", m_deg_k19, check=chk_deg_k19)

    # predictions' failing legs
    def m_p8(F, L, T):
        F["vllm_gptq_fp8kv_b16_r1.json"]["decode_tok_s"] = 1490.0
        F["vllm_gptq_fp8kv_b16_r2.json"]["decode_tok_s"] = 1492.0
    def chk_p8(R):
        v, ev = R["predictions"]["P8"]
        assert v == "REFUTED" and "B=16" in ev
        return f"fp8kv slower than kv auto at B=16 -> P8 REFUTED ({ev[:60]})"
    case("P8 refuted: fp8-KV not faster at B=16", m_p8, check=chk_p8)

    def m_p14(F, L, T):
        F["vllm_gptq_graph_nodetok_b16_r1.json"]["decode_tok_s"] = 1500.0 * 1.02
    def chk_p14(R):
        assert R["predictions"]["P14"][0] == "REFUTED"
        return "nodetok 2 % off the matched arm at B=16 (> 1 %) -> P14 REFUTED"
    case("P14 refuted: nodetok asymmetry > 1 % at B=16", m_p14, check=chk_p14)

    def m_p10(F, L, T):
        F["ttft_vllm_4096.json"]["ttft_s_median"] = 0.70
    def chk_p10(R):
        v, ev = R["predictions"]["P10"]
        assert v == "REFUTED" and "1.167" in ev
        return "TTFT-4096 vllm/e4b 1.167 > 1.0 -> P10 REFUTED"
    case("P10 refuted: TTFT-4096 ratio above 1.0", m_p10, check=chk_p10)

    def m_p11(F, L, T):
        T["energy_vllm_b1.csv"], F["energy_vllm_b1.json"] = _energy("vllm_b1", "vllm", 1, 360.0, 4.0)
    def chk_p11(R):
        assert R["predictions"]["P11"][0] == "REFUTED" and abs(R["energy"]["vllm_b1"]["j_per_token"] / R["energy"]["e4b_b1"]["j_per_token"] - 1.2) < 1e-9
        return "B=1 J/token vllm/e4b = 1.20 (> 15 %) -> P11 REFUTED"
    case("P11 refuted: B=1 J/token differs by 20 %", m_p11, check=chk_p11)

    def m_energy_bad(F, L, T):
        del T["energy_e4b_b16.csv"]
        F["energy_vllm_b16.json"].pop("tokens")
    def chk_energy_bad(R):
        assert R["energy"]["e4b_b16"]["j_per_token"] is None and R["energy"]["e4b_b16"]["why"] == "csv missing"
        assert R["energy"]["vllm_b16"]["j_per_token"] is None and "tokens" in R["energy"]["vllm_b16"]["why"]
        assert R["predictions"]["P11"][0] == "UNREAD"
        return "energy without a csv / without tokens -> J/token None, P11 UNREAD (never a number)"
    case("energy: missing csv and missing token count read as None", m_energy_bad, check=chk_energy_bad)

    # boxes B and C on their own: anchors int4, positions, substitution unread without box A
    def chk_boxb(R):
        assert R["box"] == "B" and R["positions"]["anchor"] == "e4bsched/int4_sched"
        for engine, a in (("vllm", "gptq_graph"), ("llamacpp", "q4km"), ("exl3", "4bpw"), ("lmdeploy", "w4a16")):
            for B in BATCHES:
                assert pos(R, engine, a, B)["quoted"], (engine, a, B, pos(R, engine, a, B)["why"])
        assert R["substitution"]["holds"] is None and R["licence"]["verdict"] == "UNREAD"
        Pp = {k: v for k, (v, _) in R["predictions"].items()}
        assert Pp["P4"] == "HOLD" and Pp["P5"] == "HOLD" and Pp["P5b"] == "HOLD" and Pp["P12"] == "UNREAD" and Pp["P1"] == "UNREAD", Pp
        assert not pos(R, "llamacpp", "iq4xs", 1)["quoted"]                 # single-draw speed-only row
        return "box B: four comparators quoted against int4_sched at both B; P4/P5/P5b HOLD; substitution + licence UNREAD without box A"
    case("box B alone: positions against the int4_sched anchor", box="B", check=chk_boxb)

    def chk_boxc(R):
        assert pos(R, "sglang", "gptq_matched", 1)["quoted"] and pos(R, "sglang", "gptq_matched", 16)["quoted"]
        assert pos(R, "sglang", "gptq_native", 1)["native"] and not pos(R, "sglang", "gptq_native", 1)["quoted"]
        assert R["predictions"]["P3"][0] == "HOLD", R["predictions"]["P3"]
        return "box C: sglang matched quoted at both B, native row labelled (single draw, not quoted); P3 HOLD"
    case("box C alone: SGLang against its own anchors", box="C", check=chk_boxc)

    def m_p3(F, L, T):
        for r in (1, 2):
            F[f"sglang_gptq_matched_b1_r{r}.json"]["decode_tok_s"] = 200.0 * (1 + 0.004 * (r - 1))
    def chk_p3(R):
        v, ev = R["predictions"]["P3"]
        assert v == "REFUTED" and "B=1" in ev
        return "sglang 17 % off the vllm anchor at B=1 (> 10 %) -> P3 REFUTED"
    case("P3 refuted: SGLang more than 10 % from the vLLM anchor at B=1", m_p3, box="C", check=chk_p3)

    def m_p3_unsup(F, L, T):
        for B in BATCHES:
            for r in (1, 2):
                F[f"sglang_gptq_matched_b{B}_r{r}.json"] = {"status": "unsupported", "reason": "Failed to build JIT module sgl_kernel_jit_moe_wna16_marlin"}
    def chk_p3_unsup(R):
        assert R["predictions"]["P3"][0] == "HOLD" and "UNSUPPORTED" in R["predictions"]["P3"][1]
        return "sglang UNSUPPORTED (JIT did not build) -> P3's stated alternative -> HOLD, with the reason"
    case("P3: SGLang UNSUPPORTED is the prediction's stated alternative", m_p3_unsup, box="C", check=chk_p3_unsup)

    # cross-box
    n[0] += 1
    roots = []
    for box in ("A", "B", "C"):
        F, L, T = _box_set(box)
        roots.append(_write_set(os.path.join(tmp, f"case{n[0]:02d}_{box}"), F, L, T))
    X = cross_box(roots)
    assert X["p13"][0] == "HOLD", X["p13"]
    assert set(X["boxes"]) == {"A", "B", "C"} and all(X["agree"][B]["within"] for B in BATCHES)
    RB = next(R for R in X["reductions"] if R["box"] == "B")
    assert RB["substitution"]["holds"] is True and RB["predictions"]["P13"][0] == "HOLD" and pos(RB, "vllm", "gptq_graph", 1).get("substitution", {}).get("holds") is True
    assert "licensed by box A" in _subst_label(RB["substitution"])
    print(f"CASE {n[0]:02d} [cross-box: three boxes' anchor ratios within 5 %] -> P13 HOLD; " + "; ".join(X["lines"][2:4]))
    n[0] += 1
    F, L, T = _box_set("B")
    for r in (1, 2):
        F[f"vllm_gptq_graph_b16_r{r}.json"]["decode_tok_s"] = 1512.0 * 1.08 * (1 + 0.006 * (r - 1))
    rootB = _write_set(os.path.join(tmp, f"case{n[0]:02d}_B"), F, L, T)
    F, L, T = _box_set("A")
    for r in (1, 2):
        F[f"e4b_rtn_b1_r{r}.json"]["step_ms_clean"] = 4.0 / 1.03
    rootA = _write_set(os.path.join(tmp, f"case{n[0]:02d}_A"), F, L, T)
    X = cross_box([rootA, rootB])
    assert X["p13"][0] == "REFUTED" and X["agree"][16]["within"] is False and X["agree"][1]["within"] is True, X["agree"]
    RB = next(R for R in X["reductions"] if R["box"] == "B")
    assert RB["substitution"]["holds"] is False and "RTN speed" in _subst_label(RB["substitution"])
    print(f"CASE {n[0]:02d} [cross-box: box B's B=16 anchor ratio 8 % off box A's; box A's substitution fails] -> P13 REFUTED; box B anchors labelled '{_subst_label(RB['substitution'])}'")
    print(f"REDUCE SELFTEST OK cases={n[0]} dir={tmp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="*", help="one run dir (or two/three with --cross-box)")
    ap.add_argument("--box", default=None, help="A | B | C (default: box.json's SC1_BOX, else from the dir name, else A)")
    ap.add_argument("--out-dir", default=None, help="where RESULTS-sc1-<box>-generated.md / verdict.json go (default: the run dir)")
    ap.add_argument("--box-a", default=None, help="box A's run dir, to read the substitution licence onto a box B/C reduction")
    ap.add_argument("--cross-box", action="store_true", help="two or three run dirs: the anchor-ratio comparison (P13) and the substitution licence applied")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.cross_box:
        if len(a.dirs) < 2:
            ap.error("--cross-box takes two or three run dirs")
        X = cross_box(a.dirs)
        print("# SC1 cross-box anchor ratios (vllm/gptq_graph over the box's e4b sched anchor; quoted positions only)")
        print("\n".join(X["lines"]))
        print(f"\nP13: {X['p13'][0]} -- {X['p13'][1]}")
        for R in X["reductions"]:
            md = write_outputs(R, a.out_dir or R["dir"])
            print(f"wrote {md}")
        return 0
    if len(a.dirs) != 1:
        ap.error("one run dir (or --cross-box with two or three, or --selftest)")
    R = reduce_dir(a.dirs[0], a.box, a.box_a)
    md = write_outputs(R, a.out_dir or R["dir"])
    print(render(R))
    print(f"\nwrote {md} and verdict.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
