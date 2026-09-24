#!/usr/bin/env python3
"""p68_probe.py -- lane P68 (experts4bit-qlora#725): which part of the attention makes a verify or a prefill differ from
T = 1 decode, and is the difference over the shipped bar once enough positions are read?

Lane P63 found that no position of a 16- or 17-row verify, or of a 160-row prefill, is bit-equal to incremental T = 1
decode, and that every first difference is in ATTENTION at layer 0 -- never in the experts, whose routes it registered as
row-exact. This probe keeps P63's instrument (``p63_probe``'s stack build, site capture and backends; ``p63_compare``'s
arithmetic) and adds two things (P68-PREREG.md, Instrument):

1. **Forcing arms (the ablation).** A forcing makes one component compute a multi-row call the way T = 1 decode computes
   it, one row at a time, so every row takes the T = 1 kernel with the T = 1 shapes:
   - ``proj``: the attention projections (q/k/v/o, or the fused ``qkv_proj``) called once per row;
   - ``core``: the attention core -- the SDPA entry transformers dispatches to -- called once per QUERY row, over
     exactly the key/value prefix that row's T = 1 decode sees, with no mask;
   - ``router``, ``lm_head``, ``norms``: the router module, the LM head and every RMSNorm module, once per row.
   A forcing is inert at T = 1 by construction (a one-row call passes straight through), so each arm's control is the
   unforced control -- checked bit for bit (gate G0). Every wrapped call is counted (the census), so a forcing that did
   not engage cannot pass for one that did. ``fp32red`` is the other toggle: torch's bf16 reduced-precision cuBLAS
   reduction off for the arm.
2. **The size reading.** The served configuration (default grouping), unforced, on the HF and the paged backends, over
   long rows (default 4 x 512 tokens) with a verify window every 32 positions, so KL and top-1 carry a confidence
   interval the reducer can read against the bar. P63's OVER-BAR cells were over on top-1 over 64-160 positions, where
   one flip moves top-1 by 1.6 %.

Nothing here changes a default. The receipt is ``<out>/p68_arm.json``; ``p68_reduce.py`` applies the registration.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "p63"), os.path.join(HERE, "..", "p44")):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

import p63_compare as C  # noqa: E402

FORCE_KINDS = ("proj", "core", "router", "lm_head", "norms")
PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj")

#: The registered arms (P68-PREREG.md "Arms"). ``kind``: ablate = P63's fixture, sites captured; size = long rows,
#: logits only. ``grouping``: singleton keeps the MoE on its registered row-exact route, so an ablation isolates the
#: attention; the size arms run the served default.
ARMS = [
    {"name": "hf.base", "kind": "ablate", "attn": "hf", "force": [], "fp32red": False, "grouping": "singleton"},
    {"name": "hf.proj", "kind": "ablate", "attn": "hf", "force": ["proj"], "fp32red": False, "grouping": "singleton"},
    {"name": "hf.core", "kind": "ablate", "attn": "hf", "force": ["core"], "fp32red": False, "grouping": "singleton"},
    {"name": "hf.proj_core", "kind": "ablate", "attn": "hf", "force": ["proj", "core"], "fp32red": False,
     "grouping": "singleton"},
    {"name": "hf.all", "kind": "ablate", "attn": "hf", "force": list(FORCE_KINDS), "fp32red": False,
     "grouping": "singleton"},
    {"name": "hf.fp32red", "kind": "ablate", "attn": "hf", "force": [], "fp32red": True, "grouping": "singleton"},
    {"name": "paged.base", "kind": "ablate", "attn": "paged", "force": [], "fp32red": False, "grouping": "singleton"},
    {"name": "paged.proj", "kind": "ablate", "attn": "paged", "force": ["proj"], "fp32red": False,
     "grouping": "singleton"},
    {"name": "size.hf", "kind": "size", "attn": "hf", "force": [], "fp32red": False, "grouping": "default"},
    {"name": "size.paged", "kind": "size", "attn": "paged", "force": [], "fp32red": False, "grouping": "default"},
]
STACKS = ("nf4", "int4")
L_DEFAULT = 160
WINDOWS_DEFAULT = (128, 96, 64, 32)
WIDTHS_DEFAULT = (17, 16)
SIZE_NROWS, SIZE_LEN, SIZE_STEP, SIZE_FIRST = 4, 512, 32, 64


# ------------------------------------------------------------------------------------------------------ forcing --
def _token_axis(x: torch.Tensor) -> int:
    """[B, T, ...] -> 1; [T, H] (the MoE block flattens before its router) -> 0."""
    return 1 if x.dim() >= 3 else 0


def _cat_out(outs: list, ax: int):
    o0 = outs[0]
    if torch.is_tensor(o0):
        return torch.cat(outs, ax)
    if isinstance(o0, tuple):
        return tuple(_cat_out([o[i] for o in outs], ax) for i in range(len(o0)))
    if isinstance(o0, list):
        return [_cat_out([o[i] for o in outs], ax) for i in range(len(o0))]
    raise TypeError(f"cannot concatenate a {type(o0).__name__} output row by row")


class RowWise:
    """Wraps one module's forward: a call over n > 1 token rows becomes n one-row calls, concatenated. The module's
    own forward -- including any fold that patched it -- runs unchanged on each row, exactly as at T = 1."""

    def __init__(self, mod: torch.nn.Module, kind: str, counts: dict):
        self.mod, self.kind, self.counts = mod, kind, counts
        self.had_instance = "forward" in mod.__dict__
        self.orig = mod.forward

    def install(self):
        orig, kind, counts = self.orig, self.kind, self.counts

        def fwd(x, *a, **k):
            ax = _token_axis(x)
            n = x.shape[ax]
            if n <= 1:
                counts[kind]["t1_calls"] += 1
                return orig(x, *a, **k)
            counts[kind]["split_calls"] += 1
            counts[kind]["split_rows"] += n
            return _cat_out([orig(x.narrow(ax, i, 1).contiguous(), *a, **k) for i in range(n)], ax)
        self.mod.forward = fwd

    def remove(self):
        if self.had_instance:
            self.mod.forward = self.orig
        else:
            del self.mod.forward


class RowWiseSDPA:
    """The attention core, one QUERY row at a time. Replaces transformers' ``sdpa`` entry for the arm (every attention
    forward looks it up at call time, the int4 folds included). Row i of a T-row call over S keys attends to the first
    S - T + i + 1 keys -- exactly the prefix its T = 1 decode call saw -- with no mask, as a one-query decode call is
    made. At T = 1 it passes straight through, and records whether that call carried a mask (a decode call with a mask
    would make the no-mask per-row call a different call; the census shows it)."""

    def __init__(self, counts: dict):
        self.counts = counts
        self.orig = None

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.table = ALL_ATTENTION_FUNCTIONS
        self.orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.local_before = "sdpa" in getattr(ALL_ATTENTION_FUNCTIONS, "_local_mapping", {})
        orig, counts = self.orig, self.counts

        def rowwise(module, query, key, value, attention_mask, **kw):
            T, S = query.shape[2], key.shape[2]
            if T == 1:
                counts["core"]["t1_calls"] += 1
                counts["core"]["t1_mask_present"] += int(attention_mask is not None)
                return orig(module, query, key, value, attention_mask, **kw)
            counts["core"]["split_calls"] += 1
            counts["core"]["split_rows"] += T
            kw = {k: v for k, v in kw.items() if k != "is_causal"}
            outs = []
            for i in range(T):
                n = S - T + i + 1
                o, _ = orig(module, query[:, :, i:i + 1].contiguous(), key[:, :, :n].contiguous(),
                            value[:, :, :n].contiguous(), None, **kw)
                outs.append(o)
            return torch.cat(outs, 1), None
        ALL_ATTENTION_FUNCTIONS["sdpa"] = rowwise
        return self

    def __exit__(self, *exc):
        if self.local_before:
            self.table["sdpa"] = self.orig
        else:
            del self.table["sdpa"]                     # back to the global entry, exactly as before
        assert self.table["sdpa"] is self.orig, "the sdpa entry was not restored"


def forcing_targets(model, kinds) -> list:
    """(module, kind) pairs a forcing wraps, found by structure: the attention projections by name under each
    decoder layer's ``self_attn``, the router as ``mlp.gate`` when it is a module with a weight, the LM head, and every
    module whose class name ends in RMSNorm."""
    import p63_probe as P
    layers, _norm = P._layers(model)
    out = []
    for lyr in layers:
        if "proj" in kinds:
            for nm in PROJ_NAMES:
                m = getattr(lyr.self_attn, nm, None)
                if isinstance(m, torch.nn.Module):
                    out.append((m, "proj"))
        if "router" in kinds:
            g = getattr(getattr(lyr, "mlp", None), "gate", None)
            if isinstance(g, torch.nn.Module) and getattr(g, "weight", None) is not None:
                out.append((g, "router"))
    if "lm_head" in kinds and isinstance(getattr(model, "lm_head", None), torch.nn.Module):
        out.append((model.lm_head, "lm_head"))
    if "norms" in kinds:
        out += [(m, "norms") for m in model.modules() if type(m).__name__.endswith("RMSNorm")]
    return out


def new_counts() -> dict:
    return {k: {"t1_calls": 0, "split_calls": 0, "split_rows": 0, "t1_mask_present": 0, "wrapped_modules": 0}
            for k in FORCE_KINDS}


@contextlib.contextmanager
def forcing(model, kinds, fp32red: bool, counts: dict):
    """Apply an arm's forcings for the duration of the block, and restore everything after (a failed arm leaves the
    model as it found it)."""
    wraps = [RowWise(m, k, counts) for m, k in forcing_targets(model, kinds)]
    for w in wraps:
        counts[w.kind]["wrapped_modules"] += 1
    b = torch.backends.cuda.matmul
    prev_red = b.allow_bf16_reduced_precision_reduction
    stack = contextlib.ExitStack()
    try:
        for w in wraps:
            w.install()
        if "core" in kinds:
            stack.enter_context(RowWiseSDPA(counts))
        if fp32red:
            b.allow_bf16_reduced_precision_reduction = False
        yield
    finally:
        stack.close()
        b.allow_bf16_reduced_precision_reduction = prev_red
        for w in reversed(wraps):
            w.remove()


# ---------------------------------------------------------------------------------------------------- the arms --
def _apply_grouping(sa: dict):
    from experts4bit_qlora.engines import hot_residency as hr
    hr.FORCE_SINGLETON_GROUPS[0] = sa["grouping"] == "singleton"
    hr.DEVICE_GROUPING[0] = False


def _reset_grouping():
    from experts4bit_qlora.engines import hot_residency as hr
    hr.FORCE_SINGLETON_GROUPS[0] = False
    hr.DEVICE_GROUPING[0] = False


def run_ablate(model, cap, ids, sa, a, paged, hf_impl) -> tuple:
    """One ablation arm on the P63 fixture: the T = 1 control (unforced by construction: every forcing is inert at one
    row), then each verify window and width, then the prefill, all under the arm's forcings; per site and per logit,
    against the control."""
    import p63_probe as P
    L = a.L
    windows, widths = sorted(a.windows, reverse=True), list(a.widths)
    counts = new_counts()
    _apply_grouping(sa)
    P._set_attn_impl(model, paged.impl if sa["attn"] == "paged" else hf_impl)
    cap.active = False
    cap.take()
    out = {"arm": sa, "modes": {}}
    try:
        with torch.no_grad(), forcing(model, sa["force"], sa["fp32red"], counts):
            t0 = time.time()
            if sa["attn"] == "hf":
                logits_c, snaps = P.run_control_hf(model, cap, ids, L, set(windows))
            else:
                logits_c, snaps = paged.control(cap, ids, L), None
            ctrl = cap.take()
            ctrl["logits"] = logits_c
            out["control_s"] = round(time.time() - t0, 1)
            out["counts_after_control"] = copy.deepcopy(counts)
            acc = {W: {"caps": {s: [] for s in C.SITES}, "fin": [], "logs": [], "pos": [], "rows": []} for W in widths}
            for p0 in windows:
                for W in widths:
                    if p0 + W > L:
                        continue
                    lg = (P.run_verify_hf(model, cap, ids, p0, W, snaps[p0]) if sa["attn"] == "hf"
                          else paged.verify(cap, ids, p0, W))
                    c = cap.take()
                    d = acc[W]
                    for s in C.SITES:
                        d["caps"][s].append(c[s])
                    if c.get("final_norm") is not None:
                        d["fin"].append(c["final_norm"])
                    d["logs"].append(lg)
                    d["pos"] += list(range(p0, p0 + W))
                    d["rows"] += list(range(W))
            lg_pre = P.run_prefill_hf(model, cap, ids, L) if sa["attn"] == "hf" else paged.prefill(cap, ids, L)
            pre = cap.take()
    finally:
        _reset_grouping()
        P._set_attn_impl(model, hf_impl)
    for W in widths:
        d = acc[W]
        if not d["pos"]:
            continue
        test = {s: torch.cat(d["caps"][s], 0) for s in C.SITES}
        test["final_norm"] = torch.cat(d["fin"], 0) if d["fin"] else None
        pidx = torch.tensor(d["pos"])
        site, fin = P._cmp_mode(ctrl, test, pidx)
        lcmp = C.logits_stats(ctrl["logits"][pidx], torch.cat(d["logs"], 0))
        out["modes"][f"verify{W}"] = _slim(C.summarize_mode(site, fin, lcmp, d["pos"], d["rows"]))
    pidx = torch.arange(L)
    site, fin = P._cmp_mode(ctrl, pre, pidx)
    out["modes"]["prefill"] = _slim(C.summarize_mode(site, fin, C.logits_stats(ctrl["logits"], lg_pre), list(range(L))))
    out["counts"] = counts
    return out, ctrl


def _slim(m: dict) -> dict:
    """Keep the per-layer table for layer 0 only (the site of every P63 first difference); the rest is per position."""
    m = dict(m)
    m["per_layer_l0"] = {s: {k: v[0] for k, v in d.items()} for s, d in m.pop("per_layer").items()}
    return m


def _size_windows(n: int, width: int) -> list:
    return [p for p in range(SIZE_FIRST, n - width + 1, SIZE_STEP)]


def run_size(model, rows, sa, a, paged, hf_impl) -> dict:
    """The size reading: per row, incremental T = 1 decode over the row, then a verify of each width at every window
    and a prefill of the whole row; logits only (no site capture). Per position: KL(P_T1 || P_test) in fp64 over the
    full vocabulary, argmax agreement, and the control's top-1 margin. Windows are recorded so the reducer can
    bootstrap over them (positions inside one window are not independent)."""
    import p63_probe as P

    class _NoCap:                                                     # the P63 runners take a capture object
        def begin_forward(self):
            pass

        def end_forward(self):
            pass
    nocap = _NoCap()
    widths = list(a.widths)
    _apply_grouping(sa)
    P._set_attn_impl(model, paged.impl if sa["attn"] == "paged" else hf_impl)
    res = {"arm": sa, "rows": []}
    try:
        with torch.no_grad():
            for r, row in enumerate(rows):
                t0 = time.time()
                ids = torch.tensor([row], dtype=torch.long, device=a.device)
                n = ids.shape[1]
                allw = sorted({p for W in widths for p in _size_windows(n, W)}, reverse=True)
                if sa["attn"] == "hf":
                    ctrl, snaps = P.run_control_hf(model, nocap, ids, n, set(allw))
                else:
                    ctrl, snaps = paged.control(nocap, ids, n), None
                rec = {"row": r, "len": n, "modes": {}}
                acc = {W: {"kl": [], "flip": [], "margin": [], "window": [], "pos": []} for W in widths}
                # windows DESCENDING, widths inner (P63's order): on the paged backend a verify rewinds to p0 and writes
                # p0..p0+W-1, so every later verify must sit at a smaller p0 or its prefix would hold verify K/V
                for p0 in allw:
                    for W in widths:
                        if p0 not in _size_windows(n, W):
                            continue
                        lg = (P.run_verify_hf(model, nocap, ids, p0, W, snaps[p0]) if sa["attn"] == "hf"
                              else paged.verify(nocap, ids, p0, W))
                        s = C.logits_stats(ctrl[p0:p0 + W], lg)
                        d = acc[W]
                        d["kl"] += [float(v) for v in s["kl"]]
                        d["flip"] += [int(not bool(v)) for v in s["argmax_equal"]]
                        d["margin"] += [float(v) for v in s["margin_ref"]]
                        d["window"] += [p0] * W
                        d["pos"] += list(range(p0, p0 + W))
                for W in widths:
                    rec["modes"][f"verify{W}"] = acc[W]
                lg = P.run_prefill_hf(model, nocap, ids, n) if sa["attn"] == "hf" else paged.prefill(nocap, ids, n)
                s = C.logits_stats(ctrl, lg)
                rec["modes"]["prefill"] = {"kl": [float(v) for v in s["kl"]],
                                           "flip": [int(not bool(v)) for v in s["argmax_equal"]],
                                           "margin": [float(v) for v in s["margin_ref"]],
                                           "window": [(p // SIZE_STEP) * SIZE_STEP for p in range(n)],
                                           "pos": list(range(n))}
                rec["wall_s"] = round(time.time() - t0, 1)
                res["rows"].append(rec)
                print(f"P68 size {sa['name']} row {r}: " + " ".join(
                    f"{m}: kl {sum(v['kl']) / max(1, len(v['kl'])):.3e} flips {sum(v['flip'])}/{len(v['flip'])}"
                    for m, v in rec["modes"].items()) + f" ({rec['wall_s']} s)", flush=True)
    finally:
        _reset_grouping()
        P._set_attn_impl(model, hf_impl)
    return res


# -------------------------------------------------------------------------------------------------------- main --
def _versions() -> dict:
    import importlib.metadata as md
    v = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for p in ("experts4bit-qlora", "grouped-nf4-gemm", "transformers", "triton", "bitsandbytes"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    b = torch.backends.cuda.matmul
    v["matmul_flags"] = {"allow_tf32": b.allow_tf32,
                         "allow_bf16_reduced_precision_reduction": b.allow_bf16_reduced_precision_reduction}
    return v


def load_size_rows(path: str, nrows: int, length: int, vocab: int) -> dict:
    """Rows of token ids from a JSON file with a ``prompts`` list (P64's committed wikitext rows on the box). Rows are
    taken in order and cut to ``length``; an id past the model's vocabulary refuses (a different tokenizer's rows)."""
    import hashlib
    d = json.load(open(path))
    rows = [list(map(int, r[:length])) for r in d["prompts"][:nrows]]
    if len(rows) < nrows or any(len(r) < length for r in rows):
        raise SystemExit(f"{path}: needs {nrows} rows of >= {length} ids")
    top = max(max(r) for r in rows)
    if top >= vocab:
        raise SystemExit(f"{path}: id {top} >= the model's vocabulary {vocab}: these rows are another tokenizer's")
    return {"rows": rows, "source": path, "source_sha256": d.get("prompts_sha256"),
            "rows_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(), "nrows": nrows, "len": length}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stack", choices=STACKS)
    ap.add_argument("--model")
    ap.add_argument("--source")
    ap.add_argument("--arena")
    ap.add_argument("--calib")
    ap.add_argument("--fixture", default=next((p for p in (os.path.join(HERE, "fixture.txt"),
                                                           os.path.join(HERE, "..", "p63", "fixture.txt"))
                                                if os.path.exists(p)), os.path.join(HERE, "fixture.txt")),
                    help="P63's fixture: flat beside the probe on the box, bench/p63/ in the repository")
    ap.add_argument("--size-rows", help="JSON with a `prompts` list of token-id rows (the size reading's text)")
    ap.add_argument("--size-nrows", type=int, default=SIZE_NROWS)
    ap.add_argument("--size-len", type=int, default=SIZE_LEN)
    ap.add_argument("--out")
    ap.add_argument("--L", type=int, default=L_DEFAULT)
    ap.add_argument("--windows", type=lambda s: tuple(int(x) for x in s.split(",")), default=WINDOWS_DEFAULT)
    ap.add_argument("--widths", type=lambda s: tuple(int(x) for x in s.split(",")), default=WIDTHS_DEFAULT)
    ap.add_argument("--arms", default=None, help="comma list: a subset (rehearsal); default the registered table")
    ap.add_argument("--no-fuse-qkv", action="store_true")
    ap.add_argument("--no-paged", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--list-arms", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.list_arms:
        print(json.dumps(ARMS, indent=1))
        return 0
    for req in ("stack", "model", "arena", "calib", "out"):
        if getattr(a, req) is None:
            ap.error(f"--{req} is required")
    a.source = a.source or a.model
    arms = ARMS
    if a.arms:
        want = a.arms.split(",")
        unknown = [w for w in want if w not in {s["name"] for s in ARMS}]
        if unknown:
            ap.error(f"unknown arms: {unknown}")
        arms = [s for s in ARMS if s["name"] in want]
    if a.no_paged:
        arms = [s for s in arms if s["attn"] != "paged"]
    if any(s["kind"] == "size" for s in arms) and not a.size_rows:
        ap.error("the size arms need --size-rows")
    if a.dry_run:
        print(json.dumps({"stack": a.stack, "arms": [s["name"] for s in arms], "L": a.L, "windows": a.windows,
                          "widths": a.widths, "size": [a.size_rows, a.size_nrows, a.size_len]}, indent=1))
        return 0
    import p63_probe as P
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(1689)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    fx = P.load_fixture(a.fixture, tok, a.L)
    model, census = P.build_stack(a)
    text = getattr(model.config, "text_config", None) or model.config
    size = (load_size_rows(a.size_rows, a.size_nrows, a.size_len, int(text.vocab_size))
            if any(s["kind"] == "size" for s in arms) else None)
    ids = torch.tensor([fx["ids"]], dtype=torch.long, device=a.device)
    hf_impl = census["hf_attn_impl"]
    rec = {"lane": "P68", "stack": a.stack,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "sm_count": (torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else None),
           "versions": _versions(), "env": {k: v for k, v in os.environ.items() if k.startswith(("E4B_", "GNF4_"))},
           "census": census, "fixture": {k: v for k, v in fx.items() if k != "ids"},
           "size_rows": ({k: v for k, v in size.items() if k != "rows"} if size else None),
           "config": {"L": a.L, "windows": list(a.windows), "widths": list(a.widths),
                      "size": {"step": SIZE_STEP, "first": SIZE_FIRST}},
           "arms": {}}
    need_paged = any(s["attn"] == "paged" for s in arms)
    paged = None
    if need_paged:
        try:
            plen = max(a.L, a.size_len if size else 0)
            paged = P.Paged(model, plen, max(a.widths), a.device)
        except Exception as e:
            rec["paged_unavailable"] = repr(e)[:400]
            arms = [s for s in arms if s["attn"] != "paged"]
    cap = P.SiteCapture(model)
    controls = {}
    for sa in arms:
        t0 = time.time()
        try:
            if sa["kind"] == "ablate":
                res, ctrl = run_ablate(model, cap, ids, sa, a, paged, hf_impl)
                controls[sa["name"]] = ctrl
                m = res["modes"]
                print(f"P68 {a.stack}/{sa['name']}: " + " ".join(
                    f"{k}: exact {v['exact_positions']}/{v['positions']} flips {v['argmax_flips']} "
                    f"first-diff {v['first_diff_site_hist']}" for k, v in m.items()), flush=True)
            else:
                res = run_size(model, size["rows"], sa, a, paged, hf_impl)
        except Exception as e:
            import traceback
            rec["arms"][sa["name"]] = {"error": repr(e)[:600], "tb": traceback.format_exc()[-3000:]}
            print(f"P68 {a.stack}/{sa['name']}: ERROR {e!r}", flush=True)
            continue
        res["wall_s"] = round(time.time() - t0, 1)
        rec["arms"][sa["name"]] = res
    # G0: a forcing is inert at T = 1, so every arm's control must equal its backend's unforced control bit for bit;
    # and the unforced control run twice must be identical (the determinism the whole comparison rests on)
    g0 = {}
    for backend, base in (("hf", "hf.base"), ("paged", "paged.base")):
        if base not in controls:
            continue
        ref = controls[base]
        for name, c in controls.items():
            if name == base or not name.startswith(backend + "."):
                continue
            g0[f"{name} control == {base} control"] = bool(
                all(C.bit_equal(ref[s], c[s]).all().item() for s in C.SITES)
                and bool(C.bit_equal(ref["logits"], c["logits"]).all()))
    if "hf.base" in controls:
        with torch.no_grad():
            _apply_grouping({"grouping": "singleton"})
            try:
                lg2, _ = P.run_control_hf(model, cap, ids, a.L, set())
            finally:
                _reset_grouping()
        c2 = cap.take()
        ref = controls["hf.base"]
        g0["hf.base control repeat bit-identical"] = bool(
            all(C.bit_equal(ref[s], c2[s]).all().item() for s in C.SITES) and bool(C.bit_equal(ref["logits"], lg2).all()))
    rec["g0"] = g0
    cap.remove()
    if torch.cuda.is_available():
        rec["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    with open(os.path.join(a.out, "p68_arm.json"), "w") as f:
        json.dump(rec, f)
    bad = [n for n, s in rec["arms"].items() if "error" in s]
    print("P68ARM " + json.dumps({"stack": a.stack, "arms_run": [n for n in rec["arms"] if n not in bad],
                                  "errors": bad, "g0": g0}), flush=True)
    if not all(g0.values()):
        return 14
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
