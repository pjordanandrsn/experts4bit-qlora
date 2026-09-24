#!/usr/bin/env python3
"""p63_probe.py -- lane P63 (experts4bit-qlora#708): does a token's output depend on how many rows share its forward?

One process builds ONE stack (``--stack nf4 | int4 | int4nf``) and runs its registered sub-arms in turn. A sub-arm is a
route selection made by configuration only -- the hot-residency grouping flags (``FORCE_SINGLETON_GROUPS``,
``DEVICE_GROUPING``), ``E4B_FUSE_COMBINE``, ``GNF4_GEMV_DOTPAD`` and the attention backend (HF ``sdpa`` with a
``DynamicCache``, or the serving path's paged FP8 shim). Each sub-arm is read three ways (P63-PREREG.md, Instrument):

1. **End to end, teacher-forced.** The CONTROL is incremental eager decode: every fixture token 0..L-1 through its own
   T = 1 forward over the carried cache. Then the same tokens inside a W-row verify (the control's cache cut to p0,
   one forward over tokens p0..p0+W-1) for each registered width and window, and inside one L-row prefill. Per token:
   per decoder layer, per site (``p63_compare.SITES``), bit equality / max |d| / rel L2 / significant-element ULPs;
   the final norm; the logits (bit equality, argmax, KL). The first differing (layer, site) localises the route.
2. **Module replay ("same bytes in").** During the control, each target module's T = 1 inputs and outputs are
   recorded; afterwards the first n recorded inputs are stacked into ONE n-row call. Row i must be the T = 1 output of
   token i, or the module's arithmetic depends on the row count. For single-GEMM modules at three layers and the
   lm_head the rows are also scored against the exact fp64 result of the logical operands (``p63_compare.gemm_truth``).
3. **Kernel census.** The layer-0 gate_up GEMM of the expert store, by each kernel path the dispatch can take, on the
   control's own layer-0 routing: T = 1 calls against T-token calls, each path against fp64, classified
   EXACT / REORDER / PRECISION (``p63_compare.classify_pair``). Plus combine_rows and its torch chain on random rows.

Nothing here changes a default. ``--list-subarms`` prints the registered table; ``--dry-run`` builds nothing and prints
what would run. The receipt is ``<out>/p63_arm.json`` (+ ``p63_detail.pt``, compact per-position tensors).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "p44")):   # flat on the box; bench/ + bench/p44 in repo
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

import p63_compare as C  # noqa: E402

#: The registered sub-arms per stack (P63-PREREG.md "Arms"). attn: hf | paged; grouping: default | singleton | device;
#: combine: 1 = combine_rows (the default), 0 = the torch chain (B393's control arm); dotpad: None = gnf4's default.
SUBARMS = {
    "nf4": [
        {"name": "hf.default", "attn": "hf", "grouping": "default", "combine": 1, "dotpad": None},
        {"name": "hf.singleton", "attn": "hf", "grouping": "singleton", "combine": 1, "dotpad": None},
        {"name": "hf.device", "attn": "hf", "grouping": "device", "combine": 1, "dotpad": None},
        {"name": "hf.combine0", "attn": "hf", "grouping": "default", "combine": 0, "dotpad": None},
        {"name": "hf.singleton.dotpad0", "attn": "hf", "grouping": "singleton", "combine": 1, "dotpad": "0"},
        {"name": "paged.default", "attn": "paged", "grouping": "default", "combine": 1, "dotpad": None},
    ],
    "int4": [
        {"name": "hf.default", "attn": "hf", "grouping": "default", "combine": 1, "dotpad": None},
        {"name": "hf.device", "attn": "hf", "grouping": "device", "combine": 1, "dotpad": None},
        {"name": "hf.singleton", "attn": "hf", "grouping": "singleton", "combine": 1, "dotpad": None},
        {"name": "hf.combine0", "attn": "hf", "grouping": "default", "combine": 0, "dotpad": None},
        {"name": "paged.default", "attn": "paged", "grouping": "default", "combine": 1, "dotpad": None},
        {"name": "paged.device", "attn": "paged", "grouping": "device", "combine": 1, "dotpad": None},
    ],
    "int4nf": [
        {"name": "hf.default", "attn": "hf", "grouping": "default", "combine": 1, "dotpad": None},
        {"name": "hf.device", "attn": "hf", "grouping": "device", "combine": 1, "dotpad": None},
        {"name": "paged.device", "attn": "paged", "grouping": "device", "combine": 1, "dotpad": None},
    ],
}
#: stack -> (int4 expert store, int4 attention (RTN), decode folds r1/r2/router epilogue)
STACKS = {"nf4": (False, False, False), "int4": (True, True, True), "int4nf": (True, True, False)}
GLUE_ENV = ("E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")

L_DEFAULT = 160
WINDOWS_DEFAULT = (32, 64, 96, 128)
WIDTHS_DEFAULT = (17, 16)
REPLAY_NS_DEFAULT = (1, 16, 17, L_DEFAULT)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------------ the fixture --
def load_fixture(path: str, tok, L: int) -> dict:
    raw = open(path, "rb").read()
    text = raw.decode("utf-8")
    ids = tok(text).input_ids
    if len(ids) < L:
        raise SystemExit(f"fixture tokenises to {len(ids)} tokens, fewer than L={L}")
    ids = ids[:L]
    return {"path": os.path.basename(path), "text_sha256": _sha(raw), "L": L, "ids": ids,
            "ids_sha256": _sha(json.dumps(ids).encode())}


# ------------------------------------------------------------------------------------------------------- the stack --
def _layers(model):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("no model.model.layers: the probe needs the decoder-layer list")
    return list(layers), getattr(inner, "norm", None)


def _set_attn_impl(model, name: str) -> None:
    seen = set()
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and id(cfg) not in seen and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = name
            seen.add(id(cfg))


def build_stack(a) -> tuple:
    """The served model as step_decomp builds its shipped stack: NF4 through the arena, all-VRAM hybrid tier
    (collapse_resident), then -- for the int4 stacks -- the int4 expert store (RTN, packed from the source checkpoint)
    and uncalibrated int4 attention, then fused q/k/v on Qwen3-MoE (which re-applies the env-gated folds, as
    ``fuse_qkv`` does on the harness path). The folds are env-gated; this function sets their env from the stack."""
    exp_int4, attn_int4, folds = STACKS[a.stack]
    for k in GLUE_ENV:
        os.environ[k] = "1" if folds else "0"
    import experts4bit_qlora.engines.hybrid as _hy
    if getattr(_hy.enable_hybrid_tier, "_gen_hooked", False):
        raise SystemExit("the P42 lane hook is active (usercustomize on PYTHONPATH): it would apply the int4 lanes a "
                         "second time -- run the probe without it; the probe applies the lanes itself")
    import serve_stack
    t0 = time.time()
    model, info = serve_stack.build_served_model(a.model, a.arena, a.calib, device=a.device)
    info["build_nf4_s"] = round(time.time() - t0, 1)
    mt = getattr(model.config, "model_type", "?")
    info["model_type"] = mt
    if exp_int4:
        from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4
        t1 = time.time()
        info["int4_expert_layers_enabled"] = enable_serve_experts_int4(model, a.source, model_type=mt)
        info["int4_pack_s"] = round(time.time() - t1, 1)
    if attn_int4:
        from experts4bit_qlora.engines.int4_attn import enable_serve_attn_int4
        info["int4_attn_enabled"] = enable_serve_attn_int4(model)
    info["fuse_qkv_n"] = 0
    if attn_int4 and mt == "qwen3_moe" and not a.no_fuse_qkv:
        from experts4bit_qlora.engines.qkv_fuse import fuse_qkv
        info["fuse_qkv_n"] = fuse_qkv(model)
    model.eval()
    info["hf_attn_impl"] = getattr(model.config, "_attn_implementation", None) or "sdpa"
    return model, _census(model, info)


def _census(model, info: dict) -> dict:
    """The proof-of-execution counts a reading needs (a route that silently did not apply is the number this lane must
    never publish): int4 attention modules and which carry the K16 route, int4 expert layers, patched folds."""
    from experts4bit_qlora.engines.hot_residency import target_modules
    try:
        from experts4bit_qlora.engines.int4_attn import Int4Linear
        i4 = [m for m in model.modules() if isinstance(m, Int4Linear)]
    except ImportError:
        i4 = []
    mods = target_modules(model)
    stores = [getattr(getattr(m, "_hot_residency", None), "_int4_stores", None) for m in mods]
    info.update(int4_linear=len(i4), int4_linear_k16=sum(1 for m in i4 if getattr(m, "_smallm", None) is not None),
                int4_expert_layers=sum(1 for s in stores if s), moe_layers=len(mods),
                collapse_resident=all(getattr(getattr(m, "_hot_residency", None), "collapse_resident", False) for m in mods),
                glue_env={k: os.environ.get(k) for k in GLUE_ENV})
    return info


# ------------------------------------------------------------------------------------------------------- capture --
class SiteCapture:
    """Forward hooks per decoder layer: self_attn input (attn_in), o_proj input (attn_core), self_attn output
    (attn_out), mlp input (mlp_in), mlp output (mlp_out), layer output (layer_out); plus the final norm. Each forward's
    rows are kept on the device during the forward and moved to the host once at ``end_forward``."""

    def __init__(self, model):
        self.layers, self.norm = _layers(model)
        self.hooks = []
        self.cur = None
        self.done = {s: [] for s in C.SITES}
        self.final = []
        self.active = False
        for li, lyr in enumerate(self.layers):
            attn, mlp = lyr.self_attn, lyr.mlp
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is None:
                raise RuntimeError(f"layer {li}: self_attn has no o_proj child -- attn_core cannot be captured")
            self.hooks += [
                attn.register_forward_pre_hook(self._pre("attn_in", li), with_kwargs=True),
                o_proj.register_forward_pre_hook(self._pre("attn_core", li), with_kwargs=True),
                attn.register_forward_hook(self._post("attn_out", li)),
                mlp.register_forward_pre_hook(self._pre("mlp_in", li), with_kwargs=True),
                mlp.register_forward_hook(self._post("mlp_out", li)),
                lyr.register_forward_hook(self._post("layer_out", li)),
            ]
        if self.norm is not None:
            self.hooks.append(self.norm.register_forward_hook(self._post("final_norm", -1)))

    @staticmethod
    def _rows(t):
        if isinstance(t, (tuple, list)):
            t = t[0]
        return t.detach().reshape(-1, t.shape[-1]).clone()

    def _pre(self, site, li):
        def h(mod, args, kwargs):
            if self.active:
                x = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
                self.cur[site][li] = self._rows(x)
        return h

    def _post(self, site, li):
        def h(mod, args, out):
            if not self.active:
                return
            if site == "final_norm":
                self.cur_final = self._rows(out)
            else:
                self.cur[site][li] = self._rows(out)
        return h

    def begin_forward(self):
        self.cur = {s: [None] * len(self.layers) for s in C.SITES}
        self.cur_final = None
        self.active = True

    def end_forward(self):
        self.active = False
        for s in C.SITES:
            if any(v is None for v in self.cur[s]):
                missing = [i for i, v in enumerate(self.cur[s]) if v is None]
                raise RuntimeError(f"site {s} not captured at layers {missing[:8]}: a patched forward bypassed a hook")
            self.done[s].append(torch.stack(self.cur[s], dim=1).cpu())          # [T, Ly, w]
        if self.cur_final is not None:
            self.final.append(self.cur_final.cpu())

    def take(self) -> dict:
        out = {s: (torch.cat(v, 0) if v else None) for s, v in self.done.items()}
        out["final_norm"] = torch.cat(self.final, 0) if self.final else None
        self.done = {s: [] for s in C.SITES}
        self.final = []
        return out

    def remove(self):
        for h in self.hooks:
            h.remove()


class ModuleRecorder:
    """Records each target module's T = 1 (args, kwargs, output) during the control, for ``replay``."""

    def __init__(self, targets: dict):
        self.targets = targets
        self.recs = {n: [] for n in targets}
        self.active = False
        self.hooks = [m.register_forward_hook(self._hook(n), with_kwargs=True) for n, m in targets.items()]

    @staticmethod
    def _clone(v):
        if torch.is_tensor(v):
            return v.detach().clone()
        if isinstance(v, (tuple, list)):
            return type(v)(ModuleRecorder._clone(x) for x in v)
        return v

    def _hook(self, name):
        def h(mod, args, kwargs, out):
            if self.active:
                self.recs[name].append((self._clone(args), self._clone(kwargs), self._clone(out)))
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()


def _tok_axis(t: torch.Tensor) -> int | None:
    return 1 if t.dim() >= 3 else (0 if t.dim() == 2 else None)


def _stack(vals):
    v0 = vals[0]
    if torch.is_tensor(v0):
        ax = _tok_axis(v0)
        if ax is None:
            raise ValueError(f"cannot stack a {v0.dim()}-D tensor along a token axis")
        return torch.cat(vals, dim=ax)
    if isinstance(v0, (tuple, list)):
        return type(v0)(_stack([v[i] for v in vals]) for i in range(len(v0)))
    if isinstance(v0, dict):
        return {kk: _stack([v[kk] for v in vals]) for kk in v0}
    return v0


def _flat_tensors(v):
    if torch.is_tensor(v):
        return [v]
    if isinstance(v, (tuple, list)):
        return [t for x in v for t in _flat_tensors(x)]
    return []


def _rows_first(t: torch.Tensor) -> torch.Tensor:
    ax = _tok_axis(t)
    return t.movedim(ax, 0).reshape(t.shape[ax], -1) if ax is not None else t.reshape(1, -1)


def replay_targets(model, full: bool) -> dict:
    """Module-replay targets: every layer's experts module (the DISPATCHED one -- a wrapped base is never called) and,
    when ``full``, its router, input_layernorm and attention projections, and the lm_head."""
    from experts4bit_qlora.engines.hot_residency import dispatched_modules
    layers, _ = _layers(model)
    disp = dispatched_modules(model)
    t = {}
    for li, lyr in enumerate(layers):
        mlp = lyr.mlp
        exp = getattr(mlp, "experts", None)
        if exp is not None:
            match = [d for d in disp if d is exp or getattr(d, "base", None) is exp]
            t[f"L{li}.experts"] = match[0] if match else exp
        if not full:
            continue
        gate = getattr(mlp, "gate", None)
        if gate is not None:
            t[f"L{li}.router"] = gate
        t[f"L{li}.input_layernorm"] = lyr.input_layernorm
        for cname, child in lyr.self_attn.named_children():
            if cname.endswith("_proj"):
                t[f"L{li}.attn.{cname}"] = child
    if full and getattr(model, "lm_head", None) is not None:
        t["lm_head"] = model.lm_head
    return t


def _linear_path(mod, n: int):
    """(kernel path name, exact weight fp32 [N, K]) for single-GEMM modules; None for anything else."""
    try:
        from experts4bit_qlora.engines.int4_attn import Int4Linear
    except ImportError:
        Int4Linear = ()
    if Int4Linear and isinstance(mod, Int4Linear):
        if mod.bias is not None:
            return None
        from int4_pack_ref import dequant_int4_ref
        w = dequant_int4_ref(mod.packed[0], mod.scales[0], mod.N, mod.K)
        if n <= mod.GEMV_ROWS_MAX:
            return "int4.gemv", w
        if mod._smallm is not None and n <= mod.SMALLM_ROWS_MAX:
            return "int4.k16", w
        return "int4.cublas_cache", w
    if type(mod) is torch.nn.Linear and mod.bias is None and mod.weight.dtype == torch.bfloat16:
        return "dense.cublas", mod.weight.detach()
    return None


class fp32_reduction:
    """Context: torch's ``allow_bf16_reduced_precision_reduction`` OFF (cuBLAS keeps split-K partials in fp32),
    restored on exit. Used only for the attribution reruns of ``C.CUBLAS_PATHS``; the readings run under the defaults."""

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.prev
        return False


def _score_linear(rec, path, x2, w, y, rerun):
    """fp64 accuracy of one single-GEMM call: the ratio under the default flags and, for a cuBLAS path, the ratio of
    the same call rerun with fp32 split-K reduction (``rerun()`` returns that output) -- the DEFECT line reads the
    latter. ``fp32_reduction_changes_bits`` says whether the default flag changed this call's result at all."""
    tb = C.gemm_truth(x2, w.to(x2.device), path)
    yv = y.reshape(tb["truth"].shape)
    rec["path"] = path
    rec["bound_ratio"] = C.bound_ratio(yv, tb["truth"], tb["bound"])
    rec["rel_rms_err"] = C.rel_rms_err(yv, tb["truth"])
    if path in C.CUBLAS_PATHS:
        with fp32_reduction(), torch.no_grad():
            y2 = rerun().reshape(tb["truth"].shape)
        rec["bound_ratio_fp32_reduction"] = C.bound_ratio(y2, tb["truth"], tb["bound"])
        rec["fp32_reduction_changes_bits"] = not bool(C.bit_equal(yv, y2).all())
    del tb


def replay(recorder: ModuleRecorder, ns, fp64_layers=(), device="cuda") -> dict:
    """Stack the first n recorded calls into one call per module and compare row by row with the recorded T = 1
    outputs. For single-GEMM modules on ``fp64_layers`` (and the lm_head) also score against fp64. A module whose
    replay raises is recorded with its error and the others still run (the rehearsal lost a whole stack's readings to
    one module's dtype difference before this)."""
    res = {}
    for name, mod in recorder.targets.items():
        recs = recorder.recs[name]
        if not recs:
            res[name] = {"skipped": "never called during the control"}
            continue
        ent = {"calls": len(recs), "by_n": {}}
        lin_ok = name == "lm_head" or any(name.startswith(f"L{i}.attn.") for i in fp64_layers)
        for n in ns:
            if n > len(recs):
                continue
            try:
                args = _stack([r[0] for r in recs[:n]])
                kw = _stack([r[1] for r in recs[:n]]) if recs[0][1] else {}
            except ValueError as e:
                ent["by_n"][str(n)] = {"skipped": str(e)}
                continue
            try:
                with torch.no_grad():
                    out = mod(*args, **kw)
                outs = [_rows_first(t) for t in _flat_tensors(out)]
                refs = [_rows_first(_stack([_flat_tensors(r[2])[i] for r in recs[:n]])) for i in range(len(outs))]
                parts = [C.rows_equal_to_singles(o, r) for o, r in zip(outs, refs)]
                rec = {"rows": parts[0]["rows"], "rows_bit_equal": min(p["rows_bit_equal"] for p in parts),
                       "outputs": parts}
                if any("dtype_batched" in p for p in parts):
                    rec["dtype_differs"] = True
                lp = _linear_path(mod, n) if lin_ok else None
                if lp is not None:
                    x2 = _flat_tensors(args)[0]
                    x2 = x2.reshape(-1, x2.shape[-1])

                    def _again(_m=mod, _a=args, _k=kw):
                        return _rows_first(_flat_tensors(_m(*_a, **_k))[0])
                    _score_linear(rec, lp[0], x2, lp[1], outs[0], _again)
            except Exception as e:                      # recorded; the other modules and n still run
                rec = {"error": repr(e)[:400]}
            ent["by_n"][str(n)] = rec
        one = ent["by_n"].get("1", {})
        ent["replay_faithful_at_1"] = bool(one) and "error" not in one and one.get("rows_bit_equal") == one.get("rows")
        res[name] = ent
    return res


# ------------------------------------------------------------------------------------------ the three modes, hf --
def _logits(out):
    lg = out.logits if hasattr(out, "logits") else out[0]
    return lg[0].float().cpu()


def run_control_hf(model, cap, ids, L, snaps_at):
    """Incremental eager decode, T = 1 per forward, DynamicCache carried; snapshots of the cache BEFORE token p for
    every p in ``snaps_at`` (deep copies, so a verify never mutates the control's state)."""
    past, logits, snaps = None, [], {}
    for t in range(L):
        if t in snaps_at:
            snaps[t] = copy.deepcopy(past) if past is not None else None
        cap.begin_forward()
        out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
        cap.end_forward()
        past = out.past_key_values
        logits.append(_logits(out))
    return torch.cat(logits, 0), snaps


def run_verify_hf(model, cap, ids, p0, W, snap):
    cap.begin_forward()
    out = model(input_ids=ids[:, p0:p0 + W], past_key_values=copy.deepcopy(snap), use_cache=True)
    cap.end_forward()
    return _logits(out)


def run_prefill_hf(model, cap, ids, L):
    cap.begin_forward()
    out = model(input_ids=ids[:, :L], use_cache=True)
    cap.end_forward()
    return _logits(out)


# ------------------------------------------------------------------------------------ the three modes, paged FP8 --
class Paged:
    """The serving path's attention: PagedAttentionContext over one Fp8PagedKV slot, the model run with
    use_cache=False and explicit positions (engines/paged_runner.py's contract). Decode = one query row through the
    fp8 decode kernel; verify = PREREG-s2lite's staggered-length read of the same kernel; prefill = SDPA over the
    staged bf16 K/V (paged_attention.py)."""

    def __init__(self, model, L, max_w, device):
        from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
        from experts4bit_qlora.engines.paged_attention import IMPL_NAME, PagedAttentionContext, register
        cfg = model.config
        text = getattr(cfg, "text_config", None) or cfg
        hkv = getattr(text, "num_key_value_heads", None) or text.num_attention_heads
        hd = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
        n_layers = len(_layers(model)[0])
        self.kv = Fp8PagedKV(n_layers, hkv, hd, batch=1, max_tokens_per_seq=L + max_w + 16, device=device)
        self.ctx = PagedAttentionContext(kv=self.kv, slots=[0], mode="decode")
        register(None)
        self.impl = IMPL_NAME
        self.model = model
        self.device = device
        self.tiers = [m._hot_residency for m in model.modules()
                      if hasattr(m, "_hot_residency") and hasattr(m._hot_residency, "prefill_gpu_only")]

    def _fwd(self, ids, pos, mode):
        from experts4bit_qlora.engines.paged_attention import set_context
        self.ctx.mode = mode
        prev = set_context(self.ctx)
        try:
            return self.model(input_ids=ids, position_ids=pos, use_cache=False)
        finally:
            set_context(prev)

    def control(self, cap, ids, L):
        self.kv.reset(0)
        logits = []
        for t in range(L):
            cap.begin_forward()
            out = self._fwd(ids[:, t:t + 1], torch.tensor([[t]], device=self.device), "decode")
            cap.end_forward()
            logits.append(_logits(out))
        return torch.cat(logits, 0)

    def verify(self, cap, ids, p0, W):
        self.kv.rewind(0, p0)
        cap.begin_forward()
        out = self._fwd(ids[:, p0:p0 + W], torch.arange(p0, p0 + W, device=self.device)[None], "verify")
        cap.end_forward()
        return _logits(out)

    def prefill(self, cap, ids, L):
        self.kv.reset(0)
        self.ctx.staging.clear()
        for t in self.tiers:
            t.prefill_gpu_only(True)
        try:
            cap.begin_forward()
            out = self._fwd(ids[:, :L], torch.arange(L, device=self.device)[None], "prefill")
            cap.end_forward()
        finally:
            for t in self.tiers:
                t.prefill_gpu_only(False)
            self.ctx.staging.clear()
        return _logits(out)


# ---------------------------------------------------------------------------------------------------- sub-arms --
def apply_subarm(model, sa: dict, paged: Paged | None, hf_impl: str):
    from experts4bit_qlora.engines import hot_residency as hr
    hr.FORCE_SINGLETON_GROUPS[0] = sa["grouping"] == "singleton"
    hr.DEVICE_GROUPING[0] = sa["grouping"] == "device"
    os.environ["E4B_FUSE_COMBINE"] = str(sa["combine"])
    hr._COMBINE.clear()
    if sa["dotpad"] is None:
        os.environ.pop("GNF4_GEMV_DOTPAD", None)
    else:
        os.environ["GNF4_GEMV_DOTPAD"] = sa["dotpad"]
    _set_attn_impl(model, paged.impl if sa["attn"] == "paged" else hf_impl)


def reset_subarm(model, hf_impl: str):
    from experts4bit_qlora.engines import hot_residency as hr
    hr.FORCE_SINGLETON_GROUPS[0] = False
    hr.DEVICE_GROUPING[0] = False
    os.environ.pop("E4B_FUSE_COMBINE", None)
    hr._COMBINE.clear()
    os.environ.pop("GNF4_GEMV_DOTPAD", None)
    _set_attn_impl(model, hf_impl)


def refusal(stack: str, sa: dict) -> str | None:
    """A sub-arm the installed cut cannot run safely is refused with the reason, never run."""
    if stack.startswith("int4") and sa["grouping"] == "singleton":
        from experts4bit_qlora.engines import hot_residency as hr
        if not hasattr(hr, "_int4_part_or_none"):
            return ("REFUSED: this e4b cut hands the int4 singleton GEMV a partials buffer sized for one token; at "
                    "T > 1 it writes past it (P63 rehearsal, part_oob.json). Needs the _int4_part_or_none fix.")
    return None


def _cmp_mode(ctrl, test, pos_idx):
    """Compare a mode's capture against the control rows at absolute positions ``pos_idx``."""
    ref = {s: ctrl[s][pos_idx] for s in C.SITES}
    site = C.compare_sites(ref, {s: test[s] for s in C.SITES})
    fin = (C.row_stats(ctrl["final_norm"][pos_idx], test["final_norm"])
           if ctrl.get("final_norm") is not None and test.get("final_norm") is not None else None)
    return site, fin


def _nf4_counts():
    try:
        import nf4_grouped
        return dict(nf4_grouped.dispatch_counts())
    except (ImportError, AttributeError):
        return None


def run_subarm(model, cap, ids, sa, a, paged, first: bool, stack: str, hf_impl: str) -> tuple:
    L = a.L
    windows = sorted(a.windows, reverse=True)                         # descending: a verify never overwrites a
    widths = list(a.widths)                                           # later window's prefix (paged KV rewind)
    apply_subarm(model, sa, paged, hf_impl)
    cap.active = False
    cap.take()                                                        # a failed sub-arm leaves no rows behind
    rec = ModuleRecorder(replay_targets(model, full=first))
    try:
        t0 = time.time()
        nf4_before = _nf4_counts()
        rec.active = True
        with torch.no_grad():
            if sa["attn"] == "hf":
                logits_c, snaps = run_control_hf(model, cap, ids, L, set(windows))
            else:
                logits_c, snaps = paged.control(cap, ids, L), None
        rec.active = False
        ctrl = cap.take()
        ctrl["logits"] = logits_c
        nf4_after = _nf4_counts()
        out = {"toggles": sa, "control_s": round(time.time() - t0, 1), "modes": {}, "determinism": {}}
        if nf4_before is not None:
            out["nf4_dispatch_during_control"] = {k: nf4_after[k] - nf4_before.get(k, 0) for k in nf4_after}
        if first:
            with torch.no_grad():
                if sa["attn"] == "hf":
                    lg2, _ = run_control_hf(model, cap, ids, L, set())
                else:
                    lg2 = paged.control(cap, ids, L)
            c2 = cap.take()
            same = all(C.bit_equal(ctrl[s], c2[s]).all().item() for s in C.SITES) and bool(C.bit_equal(logits_c, lg2).all())
            out["determinism"]["control_repeat_bit_identical"] = bool(same)
        detail = {}
        acc = {W: {"caps": {s: [] for s in C.SITES}, "fin": [], "logs": [], "pos": [], "rows": []} for W in widths}
        for p0 in windows:                     # descending; widths inner: every verify's prefix is still the control's
            for W in widths:
                if p0 + W > L:
                    continue
                with torch.no_grad():
                    lg = (run_verify_hf(model, cap, ids, p0, W, snaps[p0]) if sa["attn"] == "hf"
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
        if first and windows:
            # determinism: the smallest window, first width, twice -- LAST, so its KV writes precede only the prefill
            p0, W = windows[-1], widths[0]
            with torch.no_grad():
                lg_a = (run_verify_hf(model, cap, ids, p0, W, snaps[p0]) if sa["attn"] == "hf"
                        else paged.verify(cap, ids, p0, W))
                ca = cap.take()
                lg_b = (run_verify_hf(model, cap, ids, p0, W, snaps[p0]) if sa["attn"] == "hf"
                        else paged.verify(cap, ids, p0, W))
                cb = cap.take()
            out["determinism"][f"verify{W}_repeat_bit_identical"] = bool(
                all(C.bit_equal(ca[s], cb[s]).all().item() for s in C.SITES) and bool(C.bit_equal(lg_a, lg_b).all()))
        for W in widths:
            d = acc[W]
            if not d["pos"]:
                continue
            test = {s: torch.cat(d["caps"][s], 0) for s in C.SITES}
            test["final_norm"] = torch.cat(d["fin"], 0) if d["fin"] else None
            pidx = torch.tensor(d["pos"])
            site, fin = _cmp_mode(ctrl, test, pidx)
            lcmp = C.logits_stats(ctrl["logits"][pidx], torch.cat(d["logs"], 0))
            out["modes"][f"verify{W}"] = C.summarize_mode(site, fin, lcmp, d["pos"], d["rows"])
            detail[f"verify{W}"] = {"pos": pidx, "row": torch.tensor(d["rows"]), "equal": site["equal"],
                                    "rel_l2": site["rel_l2"].float()}
        with torch.no_grad():
            lg = run_prefill_hf(model, cap, ids, L) if sa["attn"] == "hf" else paged.prefill(cap, ids, L)
        pre = cap.take()
        pidx = torch.arange(L)
        site, fin = _cmp_mode(ctrl, pre, pidx)
        lcmp = C.logits_stats(ctrl["logits"], lg)
        out["modes"]["prefill"] = C.summarize_mode(site, fin, lcmp, list(range(L)))
        detail["prefill"] = {"pos": pidx, "equal": site["equal"], "rel_l2": site["rel_l2"].float()}
        t1 = time.time()
        ly = len(_layers(model)[0])
        out["replay"] = replay(rec, [n for n in a.replay_ns if n <= L],
                               fp64_layers=(0, ly // 2, ly - 1) if first else (), device=a.device)
        out["replay_s"] = round(time.time() - t1, 1)
        experts_rec = {n: rec.recs[n] for n in rec.recs if n == "L0.experts"}
    finally:
        rec.remove()
        reset_subarm(model, hf_impl)
    return out, ctrl, detail, experts_rec


# ------------------------------------------------------------------------------------------------ kernel census --
def _pair(results, singles, a_path, b_path, T):
    """Class of the route (a_path at T = 1, b_path at T tokens) from the census records: every row of b_path's
    T-token call against a_path's one-token call of the same token (``singles[a_path]``, all L tokens)."""
    a1 = results.get((a_path, 1))
    bT = results.get((b_path, T))
    if a1 is None or bT is None or bT["_rows"] is None:
        return None
    y = bT["_rows"]
    ref = singles[a_path][: y.shape[0]]
    eq = bool(C.bit_equal(y, ref).all())
    st = C.rows_equal_to_singles(y, ref)
    ra = a1.get("bound_ratio_fp32_reduction", a1["bound_ratio"])
    rb = bT.get("bound_ratio_fp32_reduction", bT["bound_ratio"])
    return {"a": f"{a_path}@T=1", "b": f"{b_path}@T={T}", "bit_equal_all_rows": eq,
            "rows": st["rows"], "rows_bit_equal": st["rows_bit_equal"], "max_rel_l2": st.get("max_rel_l2"),
            "max_ulp_sig": st.get("max_ulp_sig"), "ratio_a": ra, "ratio_b": rb,
            "class": C.classify_pair(eq, a1["model"], bT["model"], ra, rb)}


def kernel_census(model, stack: str, x_rec, Ts, device) -> dict:
    """Layer 0's gate_up GEMM on the control's layer-0 rows and routing (``x_rec``: the recorded experts-module calls),
    by every kernel path the dispatch can take. Row r of token t is x_t against expert top_k[t][j] (r = t*k + j), the
    exact rows the module computes."""
    from experts4bit_qlora.engines.hot_residency import target_modules
    st0 = target_modules(model)[0]._hot_residency
    xs = torch.cat([r[0][0] for r in x_rec], 0)                       # [L, H] bf16
    idx = torch.cat([r[0][1] for r in x_rec], 0)                      # [L, k]
    L, k = idx.shape
    Ts = [T for T in Ts if T <= L]

    def rows(T):
        return xs[:T].repeat_interleave(k, 0).contiguous(), idx[:T].reshape(-1)

    paths = {}
    if stack.startswith("int4"):
        from int4_b32 import gemm_int4_b32_grouped_captured, gemv_int4_b32, quant_x_rows
        from int4_pack_ref import dequant_int4_ref
        from nf4_grouped import build_group_tiles_device
        s = st0._int4_stores["gu"]
        P, S, N, K = s["packed"], s["scales"], s["N"], s["K"]
        E = P.shape[0]
        wx = {}

        def W(e):
            if e not in wx:
                wx[e] = dequant_int4_ref(P[e], S[e], N, K)
            return wx[e]

        def gemv(X, ids):
            q, sc = quant_x_rows(X)
            return gemv_int4_b32(q, sc, P, S, ids.to(torch.int32), N, K)

        def deq(X, ids):
            order = torch.argsort(ids)
            xs_, ids_s = X.index_select(0, order), ids.index_select(0, order)
            uniq, counts = torch.unique_consecutive(ids_s, return_counts=True)
            o = torch.empty(X.shape[0], N, dtype=torch.bfloat16, device=X.device)
            r0 = 0
            for e, m in zip(uniq.tolist(), counts.tolist()):
                o[r0:r0 + m] = xs_[r0:r0 + m].to(torch.bfloat16) @ W(e).to(torch.bfloat16).t()
                r0 += m
            out = torch.empty_like(o)
            out.index_copy_(0, order, o)
            return out

        def grouped(X, ids):
            t0, tr, tg, order, _c = build_group_tiles_device(ids, E, 16)
            q, sc = quant_x_rows(X.index_select(0, order).contiguous())
            o = gemm_int4_b32_grouped_captured(q, sc, P, S, t0, tr, tg)
            out = torch.empty_like(o)
            out.index_copy_(0, order, o)
            return out
        paths = {"int4.gemv": gemv, "int4.deq_bf16": deq, "int4.grouped_gemm": grouped}
        pairs = [("int4.gemv", "int4.deq_bf16", "default T>1 (host-grouped dequant + bf16 matmul)"),
                 ("int4.gemv", "int4.gemv", "DEVICE_GROUPING <= 256 rows, or FORCE_SINGLETON_GROUPS"),
                 ("int4.gemv", "int4.grouped_gemm", "DEVICE_GROUPING > 256 rows")]
    else:
        import nf4_grouped
        from nf4_grouped import NF4_LUT, gemm_4bit_grouped
        B, A = st0.h_gu_p, st0.h_gu_a
        E, N = B.shape[0], B.shape[1]
        K = B.shape[2] * 2
        lut = torch.tensor(NF4_LUT, dtype=torch.float64, device=B.device)
        wx = {}

        def W(e):
            if e not in wx:
                flat = B[e].reshape(-1).to(torch.int32)
                codes = torch.stack([(flat >> 4) & 0xF, flat & 0xF], 1).reshape(-1)
                wx[e] = (lut[codes] * A[e].double().reshape(-1).repeat_interleave(64)).reshape(N, K)
            return wx[e]

        def decode(X, ids):
            return gemm_4bit_grouped(X.contiguous(), B, A, [1] * X.shape[0], ids.to(torch.int32))

        def grouped(X, ids):
            order = torch.argsort(ids)
            xs_, ids_s = X.index_select(0, order).contiguous(), ids.index_select(0, order)
            uniq, counts = torch.unique_consecutive(ids_s, return_counts=True)
            o = gemm_4bit_grouped(xs_, B, A, counts.tolist(), uniq)
            out = torch.empty_like(o)
            out.index_copy_(0, order, o)
            return out

        def decode_scalar(X, ids):
            prev = os.environ.get("GNF4_GEMV_DOTPAD")
            os.environ["GNF4_GEMV_DOTPAD"] = "0"
            try:
                return decode(X, ids)
            finally:
                if prev is None:
                    os.environ.pop("GNF4_GEMV_DOTPAD", None)
                else:
                    os.environ["GNF4_GEMV_DOTPAD"] = prev
        # which decode kernel the default dispatch runs here, read off the kernel's own counters
        nf4_grouped.reset_dispatch_counts()
        X1, i1 = rows(1)
        decode(X1, i1)
        dc = dict(nf4_grouped.dispatch_counts())
        dec_model = "nf4.dotpad" if (dc.get("dotpad", 0) + dc.get("dotpad_splitk", 0)) else "nf4.gemv_scalar"
        paths = {dec_model: decode, "nf4.mtile": grouped, "nf4.gemv_scalar[dotpad=0]": decode_scalar}
        pairs = [(dec_model, dec_model, "FORCE_SINGLETON_GROUPS (decode GEMV at T*k rows)"),
                 (dec_model, "nf4.mtile", "default / DEVICE_GROUPING T>1 (M-tile, some group > 1 row)"),
                 ("nf4.gemv_scalar[dotpad=0]", "nf4.gemv_scalar[dotpad=0]", "GNF4_GEMV_DOTPAD=0 + singleton: split-K by rows")]

    def model_of(p):
        return p.split("[")[0]

    results, singles_by = {}, {}
    with torch.no_grad():
        for p, fn in paths.items():
            singles = torch.cat([fn(*_tok_rows(xs, idx, t, k)) for t in range(L)], 0)     # [L*k, N]
            singles_by[p] = singles
            for T in Ts:
                X, ids = rows(T)
                y = singles[: T * k] if T == 1 else fn(X, ids)
                if p == "nf4.mtile" and T == 1:
                    y = None                                                      # sizes are all 1 at T=1: not this path
                truth = torch.empty(X.shape[0], N, dtype=torch.float64, device=X.device)
                bound = torch.empty_like(truth)
                for e in ids.unique().tolist():
                    r = (ids == e).nonzero().view(-1)
                    tb = C.gemm_truth(X.index_select(0, r), W(e).to(X.device), model_of(p))
                    truth[r], bound[r] = tb["truth"], tb["bound"]
                ent = {"path": p, "T": T, "rows": X.shape[0], "model": model_of(p), "_rows": y}
                if y is not None:
                    ent.update(C.rows_equal_to_singles(y, singles[: T * k]))
                    ent["bound_ratio"] = C.bound_ratio(y, truth, bound)
                    ent["rel_rms_err"] = C.rel_rms_err(y, truth)
                    if model_of(p) in C.CUBLAS_PATHS:
                        with fp32_reduction():
                            y2 = fn(X, ids)                        # rows(1) is token 0's own call
                        ent["bound_ratio_fp32_reduction"] = C.bound_ratio(y2, truth, bound)
                        ent["fp32_reduction_changes_bits"] = not bool(C.bit_equal(y, y2).all())
                else:
                    ent["bound_ratio"] = None
                results[(p, T)] = ent
    out = {"layer": 0, "projection": "gate_up", "N": int(N), "K": int(K), "tokens": L, "top_k": k,
           "records": [{kk: v for kk, v in e.items() if kk != "_rows"} for e in results.values()],
           "pairs": []}
    for pa, pb, what in pairs:
        for T in Ts:
            if T == 1:
                continue
            pr = _pair(results, singles_by, pa, pb, T)
            if pr is not None:
                pr["route"] = what
                out["pairs"].append(pr)
    return out


def _tok_rows(xs, idx, t, k):
    return xs[t:t + 1].repeat_interleave(k, 0).contiguous(), idx[t]


def combine_census(k: int, H: int, Ts, device, seed: int = 0) -> dict:
    """combine_rows and its torch chain (E4B_FUSE_COMBINE=0) on random rows: token t alone vs t inside a T-token
    call, each against fp64 (B393's bound), and the kernel against the chain (B393's own question, recorded)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Tmax = max(Ts)
    dn = (torch.randn(Tmax * k, H, generator=g)).to(torch.bfloat16).to(device)
    w = torch.softmax(torch.randn(Tmax, k, generator=g), -1).reshape(-1).to(device)
    try:
        from int4_b32 import combine_rows
    except ImportError:
        combine_rows = None

    def chain(d, ww, kk):
        T = d.shape[0] // kk
        return (d.to(torch.float32) * ww[:, None]).view(T, kk, -1).sum(dim=1).to(torch.bfloat16)
    fns = {"chain": chain}
    if combine_rows is not None:
        fns["combine_rows"] = combine_rows
    recs = []
    with torch.no_grad():
        for name, fn in fns.items():
            singles = torch.cat([fn(dn[t * k:(t + 1) * k], w[t * k:(t + 1) * k], k) for t in range(Tmax)], 0)
            for T in Ts:
                y = fn(dn[:T * k], w[:T * k], k)
                tb = C.combine_truth(dn[:T * k], w[:T * k], k)
                r = {"path": name, "T": T, **C.rows_equal_to_singles(y, singles[:T]),
                     "bound_ratio": C.bound_ratio(y, tb["truth"], tb["bound"]),
                     "rel_rms_err": C.rel_rms_err(y, tb["truth"])}
                if name == "combine_rows":
                    r["bit_equal_to_chain"] = bool(C.bit_equal(y, chain(dn[:T * k], w[:T * k], k)).all())
                recs.append(r)
    return {"k": k, "H": H, "seed": seed, "records": recs, "kernel_present": combine_rows is not None}


# --------------------------------------------------------------------------------------------------------- main --
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
                         "allow_bf16_reduced_precision_reduction": b.allow_bf16_reduced_precision_reduction,
                         "allow_fp16_reduced_precision_reduction": b.allow_fp16_reduced_precision_reduction}
    v["sdp_flags"] = {"flash": torch.backends.cuda.flash_sdp_enabled(),
                      "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                      "math": torch.backends.cuda.math_sdp_enabled()}
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stack", choices=sorted(STACKS), required=False)
    ap.add_argument("--model", help="LOCAL snapshot directory of the checkpoint (the runner resolves the pinned sha)")
    ap.add_argument("--source", help="source checkpoint directory for the int4 expert pack (default: --model)")
    ap.add_argument("--arena")
    ap.add_argument("--calib")
    ap.add_argument("--fixture", default=os.path.join(HERE, "fixture.txt"))
    ap.add_argument("--out")
    ap.add_argument("--L", type=int, default=L_DEFAULT)
    ap.add_argument("--windows", type=lambda s: tuple(int(x) for x in s.split(",")), default=WINDOWS_DEFAULT)
    ap.add_argument("--widths", type=lambda s: tuple(int(x) for x in s.split(",")), default=WIDTHS_DEFAULT)
    ap.add_argument("--replay-ns", type=lambda s: tuple(int(x) for x in s.split(",")), default=REPLAY_NS_DEFAULT)
    ap.add_argument("--subarms", default=None, help="comma list to run a subset (rehearsal); default: the registered table")
    ap.add_argument("--no-fuse-qkv", action="store_true")
    ap.add_argument("--no-paged", action="store_true", help="skip paged sub-arms (a box whose fp8 kernels do not run)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--list-subarms", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.list_subarms:
        print(json.dumps(SUBARMS, indent=1))
        return 0
    for req in ("stack", "model", "arena", "calib", "out"):
        if getattr(a, req) is None:
            ap.error(f"--{req} is required")
    a.source = a.source or a.model
    subs = SUBARMS[a.stack]
    if a.subarms:
        want = a.subarms.split(",")
        unknown = [w for w in want if w not in {s["name"] for s in subs}]
        if unknown:
            ap.error(f"unknown sub-arms for {a.stack}: {unknown}")
        subs = [s for s in subs if s["name"] in want]
    if a.no_paged:
        subs = [s for s in subs if s["attn"] != "paged"]
    if a.dry_run:
        print(json.dumps({"stack": a.stack, "subarms": [s["name"] for s in subs], "L": a.L, "windows": a.windows,
                          "widths": a.widths, "replay_ns": a.replay_ns}, indent=1))
        return 0
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(1689)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    fx = load_fixture(a.fixture, tok, a.L)
    model, census = build_stack(a)
    ids = torch.tensor([fx["ids"]], dtype=torch.long, device=a.device)
    rec = {"lane": "P63", "stack": a.stack, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "sm_count": (torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else None),
           "versions": _versions(), "env": {k: v for k, v in os.environ.items() if k.startswith(("E4B_", "GNF4_"))},
           "census": census, "fixture": {k: v for k, v in fx.items() if k != "ids"},
           "config": {"L": a.L, "windows": list(a.windows), "widths": list(a.widths), "replay_ns": list(a.replay_ns)},
           "subarms": {}, "refused": {}}
    cap = SiteCapture(model)
    paged = None
    if any(s["attn"] == "paged" for s in subs):
        try:
            paged = Paged(model, a.L, max(a.widths), a.device)
        except Exception as e:                              # recorded, and the paged sub-arms read UNRUN
            rec["paged_unavailable"] = repr(e)[:400]
            subs = [s for s in subs if s["attn"] != "paged"]
    details, controls, x0, base_name = {}, {}, None, None
    first = True
    for sa in subs:
        why = refusal(a.stack, sa)
        if why:
            rec["refused"][sa["name"]] = why
            print(f"P63 {a.stack}/{sa['name']}: {why}", flush=True)
            continue
        t0 = time.time()
        try:
            res, ctrl, det, xr = run_subarm(model, cap, ids, sa, a, paged, first, a.stack, census["hf_attn_impl"])
        except Exception as e:
            import traceback
            rec["subarms"][sa["name"]] = {"error": repr(e)[:600], "tb": traceback.format_exc()[-3000:]}
            print(f"P63 {a.stack}/{sa['name']}: ERROR {e!r}", flush=True)
            continue
        res["wall_s"] = round(time.time() - t0, 1)
        rec["subarms"][sa["name"]] = res
        details[sa["name"]] = det
        controls[sa["name"]] = {"logits": ctrl["logits"], **{s: ctrl[s] for s in ("layer_out",)}}
        if first:
            controls["_first_full"] = ctrl
            base_name = sa["name"]
            x0 = xr.get("L0.experts")
        first = False
        m = res["modes"]
        print(f"P63 {a.stack}/{sa['name']}: " + " ".join(
            f"{k}: exact {v['exact_positions']}/{v['positions']} flips {v['argmax_flips']} "
            f"first-diff-layer {v['min_first_diff_layer']}" for k, v in m.items()) + f" ({res['wall_s']} s)", flush=True)
    # cross-sub-arm: the T = 1 control is expected independent of the grouping flags (they only act at T > 1) and of
    # the backend-free toggles; combine0's control against the default's sizes B393's difference end to end
    base = controls.get("_first_full")
    cross = {}
    if base is not None:
        for name, c in controls.items():
            if name.startswith("_") or name == base_name:
                continue
            lc = C.logits_stats(base["logits"], c["logits"])
            lo = C.row_stats(base["layer_out"], c["layer_out"])
            eq_layers = lo["equal"].all(0)
            cross[f"{base_name} vs {name} (T=1 controls)"] = {
                "logits_bit_equal_positions": int(lc["equal"].sum()), "positions": int(lc["equal"].numel()),
                "argmax_flips": int((~lc["argmax_equal"]).sum()), "kl_max": float(lc["kl"].max()),
                "kl_mean": float(lc["kl"].mean()),
                "first_layer_not_bit_equal": (int((~eq_layers).nonzero()[0]) if (~eq_layers).any() else None)}
    rec["cross_controls"] = cross
    if x0:
        try:
            rec["kernel_census"] = kernel_census(model, a.stack, x0, (1, 16, 17, a.L), a.device)
        except Exception as e:
            import traceback
            rec["kernel_census"] = {"error": repr(e)[:600], "tb": traceback.format_exc()[-3000:]}
    text = getattr(model.config, "text_config", None) or model.config
    k = getattr(text, "num_experts_per_tok", None) or 8
    rec["combine_census"] = combine_census(int(k), int(text.hidden_size), (1, 2, 16, 17, 64, a.L), a.device)
    cap.remove()
    if torch.cuda.is_available():
        rec["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    with open(os.path.join(a.out, "p63_arm.json"), "w") as f:
        json.dump(rec, f, indent=1)
    torch.save(details, os.path.join(a.out, "p63_detail.pt"))
    bad = [n for n, s in rec["subarms"].items() if "error" in s]
    det_fail = [n for n, s in rec["subarms"].items() if s.get("determinism") and not all(s["determinism"].values())]
    print("P63ARM " + json.dumps({"stack": a.stack, "subarms_run": [n for n in rec["subarms"] if n not in bad],
                                  "errors": bad, "refused": list(rec["refused"]), "determinism_failed": det_fail,
                                  "census": {k2: census.get(k2) for k2 in ("int4_linear", "int4_linear_k16",
                                                                            "int4_expert_layers", "fuse_qkv_n",
                                                                            "fuse_t1_glue_n", "fuse_t1_glue_r2_n",
                                                                            "fuse_router_epilogue_n")}}), flush=True)
    if det_fail:
        return 14
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
