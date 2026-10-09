# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Calibrated int4 for the serving attention projections (and, opted in
via ``E4B_SERVE_LMHEAD_INT4_CALIB=1``, the output head)
(``E4B_SERVE_ATTN_INT4_CALIB=1``).

:mod:`int4_attn` packs each attention projection by rounding every weight
to its nearest grid point. That lane was refused on quality (+0.0558 ppl
against a 0.05 gate), and the fp8 lane that followed showed why the
obvious fix is not a fix: e4m3 carried 4.6x lower WEIGHT error than int4
and bought only ~12% less perplexity cost. Weight error is not what the
gate measures.

This module keeps the same grid, the same packed bytes and the same
kernel, and changes only WHICH grid point each weight lands on. A short
calibration pass records, for every attention projection, how the
model's own activations excite each input channel (``H = 2 X X^T``);
the kernel package's ``gptq_pack_int4_b32`` then chooses grid points
that minimise the projection's OUTPUT error under that weighting,
pushing each column's rounding residual into the columns that follow.
The comparison engine's checkpoint is quantised this way (GPTQ, group
128) and serves its attention in int4 at a quality its users accept;
ours never had the calibration, only the format.

Calibration runs FORWARD only, on the text the K8 bake already carries,
in 512-token chunks; each projection's Hessian is accumulated on the
GPU in fp32 (16 MB for a 2048-wide input, 64 MB for o_proj's 4096) and
freed as soon as that projection is packed. Engagement is the banner
``ATTNINT4 calibrated: N projections`` plus census presence of the int4
GEMV, exactly as the uncalibrated lane.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable

import torch
from torch import nn

__all__ = ["calibrate_attention_hessians", "enable_serve_attn_int4_calib",
           "dump_attn_int4_artifact", "enable_serve_attn_int4_from_artifact"]


def _attention_linears(model) -> Dict[str, nn.Linear]:
    """Every ``nn.Linear`` that is a direct child of an ``*Attention``
    module -- the same structural rule :mod:`int4_attn` swaps on, so the
    calibrated lane patches exactly the set the uncalibrated one does."""
    out: Dict[str, nn.Linear] = {}
    for mname, mod in model.named_modules():
        if not type(mod).__name__.endswith("Attention"):
            continue
        for cname, child in mod.named_children():
            if type(child) is nn.Linear:
                out[f"{mname}.{cname}"] = child
    return out


_LM_HEAD_ENV = "E4B_SERVE_LMHEAD_INT4_CALIB"


def _output_head(model):
    """The model's output projection as ``(qualified_name, nn.Linear)``,
    or None. Found through ``get_output_embeddings`` (the transformers
    contract) with ``lm_head`` as the fallback, and located in the module
    tree by identity so the swap goes on the right parent."""
    head = None
    get = getattr(model, "get_output_embeddings", None)
    if callable(get):
        try:
            head = get()
        except Exception:
            head = None
    if head is None:
        head = getattr(model, "lm_head", None)
    if type(head) is not nn.Linear:
        return None
    for name, mod in model.named_modules():
        if mod is head:
            return name, head
    return None


_DENSE_MLP_ENV = "E4B_SERVE_DENSE_INT4_CALIB"


def _dense_mlp_linears(model) -> Dict[str, nn.Linear]:
    """Every ``nn.Linear`` that is a direct child of a decoder layer's
    DENSE ``mlp`` (a module whose class name ends with ``MLP``: Gemma-4's
    ``Gemma4TextMLP`` beside its routed experts). A routed block under
    the same attribute name (``*SparseMoeBlock``, ``*MoE``, ``GptOssMLP``
    with a router and fused experts) is never an MLP by this rule, so a
    router's ``gate`` Linear is not selected; nested shared experts are
    not direct children and are not selected either."""
    out: Dict[str, nn.Linear] = {}
    for lname, layer in model.named_modules():
        if not type(layer).__name__.endswith("DecoderLayer"):
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not type(mlp).__name__.endswith("MLP"):
            continue
        if any(True for _ in mlp.named_parameters(recurse=False)):
            continue
        for cname, child in mlp.named_children():
            if type(child) is nn.Linear:
                out[f"{lname}.mlp.{cname}"] = child
    return out


def _int4_targets(model, include_attention: bool = True,
                  include_head: bool = False,
                  include_dense_mlp: bool = False) -> Dict[str, nn.Linear]:
    """The projections one calibrated enable packs: the attention set
    (the structural rule above) and, opted in, the output head and the
    dense MLP beside a routed block (Gemma-4: 3 x 30 bf16 GEMVs, the
    largest unquantised slice after attention in its census).

    The output head was measured +0.18 ppl UNCALIBRATED and refused
    (``int4_attn``); the calibrated packer is what turned attention from
    -0.006 to -0.042, so the head gets the same packer under its own
    flag and its own K8 arm -- never silently alongside attention. On a
    model with tied embeddings the head's int4 store is a second copy;
    the embedding keeps its bf16 table untouched (the swap replaces the
    Linear module, not the shared Parameter)."""
    lins = _attention_linears(model) if include_attention else {}
    if include_dense_mlp:
        dense = _dense_mlp_linears(model)
        if not dense:
            raise RuntimeError(
                f"{_DENSE_MLP_ENV}=1: no decoder layer carries a dense MLP "
                "(a *MLP module under .mlp with Linear children) -- refusing "
                "a vacuous enable")
        for n, lin in dense.items():
            if lin.bias is not None:
                raise RuntimeError(
                    f"{_DENSE_MLP_ENV}=1: {n} carries a bias; this path "
                    "stores weight-only int4 -- refusing rather than dropping it")
        lins.update(dense)
    if include_head:
        found = _output_head(model)
        if found is None:
            raise RuntimeError(
                f"{_LM_HEAD_ENV}=1: the model has no nn.Linear output head "
                "to pack (get_output_embeddings / lm_head)")
        name, head = found
        if head.bias is not None:
            raise RuntimeError(
                f"{_LM_HEAD_ENV}=1: the output head carries a bias; "
                "weight-only int4 refuses rather than dropping it")
        lins[name] = head
    return lins


@torch.no_grad()
def calibrate_attention_hessians(model, batches: Iterable[torch.Tensor],
                                 device=None, hessian_device="cpu",
                                 include_attention: bool = True,
                                 include_head: bool = False,
                                 include_dense_mlp: bool = False,
                                 ) -> Dict[str, torch.Tensor]:
    """Run ``batches`` (token-id tensors ``[B, T]``) through ``model`` and
    return ``{qualified_name: H}`` with ``H = 2 X X^T`` over every input
    row each attention projection saw. Hooks are removed on exit even if
    a batch raises.

    Each batch's Gram is computed on the model's device; the running
    Hessians LIVE on ``hessian_device`` (CPU by default). Keeping them on
    the card OOMed Mixtral-8x7B's calibration on a 32 GB GPU: 128
    projections x 64 MB beside a 23 GB model (receipts P24-GEN-B)."""
    from gptq_pack import HessianAccumulator

    lins = _int4_targets(model, include_attention, include_head, include_dense_mlp)
    if not lins:
        raise RuntimeError("calibration found no attention projections")
    dev = device or next(model.parameters()).device
    accs = {n: HessianAccumulator(lin.in_features, device=hessian_device)
            for n, lin in lins.items()}
    handles = []
    for n, lin in lins.items():
        def _hook(mod, inputs, _n=n):
            accs[_n].add(inputs[0])            # Gram where the activations are
        handles.append(lin.register_forward_pre_hook(_hook))
    try:
        for ids in batches:
            model(ids.to(dev))
    finally:
        for h in handles:
            h.remove()
    return {n: a.H for n, a in accs.items()}


def enable_serve_attn_int4_calib(model, hessians: Dict[str, torch.Tensor],
                                 include_attention: bool = True,
                                 include_head: bool = False,
                                 include_dense_mlp: bool = False) -> int:
    """Swap every attention projection (and, opted in, the output head)
    for the int4 store, packed with calibration. A projection without a
    Hessian is NOT silently packed uncalibrated -- that would mix two
    quantisers under one banner and make the quality gate ambiguous; it
    raises instead."""
    from .int4_attn import resolve_smallm, resolve_wide
    smallm = resolve_smallm(None)          # K16 route (P5 read): auto by default, same rule as the RTN enable
    wide = resolve_wide(smallm)            # rows 17..64 on the same kernel: opt-in, same rule as the RTN enable
    from .int4_attn import Int4Linear, _kernels
    try:
        _kernels()
    except ImportError as e:
        raise RuntimeError(
            "E4B_SERVE_ATTN_INT4_CALIB=1 needs grouped-nf4-gemm with "
            f"int4_b32 and gptq_pack (missing: {e})") from e
    from gptq_pack import gptq_pack_int4_b32

    lins = _int4_targets(model, include_attention, include_head, include_dense_mlp)
    missing = sorted(n for n in lins if n not in hessians)
    if missing:
        raise RuntimeError(
            f"calibrated enable: {len(missing)} attention projections have "
            f"no Hessian (first: {missing[0]}); refusing to pack them "
            "uncalibrated under the calibrated banner")
    n = 0
    for name, lin in lins.items():
        if "." in name:
            parent_name, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, child = model, name        # a top-level lm_head
        # Int4Linear packs on the CPU copy of the weight; the Cholesky of
        # a 4096x4096 fp32 Hessian there is seconds, and it keeps the GPU
        # free for the store buffers being built alongside
        H = hessians[name].to("cpu")

        def packer(w, _H=H):
            return gptq_pack_int4_b32(w, _H)
        new = Int4Linear(lin, packer=packer, smallm=smallm, wide=wide)
        new._e4b_calibrated = True     # recorded per projection by dump_attn_int4_artifact
        setattr(parent, child, new)
        hessians[name] = None          # free the 16-64 MB as we go
        n += 1
    if n == 0:
        raise RuntimeError("E4B_SERVE_ATTN_INT4_CALIB=1 matched no "
                           "attention projections -- refusing a vacuous enable")
    return n


def _group_of(model, name: str, mod) -> str:
    """Which enable group a live int4 projection belongs to, from the module tree: the
    same structural rules :func:`_int4_targets` selects by."""
    # _output_head() only finds an nn.Linear, and a packed head is an Int4Linear by now
    get = getattr(model, "get_output_embeddings", None)
    try:
        head = get() if callable(get) else None
    except Exception:
        head = None
    if mod is head or mod is getattr(model, "lm_head", None):
        return "head"
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    kind = type(parent).__name__
    if kind.endswith("Attention"):
        return "attention"
    if kind.endswith("MLP"):
        return "dense_mlp"
    raise RuntimeError(f"{name}: an int4 projection under {kind!r} belongs to no enable group")


def dump_attn_int4_artifact(model, artifact_dir: str, *,
                            allow_unknown_revision: bool = False,
                            calibration_token_stream_sha: str | None = None) -> dict:
    """Serialise the live int4 attention stores (and the head / dense MLPs, when packed)
    as a hash-pinned artifact, the attention counterpart of the expert pack (#674).

    Until this existed the calibrated attention was re-derived from a Hessian pass on
    every load, so a ``pack_fingerprint`` identified only the expert half of the served
    stack -- and P55x measured the attention half as the one carrying the quality.

    Call it AFTER the calibrated enable and BEFORE ``qkv_fuse``: fused modules are
    refused, because their names do not exist on the model a licensed load installs
    onto (fuse after loading; the fused bytes are the parts' bytes concatenated).
    Refuses an unknown checkpoint revision unless ``allow_unknown_revision`` (or
    ``E4B_INT4_DUMP_ALLOW_UNKNOWN_REVISION=1``), exactly as the expert dump does. The
    returned manifest's ``pack_fingerprint`` is also attached to the model, so a
    receipt of the calibrating run names the bytes it served."""
    from .int4_attn import Int4Linear
    from .pack_manifest import (
        ATTN_LAYOUT, ATTN_PROVENANCE_ATTR, provenance_from_model, require_model_revision,
        toolchain_record, write_named_artifact,
    )
    mods = sorted((n, m) for n, m in model.named_modules() if isinstance(m, Int4Linear))
    if not mods:
        raise RuntimeError("dump_attn_int4_artifact: no int4 projections on the model")
    fused = [n for n, m in mods if getattr(m, "_e4b_fused_parts", None)]
    if fused:
        raise RuntimeError(
            f"dump_attn_int4_artifact: {len(fused)} projection(s) are fused (first: {fused[0]}); "
            "dump before qkv_fuse -- a licensed load installs onto the unfused module names")
    tensors, rows = {}, []
    for name, m in mods:
        tensors[(name, "packed")] = m.packed.reshape(m.N, m.K // 2)
        tensors[(name, "scales")] = m.scales.reshape(m.N, m.K // 32)
        if m.bias is not None:
            tensors[(name, "bias")] = m.bias
        rows.append({"name": name, "group": _group_of(model, name, m),
                     "N": int(m.N), "K": int(m.K), "bias": m.bias is not None,
                     "calibrated": bool(getattr(m, "_e4b_calibrated", False))})
    cfg = getattr(model, "config", None)
    rec = provenance_from_model(model) or {}
    allow = allow_unknown_revision or os.environ.get("E4B_INT4_DUMP_ALLOW_UNKNOWN_REVISION", "0") == "1"
    revision, missing = require_model_revision(
        rec.get("model_revision") or getattr(cfg, "_commit_hash", None), allow_unknown=allow)
    meta = {
        "model_id": rec.get("model") or getattr(cfg, "_name_or_path", None),
        "model_revision": revision,
        "model_revision_missing": True if missing else None,
        "layers": rows,
        "groups": sorted({r["group"] for r in rows}),
        "calibration_token_stream_sha": calibration_token_stream_sha,
        # informative, NOT identity: the expert pack this attention was calibrated beside --
        # its ARTIFACT's root when the build dumped one (#772), else the provenance's own
        # (the artifact's root after a licensed load; the live stores' hash otherwise)
        "expert_pack_fingerprint": rec.get("pack_artifact_fingerprint") or rec.get("pack_fingerprint"),
        "toolchain": toolchain_record(),
    }
    man = write_named_artifact(artifact_dir, tensors=tensors, meta=meta, layout=ATTN_LAYOUT)
    setattr(model, ATTN_PROVENANCE_ATTR, {
        "attn_pack_fingerprint": man["pack_fingerprint"], "attn_pack_layout": ATTN_LAYOUT,
        "attn_pack_source": "calibrated-and-dumped", "attn_pack_projections": len(rows)})
    return man


def enable_serve_attn_int4_from_artifact(model, artifact_dir: str, *,
                                         expected_fingerprint: str,
                                         include_attention: bool = True,
                                         include_head: bool = False,
                                         include_dense_mlp: bool = False) -> int:
    """Install a hash-pinned attention pack. Nothing is calibrated or re-quantised: every
    projection is built from the artifact's bytes (``Int4Linear.from_packed``), so two
    loads of one fingerprint serve the same function on any box.

    Refuses -- never falls back to calibrating -- on a fingerprint, layout or checkpoint
    revision mismatch, a corrupt payload, a live model without ``config._commit_hash``,
    or a projection set that differs from the one the enable flags select on this model
    (a partial licensed load is not a licensed load)."""
    from .int4_attn import Int4Linear, _kernels, resolve_smallm, resolve_wide
    from .pack_manifest import (
        ATTN_LAYOUT, ATTN_PROVENANCE_ATTR, PackManifestError, int4_store_dims,
        load_named_payload_tensors, parse_fingerprint, verify_artifact,
    )
    parse_fingerprint(expected_fingerprint)
    if not artifact_dir:
        raise PackManifestError("expected_fingerprint is set but artifact_dir is empty -- refusing")
    live_rev = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not live_rev:
        raise PackManifestError(
            "licensed attention-pack load requires config._commit_hash on the live model "
            "(the commit the loader recorded) -- refusing")
    man = verify_artifact(artifact_dir, expected_fingerprint=expected_fingerprint,
                          expected_model_revision=live_rev, expected_layout=ATTN_LAYOUT)
    tensors = load_named_payload_tensors(artifact_dir, man)
    rows = {r["name"]: r for r in (man.get("layers") or [])}
    targets = _int4_targets(model, include_attention, include_head, include_dense_mlp)
    if set(rows) != set(targets):
        extra, absent = sorted(set(rows) - set(targets)), sorted(set(targets) - set(rows))
        raise PackManifestError(
            f"the artifact's projections are not the set this enable selects on this model "
            f"({len(absent)} not in the artifact, first {absent[:1]}; {len(extra)} not on the "
            f"model or not selected, first {extra[:1]}) -- refusing a partial licensed load")
    try:
        _kernels()
    except ImportError as e:
        raise RuntimeError(f"attention-pack load needs grouped-nf4-gemm with int4_b32 (missing: {e})") from e
    smallm = resolve_smallm(None)
    wide = resolve_wide(smallm)
    # Validate EVERY projection before swapping ANY: a refusal on the last one must not leave
    # the model half-installed from the pack and half on its own bf16 weights.
    staged = []
    for name, lin in targets.items():
        try:
            packed, scales = tensors[(name, "packed")], tensors[(name, "scales")]
        except KeyError as e:
            raise PackManifestError(f"{name}: the artifact is missing a store tensor {e}") from e
        if packed.dtype != torch.uint8:
            raise PackManifestError(f"{name}: packed payload is {packed.dtype}, not uint8")
        N, K = int4_store_dims(packed, scales)
        row = rows[name]
        if (N, K) != (lin.out_features, lin.in_features) or (N, K) != (row["N"], row["K"]):
            raise PackManifestError(
                f"{name}: the bytes are an [{N}, {K}] store; the live projection is "
                f"[{lin.out_features}, {lin.in_features}] and the manifest says [{row['N']}, {row['K']}]")
        bias = tensors.get((name, "bias"))
        if (bias is not None) != (lin.bias is not None) or (bias is not None) != bool(row["bias"]):
            raise PackManifestError(f"{name}: the artifact and the live projection disagree on a bias")
        staged.append((name, lin, packed, scales, bias, N, K, row))
    n = 0
    for name, lin, packed, scales, bias, N, K, row in staged:
        dev = lin.weight.device
        new = Int4Linear.from_packed(packed.to(dev), scales.to(dev), N, K,
                                     bias=None if bias is None else bias.to(dev), smallm=smallm, wide=wide)
        new._e4b_calibrated = bool(row.get("calibrated"))
        if "." in name:
            parent_name, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), child, new)
        else:
            setattr(model, name, new)
        n += 1
    setattr(model, ATTN_PROVENANCE_ATTR, {
        "attn_pack_fingerprint": man["pack_fingerprint"], "attn_pack_layout": ATTN_LAYOUT,
        "attn_pack_source": "artifact", "attn_pack_projections": n})
    return n


_ARTIFACT_ENV = "E4B_SERVE_ATTN_INT4_ARTIFACT"
_FINGERPRINT_ENV = "E4B_SERVE_ATTN_INT4_FINGERPRINT"
_DUMP_ENV = "E4B_SERVE_ATTN_INT4_DUMP"


def enable_from_env(model, batches: Iterable[torch.Tensor]) -> int:
    """Harness convenience: calibrate then enable when the flags are set.
    ``E4B_SERVE_ATTN_INT4_CALIB=1`` packs the attention projections;
    ``E4B_SERVE_LMHEAD_INT4_CALIB=1`` packs the output head and
    ``E4B_SERVE_DENSE_INT4_CALIB=1`` the dense MLP beside a routed block
    (each alone, or beside attention). All off is a no-op; any flag alone
    is an enable of that set, never a silent ignore.

    ``E4B_SERVE_ATTN_INT4_ARTIFACT=<dir>`` with ``E4B_SERVE_ATTN_INT4_FINGERPRINT=sha256:...``
    installs those groups from a hash-pinned pack instead of calibrating (#674); an
    artifact without a fingerprint, or without a flag naming its groups, refuses.
    ``E4B_SERVE_ATTN_INT4_DUMP=<dir>`` writes the pack after a calibrated enable."""
    attn = os.environ.get("E4B_SERVE_ATTN_INT4_CALIB", "0") == "1"
    head = os.environ.get(_LM_HEAD_ENV, "0") == "1"
    dense = os.environ.get(_DENSE_MLP_ENV, "0") == "1"
    artifact = os.environ.get(_ARTIFACT_ENV)
    if not attn and not head and not dense:
        if artifact:
            raise RuntimeError(
                f"{_ARTIFACT_ENV} is set but no E4B_SERVE_*_INT4_CALIB flag names the groups "
                "it should install -- refusing a silent no-op")
        return 0
    if artifact:
        fp = os.environ.get(_FINGERPRINT_ENV)
        if not fp:
            raise RuntimeError(
                f"{_ARTIFACT_ENV} is set without {_FINGERPRINT_ENV}: a pack is loaded only "
                "against the fingerprint it was licensed under -- refusing")
        return enable_serve_attn_int4_from_artifact(
            model, artifact, expected_fingerprint=fp, include_attention=attn,
            include_head=head, include_dense_mlp=dense)
    dump = os.environ.get(_DUMP_ENV)
    if dump:
        batches = list(batches)        # read twice: calibration, then the token-stream hash
    hess = calibrate_attention_hessians(model, batches,
                                        include_attention=attn,
                                        include_head=head,
                                        include_dense_mlp=dense)
    n = enable_serve_attn_int4_calib(model, hess,
                                     include_attention=attn,
                                     include_head=head,
                                     include_dense_mlp=dense)
    if dump:
        from .pack_manifest import token_stream_sha
        dump_attn_int4_artifact(model, dump, calibration_token_stream_sha=token_stream_sha(batches))
    return n
