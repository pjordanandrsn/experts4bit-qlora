# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The merged-arena writer (experts4bit_qlora/merged_arena.py; bench/d0/ARENA-DESIGN.md). CPU only.

The base arena is written by grouped-nf4-gemm's own bake (``nvme_bake_nf4.bake_nf4`` on a tiny MoE, its default
quantizer on the CPU), never hand-built, so parity with it is parity with the bake. The adapter is a real
backend-adapter/2 directory whose expert fingerprint is quant_fingerprint's over training's stacks. The function check
takes the delta from ExpertsLoRA's own low-rank path (the trainer's computation), never from the writer's formula, and
gate_up is square (2I = H) so that a transposed delta is shape-legal and must be caught by value.
"""
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil

import pytest

IN_CI = os.environ.get("CI") == "true"


def _need(name):
    return importlib.import_module(name) if IN_CI else pytest.importorskip(name)


torch = _need("torch")
_need("bitsandbytes")
_need("safetensors")

from experts4bit_qlora import merged_arena as MA  # noqa: E402

L, E, INTER, H, R, ALPHA = 2, 3, 64, 128, 2, 4
REV = "c" * 40
STACK = "model.layers.{}.mlp.experts"


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    W = {(li, e, proj): (torch.randn(*shape, generator=g) * 0.02).to(torch.bfloat16)
         for li in range(L) for e in range(E)
         for proj, shape in (("gate_proj", (INTER, H)), ("up_proj", (INTER, H)), ("down_proj", (H, INTER)))}
    W[(0, 0, "gate_proj")][0, :64] = 0                                          # a zero block (absmax 0)
    W[(0, 1, "down_proj")][3, :64] = -W[(0, 1, "down_proj")][3, :64].abs() - 0.001   # an all-negative block
    return W


def _snapshot(tmp_path, W, rev=REV):
    st = _need("safetensors.torch")
    snap = tmp_path / "models--org--tiny-moe" / "snapshots" / rev
    snap.mkdir(parents=True)
    tensors = {f"model.layers.{li}.mlp.experts.{e}.{proj}.weight": t for (li, e, proj), t in W.items()}
    st.save_file(tensors, str(snap / "model-00001-of-00001.safetensors"))
    wm = {k: "model-00001-of-00001.safetensors" for k in tensors}
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}), encoding="utf-8")
    return snap


def _bake(tmp_path, W, **kw):
    nb = _need("nvme_bake_nf4")
    snap = _snapshot(tmp_path, W)
    arena = str(tmp_path / "nf4.arena")
    nb.bake_nf4(str(snap), arena, quantize_fn=nb.default_quantize_expert(dev="cpu"), log=lambda *a, **k: None, **kw)
    return snap, arena


def _trained(W, lora_scale=0.0, seed=3, attention=False):
    """Training's tree: each layer's stack quantized as training does, inside ExpertsLoRA, with A and B set (B times
    ``lora_scale``; 0 is the untrained adapter, an exact zero delta)."""
    from torch import nn

    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.lora import ExpertsLoRA
    g = torch.Generator().manual_seed(seed)
    layers = nn.ModuleList()
    for li in range(L):
        layer = nn.Module()
        layer.mlp = nn.Module()
        gu = torch.stack([torch.cat([W[(li, e, "gate_proj")], W[(li, e, "up_proj")]], 0) for e in range(E)])
        dn = torch.stack([W[(li, e, "down_proj")] for e in range(E)])
        mod = ExpertsLoRA(Experts4bit.from_float(gu, dn, quant_type="nf4", blocksize=64), r=R, alpha=ALPHA)
        with torch.no_grad():
            for which in ("gate_up", "down"):
                a, b = getattr(mod, f"{which}_lora_A"), getattr(mod, f"{which}_lora_B")
                a.copy_(torch.randn(a.shape, generator=g))
                b.copy_(torch.randn(b.shape, generator=g) * lora_scale)
        layer.mlp.experts = mod
        if attention:                                   # a served-shaped attention block (q and o), for the merge test
            layer.self_attn = nn.Module()
            for proj in ("q_proj", "o_proj"):
                lin = nn.Linear(H, H, bias=False).to(torch.bfloat16)
                lin.weight.data = torch.randn(H, H, generator=g).to(torch.bfloat16)
                setattr(layer.self_attn, proj, lin)
        layers.append(layer)
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = layers
    return root


def _adapter(tmp_path, model, *, name="ad", fingerprint=None, scale_applied="match", schema="backend-adapter/2",
             tensors=None, form="fp32", revision=REV, alpha=ALPHA, attention=None):
    """A backend-adapter/2 directory as Loggetta writes one: the expert LoRA tensors, the files' sha256, and the v2
    fields, the expert fingerprint being quant_fingerprint's over the trained stacks."""
    from safetensors.torch import save_file

    from experts4bit_qlora import adapter_schema as S
    if tensors is None:
        tensors = {}
        for li, layer in enumerate(model.model.layers):
            for which in ("gate_up", "down"):
                for ab in ("A", "B"):
                    tensors[f"{STACK.format(li)}.{which}_lora_{ab}"] = \
                        getattr(layer.mlp.experts, f"{which}_lora_{ab}").detach().clone().contiguous()
    tensors = dict(tensors, **(attention or {}))
    d = tmp_path / name
    d.mkdir()
    save_file(tensors, str(d / "adapter.safetensors"), metadata={"format": schema, "backend": "experts4bit"})
    files = {"adapter.safetensors": {"sha256": hashlib.sha256((d / "adapter.safetensors").read_bytes()).hexdigest(),
                                     "bytes": (d / "adapter.safetensors").stat().st_size}}
    man = {"schema": schema, "backend": "experts4bit", "weights": "adapter.safetensors", "files": files,
           "resolved_model_revision": revision,
           "setup": {"quant_type": "nf4", "blocksize": 64, "r": R, "alpha": alpha, "attn_4bit": False,
                     "train_experts": True, "train_attention": bool(attention)},
           "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()}}
    if schema == "backend-adapter/2":
        fp = fingerprint or S.quant_fingerprint(model, "experts")
        man.update(expert_absmax={"form": form}, train_fused_qkv=0, attn_quant={"form": "bf16"}, versions={},
                   lora_scale_applied={"attention": alpha / R if attention else None,
                                       "experts": alpha / R if scale_applied == "match" else scale_applied},
                   fingerprints={"experts": fp, **({"attention": S.quant_fingerprint(model, "attention")}
                                                   if attention else {})},
                   run_id="merged-arena-test", created_at="2026-10-23T00:00:00Z")
        if scale_applied is None:
            man["lora_scale_applied"].pop("experts")
    (d / "adapter_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return str(d)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    _need("nvme_arena")
    W = _weights()
    tmp = tmp_path_factory.mktemp("bake")
    snap, arena = _bake(tmp, W)
    return W, snap, arena


def _rows(arena):
    """``{(layer, expert, which): (blocks, absmax)}`` bytes, read at the index's offsets."""
    idx = json.load(open(arena + ".index.json"))
    geo = {g["suffix"]: g for g in idx["segments"]}
    out = {}
    with open(arena, "rb") as f:
        for lay, e, off in idx["rows"]:
            for which in MA.WHICH:
                parts = []
                for kind in ("blocks", "absmax"):
                    g = geo[f"nf4.{which}_{kind}"]
                    f.seek(off + g["seg_off"])
                    parts.append(f.read(g["length"]))
                out[(lay, e, which)] = tuple(parts)
    return out


SHAPES = {"gate_up": (2 * INTER, H), "down": (H, INTER)}


def _deq(rows, key):
    n, k = SHAPES[key[2]]
    return MA._deq(*rows[key], n, k, MA._code())


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---- parity against the real bake ------------------------------------------------------------------------------


def test_the_model_free_fingerprint_equals_the_served_path(base):
    from experts4bit_qlora import adapter_schema as S
    W, _snap, arena = base
    model = _trained(W)
    idx = json.load(open(arena + ".index.json"))
    free = S.arena_expert_fingerprint(arena, MA._stand_ins({li: STACK.format(li) for li in range(L)}, idx))
    served = S.quant_fingerprint(model, "experts", source="arena", arena=arena)
    assert free["sha256"] == served["sha256"] == S.quant_fingerprint(model, "experts")["sha256"]
    assert free["absmax_by_stack"] == served["absmax_by_stack"] and free["modules"] == 2 * L


@pytest.mark.parametrize("source", ["arena", "checkpoint"])
def test_a_zero_delta_reproduces_the_bake_byte_for_byte(base, tmp_path, source):
    W, snap, arena = base
    ad = _adapter(tmp_path, _trained(W, lora_scale=0.0))
    out = str(tmp_path / "merged.arena")
    block = MA.write(arena, ad, out, source=source, checkpoint=str(snap) if source == "checkpoint" else None,
                     log=lambda *a: None)
    assert _sha(out) == _sha(arena)                               # the whole file, padding included
    assert block["base_qfp_v1"] == block["merged_qfp_v1"]
    rec = json.load(open(out + MA.RETENTION))
    assert rec["summary"]["all"]["rho_bar"] is None and rec["summary"]["all"]["f_bar"] == 0.0


def test_the_checkpoint_source_merges_into_the_raw_weight_not_into_q(base, tmp_path):
    """D0's two sources are two different merges: NF4(W + dW) from the checkpoint, NF4(Q + dW) from the arena's rows."""
    from bitsandbytes import functional as F
    W, snap, arena = base
    model = _trained(W, lora_scale=0.05)
    ad = _adapter(tmp_path, model)
    a, c = str(tmp_path / "a.arena"), str(tmp_path / "c.arena")
    MA.write(arena, ad, a, log=lambda *x: None)
    MA.write(arena, ad, c, source="checkpoint", checkpoint=str(snap), log=lambda *x: None)
    assert len({_sha(a), _sha(c), _sha(arena)}) == 3
    rows = _rows(c)
    for li in range(L):
        for e in range(E):
            for which, raw in (("gate_up", torch.cat([W[(li, e, "gate_proj")], W[(li, e, "up_proj")]], 0)),
                               ("down", W[(li, e, "down_proj")])):
                x = raw.float() + _trainer_delta(model, li, e, which)
                q, st = F.quantize_4bit(x.contiguous(), blocksize=64, quant_type="nf4", compress_statistics=False)
                assert torch.equal(_deq(rows, (li, e, which)), F.dequantize_4bit(q, st).float().view(x.shape))
    assert json.load(open(c + ".index.json"))["merge"]["source"] == "checkpoint"


# ---- the function, against the trainer's own computation ------------------------------------------------------------


def _trainer_delta(model, li, e, which):
    """ExpertsLoRA's low-rank path on the identity: x = I gives s * (I A^T) B^T = (s B A)^T, the trainer's delta."""
    mod = model.model.layers[li].mlp.experts
    A, B = getattr(mod, f"{which}_lora_A")[e].detach(), getattr(mod, f"{which}_lora_B")[e].detach()
    k = A.shape[1]
    with torch.no_grad():
        return mod._lora(torch.eye(k, dtype=A.dtype), A, B).float().T


def _nearest(base_rows, merged_rows, model):
    """For gate_up (square), which candidate the merged rows are nearest: S + dW, S + 2 dW or S + dW^T, summed over
    every expert. The delta's norm is about a third of S's, well past NF4's rounding noise."""
    dist = {"dW": 0.0, "2dW": 0.0, "dW^T": 0.0}
    for li in range(L):
        for e in range(E):
            key = (li, e, "gate_up")
            S, M = _deq(base_rows, key), _deq(merged_rows, key)
            d = _trainer_delta(model, li, e, "gate_up")
            assert d.shape == d.T.shape
            for name, cand in (("dW", S + d), ("2dW", S + 2 * d), ("dW^T", S + d.T)):
                dist[name] += float((M - cand).pow(2).sum())
    return min(dist, key=dist.get), dist


def _sized(W):
    """B scaled so |dW| is about a third of |S| per projection (checked)."""
    model = _trained(W, lora_scale=0.0018)
    d = _trainer_delta(model, 0, 0, "gate_up")
    rel = float(d.norm() / torch.cat([W[(0, 0, "gate_proj")], W[(0, 0, "up_proj")]]).float().norm())
    assert 0.15 < rel < 0.6, rel
    return model


def test_the_merged_rows_carry_the_trainers_delta_once_and_unrotated(base, tmp_path):
    W, _snap, arena = base
    model = _sized(W)
    out = str(tmp_path / "m.arena")
    MA.write(arena, _adapter(tmp_path, model), out, log=lambda *a: None)
    nearest, dist = _nearest(_rows(arena), _rows(out), model)
    assert nearest == "dW", dist


@pytest.mark.parametrize("mutant, caught_as", [("double", "2dW"), ("transpose", "dW^T")])
def test_a_mis_scaled_or_transposed_delta_is_caught_by_value(base, tmp_path, monkeypatch, mutant, caught_as):
    W, _snap, arena = base
    model = _sized(W)
    real = MA._Merge.delta

    def bad(self, lay, e, which):
        d = real(self, lay, e, which)
        return 2 * d if mutant == "double" else (d.T if d.shape[0] == d.shape[1] else d)

    monkeypatch.setattr(MA._Merge, "delta", bad)
    out = str(tmp_path / "m.arena")
    MA.write(arena, _adapter(tmp_path, model), out, log=lambda *a: None)
    nearest, dist = _nearest(_rows(arena), _rows(out), model)
    assert nearest == caught_as, dist


# ---- graded against D0, and the receipt -------------------------------------------------------------------------


def test_the_merged_rows_are_nf4_of_s_plus_dw_as_d0_computed_and_the_receipt_is_d0s_step1(base, tmp_path):
    from bitsandbytes import functional as F
    W, _snap, arena = base
    model = _trained(W, lora_scale=0.05)
    out = str(tmp_path / "m.arena")
    MA.write(arena, _adapter(tmp_path, model), out, log=lambda *a: None)
    base_rows, merged_rows = _rows(arena), _rows(out)
    rec = json.load(open(out + MA.RETENTION))
    sums = {}
    for li in range(L):
        for e in range(E):
            for which in MA.WHICH:
                key = (li, e, which)
                S = _deq(base_rows, key)
                d = _trainer_delta(model, li, e, which)
                q, st = F.quantize_4bit((S + d).contiguous(), blocksize=64, quant_type="nf4", compress_statistics=False)
                d0 = F.dequantize_4bit(q, st).float().view(S.shape)           # D0's NF4 repack, d0lib.targets()
                M = _deq(merged_rows, key)
                assert torch.equal(M, d0), key
                D, dd = (M - S).double(), d.double()
                acc = sums.setdefault((li, which), {"dot": 0.0, "dd": 0.0, "changed": 0, "numel": 0})
                acc["dot"] += float((D * dd).sum())
                acc["dd"] += float(dd.pow(2).sum())
                acc["changed"] += int((D != 0).sum())
                acc["numel"] += D.numel()
    for row in rec["rows"]:
        want = sums[(row["layer"], row["which"])]
        assert math.isclose(row["dot"], want["dot"], rel_tol=1e-6) and math.isclose(row["dd"], want["dd"], rel_tol=1e-6)
        assert row["changed"] == want["changed"] and row["numel"] == want["numel"]
    tot = {k: sum(v[k] for v in sums.values()) for k in ("dot", "dd")}
    assert math.isclose(rec["summary"]["all"]["rho_bar"], tot["dot"] / tot["dd"], rel_tol=1e-6)


# ---- provenance -----------------------------------------------------------------------------------------------


def test_the_merge_block_is_complete_and_its_fingerprint_is_the_served_one(base, tmp_path):
    from experts4bit_qlora import adapter_schema as S
    from experts4bit_qlora.adapter_dir import verify_adapter_dir
    W, _snap, arena = base
    model = _trained(W, lora_scale=0.05)
    ad = _adapter(tmp_path, model)
    out = str(tmp_path / "m.arena")
    block = MA.write(arena, ad, out, log=lambda *a: None)
    idx, man = json.load(open(out + ".index.json")), json.load(open(out + ".manifest.json"))
    assert idx["bake_mode"] == man["bake_mode"] == MA.MERGED_BAKE_MODE and idx["merge"] == man["merge"] == block
    base_idx = json.load(open(arena + ".index.json"))
    for key in ("segments", "rows", "row_stride", "arena_bytes", "model_dims", "snapshot", "quantizer"):
        assert idx[key] == base_idx[key], key
    v = verify_adapter_dir(ad)
    assert block["adapter_id"] == v.adapter_id and block["adapter_manifest_sha256"] == v.manifest_sha256
    assert block["expert_tensors_sha256"] == MA.expert_tensors_sha256(v) and block["scale"] == ALPHA / R
    assert block["base_index_sha256"] == _sha(arena + ".index.json") and block["source_revision"] == REV
    assert block["base_qfp_v1"] == S.quant_fingerprint(model, "experts")["sha256"]
    assert block["retention_sha256"] == _sha(out + MA.RETENTION) and block["trained_base"] == "fp32-absmax"
    # the merged fingerprint is what a server measures over the merged arena (the post-load check's input)
    served = S.quant_fingerprint(model, "experts", source="arena", arena=out)
    assert block["merged_qfp_v1"] == served["sha256"] != block["base_qfp_v1"]
    assert not any(p.endswith(".tmp") for p in os.listdir(tmp_path))


# ---- refusals -------------------------------------------------------------------------------------------------


def _refused(base, tmp_path, needle, **kw):
    W, snap, arena = base
    model = _trained(W, lora_scale=0.05)
    ad = kw.pop("ad", None) or _adapter(tmp_path, model, **kw.pop("adapter", {}))
    out = str(tmp_path / "m.arena")
    with pytest.raises(MA.MergeRefused, match=needle):
        MA.write(kw.pop("arena", arena), ad, out, log=lambda *a: None, **kw)
    assert not any(n.startswith("m.arena") for n in os.listdir(tmp_path)), os.listdir(tmp_path)


@pytest.mark.parametrize("applied, needle", [(None, "records no lora_scale_applied.experts"),
                                             (ALPHA / R * 2, "is not the setup's alpha/r")])
def test_the_expert_scale_is_read_from_the_modules_and_must_be_alpha_over_r(base, tmp_path, applied, needle):
    _refused(base, tmp_path, needle, adapter={"scale_applied": applied})


def test_an_adapter_trained_on_another_base_is_refused(base, tmp_path):
    from experts4bit_qlora import adapter_schema as S
    other = S.quant_fingerprint(_trained(_weights(seed=9)), "experts")
    _refused(base, tmp_path, "not the base this adapter was trained on", adapter={"fingerprint": other})


def test_a_v1_adapter_has_no_fingerprint_to_prove_the_base(base, tmp_path):
    _refused(base, tmp_path, "records no expert fingerprint", adapter={"schema": "backend-adapter/1"})


def test_missing_or_misshapen_expert_tensors_are_refused(base, tmp_path):
    W, _snap, _arena = base
    model = _trained(W, lora_scale=0.05)
    full = {f"{STACK.format(li)}.{w}_lora_{ab}": getattr(model.model.layers[li].mlp.experts, f"{w}_lora_{ab}")
            .detach().clone().contiguous() for li in range(L) for w in MA.WHICH for ab in "AB"}
    fp = __import__("experts4bit_qlora.adapter_schema", fromlist=["x"]).quant_fingerprint(model, "experts")
    short = {k: v for k, v in full.items() if not k.endswith("1.mlp.experts.down_lora_B")}
    _refused(base, tmp_path, "the adapter lacks 1 expert LoRA tensor", ad=_adapter(tmp_path, model, name="short",
                                                                                   tensors=short, fingerprint=fp))
    extra = dict(full, **{"model.layers.7.mlp.experts.down_lora_A": full[f"{STACK.format(0)}.down_lora_A"].clone()})
    _refused(base, tmp_path, "belong to no stack of this arena and would go unmerged",
             ad=_adapter(tmp_path, model, name="extra", tensors=extra, fingerprint=fp))
    bent = dict(full, **{f"{STACK.format(0)}.down_lora_A": full[f"{STACK.format(0)}.down_lora_A"].transpose(1, 2)
                         .contiguous()})
    _refused(base, tmp_path, "ExpertsLoRA's layout", ad=_adapter(tmp_path, model, name="bent", tensors=bent,
                                                                  fingerprint=fp))


def test_a_checkpoint_that_is_not_the_base_is_refused_and_nothing_is_left(base, tmp_path):
    other = _snapshot(tmp_path / "other", _weights(seed=5))
    _refused(base, tmp_path, "the checkpoint's weight, re-quantized with no delta, does not reproduce",
             source="checkpoint", checkpoint=str(other))


def test_the_self_check_refuses_rows_that_do_not_round_trip(base, tmp_path):
    """A base whose recorded fingerprint matches but whose row does not re-quantize to itself is refused by the
    self-check, not merged. A real bake always has a +-1 code in each block (its largest element); here one block's
    codes are all NF4 index 9 (0.0796), so re-quantizing it moves the block's absmax."""
    from experts4bit_qlora import adapter_schema as S
    _W, _snap, arena = base
    bad = str(tmp_path / "bad.arena")
    shutil.copy(arena, bad)
    shutil.copy(arena + ".index.json", bad + ".index.json")
    idx = json.load(open(bad + ".index.json"))
    g = next(s for s in idx["segments"] if s["suffix"] == "nf4.gate_up_blocks")
    with open(bad, "r+b") as f:
        f.seek(idx["rows"][1][2] + g["seg_off"] + 32 * 5)             # block 5: 64 codes in 32 bytes
        f.write(b"\x99" * 32)
    fp = S.quant_fingerprint(_trained(_W), "experts")
    fp = dict(fp, sha256=S.arena_expert_fingerprint(bad, MA._stand_ins({li: STACK.format(li) for li in range(L)},
                                                                         idx))["sha256"])
    _refused(base, tmp_path, "does not reproduce the base arena's row", arena=bad, adapter={"fingerprint": fp})


def test_a_merged_arena_is_never_a_base_for_the_writer(base, tmp_path):
    W, _snap, arena = base
    model = _trained(W, lora_scale=0.05)
    first = str(tmp_path / "first.arena")
    MA.write(arena, _adapter(tmp_path, model, name="a1"), first, log=lambda *a: None)
    with pytest.raises(MA.MergeRefused, match="a merged arena is never a base"):
        MA.write(first, _adapter(tmp_path, model, name="a2"), str(tmp_path / "second.arena"), log=lambda *a: None)


def test_an_existing_output_is_never_overwritten(base, tmp_path):
    W, _snap, arena = base
    out = tmp_path / "m.arena"
    out.write_bytes(b"keep")
    with pytest.raises(MA.MergeRefused, match="refusing to overwrite"):
        MA.write(arena, _adapter(tmp_path, _trained(W)), str(out), log=lambda *a: None)
    assert out.read_bytes() == b"keep"


# ---- no reader takes a merged arena as a base ---------------------------------------------------------------------


def test_the_loader_refuses_a_merged_arena_before_reading_any_model(base, tmp_path):
    from experts4bit_qlora import loader
    W, _snap, arena = base
    out = str(tmp_path / "m.arena")
    MA.write(arena, _adapter(tmp_path, _trained(W, lora_scale=0.05)), out, log=lambda *a: None)
    with pytest.raises(ValueError, match="is a merged arena"):
        loader.load_moe_4bit_streaming("org/never-read", "cpu", torch.bfloat16, 8, 16, arena=out)


def _arena_openers():
    """``{"<module>:<function>": (first ColdTier line or None, first refusal line or None)}`` for every function in
    the package that opens an arena: calls grouped-nf4-gemm's ``load_index`` or builds a ``ColdTier``."""
    import ast
    import pathlib

    root = pathlib.Path(MA.__file__).parent
    out = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "merged_arena.py":                       # the guard itself reads the index
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = {}
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    f = node.func
                    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                    calls.setdefault(name, []).append(node.lineno)
            if "load_index" in calls or "ColdTier" in calls:
                refusal = calls.get("refuse_merged", []) + calls.get("refuse_merged_path", [])
                out[f"{path.relative_to(root).as_posix()}:{fn.name}"] = (min(calls.get("ColdTier", [0])) or None,
                                                              min(refusal) if refusal else None)
    return out


def test_every_arena_opener_in_the_package_refuses_a_merged_arena_before_any_tier():
    openers = _arena_openers()
    assert {"loader.py:load_moe_4bit_streaming", "engines/nvme_experts.py:enable_nvme_residency",
            "engines/hybrid.py:enable_hybrid_tier", "engines/nvme_train.py:enable_nvme_train_residency",
            "engines/nvme_experts.py:enable_mxfp4_nvme_residency"} <= set(openers)
    for where, (tier, refusal) in openers.items():
        assert refusal is not None, f"{where} opens an arena without refuse_merged"
        assert tier is None or refusal < tier, f"{where} builds a ColdTier before refusing a merged arena"


@pytest.mark.parametrize("entry", ["nvme_residency", "hybrid_tier"])
def test_the_coldtier_readers_refuse_a_merged_arena_before_touching_the_model(base, tmp_path, entry):
    W, _snap, arena = base
    out = str(tmp_path / "m.arena")
    MA.write(arena, _adapter(tmp_path, _trained(W, lora_scale=0.05)), out, log=lambda *a: None)
    with pytest.raises(ValueError, match="is a merged arena"):
        if entry == "nvme_residency":
            from experts4bit_qlora.engines.nvme_experts import enable_nvme_residency
            enable_nvme_residency(None, out, [], hot_rows=1, device="cpu", pinned=False)
        else:
            from experts4bit_qlora.engines.hybrid import enable_hybrid_tier
            enable_hybrid_tier(None, out, None, hot_rows=1, device="cpu")


def test_an_adapter_trained_at_another_revision_is_refused(base, tmp_path):
    _refused(base, tmp_path, "the base arena was baked from c{40}, not the adapter's base d{40}",
             adapter={"revision": "d" * 40})


def test_a_checkpoint_at_another_commit_is_refused_even_with_the_same_weights(base, tmp_path):
    W, _snap, _arena = base
    other = _snapshot(tmp_path / "rev2", W, rev="e" * 40)
    _refused(base, tmp_path, f"is commit {'e' * 40}, the base arena's {REV}", source="checkpoint",
             checkpoint=str(other))


def _bake_patched(monkeypatch, how):
    nb = _need("nvme_bake_nf4")
    real = nb.bake_nf4

    def patched(*a, **kw):
        if how == "fewer":
            kw["layers"] = [0]
        idx = real(*a, **kw)
        if how == "geometry":
            idx = dict(idx, row_stride=idx["row_stride"] + 4096)
        return idx

    monkeypatch.setattr(nb, "bake_nf4", patched)


@pytest.mark.parametrize("how, needle", [("fewer", "the bake wrote fewer projections than the base arena holds"),
                                         ("geometry", "the checkpoint bake's row_stride differs from the base arena's")])
def test_a_checkpoint_bake_that_is_not_the_base_geometry_is_refused(base, tmp_path, monkeypatch, how, needle):
    _bake_patched(monkeypatch, how)
    _refused(base, tmp_path, needle, source="checkpoint", checkpoint=str(base[1]))


def test_the_checkpoint_bake_takes_the_base_arenas_own_recorded_inputs(base, tmp_path):
    """A base baked at a non-default input (here a layers subset) re-bakes from the checkpoint with that same input, so
    its geometry matches and the merge goes through: the re-bake reads the base index's layers, projections, source and
    MoE attribute, never the bake's defaults (which would bake both layers here and be refused)."""
    from experts4bit_qlora import adapter_schema as S
    W, snap, _arena = base
    nb = _need("nvme_bake_nf4")
    sub = str(tmp_path / "layer0.arena")
    nb.bake_nf4(str(snap), sub, layers=[0], quantize_fn=nb.default_quantize_expert(dev="cpu"), log=lambda *a, **k: None)
    idx = json.load(open(sub + ".index.json"))
    assert idx["n_layers"] == 1 and {r[0] for r in idx["rows"]} == {0}
    model = _trained(W, lora_scale=0.05)
    fp = dict(S.arena_expert_fingerprint(sub, MA._stand_ins({0: STACK.format(0)}, idx)), seconds=0.0)
    tensors = {f"{STACK.format(0)}.{w}_lora_{ab}": getattr(model.model.layers[0].mlp.experts, f"{w}_lora_{ab}")
               .detach().clone().contiguous() for w in MA.WHICH for ab in "AB"}
    ad = _adapter(tmp_path, model, tensors=tensors, fingerprint=fp)
    a, c = str(tmp_path / "a.arena"), str(tmp_path / "c.arena")
    MA.write(sub, ad, a, log=lambda *x: None)
    block = MA.write(sub, ad, c, source="checkpoint", checkpoint=str(snap), log=lambda *x: None)
    assert block["source"] == "checkpoint" and json.load(open(c + ".index.json"))["rows"] == idx["rows"]
    assert _sha(c) != _sha(sub) != _sha(a)


def test_a_partial_bake_is_never_a_base(base, tmp_path):
    W, _snap, _arena = base
    _psnap, partial = _bake(tmp_path / "partial", W, limit_experts=2)
    _refused(base, tmp_path, r"a partial bake \(2 of 3 experts\)", arena=partial)


def test_only_an_nf4_quantize_bake_is_a_base(base, tmp_path):
    _W, _snap, arena = base
    other = str(tmp_path / "reloc.arena")
    shutil.copy(arena, other)
    idx = json.load(open(arena + ".index.json"))
    json.dump(dict(idx, bake_mode="relocate-expert-tensors"), open(other + ".index.json", "w"))
    _refused(base, tmp_path, "bake_mode 'relocate-expert-tensors'; the writer reads only a bake_nf4 'nf4-quantize'",
             arena=other)


def _dq_fingerprint(W):
    """What a dq-trained /2 manifest records: training's stacks with the absmax compressed (training's default)."""
    from experts4bit_qlora import adapter_schema as S
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_
    model = _trained(W, lora_scale=0.05)
    assert compress_expert_absmax_(model) == L
    return S.quant_fingerprint(model, "experts")


def test_a_double_quantized_trained_base_merges_by_its_fp32_source(base, tmp_path):
    W, _snap, arena = base
    fp = _dq_fingerprint(W)
    assert fp["sha256"] != fp["fp32_source"]["sha256"]
    out = str(tmp_path / "m.arena")
    block = MA.write(arena, _adapter(tmp_path, _trained(W, lora_scale=0.05), fingerprint=fp, form="double-quantized"),
                     out, log=lambda *a: None)
    assert block["trained_base"] == "double-quantized" and block["base_qfp_v1"] == fp["fp32_source"]["sha256"]


@pytest.mark.parametrize("src, needle", [({"sha256": "2" * 64}, "not the base this adapter was trained on"),
                                         (None, "needs the fingerprint's fp32_source")])
def test_a_double_quantized_adapter_without_its_true_fp32_source_is_refused(base, tmp_path, src, needle):
    W, _snap, _arena = base
    fp = _dq_fingerprint(W)
    fp = {k: v for k, v in fp.items() if k != "fp32_source"} if src is None else dict(fp, fp32_source=dict(
        fp["fp32_source"], **src))
    _refused(base, tmp_path, needle, adapter={"fingerprint": fp, "form": "double-quantized"})


def test_refuse_merged_names_the_adapter_and_passes_a_base():
    MA.refuse_merged({"bake_mode": MA.BASE_BAKE_MODE}, "base.arena")
    with pytest.raises(ValueError, match="adapter abcdef012345"):
        MA.refuse_merged({"bake_mode": MA.MERGED_BAKE_MODE, "merge": {"adapter_id": "abcdef0123456789"}}, "m.arena")
    with pytest.raises(ValueError, match="is a merged arena"):
        MA.refuse_merged({"bake_mode": MA.BASE_BAKE_MODE, "merge": {}}, "m.arena")
    with pytest.raises(ValueError, match="bake_mode 'nf4-merged', adapter unrecorded"):
        MA.refuse_merged({"bake_mode": MA.MERGED_BAKE_MODE}, "m.arena")       # the mode alone is enough


# ---- the server's rules (bench/d0/ARENA-DESIGN.md, point 4) ---------------------------------------------------------


@pytest.fixture(scope="module")
def merged(base, tmp_path_factory):
    """A merged arena of a live expert adapter, its adapter directory and its index."""
    W, _snap, arena = base
    tmp = tmp_path_factory.mktemp("merged")
    model = _trained(W, lora_scale=0.05)
    ad = _adapter(tmp, model)
    out = str(tmp / "m.arena")
    MA.write(arena, ad, out, log=lambda *a: None)
    return model, ad, out, json.load(open(out + ".index.json"))


def _v(ad):
    from experts4bit_qlora.adapter_dir import verify_adapter_dir
    return verify_adapter_dir(ad)


def test_refusal_13_passes_only_a_merged_arena_served_as_written(merged, base, tmp_path):
    from experts4bit_qlora import adapter_schema as S
    W, _snap, arena = base
    model, ad, _out, idx = merged
    assert MA.merged_arena_refusals(_v(ad), idx, opt_in=True) == []
    assert MA.merged_arena_refusals(_v(ad), json.load(open(arena + ".index.json")), opt_in=False) == []   # a base

    def codes_and(why, got):
        assert [c for c, _ in got] == [13] * len(got) and any(why in r for _, r in got), got

    codes_and("with no adapter", MA.merged_arena_refusals(None, idx, opt_in=True))
    codes_and("only under the opt-in E4B_PAGED_MERGED_ARENA=1", MA.merged_arena_refusals(_v(ad), idx, opt_in=False))
    other = _adapter(tmp_path, _trained(W, lora_scale=0.05, seed=11), name="other")
    got = MA.merged_arena_refusals(_v(other), idx, opt_in=True)
    codes_and("merged from adapter", got)
    codes_and("not the ones the arena was merged from", got)
    rebased = _adapter(tmp_path, model, name="rebased", fingerprint=S.quant_fingerprint(_trained(_weights(seed=9)),
                                                                                         "experts"))
    codes_and("is not the base the merge started from", MA.merged_arena_refusals(_v(rebased), idx, opt_in=True))
    rescaled = _adapter(tmp_path, model, name="rescaled", alpha=2 * ALPHA)   # the same tensors, another alpha/r
    codes_and("is not the merge's", MA.merged_arena_refusals(_v(rescaled), idx, opt_in=True))
    codes_and("writer_version 99", MA.merged_arena_refusals(
        _v(ad), dict(idx, merge=dict(idx["merge"], writer_version=99)), opt_in=True))
    codes_and("cannot read", MA.merged_arena_refusals(_v(ad), dict(idx, merge=dict(idx["merge"], writer="x")),
                                                      opt_in=True))


def _servable(ad, idx, **kw):
    from experts4bit_qlora.adapter_dir import servable_refusals
    return servable_refusals(_v(ad), served_revision=None, requested_revision=None, arena_index=idx, levers={},
                             check_served_revision=False, **kw)


def test_a_live_expert_adapter_on_its_merged_arena_is_never_refusal_1(merged, base):
    """Refusal 1's exemption is decided before the load, with 13: a live expert adapter reaches 13 or service, and on a
    base arena it stays refusal 1, as before."""
    _W, _snap, arena = base
    _model, ad, _out, idx = merged
    assert _servable(ad, idx, merged_opt_in=True) == []
    codes = [c for c, _ in _servable(ad, idx, merged_opt_in=False)]
    assert codes[0] == 13 and 1 in codes                         # not exempt: both named, 13 first
    assert [c for c, _ in _servable(ad, json.load(open(arena + ".index.json")), merged_opt_in=True)] == [1]


def test_after_the_load_the_served_experts_are_compared_with_the_merged_fingerprint(merged, base, tmp_path):
    from experts4bit_qlora.adapter_dir import base_match, fingerprint_check
    _W, _snap, arena = base
    model, ad, out, idx = merged
    refusals, report = fingerprint_check(_v(ad), model, out, merge=idx["merge"])
    assert refusals == [] and report["experts"]["status"] == "verified (merged)"
    assert report["experts"]["served"] == idx["merge"]["merged_qfp_v1"] != report["experts"]["trained"]
    assert base_match(_v(ad), fingerprints=report)["experts"].startswith("merged (qfp-v1 ")
    # the base arena's bytes under the merged index: refused, and no override covers it
    swapped = str(tmp_path / "swapped.arena")
    shutil.copy(arena, swapped)
    shutil.copy(out + ".index.json", swapped + ".index.json")
    for accept in (frozenset(), frozenset({"experts"})):
        refusals, report = fingerprint_check(_v(ad), model, swapped, merge=idx["merge"], accept_base_mismatch=accept)
        assert [c for c, _ in refusals] == [12] and "not the one its index records" in refusals[0][1]
        assert report["experts"]["status"] == "mismatch (merged)"
    # without the merge block (refusal 13 never matched it), the merged rows are a measured base mismatch, as before
    refusals, _report = fingerprint_check(_v(ad), model, out)
    assert [c for c, _ in refusals] == [12]


def test_only_the_arenas_own_merge_block_lets_a_reader_through(merged, monkeypatch):
    from experts4bit_qlora import loader
    _model, _ad, out, idx = merged
    MA.refuse_merged(idx, out, idx["merge"])
    MA.refuse_merged_path(out, idx["merge"])
    for bad in (None, dict(idx["merge"], adapter_id="0" * 64), {}):
        with pytest.raises(ValueError, match="is a merged arena"):
            MA.refuse_merged(idx, out, bad)
    with pytest.raises(ValueError, match="is a merged arena"):
        MA.refuse_merged({k: v for k, v in idx.items() if k != "bake_mode"}, out, idx["merge"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(Exception) as e:                          # past the guard, the never-read model id fails
        loader.load_moe_4bit_streaming("org/never-read", "cpu", torch.bfloat16, 8, 16, arena=out,
                                       merged_arena=idx["merge"])
    assert "is a merged arena" not in str(e.value)


def test_each_part_is_applied_once_attention_merged_at_load_experts_never(base, tmp_path):
    """The attention LoRA merges at load as for any adapter; the expert tensors are never applied by the server (they
    are in the merged rows), so merge_attention_lora merges exactly the attention projections and touches no expert."""
    from experts4bit_qlora.adapter_dir import merge_attention_lora
    W, _snap, _arena = base
    model = _trained(W, lora_scale=0.05, attention=True)
    g = torch.Generator().manual_seed(4)
    attn = {}
    for li in range(L):
        attn[f"model.layers.{li}.self_attn.q_proj.lora_A"] = torch.randn(R, H, generator=g)
        attn[f"model.layers.{li}.self_attn.q_proj.lora_B"] = torch.randn(H, R, generator=g) * 0.01
    v = _v(_adapter(tmp_path, model, attention=attn))
    assert v.expert_tensors and len(v.attention) == L
    before = {n: p.detach().clone() for n, p in model.named_parameters() if ".mlp." in n}
    report = merge_attention_lora(model, v)
    assert report.merged == L
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if ".mlp." in n)
    import experts4bit_qlora.serve_paged as sp
    src = inspect.getsource(sp)
    assert "expert_tensors" not in src and "_lora_B" not in src        # the server never reads them itself


def test_the_opt_in_without_an_adapter_is_a_loose_flag(monkeypatch):
    from experts4bit_qlora.serve_paged import PagedServeConfig
    with pytest.raises(ValueError, match="E4B_PAGED_MERGED_ARENA are set but E4B_PAGED_ADAPTER is not"):
        PagedServeConfig(model="m", merged_arena=True).validate()
    monkeypatch.setenv("E4B_PAGED_MERGED_ARENA", "yes")
    with pytest.raises(ValueError, match="expected '0' or '1'"):
        PagedServeConfig.from_env()


def _stub_paths(tmp_path, index):
    arena = tmp_path / "stub.arena"
    arena.write_bytes(b"")
    (tmp_path / "stub.arena.index.json").write_text(json.dumps(index))
    calib = tmp_path / "calib.json"
    calib.write_text("{}")
    return str(arena), str(calib)


def test_build_engine_refuses_the_opt_in_on_a_base_arena_before_reading_anything(tmp_path):
    """E4B_PAGED_MERGED_ARENA=1 on a base arena is a loose flag, refused at the index step: no adapter is verified and
    no weight is read (the model id is never resolved)."""
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    arena, calib = _stub_paths(tmp_path, {"bake_mode": MA.BASE_BAKE_MODE, "rows": []})
    cfg = PagedServeConfig(model="org/never-read", arena=arena, calib=calib, adapter=str(tmp_path / "no-such-adapter"),
                           merged_arena=True)
    with pytest.raises(ValueError, match="E4B_PAGED_MERGED_ARENA=1 but .* is not a merged arena"):
        build_engine(cfg)


def test_build_engine_refuses_a_merged_arena_without_an_adapter_before_reading_anything(tmp_path):
    from experts4bit_qlora.adapter_dir import AdapterRefused
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    arena, calib = _stub_paths(tmp_path, {"bake_mode": MA.MERGED_BAKE_MODE, "rows": [],
                                          "merge": {"writer": MA.WRITER, "adapter_id": "ab" * 32}})
    with pytest.raises(AdapterRefused, match=r"\(13\) a merged arena \(adapter abababababab\) with no adapter"):
        build_engine(PagedServeConfig(model="org/never-read", arena=arena, calib=calib))


def test_build_engine_decides_13_before_the_load_for_every_config_and_hands_the_block_on():
    """The phase order, read from build_engine itself: the index is read and refusal 13 decided (with and without an
    adapter) before the load; the loader, the fingerprint check and the hybrid tier are handed the matched block."""
    import experts4bit_qlora.serve_paged as sp
    src = inspect.getsource(sp.build_engine)
    load = src.index("load_moe_4bit_streaming(cfg.model")
    assert src.index("arena_index = json.load(f)") < src.index("if cfg.adapter:") < load
    assert src.index("merged_arena_refusals(None, arena_index, opt_in=cfg.merged_arena)") < load
    assert src.index("check_served_revision=False") < load
    assert src.index("is not a merged arena") < load
    assert "merged_arena=merge)" in src[load: load + 300]
    assert "merge=merge)" in src[src.index("fingerprint_check(adapter"):]
    assert "layers=ids, merged_arena=merge)" in src
    assert "merged_opt_in=cfg.merged_arena" in inspect.getsource(sp._adapter_refusals)
