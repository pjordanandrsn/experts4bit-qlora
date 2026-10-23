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


def _snapshot(tmp_path, W):
    st = _need("safetensors.torch")
    snap = tmp_path / "models--org--tiny-moe" / "snapshots" / REV
    snap.mkdir(parents=True)
    tensors = {f"model.layers.{li}.mlp.experts.{e}.{proj}.weight": t for (li, e, proj), t in W.items()}
    st.save_file(tensors, str(snap / "model-00001-of-00001.safetensors"))
    wm = {k: "model-00001-of-00001.safetensors" for k in tensors}
    (snap / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}), encoding="utf-8")
    return snap


def _bake(tmp_path, W):
    nb = _need("nvme_bake_nf4")
    snap = _snapshot(tmp_path, W)
    arena = str(tmp_path / "nf4.arena")
    nb.bake_nf4(str(snap), arena, quantize_fn=nb.default_quantize_expert(dev="cpu"), log=lambda *a, **k: None)
    return snap, arena


def _trained(W, lora_scale=0.0, seed=3):
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
        layers.append(layer)
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = layers
    return root


def _adapter(tmp_path, model, *, name="ad", fingerprint=None, scale_applied="match", schema="backend-adapter/2",
             tensors=None):
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
    d = tmp_path / name
    d.mkdir()
    save_file(tensors, str(d / "adapter.safetensors"), metadata={"format": schema, "backend": "experts4bit"})
    files = {"adapter.safetensors": {"sha256": hashlib.sha256((d / "adapter.safetensors").read_bytes()).hexdigest(),
                                     "bytes": (d / "adapter.safetensors").stat().st_size}}
    man = {"schema": schema, "backend": "experts4bit", "weights": "adapter.safetensors", "files": files,
           "resolved_model_revision": REV,
           "setup": {"quant_type": "nf4", "blocksize": 64, "r": R, "alpha": ALPHA, "attn_4bit": False,
                     "train_experts": True, "train_attention": False},
           "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()}}
    if schema == "backend-adapter/2":
        fp = fingerprint or S.quant_fingerprint(model, "experts")
        man.update(expert_absmax={"form": "fp32"}, train_fused_qkv=0, attn_quant={"form": "bf16"}, versions={},
                   lora_scale_applied={"attention": None,
                                       "experts": ALPHA / R if scale_applied == "match" else scale_applied},
                   fingerprints={"experts": fp}, run_id="merged-arena-test", created_at="2026-10-23T00:00:00Z")
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
    _refused(base, tmp_path, "missing", ad=_adapter(tmp_path, model, name="short", tensors=short, fingerprint=fp))
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


@pytest.mark.parametrize("module", ["experts4bit_qlora.engines.nvme_train", "experts4bit_qlora.engines.nvme_experts"])
def test_both_nvme_residencies_refuse_a_merged_arena_right_after_reading_its_index(module):
    src = inspect.getsource(importlib.import_module(module))
    i = src.index("index = load_index(arena_path)")
    assert "refuse_merged(index, arena_path)" in src[i: i + 200]
    assert src.count("load_index(") == src.count("refuse_merged(")


def test_refuse_merged_names_the_adapter_and_passes_a_base():
    MA.refuse_merged({"bake_mode": MA.BASE_BAKE_MODE}, "base.arena")
    with pytest.raises(ValueError, match="adapter abcdef012345"):
        MA.refuse_merged({"bake_mode": MA.MERGED_BAKE_MODE, "merge": {"adapter_id": "abcdef0123456789"}}, "m.arena")
    with pytest.raises(ValueError, match="is a merged arena"):
        MA.refuse_merged({"bake_mode": MA.BASE_BAKE_MODE, "merge": {}}, "m.arena")
