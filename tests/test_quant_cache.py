"""The quantized-expert cache (experts4bit_qlora/quant_cache.py; bench/quant-cache/DESIGN.md). CPU only: the key, the
format-version pin, write -> hit, the refusals, partial and concurrent writes, the disk budget and ``from_packed``. Whether
a real load's hit is byte-for-byte a fresh load is the GPU leg's question (bench/quant-cache/)."""
import hashlib
import inspect
import json
import os
import shutil

import pytest
import torch

from experts4bit_qlora import quant_cache as ec


def _fields(**over):
    f = {"checkpoint": {"model_id": "m", "commit": "c" * 40}, "model_type": "qwen3_moe",
         "layer_specs": {"0": ["nf4", 64], "1": ["nf4", 64]}, "dtype": "bfloat16", "has_gate": True,
         "shape": {"num_experts": 2}, "compress_statistics": False, "bitsandbytes": "0.50.2", "torch": "2.8.0",
         "quant_format": ec.QUANT_FORMAT_VERSION, "cache_format": ec.CACHE_FORMAT}
    f.update(over)
    return f


def _layer(seed):
    g = torch.Generator().manual_seed(seed)
    return {"gate_up_proj": torch.randint(0, 255, (2, 64), dtype=torch.uint8, generator=g),
            "down_proj": torch.randint(0, 255, (2, 32), dtype=torch.uint8, generator=g),
            "gate_up_absmax": torch.rand(2, 2, generator=g), "down_absmax": torch.rand(2, 1, generator=g)}


EST = {"schema": 1, "quant_type": "nf4", "blocksize": 64, "has_gate": True, "num_experts": 2, "hidden_dim": 64,
       "intermediate_dim": 32}


def _write(root, fields, layers=(0, 1)):
    c = ec.QuantCache(str(root), fields, log=lambda *a: None)
    assert c.mode == "miss"
    for i in layers:
        c.write_layer(i, _layer(i), EST, n_layers_expected=len(layers))
    return c


def test_the_key_is_canonical_and_moves_with_every_field():
    assert ec.cache_key(_fields()) == ec.cache_key(dict(reversed(list(_fields().items()))))
    base = ec.cache_key(_fields())
    for k, v in [("dtype", "float16"), ("has_gate", False), ("bitsandbytes", "0.51.0"), ("torch", "2.9.0"),
                 ("quant_format", ec.QUANT_FORMAT_VERSION + 1), ("layer_specs", {"0": ["nf4", 128], "1": ["nf4", 64]}),
                 ("checkpoint", {"model_id": "m", "commit": "d" * 40}), ("model_type", "olmoe")]:
        assert ec.cache_key(_fields(**{k: v})) != base, k


def test_the_quant_format_version_is_pinned_to_the_quantizer_source():
    """Changing the fuse, the quantizer call or the extra-state schema changes the bytes a checkpoint quantizes to. Bump
    quant_cache.QUANT_FORMAT_VERSION and refresh these hashes in the same change."""
    from experts4bit_qlora._vendor import experts as X
    from experts4bit_qlora.arch.moe_conventions import fuse_experts, stack_experts
    got = {n: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
           for n, f in (("fuse_experts", fuse_experts), ("stack_experts", stack_experts),
                        ("ExpertsNbit._quantize_stack", X.ExpertsNbit._quantize_stack))}
    pinned = {1: {"fuse_experts": "eff1cb1495dfa98fe6d93f36e3fd3bbc2ea295e462a0e5f0bc1c2937f498586d",
                  "stack_experts": "b9ff2552d4bb9ce1e4c91e6cf2e634a59b23664aaee7ae4ca04b49468080ef3c",
                  "ExpertsNbit._quantize_stack": "3458bbb22f9815952f24512f0b3304c26d703c6cc629ab6c33cd5c9b1258765d"}}
    assert got == pinned[ec.QUANT_FORMAT_VERSION], got
    assert (X._EXTRA_STATE_SCHEMA, X.ExpertsNbit._EXTRA_STATE_FIELDS) == (
        1, ("quant_type", "blocksize", "has_gate", "num_experts", "hidden_dim", "intermediate_dim"))


def test_write_then_hit_returns_the_same_bytes(tmp_path):
    c = _write(tmp_path, _fields())
    assert c.finish() == "wrote"
    h = ec.QuantCache(str(tmp_path), _fields(), log=lambda *a: None)
    assert h.mode == "hit" and h.has_layer(0) and h.has_layer(1) and not h.has_layer(2)
    for i in (0, 1):
        got, est = h.load_layer(i)
        assert est == EST and all(torch.equal(got[n], _layer(i)[n]) for n in ec.NAMES)
    assert not [d for d in os.listdir(tmp_path) if d.startswith(".tmp-")]


def test_only_safetensors_and_json_are_written(tmp_path):
    c = _write(tmp_path, _fields())
    c.finish()
    files = os.listdir(os.path.join(tmp_path, c.key))
    assert sorted(files) == ["layer_000.safetensors", "layer_001.safetensors", "manifest.json"]
    assert "torch.load" not in inspect.getsource(ec) and "pickle" not in inspect.getsource(ec).replace("never a pickle", "")


def test_a_flipped_byte_refuses_and_rebuild_clears(tmp_path, monkeypatch):
    c = _write(tmp_path, _fields())
    c.finish()
    f = os.path.join(tmp_path, c.key, "layer_001.safetensors")
    b = bytearray(open(f, "rb").read())
    b[-1] ^= 0xFF
    open(f, "wb").write(bytes(b))
    with pytest.raises(ec.QuantCacheCorrupt, match="do not match the manifest"):
        ec.QuantCache(str(tmp_path), _fields(), log=lambda *a: None)
    monkeypatch.setenv(ec.ENV_REBUILD, "1")
    r = ec.QuantCache(str(tmp_path), _fields(), log=lambda *a: None)
    assert r.mode == "miss" and not os.path.exists(os.path.join(tmp_path, c.key))


def test_an_edited_manifest_refuses(tmp_path):
    c = _write(tmp_path, _fields())
    c.finish()
    man = os.path.join(tmp_path, c.key, "manifest.json")
    m = json.load(open(man))
    m["fields"]["dtype"] = "float16"
    json.dump(m, open(man, "w"))
    with pytest.raises(ec.QuantCacheCorrupt, match="does not describe key"):
        ec.QuantCache(str(tmp_path), _fields(), log=lambda *a: None)


def test_a_key_change_is_a_miss_never_a_hit(tmp_path):
    _write(tmp_path, _fields()).finish()
    assert ec.QuantCache(str(tmp_path), _fields(bitsandbytes="0.51.0"), log=lambda *a: None).mode == "miss"


def test_a_partial_write_is_never_read_and_is_swept_once_stale(tmp_path):
    c = _write(tmp_path, _fields())             # no finish(): a crash mid-load
    tmp = [os.path.join(tmp_path, d) for d in os.listdir(tmp_path) if d.startswith(".tmp-")]
    assert tmp and ec.QuantCache(str(tmp_path), _fields(), log=lambda *a: None).mode == "miss"
    assert all(os.path.exists(d) for d in tmp)  # fresh: it may be a concurrent writer's, so it stays
    old = os.path.getmtime(tmp[0]) - ec._STALE_S - 60
    for root, _dirs, files in os.walk(tmp[0]):
        for f in files + [""]:
            os.utime(os.path.join(root, f) if f else root, (old, old))
    again = ec.QuantCache(str(tmp_path), _fields(), log=lambda *a: None)
    assert again.mode == "miss" and not [d for d in os.listdir(tmp_path) if d.startswith(".tmp-")]
    assert not os.path.exists(c.dir)


def test_concurrent_misses_the_loser_uses_the_winner(tmp_path):
    a = _write(tmp_path, _fields())
    b = _write(tmp_path, _fields())
    assert a.finish() == "wrote"
    assert b.finish() == "wrote (a concurrent writer's copy verified)"
    assert ec.QuantCache(str(tmp_path), _fields(), log=lambda *a: None).mode == "hit"
    assert not [d for d in os.listdir(tmp_path) if d.startswith(".tmp-")]


def test_a_full_disk_skips_the_write_and_never_fails(tmp_path, monkeypatch):
    msgs = []
    c = ec.QuantCache(str(tmp_path), _fields(), log=msgs.append)
    monkeypatch.setattr(ec.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10, 10, 1))
    c.write_layer(0, _layer(0), EST, n_layers_expected=48)
    assert c.finish() == "not-written" and any("not written" in m for m in msgs)
    assert not os.listdir(tmp_path)


def test_a_device_mismatch_becomes_a_miss_under_a_device_key(tmp_path):
    c = _write(tmp_path, _fields())
    c.finish()
    h = ec.QuantCache(str(tmp_path), _fields(), log=lambda *a: None)
    h.manifest["device"] = {"name": "another GPU", "capability": [9, 0]}
    with pytest.warns(RuntimeWarning, match="rebuilding under a device-specific key"):
        h.device_mismatch_to_miss()
    assert h.mode == "miss" and h.key != c.key and h.fields["device"] == h.device


def test_from_packed_rebuilds_the_module_and_refuses_a_mismatch():
    from experts4bit_qlora import Experts4bit
    ref = Experts4bit(2, 64, 64, has_gate=True, quant_type="nf4", blocksize=64, device="cpu")
    with torch.no_grad():
        ref.gate_up_proj.copy_(torch.randint(0, 255, ref.gate_up_proj.shape, dtype=torch.uint8))
        ref.down_proj.copy_(torch.randint(0, 255, ref.down_proj.shape, dtype=torch.uint8))
        ref.gate_up_absmax.copy_(torch.rand(ref.gate_up_absmax.shape))
        ref.down_absmax.copy_(torch.rand(ref.down_absmax.shape))
    t = {n: getattr(ref, n).detach().clone() for n in ec.NAMES}
    m = Experts4bit.from_packed(t["gate_up_proj"], t["down_proj"], t["gate_up_absmax"], t["down_absmax"],
                                ref.get_extra_state(), compute_dtype=torch.bfloat16)
    assert type(m) is Experts4bit and m.get_extra_state() == ref.get_extra_state()
    assert all(torch.equal(getattr(m, n), t[n]) for n in ec.NAMES) and torch.equal(m.code, ref.code)
    with pytest.raises(ValueError, match="does not match"):
        Experts4bit.from_packed(t["gate_up_proj"][:, :-1], t["down_proj"], t["gate_up_absmax"], t["down_absmax"],
                                ref.get_extra_state())


def test_the_loader_refuses_an_arena_and_8bit_and_ignores_resident(monkeypatch, tmp_path):
    from experts4bit_qlora import loader
    monkeypatch.setenv(ec.ENV, str(tmp_path))
    args = dict(model_id="m", snap=str(tmp_path), config=None, lm_config=None, model_type="qwen3_moe", has_gate=True,
                n_layers=2, n_exp=2, quantize_layers=None, quant_type="nf4", blocksize=64, dtype=torch.bfloat16,
                offload=True, arena_index=None, ckpt_experts={0: {}, 1: {}})
    with pytest.raises(ValueError, match="arena load"):
        loader._open_quant_cache(**{**args, "arena_index": {}})
    with pytest.raises(ValueError, match="4-bit"):
        loader._open_quant_cache(**{**args, "quant_type": "int8"})
    assert loader._open_quant_cache(**{**args, "offload": False}) is None
    monkeypatch.delenv(ec.ENV)
    assert loader._open_quant_cache(**args) is None
