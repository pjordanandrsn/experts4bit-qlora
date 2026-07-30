"""Serving dense weights from disk must be byte-for-byte, or it is pointless.

The whole reason this path exists is that the alternative way to fit a 114 GB dense
side onto a cheap card is to quantize it, which changes the model. So the tests that
matter here are equality tests, not shape tests.
"""
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from experts4bit_qlora.dense_disk import DenseDiskSource, DiskHome
from experts4bit_qlora.dense_offload import (
    _DenseOffload, dense_offload_report, enable_dense_offload)

# Same dims as test_dense_offload.py, and for the same reason: MIN_BYTES is 1 MiB, so
# a smaller toy has NO managed tensors at all and every assertion about slots passes
# or fails for the wrong reason.
H, INTER, NL = 512, 1024, 3


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, INTER, bias=False)
        self.o_proj = nn.Linear(INTER, H, bias=False)
        self.norm = nn.Parameter(torch.ones(H))

    def forward(self, x):
        return x + self.o_proj(torch.relu(self.q_proj(x * self.norm)))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(NL))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def _model(seed=11):
    torch.manual_seed(seed)
    return Toy()


def _snapshot(tmp_path, model, prefix="", shards=1):
    """Write the model's tensors as safetensors, like a real checkpoint."""
    sd = {prefix + k: v.detach().clone() for k, v in model.state_dict().items()}
    keys = sorted(sd)
    out = tmp_path / "snap"
    out.mkdir(parents=True, exist_ok=True)
    per = -(-len(keys) // shards)
    for i in range(shards):
        chunk = {k: sd[k] for k in keys[i * per:(i + 1) * per]}
        if chunk:
            save_file(chunk, str(out / f"model-{i:05d}.safetensors"),
                      metadata={"format": "pt"})
    return out


# ------------------------------------------------------------------- source ---
@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("shards", [1, 3])
def test_source_reads_are_bit_exact(tmp_path, direct, shards):
    """Byte equality against the tensor that was written, for every tensor.

    Parameterized over O_DIRECT because the aligned-window path does real offset
    arithmetic -- it reads the 4096-aligned span CONTAINING the tensor and slices the
    middle out -- and an off-by-one there would give plausible garbage rather than an
    error. Also over shard counts, because tensor offsets are per FILE.
    """
    m = _model()
    snap = _snapshot(tmp_path, m, shards=shards)
    src = DenseDiskSource(str(snap), direct=direct)
    try:
        sd = m.state_dict()
        assert set(src.tensors) == set(sd), "index does not cover the checkpoint"
        for k, want in sd.items():
            got = src.fetch(k)
            assert got.dtype == want.dtype and got.shape == want.shape, k
            assert torch.equal(got, want), f"{k} differs"
    finally:
        src.close()


def test_unaligned_tensors_are_read_correctly(tmp_path):
    """Deliberately awkward sizes so tensors do NOT start on 4096 boundaries."""
    odd = {f"t{i}": torch.randn(i * 7 + 1, 3) for i in range(1, 9)}
    odd["bf16"] = torch.randn(101, 5).to(torch.bfloat16)
    out = tmp_path / "odd"
    out.mkdir()
    save_file(odd, str(out / "m.safetensors"), metadata={"format": "pt"})
    src = DenseDiskSource(str(out))
    try:
        offsets = {n: src.tensors[n].offset % 4096 for n in odd}
        assert any(o != 0 for o in offsets.values()), offsets
        for k, want in odd.items():
            assert torch.equal(src.fetch(k), want), k
    finally:
        src.close()


def test_fetch_aliases_staging_so_views_do_not_outlive_it(tmp_path):
    """Document the sharp edge the offload path has to respect.

    `fetch` returns a view of a shared buffer. A second fetch may overwrite it. This
    is not a bug to fix -- it is what keeps the host-RAM floor at one buffer -- but
    anything that must outlive a fetch has to `.clone()`, which is why
    `DiskHome.materialize` exists and why the state_dict hook uses it.
    """
    m = _model()
    src = DenseDiskSource(str(_snapshot(tmp_path, m)))
    try:
        keys = [k for k in m.state_dict() if k.endswith("q_proj.weight")]
        first = src.fetch(keys[0])
        owned = DiskHome(src, keys[0]).materialize()
        src.fetch(keys[1])                       # overwrite the buffer
        assert torch.equal(owned, m.state_dict()[keys[0]]), "materialize() must copy"
        # `first` is a view and may now hold other bytes; assert only that the OWNED
        # copy survived, which is the property the offload path relies on.
        assert first.data_ptr() != owned.data_ptr()
    finally:
        src.close()


# ------------------------------------------------------------------ offload ---
def test_offload_uses_disk_homes_and_holds_no_host_bytes(tmp_path):
    m = _model()
    src = DenseDiskSource(str(_snapshot(tmp_path, m)))
    try:
        h = _DenseOffload(m.layers[1], "cpu", pin=False, source=src,
                          key_prefix="layers.1.", verify=True)
        assert h.slots and all(isinstance(hm, DiskHome) for _m, _a, _p, hm in h.slots)
        assert h.host_bytes == 0, "nothing should be resident on the host"
        assert h.disk_bytes == h.bytes
        assert h.verified == len(h.slots), "verify=True must check every home"
    finally:
        src.close()


def test_staged_weights_equal_the_checkpoint(tmp_path):
    """The bytes that reach the module must be the bytes in the file."""
    m = _model()
    want = {k: v.clone() for k, v in m.state_dict().items()}
    src = DenseDiskSource(str(_snapshot(tmp_path, m)))
    try:
        h = _DenseOffload(m.layers[0], "cpu", pin=False, source=src,
                          key_prefix="layers.0.")
        h.evict()
        assert m.layers[0].q_proj.weight.numel() == 0, "evict should leave placeholders"
        h.stage()
        assert torch.equal(m.layers[0].q_proj.weight, want["layers.0.q_proj.weight"])
        assert torch.equal(m.layers[0].o_proj.weight, want["layers.0.o_proj.weight"])
    finally:
        src.close()


def test_state_dict_while_evicted_is_not_a_stale_alias(tmp_path):
    """The bug this guards: a disk home's view aliases the shared staging buffer, so
    saving the view rather than a copy would serialize whichever layer was staged
    LAST. That produces a checkpoint that loads fine and holds the wrong weights --
    the same silent-empty-checkpoint class the hook was written for."""
    m = _model()
    want = {k: v.clone() for k, v in m.state_dict().items()}
    src = DenseDiskSource(str(_snapshot(tmp_path, m)))
    try:
        hs = [_DenseOffload(m.layers[i], "cpu", pin=False, source=src,
                            key_prefix=f"layers.{i}.") for i in range(NL)]
        for h in hs:
            h.evict()
        hs[NL - 1].stage()          # last layer's bytes are what the buffer holds now
        sd = m.state_dict()
        for k in ("layers.0.q_proj.weight", "layers.0.o_proj.weight"):
            assert torch.equal(sd[k], want[k]), f"{k} came back as a stale alias"
    finally:
        src.close()


def test_forward_matches_the_fully_resident_model(tmp_path):
    """End to end: disk-served dense weights must give the SAME output as the
    ordinary model, bit for bit. Same bytes and same kernels, so anything other than
    exact equality means the plumbing altered data."""
    m = _model()
    x = torch.randn(2, H)
    with torch.no_grad():
        ref = m(x).clone()

    m2 = _model()                               # identical seed
    src = DenseDiskSource(str(_snapshot(tmp_path, m2)))
    try:
        hs = enable_dense_offload(m2, "cpu", pin=False, prefetch=False,
                                  source=src, verify=True)
        rep = dense_offload_report(hs)
        assert rep["host_bytes"] == 0 and rep["disk_bytes"] > 0, rep
        assert rep["verified_bit_exact"] == rep["tensors"], rep
        with torch.no_grad():
            got = m2(x)
        assert torch.equal(got, ref), (got - ref).abs().max()
    finally:
        src.close()


def test_verify_catches_a_mismatched_checkpoint(tmp_path):
    """verify=True is the gate that says 'this file is this model'."""
    m = _model()
    snap = _snapshot(tmp_path, m)
    other = _model(seed=99)                      # different weights, same shapes
    src = DenseDiskSource(str(_snapshot(tmp_path / "b", other)))
    try:
        with pytest.raises(ValueError, match="differs from the loaded tensor"):
            _DenseOffload(m.layers[0], "cpu", pin=False, source=src,
                          key_prefix="layers.0.", verify=True)
    finally:
        src.close()
    assert snap.exists()


def test_shape_mismatch_is_refused(tmp_path):
    """A key can exist on disk at the wrong shape. Serving it would be silently
    wrong, so it must raise instead."""
    bad = {"layers.0.q_proj.weight": torch.randn(INTER + 1, H),
           "layers.0.o_proj.weight": torch.randn(H, INTER)}
    out = tmp_path / "bad"
    out.mkdir()
    save_file(bad, str(out / "m.safetensors"), metadata={"format": "pt"})
    m = _model()
    src = DenseDiskSource(str(out))
    try:
        with pytest.raises(ValueError, match="refusing to serve"):
            _DenseOffload(m.layers[0], "cpu", pin=False, source=src,
                          key_prefix="layers.0.")
    finally:
        src.close()


def test_duplicate_keys_across_shards_are_rejected(tmp_path):
    out = tmp_path / "dup"
    out.mkdir()
    t = {"a": torch.randn(4, 4)}
    save_file(t, str(out / "m-0.safetensors"), metadata={"format": "pt"})
    save_file(t, str(out / "m-1.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="two shards"):
        DenseDiskSource(str(out))


def test_stats_report_slack(tmp_path):
    """Aligned reads pull in extra bytes; the report should own that rather than
    pretend the read was exactly the tensor."""
    m = _model()
    src = DenseDiskSource(str(_snapshot(tmp_path, m)), direct=True)
    try:
        for k in list(src.tensors)[:4]:
            src.fetch(k)
        st = src.stats()
        assert st["reads"] == 4 and st["read_bytes"] >= st["slack_bytes"]
        if st["direct"]:
            assert st["slack_bytes"] > 0, "aligned reads must report their slack"
        assert json.dumps(st)
    finally:
        src.close()
