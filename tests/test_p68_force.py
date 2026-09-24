"""Lane P68's forcings on CPU (bench/p68/p68_probe.py; bench/p68/P68-PREREG.md).

A forcing turns one n-row call into n one-row calls, so it must (a) give back the rows in order, bit for bit what the
one-row calls return; (b) be inert at one row, so an arm's T = 1 control is the unforced control; (c) count every call,
so the census can show it engaged; (d) restore the module exactly, whether its forward was the class's or an instance
patch (the int4 folds patch instances). The core forcing must hand row i of a T-query call exactly the key prefix a
T = 1 decode of that row sees, with no mask. transformers is not needed: the attention table is stubbed with the
local-then-global mapping transformers' GeneralInterface uses."""
import importlib.util
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, sub):
    p = REPO / "bench" / sub / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


_load("p63_compare", "p63")
F = _load("p68_probe", "p68")


class _Table:
    """transformers' GeneralInterface: a local mapping consulted before the global one."""
    _global = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, k):
        return self._local_mapping[k] if k in self._local_mapping else self._global[k]

    def __setitem__(self, k, v):
        self._local_mapping[k] = v

    def __delitem__(self, k):
        del self._local_mapping[k]


@pytest.fixture
def table(monkeypatch):
    t = _Table()
    mu = types.ModuleType("transformers.modeling_utils")
    mu.ALL_ATTENTION_FUNCTIONS = t
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", mu)
    return t


def test_rowwise_linear_rows_are_the_one_row_calls_in_order():
    torch.manual_seed(0)
    lin = torch.nn.Linear(24, 8)
    x = torch.randn(1, 5, 24)
    counts = F.new_counts()
    w = F.RowWise(lin, "proj", counts)
    w.install()
    y = lin(x)
    singles = torch.cat([lin(x[:, i:i + 1]) for i in range(5)], 1)       # after install: one-row calls pass through
    w.remove()
    assert torch.equal(y, singles)
    assert counts["proj"]["split_calls"] == 1 and counts["proj"]["split_rows"] == 5
    assert counts["proj"]["t1_calls"] == 5                                # the five direct one-row calls above


def test_rowwise_is_inert_at_one_row():
    lin = torch.nn.Linear(8, 4)
    x = torch.randn(1, 1, 8)
    ref = lin(x)
    counts = F.new_counts()
    w = F.RowWise(lin, "proj", counts)
    w.install()
    got = lin(x)
    w.remove()
    assert torch.equal(ref, got) and counts["proj"]["split_calls"] == 0 and counts["proj"]["t1_calls"] == 1


def test_rowwise_handles_the_routers_2d_input_and_tuple_output():
    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(6, 10))

        def forward(self, h):
            logits = h @ self.weight.t()
            p = logits.softmax(-1)
            v, i = p.topk(2, -1)
            return logits, v, i
    r = Router()
    h = torch.randn(7, 10)
    counts = F.new_counts()
    w = F.RowWise(r, "router", counts)
    w.install()
    out = r(h)
    w.remove()
    rows = [r(h[i:i + 1]) for i in range(7)]
    for j in range(3):
        assert torch.equal(out[j], torch.cat([o[j] for o in rows], 0))


def test_restore_keeps_an_instance_patch_and_removes_a_class_forward_wrap():
    lin = torch.nn.Linear(4, 4)
    counts = F.new_counts()
    w = F.RowWise(lin, "proj", counts)
    w.install()
    w.remove()
    assert "forward" not in lin.__dict__                                  # back to the class's forward
    patched = torch.nn.Linear(4, 4)

    def fold(x, _m=patched):                                              # an int4 fold's instance patch
        return x * 2
    patched.forward = fold
    w2 = F.RowWise(patched, "norms", counts)
    w2.install()
    assert patched.forward is not fold
    w2.remove()
    assert patched.forward is fold


def _sdpa(module, q, k, v, mask, **kw):
    """A stand-in attention with transformers' contract: [B, H, T, D] in, ([B, T, H, D], None) out; causal when
    T > 1 and no mask is given (``sdpa_attention_forward``'s rule)."""
    T, S = q.shape[2], k.shape[2]
    att = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
    if T > 1 and mask is None:
        causal = torch.ones(T, S, dtype=torch.bool).tril(S - T)
        att = att.masked_fill(~causal, float("-inf"))
    o = att.softmax(-1) @ v
    return o.transpose(1, 2), None


def test_rowwise_core_gives_each_query_its_decode_prefix(table):
    torch.manual_seed(1)
    table._global["sdpa"] = _sdpa
    B, H, T, S, D = 1, 2, 5, 9, 4
    q, k, v = torch.randn(B, H, T, D), torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    counts = F.new_counts()
    seen = []

    def spy(module, q_, k_, v_, mask, **kw):
        seen.append((q_.shape[2], k_.shape[2], mask, q_.is_contiguous() and k_.is_contiguous()))
        return _sdpa(module, q_, k_, v_, mask, **kw)
    table._global["sdpa"] = spy
    with F.RowWiseSDPA(counts):
        out, _ = table["sdpa"](None, q, k, v, None, is_causal=True)
    assert table["sdpa"] is spy and "sdpa" not in table._local_mapping      # restored to the global entry
    assert [s[:2] for s in seen] == [(1, S - T + i + 1) for i in range(T)]
    assert all(s[2] is None and s[3] for s in seen)
    # row i equals the one-query call over its own prefix -- the T = 1 decode call
    ref = torch.cat([_sdpa(None, q[:, :, i:i + 1], k[:, :, :S - T + i + 1], v[:, :, :S - T + i + 1], None)[0]
                     for i in range(T)], 1)
    assert torch.equal(out, ref)
    # and it is the same function as the masked T-row call, up to rounding
    full, _ = _sdpa(None, q, k, v, None)
    assert torch.allclose(out, full, atol=1e-6)
    assert counts["core"]["split_calls"] == 1 and counts["core"]["split_rows"] == T


def test_rowwise_core_counts_a_masked_decode_call(table):
    table._global["sdpa"] = _sdpa
    counts = F.new_counts()
    q, k = torch.randn(1, 1, 1, 4), torch.randn(1, 1, 3, 4)
    with F.RowWiseSDPA(counts):
        table["sdpa"](None, q, k, k, torch.zeros(1, 1, 1, 3))
        table["sdpa"](None, q, k, k, None)
    assert counts["core"]["t1_calls"] == 2 and counts["core"]["t1_mask_present"] == 1


def test_forcing_finds_targets_by_structure_and_restores_everything(monkeypatch, table):
    table._global["sdpa"] = _sdpa

    class RMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8))

        def forward(self, x):
            return x * self.weight

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj, self.o_proj = torch.nn.Linear(8, 24), torch.nn.Linear(8, 8)
            self.q_norm = RMSNorm()

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))

        def forward(self, h):
            return h @ self.weight.t()

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = Attn(), torch.nn.Module()
            self.mlp.gate = Gate()
            self.input_layernorm = RMSNorm()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
            self.model.norm = RMSNorm()
            self.lm_head = torch.nn.Linear(8, 11)
    m = Model()
    p63 = sys.modules.get("p63_probe") or types.ModuleType("p63_probe")
    p63._layers = lambda model: (list(model.model.layers), model.model.norm)
    monkeypatch.setitem(sys.modules, "p63_probe", p63)
    kinds = F.forcing_targets(m, F.FORCE_KINDS)
    by = {}
    for mod, k in kinds:
        by[k] = by.get(k, 0) + 1
    assert by == {"proj": 4, "router": 2, "lm_head": 1, "norms": 5}
    counts = F.new_counts()
    before = {id(mod): ("forward" in mod.__dict__) for mod, _ in kinds}
    red = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with F.forcing(m, list(F.FORCE_KINDS), True, counts):
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
        y = m.lm_head(torch.randn(1, 3, 8))
        assert y.shape == (1, 3, 11) and counts["lm_head"]["split_calls"] == 1
        assert table["sdpa"] is not _sdpa
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is red
    assert table["sdpa"] is _sdpa
    assert all(("forward" in mod.__dict__) == before[id(mod)] for mod, _ in kinds)
    assert counts["proj"]["wrapped_modules"] == 4


def test_the_arm_table_is_the_registered_one():
    names = [a["name"] for a in F.ARMS]
    assert names == ["hf.base", "hf.proj", "hf.core", "hf.proj_core", "hf.all", "hf.fp32red", "paged.base",
                     "paged.proj", "size.hf", "size.paged"]
    for a in F.ARMS:
        assert set(a["force"]) <= set(F.FORCE_KINDS)
        assert "core" not in a["force"] or a["attn"] == "hf", "the core forcing wraps the HF sdpa entry only"
        assert (a["grouping"] == "default") == (a["kind"] == "size")


def test_size_windows_leave_room_for_the_widest_verify():
    for n in (512, 160):
        for W in (16, 17):
            ws = F._size_windows(n, W)
            assert ws and ws[0] == F.SIZE_FIRST and all(p + W <= n for p in ws)
            assert all(b - a == F.SIZE_STEP for a, b in zip(ws, ws[1:]))
