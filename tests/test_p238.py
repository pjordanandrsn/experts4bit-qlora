"""P238's kit, on CPU: the per-load premise table (bench/p238/p238_bytes.py).

The table must say SAME on the two libraries' real quantizers (bitsandbytes' Params4bit with nested statistics, as Unsloth's loader
builds it; e4b's ExpertsNbit + compress_expert_absmax_, as enable_fast_train builds it), and must fail on every way a load can
differ: one packed byte, an fp32 absmax left uncompressed, a lost layer, an empty table.
"""
import importlib.util
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
bnb = pytest.importorskip("bitsandbytes")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("p238_bytes", "bench/p238/p238_bytes.py")
E, HID, INTER = 4, 256, 64                     # gate_up [E, 2*INTER, HID], down [E, HID, INTER]: whole 64-blocks per expert


def _stacks(seed):
    g = torch.Generator().manual_seed(seed)
    gu = (torch.randn(E, 2 * INTER, HID, generator=g) * 0.02).to(torch.bfloat16)
    dn = (torch.randn(E, HID, INTER, generator=g) * 0.02).to(torch.bfloat16)
    return gu, dn


def _e4b_model(n_layers=2, compress=True):
    from experts4bit_qlora import ExpertsLoRA, ExpertsNbit
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList()
    for L in range(n_layers):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = ExpertsLoRA(ExpertsNbit.from_float(*_stacks(L)), r=8, alpha=16)
        root.layers.append(layer)
    if compress:
        assert compress_expert_absmax_(root) == n_layers
    return root


class _PeftLike(torch.nn.Module):
    """A wrapper with ``base_layer``, as PEFT's ParamWrapper presents an expert parameter."""

    def __init__(self, base):
        super().__init__()
        self.base_layer = base


def _unsloth_model(n_layers=2, wrap=True):
    from bitsandbytes.nn import Params4bit
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList()
    for L in range(n_layers):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        ex = torch.nn.Module()
        for which, W in zip(("gate_up", "down"), _stacks(L)):
            p = Params4bit(W, requires_grad=False, blocksize=64, compress_statistics=True, quant_type="nf4",
                           quant_storage=torch.uint8).to("cpu")
            setattr(ex, f"_{which}_param", p)
            object.__setattr__(ex, f"{which}_proj", _PeftLike(p) if wrap else p)
        layer.mlp.experts = ex
        root.layers.append(layer)
    return root


def test_the_two_libraries_real_quantizers_read_same_on_cpu():
    a, b = B.e4b_table(_e4b_model()), B.unsloth_table(_unsloth_model())
    assert sorted(a) == [0, 1] and set(a[0]["gate_up"]) == set(B.FIELDS)
    assert B.compare(a, b, n_layers=2) == (True, "")


def test_one_flipped_packed_byte_differs():
    m = _e4b_model()
    m.layers[1].mlp.experts.base.down_proj.data[0, 7] ^= 1
    ok, why = B.compare(B.e4b_table(m), B.unsloth_table(_unsloth_model()), n_layers=2)
    assert not ok and "layer 1 down packed" in why


def test_an_uncompressed_fp32_absmax_never_equals_the_nested_table():
    ok, why = B.compare(B.e4b_table(_e4b_model(compress=False)), B.unsloth_table(_unsloth_model()), n_layers=2)
    assert not ok and "absmax" in why


def test_a_lost_layer_fails_even_where_the_kept_layers_agree():
    a, b = B.e4b_table(_e4b_model()), B.unsloth_table(_unsloth_model())
    del b[1]
    ok, why = B.compare(a, b, n_layers=2)
    assert not ok and "not 0..1" in why
    ok, why = B.compare(a, b)
    assert not ok and "layer sets differ" in why


def test_an_empty_table_never_passes():
    assert B.compare({}, {})[0] is False
    assert B.compare(B.e4b_table(_e4b_model()), {})[0] is False


def test_the_unsloth_side_unwraps_peft_and_skips_modules_without_both_stacks():
    a = B.unsloth_table(_unsloth_model(wrap=False))
    b = B.unsloth_table(_unsloth_model(wrap=True))
    assert a == b and sorted(a) == [0, 1]


# -- p238_op.py: the routed input, the fp32 reference, the schedule, and a whole dry record ---------------------------------------
OP = _load("p238_op", "bench/p238/p238_op.py")


@pytest.mark.parametrize("rows", OP.ROWS)
def test_uniform_routing_gives_exactly_the_registered_rows_and_distinct_experts_per_token(rows):
    idx, w = OP.routing(rows, "uniform")
    n = OP.counts(idx)
    assert idx.shape == (OP.tokens_for(rows), OP.TOPK) and bool((n == rows).all())
    assert all(len(set(r.tolist())) == OP.TOPK for r in idx[:64])
    assert torch.allclose(w.sum(-1), torch.ones(w.shape[0]))


def test_the_skew_keeps_the_mean_has_an_empty_expert_and_is_seeded():
    idx, _ = OP.routing(128, "skew")
    n = OP.counts(idx)
    assert int(n.sum()) == 128 * OP.E and int(n[OP.ZERO_EXPERT]) == 0 and int(n.max()) > 4 * 128
    assert all(len(set(r.tolist())) == OP.TOPK for r in idx[:64])
    assert torch.equal(idx, OP.routing(128, "skew")[0])


def test_the_reference_equals_a_dense_per_token_loop():
    g = torch.Generator().manual_seed(1)
    Ee, H, Iw, r, T = 6, 8, 4, 2, 5
    W_gu, W_d = torch.randn(Ee, 2 * Iw, H, generator=g), torch.randn(Ee, H, Iw, generator=g)
    lora = [torch.randn(Ee, r, H, generator=g), torch.randn(Ee, 2 * Iw, r, generator=g), torch.randn(Ee, r, Iw, generator=g),
            torch.randn(Ee, H, r, generator=g)]
    x, idx = torch.randn(T, H, generator=g), torch.tensor([[0, 3], [1, 2], [5, 0], [4, 4 - 1], [2, 1]])
    w = torch.rand(T, 2, generator=g)
    got = OP.reference(x, idx, w, W_gu, W_d, lora, 2.0)
    want = torch.zeros(T, H)
    for t in range(T):
        for k in range(2):
            e = int(idx[t, k])
            h = x[t] @ (W_gu[e] + 2.0 * lora[1][e] @ lora[0][e]).t()
            gate, up = h.chunk(2)
            want[t] += w[t, k] * ((torch.nn.functional.silu(gate) * up) @ (W_d[e] + 2.0 * lora[3][e] @ lora[2][e]).t())
    assert torch.allclose(got, want, atol=1e-5)


def test_the_palindrome_balances_positions():
    assert OP.palindrome(["E_F", "E_A"], 0) == ["E_F", "E_A", "E_A", "E_F"]
    assert OP.palindrome(["U0", "U1", "UA"], 1) == ["U0", "U1", "UA", "UA", "U1", "U0"][::-1]


def test_rel_err_fails_closed_on_a_zero_reference():
    assert OP.rel_err(torch.ones(2), torch.zeros(2)) == float("inf")
    assert OP.rel_err(torch.zeros(2), torch.zeros(2)) == 0.0


@pytest.fixture
def dry_dims(monkeypatch):
    """DryCtx shrinks the module's dims; restore them after the test. The schedule is shortened for CPU time (the counts it
    produces are asserted against the shortened constants)."""
    for k in ("E", "HIDDEN", "INTER", "ZERO_EXPERT"):
        monkeypatch.setattr(OP, k, getattr(OP, k))
    monkeypatch.setattr(OP, "REPS", 2)
    monkeypatch.setattr(OP, "CALLS", 2)
    monkeypatch.setattr(OP, "WARM", 1)


def test_a_whole_dry_record(tmp_path, dry_dims):
    out = tmp_path / "op.json"
    assert OP.main(["--framework", "e4b", "--draw", "1", "--out", str(out), "--rows", "32", "--dry"]) == 0
    import json
    rec = json.loads(out.read_bytes())
    assert rec["dry"] is True and len(rec["cells"]) == 2 and rec["capture"] == "eager"
    c = rec["cells"][1]
    assert c["routing"] == "skew" and c["experts_empty"] >= 1 and c["b_restored_zero"] is True
    assert all(p["ok"] and p["bits_equal_self"] for p in c["parity"].values())
    assert len(c["visits"]) == OP.REPS * 2 * 2 and {v["arm"] for v in c["visits"]} == {"D_A", "D_B"}
    assert b"\r" not in out.read_bytes()


def test_the_dry_parity_check_sees_a_b_layout_mismatch(monkeypatch, dry_dims):
    """The non-zero B check: a module that reads B transposed per expert (a layout bug) must miss the tolerance."""
    ctx = OP.DryCtx()
    orig = ctx._run

    def swapped(x, idx, w, gup, fp32):
        if fp32:
            return orig(x, idx, w, gup, fp32)
        saved = [t.detach().clone() for t in ctx.lora]
        with torch.no_grad():
            ctx.lora[1].copy_(ctx.lora[1].flip(0))            # expert e reads expert E-1-e's B
        try:
            return orig(x, idx, w, gup, fp32)
        finally:
            with torch.no_grad():
                for t, s in zip(ctx.lora, saved):
                    t.copy_(s)

    monkeypatch.setattr(ctx, "_run", swapped)
    cell = OP.run_cell(ctx, 32, "uniform")
    assert not any(p["ok"] for p in cell["parity"].values())
    assert all(p["errs"]["out"] > OP.TOL for p in cell["parity"].values())


# -- p238_hook.py: observe only ---------------------------------------------------------------------------------------------------
HOOK = _load("p238_hook", "bench/p238/p238_hook.py")


@pytest.fixture
def fake_route(monkeypatch):
    import sys
    import types

    def train_gemm_route(dev=None, n_groups=None):
        return "dense" if n_groups is not None and n_groups <= 16 else "fused"

    nr = types.ModuleType("nf4_route")
    nr.train_gemm_route, nr.ROUTE_STATS = train_gemm_route, {"fwd": 0, "dgrad": 0, "dense_fwd": 0}
    nq = types.ModuleType("nf4_qlora")
    nq.DGRAD_STATS = {"kernel": 0, "loop_reasons": {}}
    monkeypatch.setitem(sys.modules, "nf4_route", nr)
    monkeypatch.setitem(sys.modules, "nf4_qlora", nq)
    return nr, train_gemm_route


def test_the_route_wrapper_returns_the_routes_own_answer_and_logs_it(fake_route):
    nr, orig = fake_route
    h = HOOK.make(type("A", (), {"framework": "e4b"})())
    h.post_import(None)
    assert nr.train_gemm_route is not orig and nr.train_gemm_route.__wrapped__ is orig
    for n in (None, 1, 8, 16, 17, 256):
        assert nr.train_gemm_route("cuda", n) == orig("cuda", n)
    assert h.route_log == {"None:fused": 1, "1:dense": 1, "8:dense": 1, "16:dense": 1, "17:fused": 1, "256:fused": 1}
    h.step_edge(0)
    assert h.record()["edges"][0]["route_log"]["256:fused"] == 1


def test_mutant_a_route_wrapper_that_changes_the_route_is_caught(fake_route):
    """The output-equality check the tests apply: a hook that forced 'fused' would differ from the route at n_groups <= 16."""
    nr, orig = fake_route
    forced = lambda dev=None, n_groups=None: "fused"            # noqa: E731 -- the mutant
    assert any(forced("cuda", n) != orig("cuda", n) for n in (1, 8, 16))


def test_the_forward_pre_hook_leaves_the_call_unchanged_and_counts_grad_state(monkeypatch, fake_route):
    m = _e4b_model(n_layers=1)
    m.layers[0].mlp.experts.forward = lambda *a, **k: torch.ones(1)  # a stand-in call target; the hook sees the module call
    h = HOOK.make(type("A", (), {"framework": "e4b"})())
    import types
    fake_fast = types.SimpleNamespace(FAST_TRAIN_STATS={"absmax_dq": {"compressed": 0, "required": True, "kept_fp32": None},
                                                         "patched": 1, "skipped": {}})
    monkeypatch.setitem(__import__("sys").modules, "experts4bit_qlora.engines.fast", fake_fast)
    h.post_load(None, m)
    x = torch.randn(3, HID)
    with torch.no_grad():
        out0 = m.layers[0].mlp.experts(x, None, None)
    out1 = m.layers[0].mlp.experts(x, None, None)
    assert torch.equal(out0, torch.ones(1)) and torch.equal(out1, torch.ones(1))
    assert h.fwd == {"calls": 2, "under_grad": 1}
    f = h.facts
    assert f["called_experts"] == 1 and f["absmax_compressed"] == 1 and f["fast_train_absmax_dq"]["kept_fp32"] is None
    assert f["adapters"]["0:gate_up_proj"]["A"] == [E, 8, HID] and f["adapters"]["0:down_proj"]["B"] == [E, HID, 8]


def test_the_adapter_census_matches_for_the_same_per_expert_tensors_in_the_two_layouts():
    """e4b's [E, r, in] / [E, out, r] and PEFT's (r*E, in) / (out, r*E) holding the same per-expert A read the same census."""
    from experts4bit_qlora import ExpertsLoRA, ExpertsNbit
    lo = ExpertsLoRA(ExpertsNbit.from_float(*_stacks(0)), r=8, alpha=16)
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList([torch.nn.Module()])
    root.layers[0].mlp = torch.nn.Module()
    root.layers[0].mlp.experts = lo
    e = HOOK.adapter_census(root, "e4b")

    class Lin(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight = torch.nn.Parameter(w)

    class PW(torch.nn.Module):
        def __init__(self, pn, A, out):
            super().__init__()
            self.parameter_name, self.num_experts, self.scaling = pn, E, {"default": 2.0}
            self.lora_A = torch.nn.ModuleDict({"default": Lin(A.reshape(-1, A.shape[-1]))})
            self.lora_B = torch.nn.ModuleDict({"default": Lin(torch.zeros(out, A.shape[1] * E))})

    u = torch.nn.Module()
    u.layers = torch.nn.ModuleList([torch.nn.Module()])
    u.layers[0].mlp = torch.nn.Module()
    u.layers[0].mlp.experts = torch.nn.Module()
    u.layers[0].mlp.experts.gate_up = PW("gate_up_proj", lo.gate_up_lora_A.detach(), 2 * INTER)
    u.layers[0].mlp.experts.down = PW("down_proj", lo.down_lora_A.detach(), HID)
    un = HOOK.adapter_census(u, "unsloth")
    assert e == un
