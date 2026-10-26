"""P238's kit, on CPU: the per-load premise table (bench/p238/p238_bytes.py).

The table must say SAME on the two libraries' real quantizers (bitsandbytes' Params4bit with nested statistics, as Unsloth's loader
builds it; e4b's ExpertsNbit + compress_expert_absmax_, as enable_fast_train builds it), and must fail on every way a load can
differ: one packed byte, an fp32 absmax left uncompressed, a lost layer, an empty table.
"""
import importlib.util
import os
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
    h = HOOK.make()
    h.post_import(type("A", (), {"framework": "e4b", "accum": 1})())
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
    h = HOOK.make()
    h.fw = "e4b"
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


# -- p238_reduce.py: the rule's own self-test (every gate fails closed with its evidence removed) ---------------------------------
def test_the_reducer_selftest():
    if not (ROOT / "bench" / "common" / "unsloth_moe_counter.py").exists():
        pytest.skip("bench/common/unsloth_moe_counter.py lands with e4b#2132; this PR merges after it")
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    assert R.selftest() >= 37


# -- the reducer's POSITIVE Unsloth rows: a real UnslothMoeCounter driven through 60 checkpointed steps on stand-in zoo modules -----
def _counter_helpers():
    if not (ROOT / "tests" / "test_unsloth_moe_counter.py").exists():
        pytest.skip("tests/test_unsloth_moe_counter.py (the counter's stand-ins) lands with e4b#2132; this PR merges after it")
    return _load("p238_counter_helpers", "tests/test_unsloth_moe_counter.py")


def _unsloth_row(monkeypatch, tmp_path, arm, drop_dx_step=None):
    """One U0 / U1 / UA load as the reducer reads it: the counter's real snapshot and structural count after 60 steps under Unsloth's
    checkpointing (a no-grad pass, then the replay under grad and its backward), the arm's real environment, and the cell and profile
    F15 v2 reads. The stand-in experts forward tries the NF4-fused base first (moe_utils.py:5322), else the dequant path."""
    import sys

    T = _counter_helpers()
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    for k in [k for k in os.environ if k.startswith("UNSLOTH_")]:
        monkeypatch.delenv(k)
    pins = {**R.UNSLOTH_PINS, "UNSLOTH_MOE_NF4_FUSED": R.NF4_MODE[arm]}
    for k, v in pins.items():
        monkeypatch.setenv(k, v)
    pkg, tp, mg, nf, _ = T._fake_zoo()
    zroot, uroot = tmp_path / "pkgs" / "unsloth_zoo", tmp_path / "pkgs" / "unsloth"
    tdir = zroot / "temporary_patches"
    tdir.mkdir(parents=True)
    uroot.mkdir(parents=True)
    (tdir / "knobs.py").write_text("".join(f'os.environ.get("{k}")\n' for k in pins), encoding="utf-8")
    (uroot / "__init__.py").write_text("", encoding="utf-8")
    src = T.FAKE_MOE_UTILS.replace('''def forward_native(x, base, lora_a, lora_b):               # forward_native_grouped_mm, 5244-
    _lora_rank_pad_needed(x, lora_a)
    with torch.no_grad():
        y = _grouped_mm_with_backward_fix(x, base, None)     # the dequant path's recompute forward (1100): no backward node
''', '''def forward_native(x, base, lora_a, lora_b):
    _lora_rank_pad_needed(x, lora_a)
    from unsloth_zoo.temporary_patches.moe_nf4_grouped import maybe_nf4_grouped_mm
    y = maybe_nf4_grouped_mm(x, torch.ones(base.shape[0], dtype=torch.int32), base[0], None)
    if y is None:
        with torch.no_grad():
            y = _grouped_mm_with_backward_fix(x, base, None)
''')
    assert src != T.FAKE_MOE_UTILS
    mg.__file__, nf.__file__ = str(tdir / "moe_grouped_fp16.py"), str(tdir / "moe_nf4_grouped.py")
    mu = T._module_from_source("unsloth_zoo.temporary_patches.moe_utils", src, tdir / "moe_utils.py")
    for name, mod in (("unsloth_zoo", T._package("unsloth_zoo", zroot)), ("unsloth_zoo.temporary_patches", T._package("unsloth_zoo.temporary_patches", tdir)),
                      ("unsloth_zoo.temporary_patches.moe_grouped_fp16", mg), ("unsloth_zoo.temporary_patches.moe_nf4_grouped", nf),
                      ("unsloth_zoo.temporary_patches.moe_utils", mu), ("unsloth", T._package("unsloth", uroot))):
        monkeypatch.setitem(sys.modules, name, mod)
    C = R.C
    ex = T._experts(mu.forward_moe_backend)[0]
    c = C.UnslothMoeCounter().install().patch_loaded([ex])
    for i in range(R.STEPS):
        c.mark_step(i)
        args = T._stacks()
        with torch.no_grad():
            ex(*args)
        ex(*args)
        T._backward_as_the_library_would(mg, 2, 2)              # the two LoRA GEMMs' gx and dW
        if arm != "U0" and i != drop_dx_step:
            nf.nf4_dx(torch.ones(8, 40), args[1][0])             # the fused base's dX, inside the op
    c.mark_step(R.STEPS)
    e, table, census = R._good_rows()
    E = 2 * R.N_LAYERS
    row = dict(e, framework="unsloth", arm=arm, absmax={"loaded_nested": True}, loader_used="FastModel",
               unsloth_env={k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_")},
               hook={"edges": [{}] * R.STEPS + [{"unsloth": c.snapshot()}], "structural": c.structural(1, R.STEPS)},
               cell={"n_layers": R.N_LAYERS, "accum": 1, "moe_backend_selected": "grouped_mm", "env": {"unsloth_zoo": "2026.10.8"},
                     "unsloth_backend_calls_per_step_min": {"unsloth_grouped_mm": E}, "unsloth_manual_grouped_mm_calls_per_step_max": 0},
               profile={"summary": {"profiled_steps": 3, "device_ms_method": R.DEVICE_MS_METHOD, "device_ms": 900.0}, "top_cpu": [],
                        "top_device": [{"name": "_grouped_gemm_kernel", "count": 9 * E * 3}, {"name": "_grouped_wgrad_kernel", "count": 2 * E * 3}]})
    return R, row, table, census


@pytest.mark.parametrize("arm", ["U0", "U1", "UA"])
def test_an_unsloth_row_reads_and_each_piece_removed_voids(monkeypatch, tmp_path, arm):
    import copy

    R, row, table, census = _unsloth_row(monkeypatch, tmp_path, arm)
    assert R.row_faults(row, table, census) == []
    snap = row["hook"]["edges"][-1]["unsloth"]
    assert snap["counts"]["nf4_fused_calls"] == (0 if arm == "U0" else 2 * R.STEPS)
    for mutate, label in [
        (lambda r: r["hook"].pop("structural"), "no structural count"),
        (lambda r: r["hook"]["edges"][-1].pop("unsloth"), "no counter snapshot"),
        (lambda r: r["unsloth_env"].update(UNSLOTH_MOE_FUSED_ROUTER="1"), "the router switched on"),
        (lambda r: r["unsloth_env"].pop("UNSLOTH_ENABLE_LOGGING"), "a registered pin unset"),
        (lambda r: r["unsloth_env"].update(UNSLOTH_ANYTHING="1"), "an ambient key"),
        (lambda r: r.update(loader_used="FastLanguageModel"), "another loader"),
        (lambda r: r["absmax"].update(loaded_nested=False), "fp32 absmax"),
    ] + ([(lambda r: r["profile"]["top_device"][0].update(count=1), "F15 v2 short")] if arm == "U0" else []):
        r = copy.deepcopy(row)
        mutate(r)
        assert R.row_faults(r, table, census), f"{arm}: {label} still reads clean"


@pytest.mark.parametrize("arm", ["U1", "UA"])
def test_a_fused_dx_that_never_launched_voids_the_row(monkeypatch, tmp_path, arm):
    R, row, table, census = _unsloth_row(monkeypatch, tmp_path, arm, drop_dx_step=17)
    assert any("launches" in w for w in R.row_faults(row, table, census))


# -- p238_schedule.py and p238_drive.sh -----------------------------------------------------------------------------------------------
SCH = _load("p238_schedule", "bench/p238/p238_schedule.py")


def test_the_schedule_is_44_processes_with_each_points_draws_a_palindrome():
    s = SCH.schedule()
    assert [r["index"] for r in s] == list(range(1, 45))
    assert sum(r["kind"] == "op" for r in s) == 4 and sum(r["kind"] == "step" for r in s) == 40
    assert [r["framework"] for r in s if r["kind"] == "op"] == ["e4b", "unsloth", "unsloth", "e4b"]
    for pt in SCH.POINTS:
        d1 = [r["arm"] for r in s if r["point"] == pt and r["draw"] == 1]
        d2 = [r["arm"] for r in s if r["point"] == pt and r["draw"] == 2]
        assert d1 == list(SCH.ARMS) and d2 == list(reversed(SCH.ARMS))
    steps1 = [r["point"] for r in s if r["kind"] == "step" and r["draw"] == 1]
    steps2 = [r["point"] for r in s if r["kind"] == "step" and r["draw"] == 2]
    assert steps1[::5] == list(SCH.POINTS) and steps2[::5] == list(reversed(SCH.POINTS))


def test_the_driver_parses_and_refuses_unregistered_ids(tmp_path):
    import subprocess
    drive = ROOT / "bench" / "p238" / "p238_drive.sh"
    assert subprocess.run(["bash", "-n", str(drive)]).returncode == 0
    env = {**os.environ, "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
           "E4B_RENT_DEADLINE_EPOCH": "1"}
    for rid in ("p238-5090-3", "p238-shake-2", "p238-5090-1c", "p236-5090-1", "", "p238-5090-1a", "p238-5090-1-4-5"):
        r = subprocess.run(["bash", str(drive)], env={**env, "E4B_RENT_RUN_ID": rid}, capture_output=True, text=True)
        assert r.returncode == 78, (rid, r.stdout, r.stderr)


@pytest.mark.parametrize("files,want", [
    ({}, 23),                                                     # no exit file fetched
    ({"P238_EXIT_CODE.n1": "x\n"}, 23),                          # unreadable
    ({"P238_EXIT_CODE.n1": "0\n", "P238_SUCCESS.n1": ""}, 0),    # a success
    ({"P238_EXIT_CODE.n1": "0\n"}, 24),                          # 0 without the success marker: never success
    ({"P238_EXIT_CODE.n1": "0\n", "P238_SUCCESS.n0": ""}, 24),   # another run's marker does not count
    ({"P238_EXIT_CODE.n1": "13\n"}, 13),                         # disk: the launcher excludes the machine
    ({"P238_EXIT_CODE.n1": "18\n"}, 18),                         # card floor / CUDA: excluded
    ({"P238_EXIT_CODE.n1": "14\n"}, 14),                         # a launcher host code P238 does not register: still passes
    ({"P238_EXIT_CODE.n1": "17\n"}, 17),
    ({"P238_EXIT_CODE.n1": "15\n"}, 15),                         # card: passed through for the pin walk
    ({"P238_EXIT_CODE.n1": "16\n"}, 16),
    ({"P238_EXIT_CODE.n1": "27\n"}, 27),                         # VOID
    ({"P238_EXIT_CODE.n1": "41\n"}, 41),                         # fit
    ({"P238_EXIT_CODE.n1": "130\n"}, 130),
    ({"P238_EXIT_CODE.n1": "5\n"}, 26),                          # not a registered code
])
def test_the_driver_exit_passes_box_codes_through(tmp_path, files, want):
    import subprocess
    for name, body in files.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    drive = ROOT / "bench" / "p238" / "p238_drive.sh"
    r = subprocess.run(["bash", str(drive)], env={**os.environ, "P238_DRIVE_TAIL_DIR": str(tmp_path), "P238_DRIVE_TAIL_NONCE": "n1"},
                       capture_output=True, text=True)
    assert r.returncode == want, (files, r.stdout, r.stderr)


def test_mutant_the_old_tail_collapses_host_codes_to_23(tmp_path):
    """P236's tail (`[ "$RC" = 0 ] && exit 0; exit 23`) turns 13 and 18 into 23, so the launcher never excludes the machine."""
    import subprocess
    old = '[ "$RC" = 0 ] && exit 0\nexit 23\n'
    for rc in ("13", "18"):
        r = subprocess.run(["bash", "-c", f"RC={rc}\n" + old])
        assert r.returncode == 23 != int(rc)


def test_the_factory_takes_no_arguments_and_post_import_reads_the_framework(fake_route):
    """tc1_arm's load_arm_hook calls ``getattr(mod, fac)()``: a factory that needs an argument would exit 10 on the box."""
    import inspect
    assert list(inspect.signature(HOOK.make).parameters) == []
    h = HOOK.make()
    assert h.fw is None
    h.post_import(type("A", (), {"framework": "e4b", "accum": 3})())
    assert (h.fw, h.micro_batches) == ("e4b", 3)
    h.step_edge(0)
    e = h.record()["edges"][0]
    assert "energy_mj" in e and "t" in e                         # the counter's reading (None without NVML) is always recorded


# -- p238_reduce.py --work: the receipts a box leaves on disk, read end to end ------------------------------------------------------
def _write(path, obj):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj) + "\n").encode() if not isinstance(obj, str) else obj.encode())


def _work_dir(tmp_path):
    """A box's run directory with one op process (op-e4b, index 1) and one step load (E_F at 2k, index 3), built from the
    self-test's clean row: the tc1_arm receipt (arm_hook.record carrying the hook's facts and edges), the full profile, a gpuclk log
    in card_clock's format, box.json, the release proof's JSON lines, and the op record."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    e, table, census = R._good_rows()
    w = tmp_path / "p238"
    _write(w / "schedule.txt", "1 op 1 - e4b -\n3 step 1 2k E_F 0\n")
    cell = {"status": "ok", "env": {**{k: v for k, v in R.PINS.items()}, "torch": "2.12.1+cu130", "unsloth_env": {}},
            "absmax_dq_modules": R.N_LAYERS, "tokens_per_step": e["tokens_per_step"], "step_ms": e["step_ms"],
            "loss_first": 2.0, "eval_loss_final": 1.5, "peak_vram_gb": 28.0, "loader_used": None, "profile_warm": 20, "profile_steps": 3,
            "arm_hook": {"record": {"facts": {**e["hook"]["facts"], "premise_table": table, "adapters": census,
                                              "absmax_compressed": R.N_LAYERS, "fast_train_absmax_dq": e["absmax"]["fast_train"]},
                                    "edges": e["hook"]["edges"], "micro_batches": 1}}}
    base = w / "loads" / "3" / "qwen3_5_e4b_p238_3_E_F_2k"
    _write(base.with_name(base.name + ".json"), cell)
    _write(base.with_name(base.name + "_profile.json"), e["profile"])
    _write(w / "loads" / "gpuclk_3.txt", "".join(f"{1700000000 + i} 2800, 14001, 61, 0x0000000000000000\n" for i in range(200)))
    _write(w / "box.json", e["card"])
    _write(w / "env_box.txt", "".join(f"{k}={v}\n" for k, v in sorted(e["env_box"].items())))
    _write(w / "release_proof.json", "".join(f'{{"dist": "{d}", "record_equal": true}}\n' for d in R.RELEASES))
    op = {"p238_op": 1, "dry": False, "capture": "eager", "framework": "e4b", "arms": ["E_F", "E_A"],
          "facts": {"premise_table": table, "adapters": census},
          "cells": [{"rows": r, "routing": k, "b_restored_zero": True,
                     "parity": {a_: {"errs": {"out": 1e-3}, "ok": True} for a_ in ("E_F", "E_A")},
                     "visits": [{"device_ms_per_call": 1.0, "wall_ms_per_call": 1.1}] * (R.OP_REPS * 4)}
                    for r in R.OP_ROWS for k in R.OP_ROUTINGS]}
    _write(w / "ops" / "op_1_e4b.json", op)
    return R, w


def test_reduce_work_reads_a_box_directory_end_to_end(tmp_path):
    R, w = _work_dir(tmp_path)
    out = R.reduce_work(str(w), mode="shake")
    assert out["verdict"] == "SHAKE_OK", out                     # a partial box directory: its own faults, never read
    assert (out["n_rows"], out["n_op_rows"], out["n_void"]) == (1, 1, 0)
    assert R.reduce_work(str(w))["verdict"] == "VOID"             # read as the reading, a partial box is no reading


@pytest.mark.parametrize("remove", ["loads/3/qwen3_5_e4b_p238_3_E_F_2k.json", "loads/3/qwen3_5_e4b_p238_3_E_F_2k_profile.json",
                                    "loads/gpuclk_3.txt", "box.json", "release_proof.json", "ops/op_1_e4b.json", "env_box.txt"])
def test_reduce_work_voids_when_any_file_is_missing(tmp_path, remove):
    R, w = _work_dir(tmp_path)
    (w / remove).unlink()
    out = R.reduce_work(str(w), mode="shake")
    assert out["verdict"] == "VOID" and out["n_void"] >= 1


def test_the_runner_exits_only_registered_codes_and_host_codes_only_for_host_faults():
    """The launcher excludes a machine on 13 14 17 18 (the driver passes them through): the runner may exit 13 only at the disk
    floor and 18 only at a refused host fault, never 14 or 17, and every other exit is a code the PREREG registers."""
    import re
    text = (ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8")
    sites = [(m.group(1), text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.end())]) for m in re.finditer(r"finish (\d+)", text)]
    drive = (ROOT / "bench" / "p238" / "p238_drive.sh").read_text(encoding="utf-8")
    box_codes = set(re.search(r'^BOX_CODES="([0-9 ]+)"$', drive, re.M).group(1).split())
    assert {c for c, _ in sites} - {"0"} == box_codes, (sorted({c for c, _ in sites}), sorted(box_codes))   # both directions
    registered = {"0", "9", "10", "13", "15", "16", "18", "27", "41", "78", "130"}
    assert sites and all(code in registered for code, _ in sites), sorted({c for c, _ in sites} - registered)
    host = [(c, line) for c, line in sites if c in ("13", "14", "17", "18")]
    assert all(c in ("13", "18") for c, _ in host)
    assert [line for c, line in host if c == "13"] and all("disk" in line and "REFUSED" in line for c, line in host if c == "13")
    assert len([1 for c, _ in host if c == "13"]) == 1
    assert all("REFUSED" in line for c, line in host if c == "18")


def test_an_env_box_with_an_extra_key_voids_and_an_image_key_is_information(tmp_path):
    """E2: env_box.txt's key set must EQUAL the registered one (an ambient E4B_TRAIN_DETERMINISTIC VOIDs the box); a PYTORCH_* key
    from the image lands in env_image.txt as information and never VOIDs."""
    R, w = _work_dir(tmp_path)
    _write(w / "env_image.txt", "PYTORCH_VERSION=2.12.1\nNV_CUDA_LIB_VERSION=13.0.1\n")
    out = R.reduce_work(str(w), mode="shake")
    assert out["verdict"] == "SHAKE_OK" and out["image_env"]["PYTORCH_VERSION"] == "2.12.1", out
    body = (w / "env_box.txt").read_text(encoding="utf-8")
    for extra in ("E4B_TRAIN_DETERMINISTIC=1\n", "PYTORCH_VERSION=2.12.1\n", "not an env line\n"):
        (w / "env_box.txt").write_text(body + extra, encoding="utf-8")
        out = R.reduce_work(str(w), mode="shake")
        assert out["verdict"] == "VOID" and any("unregistered" in y for r in out["rows"] for y in r["why"]), (extra, out["rows"])


def _ambient_block(run=None):
    """The runner's marked block, from the line after its opening marker to its closing marker."""
    run = run or (ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8")
    return run.split("# ---- ambient knobs", 1)[1].split("\n", 1)[1].split("# ---- end ambient knobs", 1)[0]


@pytest.mark.parametrize("key,val,want", [
    ("E4B_TRAIN_DETERMINISTIC", "1", 78), ("GNF4_TRAIN_GEMM", "fused", 78), ("NF4_QLORA_ANYTHING", "1", 78),
    ("UNSLOTH_MOE_NF4_FUSED", "1", 78), ("E4B_ABSMAX_DQ", "1", 78),
    ("E4B_SHA", "a" * 40, 0), ("UNSLOTH_MOE_NF4_AUTOTUNE", "0", 0), ("PYTORCH_VERSION", "2.12.1", 0), (None, None, 0),
])
def test_the_runners_ambient_block_refuses_every_unregistered_key(key, val, want):
    """E2 at the box: the runner's own marked block, run as written: any UNSLOTH_* / E4B_* / GNF4_* / NF4_QLORA_* key outside the
    registered set refuses 78 before STARTED; E4B_SHA (the driver's) and image keys pass."""
    import re
    import subprocess
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    env = {k: v for k, v in os.environ.items() if not re.match(R.ENV_REFUSE_GREP, k)}
    if key:
        env[key] = val
    script = 'say(){ echo "$*"; }\nfinish(){ exit "$1"; }\n' + _ambient_block() + "exit 0\n"
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == want, (key, r.stdout, r.stderr)
    if want:
        assert key in r.stdout


def _env_disagreements(run, R):
    """Where the runner's ambient refusal, its env_box.txt grep, its exports and the reducer's registered set disagree."""
    import re
    bad = []
    block = _ambient_block(run)
    refuse = re.search(r"env \| grep -oE '([^']+)' \| sort -u", block)
    if not refuse or refuse.group(1) != R.ENV_REFUSE_GREP:
        bad.append(("refusal grep", refuse and refuse.group(1)))
    allowed = set()
    for labels in re.findall(r"^\s*(?:case \"\$k\" in )?([A-Z0-9_|]+)\) ;;", block, re.M):
        allowed |= set(labels.split("|"))
    if allowed != set(R.ENV_BOX) - set(R.ENV_EXACT):
        bad.append(("refusal allowlist", sorted(allowed ^ (set(R.ENV_BOX) - set(R.ENV_EXACT)))))
    dump = re.search(r"^env \| grep -E '([^']+)' \| sort > \$W/env_box\.txt$", run, re.M)
    if not dump or dump.group(1) != R.ENV_BOX_GREP:
        bad.append(("env_box grep", dump and dump.group(1)))
    exported = {}
    for line in re.findall(r"^export (.+)$", run, re.M):
        for kv in line.split():
            k, _, v = kv.partition("=")
            if re.match(R.ENV_BOX_GREP, k + "="):
                exported[k] = v
    if exported != {k: v for k, v in R.ENV_BOX.items() if v is not None}:
        bad.append(("exports", exported))
    if not all(re.match(R.ENV_BOX_GREP, k + "=") for k in R.ENV_BOX):
        bad.append(("a registered key the dump would not record", sorted(R.ENV_BOX)))
    return bad


def test_one_env_list_binds_the_refusal_the_dump_and_the_reducer(monkeypatch):
    """The maintainer's binding: the runner's box-side refusal, the env_box.txt grep and the reducer's registered set come from ONE
    list (the reducer's ENV_PREFIXES / ENV_EXACT / ENV_BOX). A mutant that widens or narrows any one of them fails here."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    run = (ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8")
    assert _env_disagreements(run, R) == []
    mutants = [
        ("    E4B_SHA) ;;", "    E4B_SHA|E4B_TRAIN_DETERMINISTIC) ;;"),                      # a wider box allowlist
        ("grep -oE '^(UNSLOTH|E4B|GNF4|NF4_QLORA)_", "grep -oE '^(UNSLOTH|E4B|GNF4)_"),          # a narrower refusal
        ("|^PYTORCH_CUDA_ALLOC_CONF=' | sort > $W/env_box.txt", "|^PYTORCH_' | sort > $W/env_box.txt"),   # a wider dump
        ("UNSLOTH_ENABLE_LOGGING=1\n", "UNSLOTH_ENABLE_LOGGING=1 UNSLOTH_MOE_NF4_FUSED=1\n"),     # an extra export
    ]
    for old, new in mutants:
        assert run.count(old) == 1, old
        assert _env_disagreements(run.replace(old, new), R), f"mutant survives: {new}"
    monkeypatch.setattr(R, "ENV_BOX", {**R.ENV_BOX, "E4B_TRAIN_DETERMINISTIC": "1"})        # a wider reducer set
    assert _env_disagreements(run, R)


def _energy_row(R):
    e, _, _ = R._good_rows()
    step_s = 1.0
    edges, t, mj = [], 1.7e9, 0
    for i in range(R.STEPS + 1):
        edges.append({"i": i, "t": t, "energy_mj": mj})
        t += step_s + 0.01                                                 # the step and a little between steps
        mj += 300_000                                                      # 300 J per step
    e["hook"] = {"edges": edges}
    e["step_ms"] = [step_s * 1e3] * R.STEPS
    e["profile_warm"], e["profile_steps"] = 20, 3                         # the runner's --profile-warm 20 --profile-steps 3
    return e


def _jump(e, at, secs=6.0, mj=2_400_000):
    """An eval- or profiler-sized jump inside edge pair (at - 1) -> at: every edge from ``at`` on is later and further along."""
    for edge in e["hook"]["edges"]:
        if edge["i"] >= at:
            edge["t"] += secs
            edge["energy_mj"] += mj
    return e


def test_the_energy_reads_two_spans_around_the_profile_window():
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    out = R.energy(_energy_row(R))
    assert out["status"] == "READ" and out["j_per_step"] == 300.0 and out["steps"] == 46, out
    assert out["spans"] == [[10, 20], [23, 59]] and out["j_per_token"] == round(300.0 / 2048, 6)


@pytest.mark.parametrize("at", [21, 22, 23])
def test_a_profiler_sized_jump_in_the_profile_window_moves_nothing(at, monkeypatch):
    """tc1_arm starts the profiler between edges 20 and 21 and stops it (plus summarize_profile) between 22 and 23, outside
    step_ms: several seconds and kJ there leave the energy READ and unchanged. The mutant that sums 10..59 straight fails here."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    clean = R.energy(_energy_row(R))["j_per_step"]
    out = R.energy(_jump(_energy_row(R), at))
    assert out["status"] == "READ" and out["j_per_step"] == clean, (at, out)
    monkeypatch.setattr(R, "ENERGY_SPANS", ((10, 59),))                  # mutant: one straight window over the profiler
    bad = R.energy(_jump(_energy_row(R), at))
    assert not (bad["status"] == "READ" and bad["j_per_step"] == clean), (at, bad)


@pytest.mark.parametrize("warm,steps", [(10, 3), (20, 5), (None, 3), (20, None)])
def test_a_row_whose_profile_flags_differ_reads_unread(warm, steps):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    e = _energy_row(R)
    e["profile_warm"], e["profile_steps"] = warm, steps
    out = R.energy(e)
    assert out["status"] == "UNREAD" and "profile window" in out["why"], out


def test_an_eval_sized_jump_never_moves_the_energy():
    """The E1 tests: an eval between edges 19 and 20 (tc1_arm's old default, --eval-every 20) makes the energy UNREAD,
    never a moved number; the registered held-out eval between edges 59 and 60 lies outside the window and moves nothing."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    clean = R.energy(_energy_row(R))["j_per_step"]
    for at, want in ((20, "UNREAD"), (60, "READ"), (40, "UNREAD"), (11, "UNREAD"), (59, "UNREAD")):
        out = R.energy(_jump(_energy_row(R), at))                          # a 48-row eval pass: seconds of wall and its joules
        assert out["status"] == want, (at, out)
        assert out["j_per_step"] in (None, clean), (at, out)              # never a different number
        if want == "UNREAD":
            assert "non-step wall" in out["why"]


@pytest.mark.parametrize("hit", ["pop_10", "pop_59", "pop_30", "pop_20", "pop_23", "none_10", "none_20", "none_23", "none_59",
                                 "t_59", "short_window", "no_step_ms"])
def test_a_removed_edge_or_counter_reads_unread_never_zero(hit):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    e = _energy_row(R)
    edges = e["hook"]["edges"]
    kind, _, n = hit.rpartition("_")
    if hit == "short_window":
        for edge in edges:
            edge["t"] = 1.7e9 + edge["i"] * 0.1
        e["step_ms"] = [100.0] * R.STEPS
    elif hit == "no_step_ms":
        e["step_ms"] = e["step_ms"][:30]
    elif kind == "pop":
        edges[:] = [x for x in edges if x["i"] != int(n)]                  # edges by their own index: a gap, not a shift
    elif kind == "none":
        edges[int(n)]["energy_mj"] = None
    else:
        edges[int(n)]["t"] = None
    out = R.energy(e)
    assert out["status"] == "UNREAD" and out["j_per_step"] is None and out["j_per_token"] is None, (hit, out)


def _load_s_lines(run=None):
    run = run or (ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8")
    lines = run.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("LOAD_S=${P238_LOAD_S_REGISTERED"))
    return "\n".join(lines[i:i + 3]) + "\n"


@pytest.mark.parametrize("val,want", [("0", 78), ("299", 78), ("300", 0), ("900", 0), (None, 0), ("", 0), ("abc", 78),
                                      ("9e2", 78), ("-5", 78), ("0300", 78), ("0900", 78), ("300 ", 78)])
def test_the_registered_load_wall_is_whole_seconds_over_the_floor(val, want):
    """The guard fit's per-load wall: whole decimal seconds, no leading 0 (bash arithmetic reads 0300 as octal), at least 300 s,
    else 78; unset or empty takes the 900 s default. The runner's own lines, run as written; the mutant floor -ge 0 fails here."""
    import subprocess
    pre = 'say(){ echo "$*"; }\nfinish(){ exit "$1"; }\nset -u\n'
    env = {k: v for k, v in os.environ.items() if k != "P238_LOAD_S_REGISTERED"}
    if val is not None:
        env["P238_LOAD_S_REGISTERED"] = val
    r = subprocess.run(["bash", "-c", pre + _load_s_lines() + 'echo "LOAD_S=$LOAD_S"\n'], env=env, capture_output=True, text=True)
    assert r.returncode == want, (val, r.stdout, r.stderr)
    if val == "299":                                                   # "0" falls to the no-leading-0 rule first; 299 only to the floor
        mutant = _load_s_lines().replace('-ge 300 ]', '-ge 0 ]')
        assert mutant != _load_s_lines()
        m = subprocess.run(["bash", "-c", pre + mutant], env=env, capture_output=True, text=True)
        assert m.returncode == 0                                       # the floor is what refuses 299


def _box_and_driver_codes(drive, tail):
    import re
    box = set(re.search(r'^BOX_CODES="([0-9 ]+)"$', drive, re.M).group(1).split())
    own = set(re.findall(r"\bexit (\d+)\b", drive)) | set(re.findall(r"\bexit (\d+)\b", tail))
    return box, own - {"0"}


def test_no_box_code_is_also_one_of_the_drivers_own_exits():
    """A box code equal to a code the driver (or drive_tail) exits by itself is ambiguous at the launcher, as 23/24/26 would be:
    the two sets meet only at 78, a config refusal on both sides. The mutant that registers the driver's 21 as a box code fails."""
    drive = (ROOT / "bench" / "p238" / "p238_drive.sh").read_text(encoding="utf-8")
    tail = (ROOT / "bench" / "common" / "drive_tail.sh").read_text(encoding="utf-8")
    box, own = _box_and_driver_codes(drive, tail)
    assert {"20", "21", "22", "23", "24", "26", "78"} <= own
    assert box & own == {"78"}, sorted(box & own)
    mbox, mown = _box_and_driver_codes(drive.replace('BOX_CODES="9 ', 'BOX_CODES="9 21 '), tail)
    assert mbox & mown != {"78"}


def test_the_fetch_brings_the_evidence_and_no_tensors_venv_or_bulk(tmp_path):
    """E3: the driver's fetch, its own exclude array, run by real rsync over a stub box tree holding everything the runner leaves
    in /root/p238. Fetched: every file the reducer reads, and the logs. Not fetched: the venv, adapters (tc1_arm --adapter-dir
    safetensors), the 40 MB alpaca source, compile caches, any tensor file, anything over 10 MiB."""
    import re
    import shutil
    import subprocess
    if not shutil.which("rsync"):
        pytest.fail("rsync is required: the driver's fetch is rsync")
    drive = (ROOT / "bench" / "p238" / "p238_drive.sh").read_text(encoding="utf-8")
    arr = re.search(r"^P238_RSYNC_EXCLUDES=\(.*?\)$", drive, re.M | re.S).group(0)
    assert re.search(r'^rsync -az --no-links .*"\$\{P238_RSYNC_EXCLUDES\[@\]\}" "root@\$HOST:\$W/" "\$OUT/"', drive, re.M)
    box, out = tmp_path / "box", tmp_path / "out"
    keep = ["schedule.txt", "box.json", "env_box.txt", "env_image.txt", "release_proof.json", "summary.txt", "forensics.txt",
            "staged-p238.sha256", "logs/load_3.log", "loads/3/qwen3_5_e4b_p238_3_E_F_2k.json",
            "loads/3/qwen3_5_e4b_p238_3_E_F_2k_profile.json", "loads/gpuclk_3.txt", "loads/power_3.txt", "ops/op_1_e4b.json",
            "data/tokens_packed.json", "data/ds_alpaca.json", "P238_EXIT_CODE.n1", "P238_SUCCESS.n1"]
    drop = ["venv-unsloth/bin/python", "venv-unsloth/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so",
            "loads/3/adapters/adapter_model.safetensors", "loads/3/adapters/adapter_config.json", "data/alpaca_data_cleaned.json",
            "unsloth_compiled_cache/UnslothSFTTrainer.py", "__pycache__/x.cpython-312.pyc", ".cache/pip/x", "stray/w.pt",
            "stray/w.bin", "stray/w.pth", "stray/big.json"]
    for f in keep + drop:
        p = box / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * (11 * 1024 * 1024 if f == "stray/big.json" else 64))
    r = subprocess.run(["bash", "-c", arr + '\nrsync -a --no-links "${P238_RSYNC_EXCLUDES[@]}" "$1/" "$2/"', "_", str(box), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = sorted(str(p.relative_to(out)).replace(os.sep, "/") for p in out.rglob("*") if p.is_file())
    assert got == sorted(keep), (sorted(set(got) - set(keep)), sorted(set(keep) - set(got)))
    assert not any(p.stat().st_size > 10 * 1024 * 1024 for p in out.rglob("*") if p.is_file())


@pytest.mark.parametrize("reset_at,span_end_lower", [(15, True), (40, True), (15, False), (40, False)])
def test_a_counter_that_goes_backwards_in_a_span_reads_unread(reset_at, span_end_lower):
    """An NVML / driver reset inside a span: the counter drops at edge ``reset_at``. Whether the span then ends below its start
    (eb < ea) or above it (a reset early in a long span), the energy reads UNREAD, never a negative or understated J/step."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    e = _energy_row(R)
    drop = 10 ** 12 if span_end_lower else 1_000_000                 # far below the start, or just one step's worth less
    for edge in e["hook"]["edges"]:
        if edge["i"] >= reset_at:
            edge["energy_mj"] -= drop
    end = 20 if reset_at < 20 else 59
    start = 10 if reset_at < 20 else 23
    by_i = {x["i"]: x for x in e["hook"]["edges"]}
    assert (by_i[end]["energy_mj"] < by_i[start]["energy_mj"]) is span_end_lower
    out = R.energy(e)
    assert out["status"] == "UNREAD" and out["j_per_step"] is None, (reset_at, out)


SPEED = {"E_F": 1.0, "E_A": 1.02, "U0": 1.3, "U1": 1.2, "UA": 1.21}


def _grid(R, monkeypatch, oom=(), **status):
    """Every step row of a two-draw box (4 points x 5 arms x 2 draws) from the self-test's clean row, each arm at its own speed.
    ``oom``: (point, arm, draw) rows that are tc1_arm's OOM stub (status oom, exit 5, phase train, a partial step list).
    ``status``: {"<point>:<arm>:<draw>": (status, rc)} for other stubs. row_faults is narrowed to check_present, so these tests
    read the OOM rule and the readings alone (the row checks have their own tests)."""
    import copy
    monkeypatch.setattr(R, "row_faults", lambda r, t, c: R.check_present(r))
    e, _, _ = R._good_rows()
    rows = []
    for s in R._sched().schedule():                    # the schedule's own indices, so boxes are its real blocks
        if s["kind"] != "step":
            continue
        idx, draw, point, arm = s["index"], s["draw"], s["point"], s["arm"]
        r = copy.deepcopy(e)
        r.update(index=idx, draw=draw, point=point, arm=arm, framework=R.ARMS[arm], step_ms=[1000.0 * SPEED[arm]] * R.STEPS,
                 rc=0, phase=None, peak_vram_gb=20.0 + idx / 100)
        r["profile"]["summary"]["device_ms"] = 900.0 * SPEED[arm]
        if (point, arm, draw) in oom:                  # a genuine OOM a few steps in: a short step list and a short clock log
            r.update(status="oom", rc=R.OOM_EXIT, phase="train", step_ms=[1000.0] * 3,
                     clock={"busy_median_mhz": None, "hard_reason_any": None, "swpowercap_clear_frac": None, "busy_samples": 3})
        if f"{point}:{arm}:{draw}" in status:
            st, rc = status[f"{point}:{arm}:{draw}"]
            r.update(status=st, rc=rc, phase="train")
        rows.append(r)
    return rows


def test_a_reproduced_oom_is_a_read_outcome_kept_out_of_every_ratio(monkeypatch):
    """The PREREG: "An OOM is a result". Both draws of U1 OOM at 8k: the box READs, the 8k pairs with U1 read OOM_DIFFERENCE
    (E_F fits, with its peak VRAM), and every other point's ratios are exactly the no-OOM box's. The mutant that is the old
    reducer (status != ok -> VOID, no OOM outcome) VOIDs the whole box and fails here."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    base = R.reduce(_grid(R, monkeypatch), [])
    assert base["verdict"] == "READ" and base["n_oom"] == 0
    assert base["readings"]["2k:E_F/U1"]["status"] == "RATIO" and base["readings"]["2k:E_F/U1"]["wall_ratio"] == round(1.0 / 1.2, 4)
    oom = {("8k", "U1", 1), ("8k", "U1", 2)}
    out = R.reduce(_grid(R, monkeypatch, oom), [])
    assert out["verdict"] == "READ" and out["n_oom"] == 2 and out["n_void"] == 0, [o for o in out["rows"] if o["verdict"] != "READ"]
    rd = out["readings"]
    assert rd["8k:E_F/U1"]["status"] == "OOM_DIFFERENCE" and (rd["8k:E_F/U1"]["oom"], rd["8k:E_F/U1"]["fits"]) == ("U1", "E_F")
    assert rd["8k:E_F/U1"]["fits_peak_vram_gb"] and rd["8k:U1/U0"]["status"] == "OOM_DIFFERENCE" and rd["8k:U1/U0"]["fits"] == "U0"
    assert rd["8k:E_A/UA"] == base["readings"]["8k:E_A/UA"]                    # the pairs without U1 at 8k are untouched
    assert {k: v for k, v in rd.items() if not k.startswith("8k:")} == {k: v for k, v in base["readings"].items() if not k.startswith("8k:")}
    assert all(o["s_per_step"] is None for o in out["rows"] if o["verdict"] == "OOM")
    monkeypatch.setattr(R, "is_oom_row", lambda r: False)                      # mutant: today's code, an OOM row is just not ok
    assert R.reduce(_grid(R, monkeypatch, oom), [])["verdict"] == "VOID"


def test_an_ooms_short_clock_log_never_voids_it(monkeypatch):
    """An OOM a few steps in leaves fewer than 10 busy clock samples: the OOM rule reads the card's identity, never its loaded
    clock (the box's other rows carry that). The mutant that reads the whole check_card VOIDs every genuine OOM and fails here."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    oom = {("8k", "U1", 1), ("8k", "U1", 2)}
    assert R.reduce(_grid(R, monkeypatch, oom), [])["verdict"] == "READ"
    import types
    path = ROOT / "bench" / "p238" / "p238_reduce.py"
    src = path.read_text(encoding="utf-8")
    mut = src.replace("return why + check_env_box(row) + check_card_identity(row)", "return why + check_env_box(row) + check_card(row)")
    assert mut != src
    M = types.ModuleType("p238_reduce_mutant")
    M.__file__ = str(path)
    exec(compile(mut, str(path), "exec"), M.__dict__)
    out = M.reduce(_grid(M, monkeypatch, oom), [])
    assert out["verdict"] == "VOID" and out["n_oom"] == 0


def test_both_sides_oom_at_a_point_leaves_it_out_of_that_pairs_step_read(monkeypatch):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    oom = {(p, a, d) for p in ("8k",) for a in ("E_F", "U1") for d in (1, 2)}
    out = R.reduce(_grid(R, monkeypatch, oom), [])
    assert out["verdict"] == "READ" and out["readings"]["8k:E_F/U1"]["status"] == "BOTH_OOM"
    assert out["readings"]["8k:E_A/E_F"]["status"] == "OOM_DIFFERENCE" and out["readings"]["8k:E_A/E_F"]["fits"] == "E_A"


def test_an_oom_in_one_draw_only_voids_both_draws(monkeypatch):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    out = R.reduce(_grid(R, monkeypatch, {("8k", "U1", 1)}), [])
    pair = [o for o in out["rows"] if (o["point"], o["arm"]) == ("8k", "U1")]
    assert [o["verdict"] for o in pair] == ["VOID", "VOID"] and all("did not reproduce" in o["why"][-1] for o in pair)
    assert out["verdict"] == "VOID" and out["readings"]["8k:E_F/U1"]["status"] == "VOID"


@pytest.mark.parametrize("st,rc,phase,fault", [
    ("harness_error", 10, "train", "status 'harness_error'"),     # a hook fault (tc1_arm's arm_hook_call stub, exit 10)
    ("oom", 10, "train", "not tc1_arm's OOM exit 5"),             # an oom status with another exit
    ("oom", None, "train", "not tc1_arm's OOM exit 5"),           # no PROC line for it
    ("oom", 5, None, "no phase"),
    ("refused", 3, "train", "status 'refused'"),
])
def test_anything_else_not_ok_stays_void(monkeypatch, st, rc, phase, fault):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    rows = _grid(R, monkeypatch)
    for r in rows:
        if (r["point"], r["arm"]) == ("8k", "U1"):
            r.update(status=st, rc=rc, phase=phase)
    out = R.reduce(rows, [])
    pair = [o for o in out["rows"] if (o["point"], o["arm"]) == ("8k", "U1")]
    assert out["verdict"] == "VOID" and all(o["verdict"] == "VOID" and any(fault in w for w in o["why"]) for o in pair), pair


def test_an_oom_on_an_unregistered_box_is_void(monkeypatch):
    """An OOM is a result only on the registered box: an extra env_box key, another card or a refused release proof VOIDs it."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    oom = {("8k", "U1", 1), ("8k", "U1", 2)}
    for spoil in (lambda r: r["env_box"].update(E4B_TRAIN_DETERMINISTIC="1"), lambda r: r["card"].update(sm_count=128),
                  lambda r: r["versions"].update(release_proof_ok=False)):
        rows = _grid(R, monkeypatch, oom)
        for r in rows:
            if r["status"] == "oom":
                spoil(r)
        out = R.reduce(rows, [])
        assert out["verdict"] == "VOID" and out["n_oom"] == 0


def test_the_runners_proc_lines_carry_each_loads_exit(tmp_path):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    run = (ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8")
    line = next(x.strip() for x in run.splitlines() if x.strip().startswith('echo "PROC $idx step '))
    assert line.endswith(" | tee -a summary.txt")
    import subprocess                                                              # the writer, rendered by bash, read by proc_rcs
    out = subprocess.run(["bash", "-c", "idx=17; who=U1; point=8k; draw=2; pos=3; rc=5\n" + line[:-len(" | tee -a summary.txt")]],
                         capture_output=True, text=True).stdout
    (tmp_path / "rendered.txt").write_text(out, encoding="utf-8")
    assert R.proc_rcs(str(tmp_path / "rendered.txt")) == {17: 5}, out
    (tmp_path / "summary.txt").write_text("PROC 3 step U1 8k draw=1 pos=2 rc=5\nPROC 1 op e4b draw=1 rc=0\nPROC 4 step E_F 2k draw=1 "
                                          "pos=0 rc=0\nPROC 5 step E_F 2k draw=1 pos=0 rc=x\n", encoding="utf-8")
    assert R.proc_rcs(str(tmp_path / "summary.txt")) == {3: 5, 4: 0}
    assert R.proc_rcs(str(tmp_path / "missing.txt")) == {}


def _ops(R, monkeypatch):
    """The schedule's op processes as op rows whose checks pass (op_faults, the premise and the census narrowed), so the box tests
    read the merge rules alone."""
    monkeypatch.setattr(R, "op_faults", lambda rec: [])
    monkeypatch.setattr(R, "check_premise", lambda o, t: [])
    monkeypatch.setattr(R, "check_census", lambda o, c: [])
    return [{"index": s["index"], "framework": s["framework"], "draw": s["draw"], "record": {}}
            for s in R._sched().schedule() if s["kind"] == "op"]


def _parts(R, monkeypatch, k, oom=(), machines=None):
    """The K boxes of a reading (p238_schedule.blocks(K); K = 1 is one box), each its block's step rows and op rows."""
    rows, ops = _grid(R, monkeypatch, oom), _ops(R, monkeypatch)
    out = []
    for i, b in enumerate(R._sched().blocks(k)):
        want = set(b)
        rs, os_ = [r for r in rows if r["index"] in want], [o for o in ops if o["index"] in want]
        out.append({"want": want, "draws": sorted({x["draw"] for x in rs + os_}), "rows": rs, "ops": os_,
                    "machine": (machines or ["m-1"] * k)[i], "image_env": None})
    return out


def test_the_blocks_cover_the_schedule_once_by_draw_and_point_never_by_arm():
    S = _load("p238_schedule", "bench/p238/p238_schedule.py")
    sched = {r["index"]: r for r in S.schedule()}
    for k in S.SPLITS:
        b = S.blocks(k)
        assert len(b) == k and sorted(i for x in b for i in x) == sorted(sched)
        if k == 1:
            continue
        firsts = {}
        for n, x in enumerate(b):
            rows = [sched[i] for i in x]
            assert len({r["draw"] for r in rows}) == 1                                  # one draw per box
            for p in {r["point"] for r in rows if r["kind"] == "step"}:                 # every arm of every point it holds
                assert sorted(r["arm"] for r in rows if r["point"] == p) == sorted(S.ARMS)
            firsts.setdefault(rows[0]["draw"], n)
            assert any(r["kind"] == "op" for r in rows) == (firsts[rows[0]["draw"]] == n)   # a draw's ops ride in its first box
        for i in range(1, k + 1):
            assert [r["index"] for r in S.box_rows(f"{k}-{i}")] == b[i - 1]
    assert [r["index"] for r in S.box_rows("all")] == sorted(sched)
    for bad in ("2-3", "4-0", "3-1", "16-1", "1-1", "x", "2-", "all2"):
        with pytest.raises(ValueError):
            S.box_rows(bad)


def test_the_plan_picks_the_fewest_k_under_the_box_cap():
    """The box cap: min(the policy's per-run wallclock maximum, 18 h; the hours USD 15 buys at USD 1.05/h after USD 1.64 of
    downloads) -- one constant, pinned here. The plan is the fewest K whose largest box fits it; K = 1 whenever one box fits."""
    import re
    import subprocess
    import sys as _sys
    S = _load("p238_schedule", "bench/p238/p238_schedule.py")
    assert (S.POLICY_MAX_WALLCLOCK_H, S.RUN_OVERRIDE_MAX_H, S.RUN_WORST_USD, S.USD_PER_H_MAX, S.DOWNLOAD_USD) == (6, 18, 15.0, 1.05, 1.64)
    assert not S.needs_override(7, 900) and S.needs_override(44, 480)            # the shakedown fits 6 h; a one-box reading does not
    assert S.box_cap_s() == int(min(18, (15.0 - 1.64) / 1.05) * 3600)
    run = (ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8")
    m = re.search(r"^INSTALL_S=(\d+); FETCH_S=(\d+); OP_TIMING_S=(\d+); RESERVE=(\d+)$", run, re.M)
    assert tuple(map(int, m.groups())) == (S.INSTALL_S, S.FETCH_S, S.OP_TIMING_S, S.RESERVE_S)
    assert "FIT_NEED=$((FIT * 125 / 100))" in run and S.HEADROOM_PCT == 125
    for load_s, k, want in ((200, 1, 1), (480, 1, 1), (600, 1, 1), (700, 2, 2), (900, 2, 2), (1500, 4, "NEEDS_APPROVAL"),
                            (3000, 8, "NEEDS_APPROVAL"), (5000, None, None)):
        assert S.fewest_k(load_s) == k and S.plan(load_s) == want, (load_s, S.fewest_k(load_s), S.plan(load_s))
        fits = [max(S.fit_s(len(b), load_s) for b in S.blocks(x)) <= S.box_cap_s() for x in S.SPLITS]
        assert k == next((x for x, f in zip(S.SPLITS, fits) if f), None)            # the fewest K that fits
    assert S.APPROVED_K_MAX == 2                                                       # the lane ceiling approved for this registration
    out = subprocess.run([_sys.executable, str(ROOT / "bench" / "p238" / "p238_schedule.py"), "--plan", "900"], capture_output=True, text=True)
    assert out.stdout.strip() == "2"
    out = subprocess.run([_sys.executable, str(ROOT / "bench" / "p238" / "p238_schedule.py"), "--plan", "5000"], capture_output=True, text=True)
    assert out.stdout.strip() == "NO_FIT"
    out = subprocess.run([_sys.executable, str(ROOT / "bench" / "p238" / "p238_schedule.py"), "--plan", "1500"], capture_output=True, text=True)
    assert out.stdout.strip() == "NEEDS_APPROVAL"


@pytest.mark.parametrize("k", [2, 4, 8])
def test_a_split_reproduces_the_one_box_verdict_on_the_same_rows(monkeypatch, k):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    one = R.reduce_boxes(_parts(R, monkeypatch, 1))
    split = R.reduce_boxes(_parts(R, monkeypatch, k))
    assert one["verdict"] == split["verdict"] == "READ" and one["readings"] == split["readings"], split["merge_why"]
    assert split["k"] == k and [x["part_verdict"] for x in split["boxes"]] == ["PART_OK"] * k
    assert [x["block"] for x in split["boxes"]] == [f"{k}-{i}" for i in range(1, k + 1)]
    for p in _parts(R, monkeypatch, k):
        assert R.reduce_boxes([p], mode="part")["verdict"] == "PART_OK"


def test_a_part_alone_or_missing_is_no_reading(monkeypatch):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    parts = _parts(R, monkeypatch, 4)
    out = R.reduce_boxes(parts[:3])
    assert out["verdict"] == "VOID" and "a reading is 1, 2, 4, 8 boxes" in out["merge_why"][0], out["merge_why"]
    two = _parts(R, monkeypatch, 2)
    out = R.reduce_boxes(two[:1])
    assert out["verdict"] == "VOID" and any("in no box" in w for w in out["merge_why"]), out["merge_why"]
    out = R.reduce_boxes(parts[:2])                                                    # two of four: a 2-box reading's shape, not its blocks
    assert out["verdict"] == "VOID" and any("in no box" in w for w in out["merge_why"]), out["merge_why"]
    assert R.reduce_boxes(two, mode="part")["verdict"] == "VOID"                       # a part is one box
    whole = _parts(R, monkeypatch, 1)[0]
    assert R.reduce_boxes([whole], mode="part")["verdict"] == "VOID"                   # the whole schedule is no part
    two[0]["want"] = None
    assert "without its schedule.txt" in " ".join(R.reduce_boxes(two)["merge_why"])


def test_an_index_in_two_boxes_or_in_none_or_not_whole_blocks_reads_void(monkeypatch):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    parts = _parts(R, monkeypatch, 4)
    dup = [dict(p) for p in parts]
    dup[1]["want"] = dup[1]["want"] | {min(dup[0]["want"])}
    assert any("in two boxes" in w for w in R.reduce_boxes(dup)["merge_why"])
    gap = [dict(p) for p in parts]
    gap[3]["want"] = gap[3]["want"] - {max(gap[3]["want"])}
    assert any("in no box" in w for w in R.reduce_boxes(gap)["merge_why"])
    swap = [dict(p) for p in parts]                                                    # every index once, but a point cut across boxes
    a, b = max(swap[0]["want"]), min(swap[1]["want"])
    swap[0]["want"], swap[1]["want"] = (swap[0]["want"] - {a}) | {b}, (swap[1]["want"] - {b}) | {a}
    assert any("not the whole (draw, point) blocks" in w for w in R.reduce_boxes(swap)["merge_why"])


@pytest.mark.parametrize("k", [2, 8])
def test_an_oom_in_draw_1_on_one_box_and_draw_2_on_another_reproduces(monkeypatch, k):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    parts = _parts(R, monkeypatch, k, {("8k", "U1", 1), ("8k", "U1", 2)})
    holders = [p for p in parts if any(r["status"] == "oom" for r in p["rows"])]
    assert len(holders) == 2 and all(R.reduce_boxes([p], mode="part")["verdict"] == "PART_OK" for p in holders)   # waits for the merge
    out = R.reduce_boxes(parts)
    assert out["verdict"] == "READ" and out["n_oom"] == 2 and out["readings"]["8k:E_F/U1"]["status"] == "OOM_DIFFERENCE"
    assert R.reduce_boxes(_parts(R, monkeypatch, k, {("8k", "U1", 1)}))["verdict"] == "VOID"   # draw 1 only: not reproduced


@pytest.mark.parametrize("machines,cross", [(("m-1",) * 4, False), (("m-1", "m-1", "m-2", "m-1"), True),
                                            (("m-1", None, "m-1", "m-1"), None), ((None,) * 4, None)])
def test_cross_host_is_set_exactly_when_the_boxes_machines_differ(monkeypatch, machines, cross):
    """And the verdict never names a machine: the ids are compared only (store-only), the verdict carries cross_host alone. A box
    without its rental record's machine VOIDs the merge (the band cannot be placed)."""
    import json
    import re
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    out = R.reduce_boxes(_parts(R, monkeypatch, 4, machines=list(machines)))
    if cross is None:
        assert out["verdict"] == "VOID" and any("rental record" in w for w in out["merge_why"])
    else:
        assert out["verdict"] == "READ" and out["cross_host"] is cross
        assert out["stability_scope"] == ("cross_host" if cross else "within_host")
    text = json.dumps(out, default=str)
    assert "m-1" not in text and "m-2" not in text

    def keys(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield str(k)
                yield from keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from keys(v)
    assert not [k for k in keys(out) if re.search(r"machine|instance|uuid|host_id|offer", k, re.I)]


def test_the_rental_machine_is_read_from_the_controllers_receipt_beside_the_box(tmp_path):
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    w = tmp_path / "run" / "p238"
    w.mkdir(parents=True)
    assert R.rental_machine(str(w)) is None                                         # on the box: never known
    _write(tmp_path / "run" / "receipt.json", {"environment": {"vast_machine_id": 1234}})
    assert R.rental_machine(str(w)) == "1234"
    _write(tmp_path / "run" / "receipt.json", {"environment": {"vast_machine_id": ""}})
    assert R.rental_machine(str(w)) is None


def test_quality_and_stability_are_read_per_pair(monkeypatch):
    """QUALITY_FAIL (a cross-framework held-out gap over 0.05 in one draw) keeps the ratio beside it, never quoted; an UNSTABLE arm
    (draws more than 10 % apart for Unsloth) keeps the pair out of the headline; UA against U0 carries the quality bars only."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    rows = _grid(R, monkeypatch)
    for r in rows:
        if (r["point"], r["arm"], r["draw"]) == ("2k", "U1", 1):
            r["heldout_loss"] = 1.6
        if (r["point"], r["arm"], r["draw"]) == ("4k", "U1", 2):
            r["step_ms"] = [1000.0 * SPEED["U1"] * 1.2] * R.STEPS
    rd = R.reduce(rows, [])["readings"]
    q = rd["2k:E_F/U1"]
    assert q["status"] == "QUALITY_FAIL" and not q["quoted"] and q["wall_ratio"] and "draw 1" in q["quality_why"][0]
    assert not q["headline_eligible"] and rd["2k:E_A/UA"]["headline_eligible"]
    s = rd["4k:E_F/U1"]
    assert s["status"] == "RATIO" and s["quoted"] and s["unstable"] and s["unstable"][0].startswith("U1") and not s["headline_eligible"]
    assert set(s["per_draw_wall_ratio"]) == {"1", "2"} and s["per_draw_wall_ratio"]["1"] != s["per_draw_wall_ratio"]["2"]
    assert rd["2k:UA/U0"]["ratio_registered"] is False and rd["2k:E_F/U1"]["ratio_registered"] is True


@pytest.mark.parametrize("v,box,want", [("READ", "all", 0), ("PART_OK", "2-1", 0), ("PART_OK", "4-3", 0), ("PART_OK", "8-8", 0),
                                         ("VOID", "2-2", 27), ("READ", "2-1", 27), ("PART_OK", "all", 27), ("VOID", "all", 27),
                                         ("SHAKE_OK", "all", 27)])
def test_the_runners_finish_reads_a_part_as_part_ok_and_a_reading_as_read(v, box, want):
    """A clean box of a split finishes 0 (PART_OK, never a strike), a faulted one 27; a one-box reading finishes 0 only on READ.
    The runner's own lines, as written; and the mode it passes the reducer per box."""
    import subprocess
    run = (ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8")
    fin = next(line for line in run.splitlines() if line.startswith('case "$V:$P238_BOX" in'))
    mode = next(line for line in run.splitlines() if line.startswith('case "$P238_BOX" in all) MODE=read'))
    pre = 'say(){ :; }\nfinish(){ exit "$1"; }\n'
    r = subprocess.run(["bash", "-c", pre + f"V={v}; P238_BOX={box}\n" + fin + "\n"], capture_output=True, text=True)
    assert r.returncode == want, (v, box, r.stderr)
    for b, m in (("all", "read"), ("2-1", "part"), ("8-8", "part"), ("shake", "shake")):
        out = subprocess.run(["bash", "-c", f"P238_BOX={b}\n" + mode + '\necho "$MODE"\n'], capture_output=True, text=True)
        assert out.stdout.strip() == m, (b, out.stdout)


def _id_block(text, var):
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith(f'case "${var}" in'))
    j = next(n for n in range(i, len(lines)) if lines[n].strip() == "esac")
    return "\n".join(lines[i:j + 1])


VALID_IDS = ["p238-shake-1", "p238-shake-r2", "p238-5090-1", "p238-5090-2", "p238-5090-r1", "p238-5090-r3", "p238-5090-1-2-1",
             "p238-5090-1-2-2", "p238-5090-2-4-4", "p238-5090-1-4-1", "p238-5090-1-8-8", "p238-5090-2-8-1"]
BAD_IDS = ["p238-5090-1a", "p238-5090-1b", "p238-5090-1-2-3", "p238-5090-1-4-5", "p238-5090-1-3-1", "p238-5090-3-2-1",
           "p238-5090-1-8-9", "p238-5090-1-16-1", "p238-5090-1-2-1x", "p238-5090-1-1-1", "p238-shake-3", ""]


def test_the_runner_and_the_driver_map_every_id_to_the_same_box():
    """The driver derives P238_BOX from the run id and the runner refuses a mismatch: both case blocks, run as written, take the
    same ids to the same box and refuse the same others (78); every split box id names a real block of the schedule."""
    import subprocess
    S = _load("p238_schedule", "bench/p238/p238_schedule.py")
    run = _id_block((ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8"), "P238_RUN_ID")
    drv = _id_block((ROOT / "bench" / "p238" / "p238_drive.sh").read_text(encoding="utf-8"), "E4B_RENT_RUN_ID")
    pre = 'say(){ :; }\nfinish(){ exit "$1"; }\n'
    for rid in VALID_IDS + BAD_IDS:
        a = subprocess.run(["bash", "-c", pre + f"P238_RUN_ID='{rid}'\n" + run + '\necho "$WANT_BOX"\n'], capture_output=True, text=True)
        b = subprocess.run(["bash", "-c", pre + f"E4B_RENT_RUN_ID='{rid}'\n" + drv + '\necho "$BOX"\n'], capture_output=True, text=True)
        if rid in BAD_IDS:
            assert a.returncode == b.returncode == 78, (rid, a.returncode, b.returncode)
            continue
        assert a.returncode == b.returncode == 0 and a.stdout == b.stdout, (rid, a.stdout, b.stdout)
        box = a.stdout.strip()
        assert box in ("all", "shake") or S.box_rows(box), (rid, box)


def test_the_runner_passes_eval_every_steps():
    """E1(a): tc1_arm's default --eval-every is 20; the runner's TC1_FLAGS must pass --eval-every $STEPS (one eval, after the
    last step), and STEPS is 60, the reducer's."""
    import re
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    run = (ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8")
    assert re.search(r"--steps \$STEPS .*--eval-every \$STEPS ", run)
    assert re.search(r"^STEPS=60\b", run, re.M) and R.STEPS == 60 and R.ENERGY_SPANS == ((10, 20), (23, 59))
    assert re.search(r"--profile-steps 3 --profile-warm 20\b", run) and R.PROFILED == (20, 23)


STAGED = ["bench/p238/p238_run.sh", "bench/p238/p238_op.py", "bench/p238/p238_hook.py", "bench/p238/p238_bytes.py",
          "bench/p238/p238_reduce.py", "bench/p238/p238_schedule.py", "bench/common/unsloth_moe_counter.py", "bench/common/card_clock.py",
          "bench/tc1/tc1_arm.py", "bench/tc1/f15v2.py", "bench/tc1/release_proof.py", "bench/tp4/tp4_alpaca.py"]


def test_the_staged_pin_is_the_repos_bytes_and_the_drivers_list():
    """staged-p238.sha256 names every file the driver stages (but itself), by the hash of the bytes in this tree. A kit file changed
    without re-pinning fails here, before any box (mutant loops on staged files deselect this test, by design)."""
    import hashlib
    import re
    pin = {}
    for line in (ROOT / "bench" / "p238" / "staged-p238.sha256").read_text(encoding="utf-8").splitlines():
        sha, name = line.split()
        pin[name] = sha
    assert pin == {pathlib.Path(f).name: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in STAGED}
    drive = (ROOT / "bench" / "p238" / "p238_drive.sh").read_text(encoding="utf-8")
    staged = set(re.findall(r"(bench/[a-z0-9_/-]+\.(?:sha256|py|sh))\b", drive.split("for f in", 1)[1].split("; do", 1)[0]))
    assert staged == set(STAGED) | {"bench/p238/staged-p238.sha256"}
    run = (ROOT / "bench" / "p238" / "p238_run.sh").read_text(encoding="utf-8")
    assert all(pathlib.Path(f).name in run for f in STAGED)        # the runner checks every staged file it is handed


def test_tc1_arm_loads_the_p238_hook_by_its_staged_sha(monkeypatch, tmp_path):
    """tc1_arm.py's own loader (e4b#2146) takes P238's hook at the sha the staged pin names, finds every phase, and refuses (10) any
    other bytes: the runner's TC1_ARM_HOOK / TC1_ARM_HOOK_SHA256 seam, end to end on CPU."""
    import hashlib
    T = _load("tc1_arm_for_p238", "bench/tc1/tc1_arm.py")
    monkeypatch.syspath_prepend(str(ROOT / "bench" / "p238"))
    pin = dict(reversed(line.split()) for line in (ROOT / "bench" / "p238" / "staged-p238.sha256").read_text(encoding="utf-8").splitlines())
    sha = hashlib.sha256((ROOT / "bench" / "p238" / "p238_hook.py").read_bytes()).hexdigest()
    assert pin["p238_hook.py"] == sha
    monkeypatch.setenv("TC1_ARM_HOOK", "p238_hook:make")
    monkeypatch.setenv("TC1_ARM_HOOK_SHA256", sha)
    a = T.build_parser().parse_args(["--out", str(tmp_path), "--tag", "p238_hook_seam"])
    h = T.load_arm_hook(a)
    assert h["phases"] == ["post_import", "post_load", "step_edge", "record"] and h["sha256"] == sha and h["file"] == "p238_hook.py"
    monkeypatch.setenv("TC1_ARM_HOOK_SHA256", "0" * 64)
    with pytest.raises(SystemExit) as e:
        T.load_arm_hook(a)
    assert e.value.code == 10
