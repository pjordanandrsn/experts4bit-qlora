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
    for rid in ("p238-5090-3", "p238-shake-2", "p238-5090-1c", "p236-5090-1", ""):
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
