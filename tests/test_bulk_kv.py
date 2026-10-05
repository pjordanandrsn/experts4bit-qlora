# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Bulk KV bookkeeping (``E4B_PAGED_BULK_KV``) and the per-step trace (``E4B_PAGED_STEP_TRACE``).

The bulk forms -- :meth:`Fp8PagedKV.reset_all_layers`, :meth:`~Fp8PagedKV.claim_blocks`,
:meth:`~Fp8PagedKV.append_prompt` -- must leave EXACTLY what the per-layer forms leave: every valid token's pool bytes
(payload and scales), the block tables and their host mirror, the lengths, and the free lists. With the same sequence
of requests the two paths give a slot the same rows, so the whole pool is compared byte for byte, not just the
tokens a reader can see. End to end, a tiny model served through the scheduler decodes the same tokens either way.
CPU-runnable (the paged attention is replaced by a reference read of the pool, as in ``test_linear_state``).
"""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm N-series")
pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BT = 16


def _kv(L=3, H=2, D=32, k_groups=None, batch=3, max_tokens=96, scratch=0):
    return Fp8PagedKV(L, H, D, batch=batch, max_tokens_per_seq=max_tokens, k_groups=k_groups, device=DEV,
                      scratch_slots=scratch)


def _prompt(kv, layers, T, seed):
    g = torch.Generator().manual_seed(seed)
    ks = [(torch.randn(T, kv.Hs[i], kv.Ds[i], generator=g) * 1.5).to(DEV, torch.bfloat16) for i in layers]
    vs = [torch.randn(T, kv.Hs[i], kv.Ds[i], generator=g).to(DEV, torch.bfloat16) for i in layers]
    return ks, vs


def _library_flush(kv, slot, layers, ks, vs, claim_all=False):
    for layer, k, v in zip(layers, ks, vs):
        kv.append(layer, slot, k.contiguous(), v.contiguous())
    if claim_all:                                     # PagedModelRunner._ensure_graph_ready, per layer
        for layer in layers:
            kv._ensure_blocks(layer, slot, kv.blocks_per_seq - 1)


def _bulk_flush(kv, slot, layers, ks, vs, claim_all=False):
    if claim_all:                                     # PagedModelRunner._flush_bulk claims first, then writes
        kv.claim_blocks(slot, kv.blocks_per_seq - 1, layers)
    kv.append_prompt(slot, layers, ks, vs)


def _state(kv):
    if DEV == "cuda":
        torch.cuda.synchronize()
    return {"k": kv.kp.dev.clone(), "v": kv.vp.dev.clone(), "bt": kv._bt_all.clone(), "lens": kv.seq_lens.clone(),
            "rows": {k: list(v) for k, v in kv._rows.items()}, "free": [list(f) for f in kv._free],
            "seen": [list(s) for s in kv._seen]}


def _assert_same_state(a, b):
    assert a["rows"] == b["rows"], "a slot was given different rows"
    assert a["free"] == b["free"], "the free lists differ"
    assert a["seen"] == b["seen"], "the host length mirrors differ"
    assert torch.equal(a["bt"], b["bt"]), "the device block tables differ"
    assert torch.equal(a["lens"], b["lens"]), "the device lengths differ"
    assert torch.equal(a["k"], b["k"]), "the key pool bytes differ"
    assert torch.equal(a["v"], b["v"]), "the value pool bytes differ"


def _tables_match_mirror(kv):
    for (layer, slot), rows in kv._rows.items():
        assert kv._bt_all[layer, slot, :len(rows)].tolist() == rows, (layer, slot)


GEOMS = {
    "uniform": dict(L=3, H=2, D=32),
    "uniform-k4": dict(L=3, H=2, D=32, k_groups=4),
    "mixed": dict(L=3, H=[4, 2, 4], D=[32, 64, 32]),       # Gemma-4-like: per-layer (kv heads, head_dim)
}


@pytest.mark.parametrize("geom", sorted(GEOMS))
@pytest.mark.parametrize("T", [1, 15, 16, 17, 40, 48])
@pytest.mark.parametrize("claim_all", [False, True])
def test_bulk_leaves_the_pool_exactly_as_the_per_layer_path(geom, T, claim_all):
    """The same requests through both paths on two fresh pools: identical pools, tables, lengths and free lists --
    including a recycled slot and a second resident slot."""
    out = []
    for flush, reset in ((_library_flush, "reset"), (_bulk_flush, "reset_all_layers")):
        kv = _kv(**GEOMS[geom])
        layers = list(range(kv.L))
        for slot, seed, t in ((0, 1, T), (1, 2, max(1, T - 3)), (0, 3, T)):   # slot 0 is recycled for the third
            getattr(kv, reset)(slot)
            ks, vs = _prompt(kv, layers, t, seed)
            flush(kv, slot, layers, ks, vs, claim_all=claim_all)
        _tables_match_mirror(kv)
        out.append(_state(kv))
    _assert_same_state(*out)


@pytest.mark.parametrize("group_layers", [1, 2])
def test_groups_bounded_by_bytes_leave_the_same_pool(group_layers, monkeypatch):
    """A prompt too large for one group splits into groups of layers (BULK_GROUP_BYTES per side); the pool is the
    same however the layers are grouped."""
    from experts4bit_qlora.engines import fp8_paged_kv
    T, out = 40, []
    for flush in (_library_flush, _bulk_flush):
        kv = _kv(**GEOMS["mixed"])
        layers = list(range(kv.L))
        monkeypatch.setattr(fp8_paged_kv, "BULK_GROUP_BYTES", group_layers * T * 2 * 64 * 2)
        ks, vs = _prompt(kv, layers, T, 9)
        flush(kv, 1, layers, ks, vs, claim_all=True)
        out.append(_state(kv))
    _assert_same_state(*out)


def test_bulk_flush_dequantizes_to_the_direct_quantization():
    from fp8_kv import dequant_kv_fp8_ref, quantize_kv_fp8
    kv = _kv(k_groups=4)
    layers = list(range(kv.L))
    ks, vs = _prompt(kv, layers, 37, 7)
    kv.append_prompt(2, layers, ks, vs)
    for layer, k, v in zip(layers, ks, vs):
        got_k, got_v = kv.reference_kv(layer, 2)
        qk, sk = quantize_kv_fp8(k, group=32 // 4)
        qv, sv = quantize_kv_fp8(v)
        assert torch.equal(got_k, dequant_kv_fp8_ref(qk, sk)) and torch.equal(got_v, dequant_kv_fp8_ref(qv, sv))


def test_reset_all_layers_equals_reset():
    out = []
    for reset in ("reset", "reset_all_layers"):
        kv = _kv()
        layers = list(range(kv.L))
        for slot in (0, 1):
            ks, vs = _prompt(kv, layers, 23 + slot, slot)
            _library_flush(kv, slot, layers, ks, vs)
        getattr(kv, reset)(0)
        out.append(_state(kv))
    _assert_same_state(*out)
    assert (0, 0) not in out[1]["rows"] and out[1]["seen"][0][0] == 0


def test_a_subset_of_layers_is_written_alone():
    """A hybrid model's pool keeps a row per model layer index but only its attention layers append."""
    out = []
    for flush in (_library_flush, _bulk_flush):
        kv = _kv(L=4)
        ks, vs = _prompt(kv, [1, 3], 21, 5)
        flush(kv, 0, [1, 3], ks, vs, claim_all=True)
        out.append(_state(kv))
    _assert_same_state(*out)
    assert out[1]["seen"][0][0] == 0 and out[1]["seen"][1][0] == 21


def test_a_slot_that_already_holds_tokens_takes_the_per_layer_path():
    out = []
    for flush in (_library_flush, _bulk_flush):
        kv = _kv()
        layers = list(range(kv.L))
        ks, vs = _prompt(kv, layers, 10, 1)
        _library_flush(kv, 0, layers, ks, vs)
        ks, vs = _prompt(kv, layers, 9, 2)
        flush(kv, 0, layers, ks, vs)                    # a second chunk onto history: append per layer
        out.append(_state(kv))
    _assert_same_state(*out)


def test_an_overflowing_prompt_is_refused_before_anything_moves():
    kv = _kv(max_tokens=32)
    layers = list(range(kv.L))
    before = _state(kv)
    ks, vs = _prompt(kv, layers, 33, 1)
    with pytest.raises(ValueError, match="overflows"):
        kv.append_prompt(0, layers, ks, vs)
    _assert_same_state(before, _state(kv))


def test_wrong_geometry_is_refused():
    kv = _kv()
    ks, vs = _prompt(kv, [0, 1, 2], 8, 1)
    ks[1] = ks[1][:, :1]
    with pytest.raises(ValueError, match="expected"):
        kv.append_prompt(0, [0, 1, 2], ks, vs)


def test_claim_blocks_with_layers_out_of_step_takes_the_per_layer_path():
    out = []
    for claim in ("library", "bulk"):
        kv = _kv()
        kv._ensure_blocks(1, 0, 2)                       # layer 1 already holds three blocks of slot 0
        if claim == "library":
            for layer in range(kv.L):
                kv._ensure_blocks(layer, 0, 4)
        else:
            kv.claim_blocks(0, 4)
        out.append(_state(kv))
    _assert_same_state(*out)


def test_claim_past_capacity_is_refused_before_anything_moves():
    kv = _kv(batch=1, max_tokens=32)                     # two blocks per layer in the whole arena
    before = _state(kv)
    with pytest.raises(RuntimeError, match="out of KV blocks"):
        kv.claim_blocks(0, 2)
    _assert_same_state(before, _state(kv))


# ------------------------------------------------------------------------------------------------ end to end --

def _tiny_model(seed=0):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, vocab_size=128, max_position_embeddings=256)
    return Qwen3ForCausalLM(cfg).eval()


def _runner(model, bulk, max_seqs=3):
    from experts4bit_qlora.engines import paged_attention
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    cfg = model.config
    paged_attention.register(model)
    kv = Fp8PagedKV(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, batch=max_seqs,
                    max_tokens_per_seq=64, device="cpu")

    def reference_attention(layer, q, slots=None, sm_scale=None, window=None, sinks=None, **_):
        outs = []
        for b, slot in enumerate(slots):
            kr, vr = kv.reference_kv(layer, slot)
            outs.append(torch.nn.functional.scaled_dot_product_attention(
                q[b][None, :, None].float(), kr.permute(1, 0, 2)[None].float(), vr.permute(1, 0, 2)[None].float(),
                scale=sm_scale, enable_gqa=True)[0, :, 0].to(q.dtype))
        return torch.stack(outs)

    kv.attention = reference_attention
    return PagedModelRunner(model, kv, device="cpu", bulk_kv=bulk)


PROMPTS = [[3, 17, 42, 9, 11, 5, 88, 23, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 1, 99, 64, 2],
           [120, 7, 7, 31, 2, 9, 14, 60, 3, 3, 8, 1, 30, 31, 32, 33, 34], [15, 100, 4, 4, 81, 6, 12], [44, 45, 46, 47]]
MAX_NEW = [9, 7, 5, 3, 6]


def _serve(bulk, tracer=None):
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler
    model = _tiny_model()
    runner = _runner(model, bulk)
    sched = ContinuousScheduler(runner=runner, max_seqs=3, kv_slots=3, chunk_tokens=8)   # multi-chunk prompts
    if tracer is not None:
        sched.tracer = runner.tracer = tracer
    rids = [sched.add_request(p, m) for p, m in zip(PROMPTS, MAX_NEW)]
    with torch.no_grad():
        while sched.queue or sched.active:
            if tracer is not None:
                tracer.begin()
            plan = sched.step()
            if tracer is not None:
                tracer.discard() if plan.is_empty else tracer.end()
    out = {r.rid: list(r.out) for r in sched.done}
    return [out[r] for r in rids], runner


def test_a_tiny_model_serves_the_same_tokens_with_bulk_bookkeeping():
    ref, r0 = _serve(False)
    got, r1 = _serve(True)
    assert [len(t) for t in got] == MAX_NEW
    assert got == ref, f"bulk changed the decoded tokens:\n  per-layer={ref}\n  bulk={got}"
    # the engagement counters a lane gates on: every request's flush took the configured path
    n = len(PROMPTS)
    assert r0.kv_bookkeeping_stats() == {"bulk": False, "flush_layers": n, "flush_bulk": 0, "ready_layers": 0,
                                         "ready_bulk": 0, "ready_at_flush": 0}
    assert r1.kv_bookkeeping_stats() == {"bulk": True, "flush_layers": 0, "flush_bulk": n, "ready_layers": 0,
                                         "ready_bulk": 0, "ready_at_flush": 0}
    # every slot freed and returned: both pools end empty with full free lists
    for r in (r0, r1):
        assert not r.kv._rows and all(len(f) == r.kv._n_rows for f in r.kv._free)


def test_the_runner_claims_every_reachable_block_at_the_flush_when_graphs_are_on():
    """With decode graphs the bulk flush claims the slot's every block in one write and marks it ready, so its first
    graphed decode claims nothing; the per-layer path claims them at that decode. Same rows either way."""
    out = []
    for bulk in (False, True):
        runner = _runner(_tiny_model(), bulk)
        runner._graphs, runner._graph_ready = {}, set()        # graphs on, no capture needed for the bookkeeping
        runner.bind(0, 1, PROMPTS[0])
        with torch.no_grad():
            runner.run_prefill([(0, 0, len(PROMPTS[0]))])
        assert (1 in runner._graph_ready) is bulk
        runner._ensure_graph_ready(1)
        st = runner.kv_bookkeeping_stats()
        assert (st["ready_at_flush"], st["ready_layers"], st["ready_bulk"]) == ((1, 0, 0) if bulk else (0, 1, 0))
        out.append(_state(runner.kv))
    _assert_same_state(*out)


# ------------------------------------------------------------------------------------------------ the knobs --

def test_bulk_kv_env_values(monkeypatch):
    from experts4bit_qlora import serve_paged
    monkeypatch.setattr(serve_paged, "_capability", lambda device: None)
    monkeypatch.setenv("E4B_PAGED_MODEL", "tiny/moe")
    for val, want in (("", False), ("0", False), ("1", True), (" 1 ", True)):
        monkeypatch.setenv("E4B_PAGED_BULK_KV", val)
        assert serve_paged.PagedServeConfig.from_env().bulk_kv is want
    monkeypatch.delenv("E4B_PAGED_BULK_KV")
    assert serve_paged.PagedServeConfig.from_env().bulk_kv is False
    for bad in ("on", "true", "2", "auto"):
        monkeypatch.setenv("E4B_PAGED_BULK_KV", bad)
        with pytest.raises(ValueError, match="E4B_PAGED_BULK_KV"):
            serve_paged.PagedServeConfig.from_env()


# --------------------------------------------------------------------------------------------- the step trace --

def test_the_step_trace_accounts_for_every_step_and_token(tmp_path):
    from experts4bit_qlora.engines.step_trace import StepTrace
    path = tmp_path / "steps.jsonl"
    tr = StepTrace(str(path), cuda=False, flush_every=4)
    got, _ = _serve(True, tracer=tr)
    tr.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and [r["step"] for r in rows] == list(range(len(rows)))
    assert sum(r.get("prefill_tokens", 0) for r in rows) == sum(map(len, PROMPTS))
    # every output token is a request's first token (its prefill completing) or a decode row
    assert sum(r.get("decode_rows", 0) for r in rows) + len(PROMPTS) == sum(MAX_NEW)
    for r in rows:
        assert r["step_ms"] >= 0 and all(v >= 0 for v in r["seg"].values())
        assert abs(sum(r["seg"].values()) - r["step_ms"]) < 1e-2 * max(1.0, r["step_ms"])
        assert "plan" in r["seg"] and "retire" in r["seg"] and "dispatch" in r["seg"]
        if r.get("prefill_tokens"):
            assert "pf_forward" in r["seg"] and "pf_emit" in r["seg"]
        if r.get("decode_rows"):
            assert "dec_issue" in r["seg"] and "dec_sync" in r["seg"]
    assert any("pf_flush" in r["seg"] for r in rows)


def test_the_server_writes_a_step_trace_and_skips_idle_steps(tmp_path):
    from test_serve_paged import PROMPT, ScriptedRunner, _client
    runner = ScriptedRunner(step_delay=0.002)
    client, engine = _client(runner, step_trace_path=str(tmp_path / "steps.jsonl"))
    with client as c:
        r = c.post("/v1/completions", json={"model": "tiny/moe", "prompt": PROMPT, "max_tokens": 5})
        assert r.status_code == 200
        h = c.get("/health").json()
    assert h["step_trace_path"] == str(tmp_path / "steps.jsonl") and h["engine"]["bulk_kv"] is False
    assert h["kv_bookkeeping"] == {"requested": False}          # a scripted runner keeps no KV
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == 5                         # one prefill step (first token) + four decode steps; no idle rows
    assert rows[0]["admitted"] == 1 and rows[0]["ops"] == 1
    assert all("ops" in r["seg"] and "dispatch" in r["seg"] for r in rows)


def test_the_step_trace_never_waits_on_a_step_the_gpu_has_not_finished(tmp_path):
    """A step with no sync after its last event (a prefill chunk that does not complete its prompt, nothing decoding)
    is held, in step order, until its events have completed. Only close() may wait for it."""
    from experts4bit_qlora.engines.step_trace import StepTrace

    class Ev:
        def __init__(self, t):
            self.t, self.done, self.waited = t, False, False

        def query(self):
            return self.done

        def synchronize(self):
            self.waited = not self.done
            self.done = True

        def elapsed_time(self, other):
            return other.t - self.t

    path = tmp_path / "steps.jsonl"
    tr = StepTrace(str(path), cuda=False, flush_every=1)
    b0, f0 = Ev(0.0), Ev(40.0)                 # step 0: a chunk whose forward is still on the GPU
    tr.begin()
    tr._ev = [("begin", b0), ("pf_forward", f0)]
    tr.end()
    assert not path.exists() and len(tr._pending) == 1 and not f0.waited
    b1, f1 = Ev(50.0), Ev(52.0)                # step 1 finished, but step 0 is still ahead of it in order
    b1.done = f1.done = True
    tr.begin()
    tr._ev = [("begin", b1), ("dec_issue", f1)]
    tr.end()
    assert not path.exists() and len(tr._pending) == 2
    f0.done = True                             # a later sync has passed step 0's events
    tr.begin()
    tr.end()
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2] and rows[0]["gpu"] == {"pf_forward": 40.0}
    assert rows[1]["gpu"] == {"dec_issue": 2.0} and not f0.waited
    b3, f3 = Ev(0.0), Ev(9.0)                  # close() resolves what is left, waiting if it must
    tr.begin()
    tr._ev = [("begin", b3), ("pf_forward", f3)]
    tr.end()
    tr.close()
    assert f3.waited and json.loads(path.read_text().splitlines()[-1])["gpu"] == {"pf_forward": 9.0}


def test_the_bulk_flush_peak_bound_covers_what_it_allocates():
    """append_prompt_peak_bytes bounds the device memory a bulk flush allocates beyond its inputs. On CUDA the bound is
    checked against the allocator's measured peak (bytes, not time); on CPU its arithmetic is pinned."""
    kv = _kv(**GEOMS["mixed"], max_tokens=96)
    layers = list(range(kv.L))
    b1, b40 = kv.append_prompt_peak_bytes(1, layers), kv.append_prompt_peak_bytes(40, layers)
    assert 0 < b1 < b40 and kv.append_prompt_peak_bytes(0, layers) == 0 and kv.append_prompt_peak_bytes(40, []) == 0
    # the (4 x 32) layers 0 and 2 form the largest group (128 values a token, two layers): 7x its bf16 input per side;
    # every layer's FP8 K and V payload plus their scales (key groups as this build resolved them) are held
    T = 40
    held = sum(T * (2 * kv.Hs[i] * kv.Ds[i] + kv.Hs[i] * kv.kgs[i] * 4 + kv.Hs[i] * 4) for i in layers)
    assert kv.append_prompt_peak_bytes(T, layers) == 7 * 2 * T * 128 * 2 + held
    if DEV != "cuda":
        return
    big = Fp8PagedKV(48, 4, 128, batch=2, max_tokens_per_seq=2048, device=DEV)
    lay = list(range(48))
    ks, vs = _prompt(big, lay, 2048, 3)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    big.append_prompt(0, lay, ks, vs)
    torch.cuda.synchronize()
    used = torch.cuda.max_memory_allocated() - base
    assert used <= big.append_prompt_peak_bytes(2048, lay), (used, big.append_prompt_peak_bytes(2048, lay))
