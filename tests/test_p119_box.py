# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P119's census box (``bench/p119/p119_box.py``) on CPU, before any GPU is rented.

The brackets need the fused KV append and CUDA kernels, so this stands the RUNNER and the pool in: ``_StandIn``
decodes each piece with a full-sequence forward of a tiny Qwen3-MoE, splits an active set into pieces of the largest
bucket and pads each to its bucket as ``PagedModelRunner`` does, and keeps its per-bucket statistics. What it checks:
- ``kernel_table``: device kernels only, per step, classes by the frozen rules, device-to-device copies counted;
- every decode bracket profiles exactly ``steps`` steps on its registered split (d64 one 64-row piece a step, d64x4
  four 16-row pieces), after ``warm`` unprofiled ones;
- the prefill brackets pass ``last_logits`` and ``bulk_kv`` through, and a refusal is recorded, not raised;
- the head bracket reads the LM head's shape and dtype.
Kernel times themselves are read on the rental.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p119", "p117", "p108", "p97")]


def _load():
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("p119_box", LANES[0] / "p119_box.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Avg:
    def __init__(self, key, count, us):
        self.key, self.count, self.self_device_time_total = key, count, us


def test_kernel_table_counts_device_kernels_per_step_by_class():
    box = _load()
    avgs = [_Avg("aten::mm", 8, 0.0),                                      # a host op: no device time of its own
            _Avg("_gemm_int4_b32_grouped_smallm_kernel", 96, 4000.0),
            _Avg("nvjet_tst_128x64_64x4_1x2_h_bz_TNT", 16, 1200.0),
            _Avg("Memcpy DtoD (Device -> Device)", 400, 800.0),
            _Avg("_fp8_paged_decode_kernel", 48, 600.0),
            _Avg("void at::native::vectorized_elementwise_kernel<4>", 32, 400.0)]
    t = box.kernel_table(avgs, per=4)
    assert [k[0] for k in t["kernels"]][0] == "_gemm_int4_b32_grouped_smallm_kernel"
    assert t["kernels"][0][1:] == [24.0, 1.0]                              # per step: 96 / 4 calls, 4 ms / 4
    assert t["device_ms"] == 1.75 and t["dtod"] == 100.0
    assert set(t["classes"]) == {"expert_int4", "dense_gemm", "memcpy_dtod", "attn", "elementwise"}
    assert box.kclass("SomethingNew") == "other" and box.kclass("_tile_table_r1") == "moe_route"


def _tiny():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
                         max_position_embeddings=128, decoder_sparse_step=1, mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


class _StandIn:
    """``PagedModelRunner``'s surface the brackets drive, decoding by full-sequence forwards."""
    made = []

    def __init__(self, model, kv, device="cpu", bulk_kv=False, last_logits=False, **_):
        if last_logits and getattr(model, "_refuse_last_logits", False):
            raise ValueError("last-logits prefill requires an explicit logits_to_keep or num_logits_to_keep keyword")
        self.model, self.tokens, self.graph_stats, self.plen = model, {}, {}, {}
        self.bulk_kv, self.last_logits, self.prefills = bulk_kv, last_logits, 0
        _StandIn.made.append(self)

    def enable_decode_graphs(self, buckets, *, capture=True, warmup=2, verbose=True):
        self.buckets = tuple(sorted(int(b) for b in buckets))
        self.graph_stats = {b: {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in self.buckets}
        return {b: ("graph" if capture else "eager: capture=False") for b in self.buckets}

    def bind(self, rid, slot, prompt):
        self.tokens[rid], self.plen[rid] = list(prompt), len(prompt)

    def run_prefill(self, chunks):
        for rid, s, n in chunks:
            self.prefills += 1
            o = self.model(input_ids=torch.tensor([self.tokens[rid][:s + n]]))
            self.tokens[rid].append(int(o.logits[0, -1].argmax()))

    def run_decode(self, rids):
        from experts4bit_qlora.engines.paged_runner import bucket_for, chunk_rows
        for piece in chunk_rows(rids, self.buckets[-1]):
            b = bucket_for(len(piece), self.buckets)
            ids = torch.tensor([self.tokens[r] for r in piece] + [self.tokens[piece[0]]] * (b - len(piece)))
            logits = self.model(input_ids=ids).logits
            st = self.graph_stats[b]
            st["eager_steps"] += 1
            st["rows"] += len(piece)
            st["pad_rows"] += b - len(piece)
            for j, r in enumerate(piece):
                self.tokens[r].append(int(logits[j, -1].argmax()))

    def last_logits_stats(self):
        return {"status": "on" if self.last_logits else "off"}


P, ROWS, WARM, STEPS, REPS = 6, 20, 2, 3, 2
DECODE = (("d16", (1, 2, 4, 8, 16), 16), ("d20", (1, 2, 4, 8, 16, 32), 20), ("d20x2", (1, 2, 4, 8, 16), 20))


@pytest.fixture(scope="module")
def run():
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    box = _load()
    from experts4bit_qlora.engines import paged_runner
    real_runner, real_pool = paged_runner.PagedModelRunner, box._pool
    paged_runner.PagedModelRunner = _StandIn
    box._pool = lambda model, rows, tokens, scratch, device: (object(), 2)   # kv_layers() returns an int (amendment 1)
    _StandIn.made.clear()
    try:
        g = torch.Generator().manual_seed(1)
        ws = [torch.randint(0, 256, (P,), generator=g).tolist() for _ in range(ROWS)]
        model = _tiny()
        rec = box.measure(model, ws, prompt=P, device="cpu", warm=WARM, steps=STEPS, reps=REPS, decode=DECODE,
                          bulk_kv=True)
        model._refuse_last_logits = True
        refused = box.prefill_bracket(model, ws, P, "cpu", last_logits=True, reps=REPS)
    finally:
        paged_runner.PagedModelRunner, box._pool = real_runner, real_pool
    return box, rec, refused


def test_every_decode_bracket_profiles_its_registered_split(run):
    _box, rec, _r = run
    d = rec["decode"]
    assert d["d16"]["profiled"]["16"] == {"replays": 0, "eager_steps": STEPS, "rows": 16 * STEPS, "pad_rows": 0}
    assert d["d20"]["profiled"]["32"] == {"replays": 0, "eager_steps": STEPS, "rows": 20 * STEPS, "pad_rows": 12 * STEPS}
    x2 = d["d20x2"]["profiled"]
    assert x2["16"]["eager_steps"] == STEPS and x2["4"]["eager_steps"] == STEPS     # 20 rows: a 16-row and a 4-row piece
    assert all(v["graph_status"] == {str(b): "eager: capture=False" for b in v["buckets"]} for v in d.values())
    assert all(v["steps"] == STEPS and v["warm"] == WARM and v["bulk_kv"] is True and v["layers"] == 2 for v in d.values())


def test_the_prefill_brackets_pass_their_mode_through_and_record_a_refusal(run):
    _box, rec, refused = run
    off, on = rec["prefill"]["p512_off"], rec["prefill"]["p512_on"]
    assert off["last_logits_stats"] == {"status": "off"} and on["last_logits_stats"] == {"status": "on"}
    assert off["reps"] == REPS and off["bulk_kv"] is True and off["tokens"] == P
    made = [m for m in _StandIn.made if not hasattr(m, "buckets")]
    assert [m.prefills for m in made[:2]] == [REPS + 1, REPS + 1]                 # one warm prefill, then the profiled ones
    assert refused == {"last_logits": True, "refused": "last-logits prefill requires an explicit logits_to_keep or "
                                                       "num_logits_to_keep keyword"}


def test_the_head_bracket_reads_the_lm_head(run):
    _box, rec, _r = run
    h = rec["head"]
    assert h["module"] == "Linear" and h["weight_shape"] == [256, 64] and h["weight_dtype"] == "torch.float32"
    assert set(h["rows"]) == {"1", "16", "64", "512"}
