# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P117's measurement (``bench/p117/p117_box.py``) on CPU, before any GPU is rented.

Every P117 arm decodes through the bucketed path, which needs the fused KV append (a GPU kernel). So this test stands
the RUNNER in: ``_StandIn`` decodes each piece with a full-sequence forward of a tiny Qwen3-MoE (no KV cache), splits
an active set into pieces of the largest bucket and pads each to its bucket exactly as ``PagedModelRunner`` does, keeps
its per-bucket statistics, and under ``capture`` emits tokens without calling the patched forward (a replay never calls
it). What it checks is the box's own bookkeeping:
- every arm scores its windows (W64pad the first 48), and every arm maps its rows back to the right windows: with no
  arithmetic difference left, every arm matches R window by window, across pieces, padding, halves and reversal;
- the box reads one forward per piece, and the statistics count pieces, padding and replays as registered;
- G64's emitted tokens equal W64's at every position (the function gate's comparison), and differ when they should.
The decode attention, the scale mutant and the real bucket path run on the proving rental.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p117", "p108", "p97")]


def _load():
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("p117_box", LANES[0] / "p117_box.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
                         max_position_embeddings=128, decoder_sparse_step=1, mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


class _StandIn:
    """``PagedModelRunner``'s surface the box drives, decoding by full-sequence forwards."""

    def __init__(self, model, kv, device="cpu", **_):
        self.model, self.tokens, self.graph_stats, self.prompt_len = model, {}, {}, {}

    def enable_decode_graphs(self, buckets, *, capture=True, warmup=2, verbose=True):
        self.buckets, self.capture = tuple(sorted(int(b) for b in buckets)), capture
        self.graph_stats = {b: {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in self.buckets}
        return {b: ("graph" if capture else "eager: capture=False") for b in self.buckets}

    def bind(self, rid, slot, prompt):
        self.tokens[rid], self.prompt_len[rid] = list(prompt), len(prompt)

    def run_prefill(self, chunks):
        out = {}
        for rid, s, n in chunks:
            o = self.model(input_ids=torch.tensor([self.tokens[rid][:s + n]]))
            if s + n >= self.prompt_len[rid]:
                out[rid] = int(o.logits[0, -1].argmax())
                self.tokens[rid].append(out[rid])
        return out

    def run_decode(self, rids):
        from experts4bit_qlora.engines.paged_runner import bucket_for, chunk_rows
        got = {}
        for piece in chunk_rows(rids, self.buckets[-1]):
            b = bucket_for(len(piece), self.buckets)
            ids = torch.tensor([self.tokens[r] for r in piece] + [self.tokens[piece[0]]] * (b - len(piece)))
            st = self.graph_stats[b]
            if self.capture:                       # a replay: no Python forward runs
                logits = type(self.model).forward(self.model, input_ids=ids).logits
                st["replays"] += 1
            else:
                logits = self.model(input_ids=ids).logits
                st["eager_steps"] += 1
            st["rows"] += len(piece)
            st["pad_rows"] += b - len(piece)
            for j, r in enumerate(piece):
                got[r] = int(logits[j, -1].argmax())
                self.tokens[r].append(got[r])
        return got


P, C, N = 12, 5, 40
CPU_ARMS = ("R", "rep", "half", "chunk", "rev", "W32", "W64", "W64pad", "G64")


@pytest.fixture(scope="module")
def run():
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    box = _load()
    from experts4bit_qlora.engines import paged_runner
    real = paged_runner.PagedModelRunner
    paged_runner.PagedModelRunner = _StandIn
    try:
        g = torch.Generator().manual_seed(1)
        windows = [torch.randint(0, 256, (P + C,), generator=g).tolist() for _ in range(N)]
        rec = box.measure(_tiny(), windows, prompt=P, cont=C, chunk=P, floor_chunk=4, device="cpu", stand_in=True,
                          arms=CPU_ARMS)
    finally:
        paged_runner.PagedModelRunner = real
    return box, windows, rec


def test_every_arm_scores_its_windows(run):
    _box, _w, rec = run
    per = rec["per_window"]
    assert [len(per[a]) for a in CPU_ARMS if a != "G64"] == [N] * 7 + [min(N, 48)]
    assert rec["rep_identical"] is True and rec["layers"] == 2 and rec["group"] == N


def test_every_arm_maps_its_rows_back_to_the_right_windows(run):
    _box, _w, rec = run
    per = rec["per_window"]
    r = {x["window"]: x["nll"] for x in per["R"]}
    for arm in ("half", "rev", "W32", "W64", "W64pad"):
        for x in per[arm]:
            assert abs(x["nll"] - r[x["window"]]) < 1e-4, (arm, x)
            assert x["argmax_agree"] == 1.0, (arm, x)
    assert all(abs(x["nll"] - r[x["window"]]) < 1e-3 for x in per["chunk"])


def test_pieces_padding_and_replays_are_counted_as_registered(run):
    _box, _w, rec = run
    e = {a: rec["engagement"][a][0] for a in CPU_ARMS}
    steps = C - 1
    # 40 rows: buckets <= 16 -> 16 + 16 + 8; <= 8 -> five 8s; <= 32 -> 32 + 8; <= 64 -> one 40 padded to 64
    assert e["R"]["graph_stats"]["16"]["eager_steps"] == 2 * steps and e["R"]["graph_stats"]["8"]["eager_steps"] == steps
    assert e["half"]["graph_stats"]["8"]["eager_steps"] == 5 * steps
    assert e["W32"]["graph_stats"]["32"]["eager_steps"] == steps and e["W32"]["graph_stats"]["8"]["eager_steps"] == steps
    assert e["W64"]["graph_stats"]["64"] == {"replays": 0, "eager_steps": steps, "rows": N * steps, "pad_rows": (64 - N) * steps}
    assert e["G64"]["graph_stats"]["64"]["replays"] == steps and e["G64"]["graph_stats"]["64"]["eager_steps"] == 0
    assert all(e[a]["grouping_flags_in_pass"]["device_grouping"] for a in CPU_ARMS)
    assert e["rev"]["reverse"] and e["W64"]["buckets"] == [1, 2, 4, 8, 16, 32, 64]
    from experts4bit_qlora.engines import hot_residency as hr
    assert hr.DEVICE_GROUPING[0] is False                     # restored after every pass


def test_the_function_gate_compares_every_emitted_token(run):
    _box, _w, rec = run
    assert rec["function"] == {"positions": N * (C - 1), "differ": 0}


def test_the_box_refuses_a_piece_whose_forward_it_did_not_see(run):
    box, windows, _rec = run
    from experts4bit_qlora.engines import paged_runner

    class Skips(_StandIn):
        def run_decode(self, rids):
            got = super().run_decode(rids)
            self.model.forward(input_ids=torch.tensor([windows[0][:P]]))     # one forward more than there are pieces
            return got
    real = paged_runner.PagedModelRunner
    paged_runner.PagedModelRunner = Skips
    try:
        with pytest.raises(RuntimeError, match="forwards for"):
            box.paged_pass(_tiny(), windows[:4], P, C, P, "cpu", buckets=(1, 2, 4), stand_in=True)
    finally:
        paged_runner.PagedModelRunner = real
