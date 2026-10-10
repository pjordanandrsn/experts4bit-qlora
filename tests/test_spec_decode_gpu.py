# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Speculative decode on the card (lane SD2, ``bench/sd2/PREREG-sd2.md``): the verify bucket on alias slots, the
rollback, the post-verify graphs and the batching transition, on a tiny Qwen3 through the real FP8 pool and graphs.

These need sm_89+ (the FP8 paged KV) and run on the proof box (``sd2-prove-N``), not in CI. What they hold:
- a captured verify replays exactly as its padded eager step (the runner's standing oracle), and the post-verify step's
  graphs exactly as its eager form;
- with an oracle drafter (the plain stream's own continuation, so accepts run long), speculation emits the plain
  stream wherever the verify's arithmetic does not move an argmax. Agreement is high and the first divergence is
  reported. The tiny model's ``lm_head`` is scaled so argmax margins are wide;
- after every speculative step the slot's device lengths equal the host mirror, on every layer;
- a request batched mid-flight drops its draft, keeps decoding at T == 1 with consistent lengths, and the census counts
  it (the maintainer's requirement);
- the EAGLE-3 drafter, a random head shaped to the tiny model, runs the whole cycle under graphs.
"""
import pytest
import torch

from experts4bit_qlora.engines.paged_runner import PagedModelRunner

needs_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the fp8 paged KV needs native e4m3 (sm_89+)")

PROMPT = [3, 17, 42, 9, 11, 5, 88, 23, 7, 61, 2, 19]
N_NEW = 40
HID, NH, NKV, HD, V, LAYERS = 256, 4, 2, 64, 256, 8


def _model():
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's fp8 paged attention")
    fp8_kv = pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm's fp8 KV")
    if not hasattr(fp8_kv, "fp8_kv_append_bt1"):
        pytest.skip("this grouped-nf4-gemm has no fused batch KV append")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    from experts4bit_qlora.engines import paged_attention

    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=HID, intermediate_size=512, num_hidden_layers=LAYERS, num_attention_heads=NH,
                      num_key_value_heads=NKV, head_dim=HD, vocab_size=V, max_position_embeddings=512)
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(8.0)                     # wide argmax margins: bf16 at M = k + 1 rarely flips one
    model = model.to("cuda", torch.bfloat16)
    paged_attention.register(model)
    return model


def _head(seed=1):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    inter = 512
    return {"d2t": torch.zeros(V, dtype=torch.long), "embed_tokens.weight": r(V, HID) / 8, "fc.weight": r(HID, 3 * HID) / 16,
            "layers.0.hidden_norm.weight": torch.ones(HID), "layers.0.input_layernorm.weight": torch.ones(HID),
            "layers.0.mlp.down_proj.weight": r(HID, inter) / 16, "layers.0.mlp.gate_proj.weight": r(inter, HID) / 16,
            "layers.0.mlp.up_proj.weight": r(inter, HID) / 16, "layers.0.post_attention_layernorm.weight": torch.ones(HID),
            "layers.0.self_attn.k_proj.weight": r(NKV * HD, 2 * HID) / 16,
            "layers.0.self_attn.o_proj.weight": r(HID, NH * HD) / 16,
            "layers.0.self_attn.q_proj.weight": r(NH * HD, 2 * HID) / 16,
            "layers.0.self_attn.v_proj.weight": r(NKV * HD, 2 * HID) / 16,
            "lm_head.weight": r(V, HID), "norm.weight": torch.ones(HID)}


class OracleDrafter:
    """Drafts the plain run's own continuation: accepts run long wherever the streams agree."""

    def __init__(self, k, stream):
        self.k, self.stream = k, stream

    def _d(self, pos):
        return torch.tensor([self.stream[pos + 1 + i] if pos + 1 + i < len(self.stream) else 0
                             for i in range(self.k)], device="cuda")

    def prefill(self, tok, aux):
        return self._d(int(tok.shape[0]))

    def extend_and_draft(self, g, aux_v, base, a):
        # inside a graph this would bake its python ints; the oracle runs the post step eagerly (no capture)
        return self._d(int(base) + int(a) + 1)


def _build(k, *, spec=True, drafter="eagle3", stream=None, capture=True, buckets=(1, 2, 4)):
    from experts4bit_qlora.engines.eagle3_draft import AuxStates, Eagle3Drafter
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler
    from experts4bit_qlora.engines.spec_decode import SpecDecoder

    model = _model()
    kv = Fp8PagedKV(LAYERS, NKV, HD, batch=4, max_tokens_per_seq=128, scratch_slots=max(buckets),
                    alias_slots=k if spec else 0)
    runner = PagedModelRunner(model, kv)
    dec = None
    if spec:
        cap = kv.blocks_per_seq * kv.bt
        if drafter == "eagle3":
            d = Eagle3Drafter(_head(), k=k, max_positions=cap + 2 * k + 1, n_heads=NH, n_kv=NKV, head_dim=HD,
                              device="cuda")
        else:
            d = OracleDrafter(k, stream)
        aux = AuxStates(model, max_rows=max(buckets), max_prompt=cap)
        dec = SpecDecoder(drafter=d, aux=aux, kv=kv, k=k, capacity=cap, device="cuda")
        assert runner.enable_speculation(dec) == k
        assert aux.install() == 3
        if drafter != "eagle3":
            dec.capture_post = lambda n, warmup=2: None   # the oracle's python ints cannot be captured
    status = runner.enable_decode_graphs(buckets, capture=capture, verbose=False)
    sched = ContinuousScheduler(runner=runner, max_seqs=4, kv_slots=4, chunk_tokens=64)
    return runner, sched, dec, status


def _decode_alone(k, **kw):
    runner, sched, dec, status = _build(k, **kw)
    rid = sched.add_request(PROMPT, N_NEW)
    sched.run_until_idle(max_steps=500)
    (req,) = sched.done
    assert req.rid == rid
    return list(req.out), runner, dec, status


def _agreement(a, b):
    same = sum(x == y for x, y in zip(a, b))
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return same / max(len(a), 1), first


@needs_fp8
@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_verify_replays_as_its_padded_eager_step(k):
    graph, runner, dec, status = _decode_alone(k)
    assert all(status[b] == "graph" for b in runner._verify_buckets), status
    assert all(v == "graph" for v in runner.spec_graph_status.values()), runner.spec_graph_status
    eager, runner_e, dec_e, _ = _decode_alone(k, capture=False)
    assert graph == eager, f"replay != padded eager:\n  graph={graph}\n  eager={eager}"
    assert dec.census()["steps"] > 0 and dec.census()["post_replays"] > 0
    assert dec_e.census()["post_eager"] > 0


@needs_fp8
@pytest.mark.parametrize("k", [1, 2, 3])
def test_speculation_tracks_the_plain_stream(k):
    plain, _, _, _ = _decode_alone(k, spec=False)
    spec, runner, dec, _ = _decode_alone(k, drafter="oracle", stream=PROMPT + plain)
    agree, first = _agreement(spec, plain)
    c = dec.census()
    print(f"k={k} agreement {agree:.3f} first divergence {first} census {c}")
    assert len(spec) == N_NEW and agree >= 0.9
    assert c["accepted"] > 0                               # the oracle's drafts were taken: rollback at a > 0 ran


@needs_fp8
@pytest.mark.parametrize("k", [1, 3])
def test_the_lengths_after_every_speculative_step_equal_the_host_mirror(k, monkeypatch):
    from experts4bit_qlora.engines import spec_decode as sd

    checks = {"n": 0}
    real = sd.SpecDecoder.step

    def checked(self, rid, budget):
        got = real(self, rid, budget)
        if got is not None:
            torch.cuda.synchronize()
            slot, base = self.state[rid]["slot"], self.state[rid]["base"]
            for layer in range(self.kv.L):
                assert int(self.kv.seq_lens[layer, slot]) == self.kv._seen[layer][slot] == base
            checks["n"] += 1
        return got

    monkeypatch.setattr(sd.SpecDecoder, "step", checked)
    _decode_alone(k)
    assert checks["n"] > 0


@needs_fp8
def test_a_request_batched_mid_flight_drops_its_draft_and_keeps_decoding():
    """The maintainer's requirement, on the card: request A speculates alone; B is admitted mid-flight. A drops its
    draft, decodes at T == 1 through the plain buckets with lengths that match the host mirror, and both finish."""
    k = 2
    runner, sched, dec, _ = _build(k)
    a = sched.add_request(PROMPT, N_NEW)
    for _ in range(6):
        sched.step()
    assert dec.census()["steps"] > 0 and dec.eligible(a)
    b = sched.add_request([5, 1, 99, 64, 2], 12)
    sched.run_until_idle(max_steps=500)
    out = {r.rid: list(r.out) for r in sched.done}
    assert len(out[a]) == N_NEW and len(out[b]) == 12
    c = dec.census()
    assert c["dropped_batched"] == 1 and not dec.state
    # the plain buckets carried the batched steps; bucket 3 (verify only) never took a plain step
    assert runner.graph_stats[2]["rows"] > 0
    assert runner.graph_stats[3]["rows"] % 3 == 0          # only whole 3-row verifies, never a plain step
