# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Uninitialized memory must never reach the loss or a gradient from a padded or unwritten region.

The failure class: a training kernel or custom backward allocates with ``torch.empty`` (or ``empty_like`` / ``new_empty``),
writes only the real rows, and lets the padded rows reach a reduction or a gradient. Recycled allocator bytes then make the
gradient vary from step to step, and spike when they happen to be large or non-finite. Nothing raises.

The fixture: inside one training step, every ``torch.empty`` / ``empty_like`` / ``empty_strided`` / ``Tensor.new_empty`` /
``new_empty_strided`` is filled with a poison (floats NaN, then a finite sentinel; integers a bit pattern). A buffer whose
unwritten part is read then shows as a non-finite or changed loss or adapter gradient. The step runs on variable-length padded
batches (attention mask 0, labels -100 at the pads) with gradient checkpointing on. Per configuration:

- two clean steps, whose difference is the path's own run-to-run spread (zero where the path is deterministic);
- a NaN step and a sentinel step, then a clean step (a poisoned buffer cached and reused would show there);
- a fresh build whose FIRST step is poisoned (buffers cached at first use), then a clean step.

Each poisoned or later step must be finite and within twice the spread of the first clean step (equal where the spread is
zero). Each configuration also asserts that the path it names ran in the poisoned step, from the engines' own counters, so a
pass cannot come from a path that never engaged.

Controls: a planted bug of exactly this class must be caught. On CPU, the batched path's padded LoRA block is allocated with
``empty_like`` and only its real rows are written; on CUDA, the fused RMSNorm is launched over all but the last rows.

CPU (CI): the reference path, the chunked LM loss and the batched path on a small Qwen3-MoE. CUDA: ``enable_fast_train``'s
routes and options on a small Qwen3-MoE, and Qwen3.5-MoE's linear-attention layers.
"""
from __future__ import annotations

import contextlib
import gc
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("bitsandbytes")

from quant_guard import load_or_skip  # noqa: E402

CUDA = torch.cuda.is_available()

POISON_INT = {torch.uint8: 0xA5, torch.int8: -91, torch.int16: -23131, torch.int32: -1515870811,
              torch.int64: -6510615555426900571}
_TARGETS = [(torch, "empty"), (torch, "empty_like"), (torch, "empty_strided"), (torch.Tensor, "new_empty"),
            (torch.Tensor, "new_empty_strided")]


@contextlib.contextmanager
def poisoned(value):
    """Fill every empty-family allocation made inside the block: floats with ``value``, integers with a bit pattern."""
    saved = [(o, n, getattr(o, n)) for o, n in _TARGETS]
    stats = {"filled": 0}

    def wrap(f):
        def g(*a, **k):
            t = f(*a, **k)
            if isinstance(t, torch.Tensor) and t.device.type != "meta" and t.numel():
                if t.is_floating_point():
                    t.fill_(value)
                elif t.dtype in POISON_INT:
                    t.fill_(POISON_INT[t.dtype])
                else:
                    return t
                stats["filled"] += 1
            return t
        return g

    for o, n, f in saved:
        setattr(o, n, wrap(f))
    try:
        yield stats
    finally:
        for o, n, f in saved:
            setattr(o, n, f)


def test_the_poison_fills_every_empty_family_allocation_and_restores_them():
    originals = [getattr(o, n) for o, n in _TARGETS]
    with poisoned(float("nan")) as st:
        a = torch.empty(3)
        b = torch.empty_like(torch.ones(2))
        c = torch.empty_strided((2, 2), (2, 1))
        d = torch.ones(1).new_empty(4)
        e = torch.ones(1).new_empty_strided((2,), (1,))
        i = torch.empty(2, dtype=torch.int64)
    assert all(bool(torch.isnan(t).all()) for t in (a, b, c, d, e)) and bool((i == POISON_INT[torch.int64]).all())
    assert st["filled"] == 6
    assert [getattr(o, n) for o, n in _TARGETS] == originals


# --- the step ------------------------------------------------------------------------------------------------------------

def _batch(vocab, device, L, left=False, seed=7):
    g = torch.Generator().manual_seed(seed)
    lens = [L, L - 23, L - 41]
    ids = torch.randint(4, vocab, (3, L), generator=g)
    mask = torch.zeros(3, L, dtype=torch.long)
    for i, n in enumerate(lens):
        if left:
            mask[i, L - n:] = 1
        else:
            mask[i, :n] = 1
    ids = ids.masked_fill(mask == 0, 0)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "labels": ids.masked_fill(mask == 0, -100).to(device)}


def _step(model, b):
    model.zero_grad(set_to_none=True)
    out = model(**b)
    out.loss.backward()
    if b["input_ids"].is_cuda:
        torch.cuda.synchronize()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    assert grads, "no adapter gradient: nothing was trained"
    return out.loss.detach().float().item(), grads


def _finite(x):
    loss, grads = x
    return loss == loss and abs(loss) != float("inf") and all(bool(torch.isfinite(g).all()) for g in grads.values())


def _maxrel(ga, gb):
    worst = 0.0
    for n in gb:
        d = (ga[n] - gb[n]).abs().max().item() / max(gb[n].abs().max().item(), 1e-30)
        if d != d:
            return float("nan")
        worst = max(worst, d)
    return worst


def _counters():
    """Every numeric leaf of every module-level ``*_STATS`` dict in e4b's engines and the kernel package, by dotted name."""
    import pkgutil

    import experts4bit_qlora.engines as eng
    mods = [sys.modules[f"experts4bit_qlora.engines.{m.name}"] for m in pkgutil.iter_modules(eng.__path__)
            if f"experts4bit_qlora.engines.{m.name}" in sys.modules]
    mods += [sys.modules[n] for n in ("nf4_qlora", "nf4_grouped", "nf4_route") if n in sys.modules]
    out = {}

    def leaves(d, prefix):
        for k, v in d.items():
            if isinstance(v, dict):
                leaves(v, f"{prefix}.{k}")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{prefix}.{k}"] = v

    for m in mods:
        for k, v in vars(m).items():
            if k.endswith("_STATS") and isinstance(v, dict):
                leaves(v, f"{m.__name__.rsplit('.', 1)[-1]}.{k}")
    return out


def _batched_calls(model):
    return sum(getattr(m, "_e4b_batched_stats", {}).get("batched", 0) for m in model.modules())


def _free():
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()


def _audit(build, device, L, left=False):
    """The configuration's verdict and evidence: run as the module docstring says."""
    b = _batch(2048 if device == "cuda" else 512, device, L, left=left)
    m = build()
    A, A2 = _step(m, b), _step(m, b)
    c0, n0 = _counters(), _batched_calls(m)
    with poisoned(float("nan")) as sp:
        P = _step(m, b)
    c1 = _counters()
    engaged = {k: v - c0.get(k, 0) for k, v in c1.items() if v != c0.get(k, 0)}
    engaged["batched_calls"] = _batched_calls(m) - n0
    with poisoned(1.0e4):
        Q = _step(m, b)
    A3 = _step(m, b)
    del m
    _free()
    m = build()
    with poisoned(float("nan")):
        P0 = _step(m, b)
    A4 = _step(m, b)
    del m
    _free()
    spread = _maxrel(A2[1], A[1])
    runs = {"P": P, "Q": Q, "A3": A3, "P0": P0, "A4": A4}
    rel = {k: _maxrel(x[1], A[1]) for k, x in runs.items()}
    ok = {k: _finite(x) and rel[k] == rel[k] and (rel[k] <= 2 * spread if spread > 0 else (rel[k] == 0 and x[0] == A[0]))
          for k, x in runs.items()}
    return {"clean": all(ok.values()), "ok": ok, "rel": rel, "spread": spread, "finite_P": _finite(P), "filled": sp["filled"],
            "engaged": engaged}


def _assert_clean(r, *expect):
    assert r["filled"] > 0, "nothing was poisoned"
    for key in expect:
        assert any(key in k and v > 0 for k, v in r["engaged"].items()), f"{key} did not engage: {r['engaged']}"
    assert r["clean"], r


# --- models --------------------------------------------------------------------------------------------------------------

def _qwen3_moe(path, **dims):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(decoder_sparse_step=1, mlp_only_layers=[], max_position_embeddings=512, norm_topk_prob=True,
                         tie_word_embeddings=False, num_hidden_layers=2, **dims)
    torch.manual_seed(0)
    Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def cpu_model(tmp_path_factory):
    return _qwen3_moe(tmp_path_factory.mktemp("q3cpu"), vocab_size=512, hidden_size=256, intermediate_size=512,
                      moe_intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=64, num_experts=8,
                      num_experts_per_tok=2)


@pytest.fixture(scope="module")
def cuda_model(tmp_path_factory):
    return _qwen3_moe(tmp_path_factory.mktemp("q3cuda"), vocab_size=2048, hidden_size=512, intermediate_size=1024,
                      moe_intermediate_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=128, num_experts=16,
                      num_experts_per_tok=4)


@pytest.fixture(scope="module")
def cuda_linear_attention_model(tmp_path_factory):
    C = getattr(transformers, "Qwen3_5MoeTextConfig", None)
    M = getattr(transformers, "Qwen3_5MoeForCausalLM", None)
    if C is None or M is None:
        pytest.skip("this transformers has no Qwen3.5-MoE")
    path = tmp_path_factory.mktemp("q35cuda")
    cfg = C(vocab_size=2048, hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
            moe_intermediate_size=256, shared_expert_intermediate_size=256, num_experts=16, num_experts_per_tok=4,
            layer_types=["linear_attention", "full_attention"], max_position_embeddings=512, tie_word_embeddings=False)
    torch.manual_seed(0)
    M(cfg).to(torch.bfloat16).save_pretrained(path)
    return str(path)


def _builder(path, device, monkeypatch, *, attn4=True, enabler="reference", adapter_dtype=torch.float32, env=None, patch=None,
             absmax_dq=None):
    for k, v in {"E4B_CHUNKED_LM_LOSS": "1", **(env or {})}.items():
        monkeypatch.setenv(k, v)

    def build():
        from experts4bit_qlora.lora import add_attention_lora, quantize_attention_projections_4bit
        torch.manual_seed(0)
        model, _ = load_or_skip(path, device, torch.bfloat16, r=8, alpha=16, what="NF4 (poisoned-empty audit)")
        model.to(device)
        if attn4:
            quantize_attention_projections_4bit(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        add_attention_lora(model, 8, 16, adapter_dtype)
        torch.manual_seed(1)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data.normal_(0, 0.02)
        if patch:
            patch()
        n = 1
        if enabler == "fast":
            from experts4bit_qlora import enable_fast_train
            n = enable_fast_train(model, absmax_dq=absmax_dq)
        elif enabler == "fast_dgrad":
            from experts4bit_qlora import enable_fast_train
            n = enable_fast_train(model, dgrad=True, absmax_dq=absmax_dq)
        elif enabler == "batched":
            from experts4bit_qlora import enable_batched_train
            n = enable_batched_train(model)
        elif enabler == "chunked":
            from experts4bit_qlora.engines.chunked_lm_loss import enable_chunked_lm_loss
            n = enable_chunked_lm_loss(model, chunk=16)
        assert n > 0, f"{enabler} patched nothing"
        model.train()
        return model
    return build


# --- CPU (CI) --------------------------------------------------------------------------------------------------------------

def test_cpu_reference_path_is_clean(cpu_model, monkeypatch):
    _assert_clean(_audit(_builder(cpu_model, "cpu", monkeypatch, attn4=False), "cpu", 48))


def test_cpu_chunked_lm_loss_is_clean(cpu_model, monkeypatch):
    _assert_clean(_audit(_builder(cpu_model, "cpu", monkeypatch, attn4=False, enabler="chunked"), "cpu", 48),
                  "CHUNKED_LM_LOSS_STATS.chunked_calls")


@pytest.mark.parametrize("left", [False, True], ids=["right-padded", "left-padded"])
def test_cpu_batched_path_is_clean(cpu_model, monkeypatch, left):
    _assert_clean(_audit(_builder(cpu_model, "cpu", monkeypatch, attn4=False, enabler="batched"), "cpu", 48, left=left),
                  "batched_calls")


def test_cpu_control_a_planted_padded_block_bug_is_caught(cpu_model, monkeypatch):
    """The batched path's padded LoRA block allocated with ``empty_like`` and only its real rows written -- this bug class,
    planted. The audit must call it, with a non-finite gradient under the NaN poison."""
    import experts4bit_qlora.engines.batched as batched
    real = batched._lora_delta_padded

    def planted(x_pad, *a, **k):
        rows = x_pad.abs().sum(-1) != 0
        bad = torch.empty_like(x_pad)
        bad[rows] = x_pad[rows]
        return real(bad, *a, **k)

    r = _audit(_builder(cpu_model, "cpu", monkeypatch, attn4=False, enabler="batched",
                        patch=lambda: monkeypatch.setattr(batched, "_lora_delta_padded", planted)), "cpu", 48)
    assert r["engaged"]["batched_calls"] > 0, r["engaged"]
    assert not r["clean"] and not r["finite_P"], r


# --- CUDA ------------------------------------------------------------------------------------------------------------------

needs_kernels = pytest.mark.skipif(not CUDA, reason="the fused training kernels need CUDA")


def _kernels():
    pytest.importorskip("nf4_qlora", reason="enable_fast_train needs grouped-nf4-gemm's training kernels")


def _tiny_combine_chunks(monkeypatch):
    import experts4bit_qlora.engines.fast as fast

    def patch():
        monkeypatch.setattr(fast, "COMBINE_CHUNK_MIN_BYTES", 0)
        monkeypatch.setattr(fast, "COMBINE_CHUNK_BYTES", 1 << 14)
        monkeypatch.setattr(fast, "COMBINE_CHUNK_MIN_ROWS", 4)
    return patch


FUSED = {"GNF4_TRAIN_GEMM": "fused"}

#: (id, builder keywords, counters that must move in the poisoned step). ``auto`` takes the dense route at these sizes off sm_90;
#: the fused kernels run with ``GNF4_TRAIN_GEMM=fused`` (and under ``auto`` at production sizes).
CUDA_CONFIGS = [
    ("auto", {}, ("CHUNKED_LM_LOSS_STATS.chunked_calls", "TRAIN_QKV_STATS.calls", "RMSNORM_TRAIN_STATS.calls")),
    ("auto/bf16-adapters", {"adapter_dtype": torch.bfloat16}, ("TRAIN_QKV_STATS.calls",)),
    ("auto/absmax-dq", {"absmax_dq": True}, ()),
    ("auto/ckpt-offload=1", {"env": {"E4B_CKPT_OFFLOAD": "1"}}, ()),
    ("auto/moe-keep=1", {"env": {"E4B_MOE_KEEP_LAYERS": "1"}}, ()),
    ("auto/fuse-qkv=0", {"env": {"E4B_TRAIN_FUSE_QKV": "0"}}, ()),
    ("fused", {"env": FUSED}, ("DGRAD_STATS.kernel",)),
    ("fused/dgrad", {"env": FUSED, "enabler": "fast_dgrad"}, ("DGRAD_STATS.kernel",)),
    ("fused/combine-chunked", {"env": FUSED, "combine": True}, ("COMBINE_STATS.chunked_fwd", "COMBINE_STATS.chunked_bwd")),
    ("fused/pad-buckets=1", {"env": {**FUSED, "NF4_QLORA_PAD_BUCKETS": "1"}},
     ("LORA_PATH_STATS.padded_bucketed", "COMPACT_BUCKETS_STATS.calls")),
    ("fused/pad-buckets=1,compact=0", {"env": {**FUSED, "NF4_QLORA_PAD_BUCKETS": "1", "NF4_QLORA_COMPACT_BUCKETS": "0"}},
     ("LORA_PATH_STATS.padded_bucketed",)),
    ("fused/pad-buckets=0", {"env": {**FUSED, "NF4_QLORA_PAD_BUCKETS": "0"}}, ("LORA_PATH_STATS.padded",)),
    ("decoded", {"env": {"GNF4_TRAIN_GEMM": "decoded"}}, ("DGRAD_STATS.decoded",)),
]


@needs_kernels
@pytest.mark.parametrize("cfg", CUDA_CONFIGS, ids=[c[0] for c in CUDA_CONFIGS])
def test_cuda_fast_train_is_clean(cuda_model, monkeypatch, cfg):
    _kernels()
    _name, kw, expect = cfg
    kw = dict(kw)
    patch = _tiny_combine_chunks(monkeypatch) if kw.pop("combine", False) else None
    kw.setdefault("enabler", "fast")
    _assert_clean(_audit(_builder(cuda_model, "cuda", monkeypatch, patch=patch, **kw), "cuda", 96), *expect)
    if _name == "auto/ckpt-offload=1":
        from experts4bit_qlora.engines.ckpt_offload import CKPT_OFFLOAD_STATS
        assert CKPT_OFFLOAD_STATS["layers"] > 0
    if _name == "auto/absmax-dq":
        import experts4bit_qlora.engines.fast as fast
        assert (fast.FAST_TRAIN_STATS["absmax_dq"] or {}).get("compressed", 0) > 0, fast.FAST_TRAIN_STATS["absmax_dq"]
    if _name == "auto/moe-keep=1":
        from experts4bit_qlora.engines.moe_keep import MOE_KEEP_STATS
        assert MOE_KEEP_STATS["layers"] > 0


@needs_kernels
def test_cuda_left_padded_batch_is_clean(cuda_model, monkeypatch):
    _kernels()
    _assert_clean(_audit(_builder(cuda_model, "cuda", monkeypatch, enabler="fast", env=FUSED), "cuda", 96, left=True),
                  "DGRAD_STATS.kernel")


@needs_kernels
@pytest.mark.parametrize("enabler", ["batched", "reference"])
def test_cuda_batched_and_reference_paths_are_clean(cuda_model, monkeypatch, enabler):
    r = _audit(_builder(cuda_model, "cuda", monkeypatch, enabler=enabler), "cuda", 96)
    _assert_clean(r, *(("batched_calls",) if enabler == "batched" else ()))


@needs_kernels
@pytest.mark.parametrize("env", [{}, FUSED], ids=["auto", "fused"])
def test_cuda_linear_attention_family_is_clean(cuda_linear_attention_model, monkeypatch, env):
    _kernels()
    _assert_clean(_audit(_builder(cuda_linear_attention_model, "cuda", monkeypatch, attn4=False, enabler="fast", env=env),
                         "cuda", 96), "CHUNKED_LM_LOSS_STATS.chunked_calls")


@needs_kernels
def test_cuda_control_a_planted_rmsnorm_tail_bug_is_caught(cuda_model, monkeypatch):
    """The fused RMSNorm launched as ``M // 5`` programs of 5 rows leaves the last ``M % 5`` rows -- padded positions here --
    unwritten: this bug class, planted. The audit must call it, with a non-finite gradient under the NaN poison."""
    _kernels()
    import experts4bit_qlora.engines.rmsnorm_train as rt
    r = _audit(_builder(cuda_model, "cuda", monkeypatch, enabler="fast",
                        patch=lambda: monkeypatch.setattr(rt, "_rows_per_prog", lambda n_rows, N: 5)), "cuda", 96)
    assert any("RMSNORM_TRAIN_STATS.calls" in k and v > 0 for k, v in r["engaged"].items()), r["engaged"]
    assert not r["clean"] and not r["finite_P"], r
