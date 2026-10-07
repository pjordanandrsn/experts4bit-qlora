"""``offload_plan`` prices what ``enable_dense_offload`` does without building it: the same selection as a real handle on
CPU, the pinned allocator's power-of-two rounding (exactly DQ3's measured host reservation at Qwen3-32B), the staged slots
and the link traffic. ``late_bound_4bit_refusal`` and ``chunked_loss_bytes`` are the two other questions a planner asks.
"""
from __future__ import annotations

import pytest
import torch

from experts4bit_qlora.engines import dense_offload
from experts4bit_qlora.engines.chunked_lm_loss import CHUNK_BYTES_PER_LOGIT, chunked_loss_bytes
from experts4bit_qlora.engines.dense_offload import _DenseOffload, late_bound_4bit_refusal, offload_plan
from test_dense_offload_trainable import _offload, _toy


@pytest.fixture(autouse=True)
def _clean_class_state():
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()
    yield
    _DenseOffload._staged_now.clear()
    _DenseOffload._resident.clear()


def _walk(m):
    """Every parameter and buffer of each decoder layer, as a handle walks them: (nbytes, ndim, trainable, is_param)."""
    out = []
    for lay in m.layers:
        out.append([(t.numel() * t.element_size(), t.dim(), bool(getattr(t, "requires_grad", False)) and is_param, is_param)
                    for _n, mod in lay.named_modules()
                    for store, is_param in ((mod._parameters, True), (mod._buffers, False))
                    for t in store.values() if t is not None])
    return out


@pytest.mark.parametrize("freeze", ["none", "all", "base+one", "layer"])
def test_the_plan_streams_exactly_what_a_real_handle_streams(freeze):
    m = _toy(freeze)
    walked = _walk(m)                                                       # before offload evicts the streamed tensors
    planned = offload_plan(walked, pin=False, train_prefetch=False)
    handles, _warnings = _offload(m)
    assert planned["layers"] == len(handles)
    assert planned["streamed"] == sum(h.bytes for h in handles) == sum(h.host_bytes for h in handles)
    assert planned["largest_layer"] == max(h.bytes for h in handles)
    assert planned["host_reserved"] == planned["streamed"]                 # nothing pinned
    total = sum(e[0] for layer in walked for e in layer)
    assert planned["stays_on_device"] == total - planned["streamed"]


def test_a_frozen_buffer_does_not_keep_trainable_weights_on_the_device():
    # enable_dense_offload decides skip_trainable over PARAMETERS only. A fully trainable model with a large 2-D buffer streams
    # its trainable weights; the plan must say the same when buffers are marked (is_param False), and the 3-tuple form, which
    # cannot tell a buffer from a frozen parameter, is the case the marker exists for.
    m = _toy("none")
    for lay in m.layers:
        lay.register_buffer("table", torch.zeros(512, 512))                 # 1 MiB fp32, 2-D: streamable, never trainable
    walked = _walk(m)
    planned = offload_plan(walked, pin=False, train_prefetch=False)
    unmarked = offload_plan([[e[:3] for e in layer] for layer in walked], pin=False, train_prefetch=False)
    handles, _warnings = _offload(m)
    assert planned["streamed"] == sum(h.bytes for h in handles)
    assert unmarked["streamed"] < planned["streamed"]                     # without the marker the trainable weights "stay"


def test_pinned_rounding_reproduces_dq3s_host_reservation_at_qwen3_32b():
    """One Qwen3-32B decoder layer in NF4 (packed uint8, half a byte per weight) with PEFT LoRA r16 in fp32 on all seven
    projections, as DQ3 built it: DQ3 requested 15,602,810,880 B of pinned homes and the allocator reserved 17,716,740,096 B
    (bench/dq3, dq3-5090-5)."""
    H, INTER, Q, KV, r = 5120, 25600, 8192, 1024, 16
    proj = {"q": (H, Q), "k": (H, KV), "v": (H, KV), "o": (Q, H), "gate": (H, INTER), "up": (H, INTER), "down": (INTER, H)}
    layer = [(i * o // 2, 2, False) for i, o in proj.values()]              # packed NF4 weights, frozen
    layer += [(r * i * 4, 2, True) for i, _o in proj.values()]              # lora_A [r, in] fp32
    layer += [(o * r * 4, 2, True) for _i, o in proj.values()]              # lora_B [out, r] fp32 (1.6 MB at 25600: kept)
    layer += [(H * 2, 1, False), (H * 2, 1, False), (128 * 2, 1, False), (128 * 2, 1, False)]   # norms
    plan = offload_plan([layer] * 64)
    assert plan["streamed"] == 15_602_810_880 and plan["host_reserved"] == 17_716_740_096
    assert plan["largest_layer"] == 243_793_920                             # DQ3's per-layer streamed bytes
    assert plan["resident_slots"] == 2 * 243_793_920                        # train_prefetch: two staged
    assert offload_plan([layer] * 64, train_prefetch=False)["resident_slots"] == 243_793_920
    assert plan["link_per_microbatch"] == 2 * 15_602_810_880
    assert offload_plan([layer] * 64, pin=False)["host_reserved"] == 15_602_810_880


def test_a_trainable_matrix_streams_only_in_an_unfrozen_model():
    big = 2 << 20
    frozen_and_lora = [[(big, 2, False), (big, 2, True)]]
    assert offload_plan(frozen_and_lora)["streamed"] == big                 # skip_trainable decided True: LoRA stays
    assert offload_plan([[(big, 2, True), (big, 2, True)]])["streamed"] == 2 * big   # nothing frozen: all streams
    assert offload_plan(frozen_and_lora, skip_trainable=False)["streamed"] == 2 * big


def test_late_bound_refusal_names_the_mismatched_sources(monkeypatch):
    monkeypatch.setattr(dense_offload, "_bnb_mirror_mismatches", lambda: [])
    assert late_bound_4bit_refusal() is None
    monkeypatch.setattr(dense_offload, "_bnb_mirror_mismatches", lambda: ["MatMul4Bit.backward"])
    why = late_bound_4bit_refusal()
    assert "MatMul4Bit.backward" in why and "frees no VRAM" in why


def test_chunked_loss_workspace_is_one_chunk_of_logits_plus_the_gathered_rows():
    V, H = 151_936, 5120
    assert chunked_loss_bytes(4096, V) == 512 * V * CHUNK_BYTES_PER_LOGIT
    assert chunked_loss_bytes(300, V) == 300 * V * CHUNK_BYTES_PER_LOGIT    # fewer supervised rows than a chunk
    assert chunked_loss_bytes(4096, V, hidden=H) == 512 * V * CHUNK_BYTES_PER_LOGIT + 4096 * H * 2
    assert chunked_loss_bytes(0, V) == 0
    assert chunked_loss_bytes(4096, V) < 4096 * V * CHUNK_BYTES_PER_LOGIT  # what stock logits would take
    with pytest.raises(ValueError):
        chunked_loss_bytes(10, V, chunk=0)


def test_cpu_only_runs_need_no_cuda():
    assert torch.device("cpu").type == "cpu"                                # the selection test above ran on CPU
