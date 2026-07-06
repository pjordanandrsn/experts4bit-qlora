"""The HTTP serving shim (:mod:`experts4bit_qlora.serve`) — everything testable without a GPU.

Covers the pieces whose failure modes are silent-wrong rather than loud: the adapter registry
(non-lora rejection, unknown-key/shape validation, and the zero-fill completion that stops one
adapter's attention LoRA leaking into another), request admission/backpressure (503 +
Retry-After, never an unbounded queue), request clamping (max_new_tokens cap, 413 on oversized
prompts, 404 on unknown adapters), and the OpenAI-compat response shape. Generation itself is
exercised through a fake engine — the real GPU path is the deploy-time verification checklist,
not a unit test.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from experts4bit_qlora.serve import (  # noqa: E402
    BusyError,
    Engine,
    ServeConfig,
    _complete_adapter,
    build_registry,
    create_app,
    parse_adapter_spec,
    validate_adapter,
)

# ---------------------------------------------------------------------------
# parse_adapter_spec
# ---------------------------------------------------------------------------


def test_parse_adapter_spec_basic():
    assert parse_adapter_spec("") == {}
    assert parse_adapter_spec("a=/x/a.pt, b=/y/b.pt") == {"a": "/x/a.pt", "b": "/y/b.pt"}


@pytest.mark.parametrize("spec", ["base=/x.pt", "a=/x.pt,a=/y.pt", "nopath", "=path", "a="])
def test_parse_adapter_spec_rejects(spec):
    with pytest.raises(ValueError):
        parse_adapter_spec(spec)


# ---------------------------------------------------------------------------
# Adapter registry: validation + zero-fill completion
# ---------------------------------------------------------------------------

INIT = {
    "model.layers.0.mlp.experts.gate_up_lora_A": torch.randn(4, 8, 16, dtype=torch.bfloat16),
    "model.layers.0.mlp.experts.gate_up_lora_B": torch.zeros(4, 32, 8, dtype=torch.bfloat16),
    "model.layers.0.self_attn.q_proj.lora_A": torch.randn(8, 16, dtype=torch.bfloat16),
    "model.layers.0.self_attn.q_proj.lora_B": torch.zeros(16, 8, dtype=torch.bfloat16),
}


def test_validate_adapter_rejects_non_lora_keys():
    sd = {"model.layers.0.mlp.experts.base.gate_up_proj": torch.zeros(2)}
    with pytest.raises(ValueError, match="non-lora"):
        validate_adapter("bad", sd, INIT)


def test_validate_adapter_rejects_unknown_and_shape_mismatch():
    with pytest.raises(ValueError, match="not in this model"):
        validate_adapter("bad", {"model.layers.9.mlp.experts.gate_up_lora_A": torch.zeros(4, 8, 16)}, INIT)
    with pytest.raises(ValueError, match="shape mismatch"):  # an R=16 adapter against an R=8 model
        validate_adapter("bad", {"model.layers.0.mlp.experts.gate_up_lora_A": torch.zeros(4, 16, 16)}, INIT)


def test_complete_adapter_zero_fills_missing_keys():
    """The attention-leak case: an adapter WITHOUT attention keys must swap in the INIT attention
    tensors (lora_B == 0 => delta 0), not inherit whatever the previously-active adapter left."""
    experts_only = {"model.layers.0.mlp.experts.gate_up_lora_A": torch.ones(4, 8, 16, dtype=torch.float32)}
    done = _complete_adapter(experts_only, INIT, pin=False)
    assert set(done) == set(INIT)  # completed to the FULL lora key-set
    assert done["model.layers.0.self_attn.q_proj.lora_B"].abs().sum() == 0
    assert torch.equal(
        done["model.layers.0.self_attn.q_proj.lora_A"].float(),
        INIT["model.layers.0.self_attn.q_proj.lora_A"].float(),
    )
    # Provided tensors are cast to the init dtype (files may be saved fp32, params run bf16).
    assert done["model.layers.0.mlp.experts.gate_up_lora_A"].dtype == torch.bfloat16


def test_build_registry_always_serves_base():
    reg = build_registry({}, INIT, pin=False)
    assert set(reg) == {"base"}
    assert all(torch.equal(reg["base"][k].float(), INIT[k].float()) for k in INIT)


# ---------------------------------------------------------------------------
# Engine admission (real Engine, no model)
# ---------------------------------------------------------------------------


def test_admit_enforces_capacity():
    eng = Engine(ServeConfig(queue_max=2))
    try:
        for _ in range(3):  # 1 running + 2 queued
            eng.admit()
        with pytest.raises(BusyError):
            eng.admit()
        eng.release()
        eng.admit()  # capacity frees up again
    finally:
        eng.shutdown()


# ---------------------------------------------------------------------------
# Endpoints via a fake engine
# ---------------------------------------------------------------------------


class FakeEngine(Engine):
    """Real admission/queue logic, faked GPU path. `submit` records the job it would have run."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.state = "ready"
        self.registry = {"base": {}, "alpaca": {}}
        self.active_adapter = "base"
        self.last_job = None

    def start(self, loop):
        self._loop = loop

    def count_tokens(self, prompt):
        return len(prompt.split())

    def submit(self, streamer, **job):
        self.last_job = job

        async def run():
            return {
                "text": "ok",
                "adapter": job["adapter"],
                "prompt_tokens": 3,
                "tokens": 2,
                "tok_per_s": 1.4,
                "swap_ms": 0.0,
                "stopped": "eos",
            }

        return run()


@pytest.fixture()
def client():
    cfg = ServeConfig(max_input_tokens=8, max_new_tokens=64, queue_max=1)
    eng = FakeEngine(cfg)
    with TestClient(create_app(cfg, engine=eng)) as c:
        c.engine = eng
        yield c
    eng.shutdown()


def test_health_shape(client):
    body = client.get("/health").json()
    assert body["status"] == "ready"
    assert body["adapters"] == ["alpaca", "base"]
    assert body["offload"] == {"enabled": True, "pinned": None}
    assert body["queue_depth"] == 0


def test_generate_roundtrip_and_clamp(client):
    r = client.post("/generate", json={"prompt": "hi there", "adapter": "alpaca", "max_new_tokens": 10_000})
    assert r.status_code == 200
    assert r.json()["adapter"] == "alpaca"
    assert client.engine.last_job["max_new_tokens"] == 64  # clamped to E4B_MAX_NEW_TOKENS
    assert client.engine.queue_depth == 0  # released after completion


def test_generate_404_on_unknown_adapter(client):
    assert client.post("/generate", json={"prompt": "hi", "adapter": "nope"}).status_code == 404


def test_generate_413_on_long_prompt(client):
    r = client.post("/generate", json={"prompt": "w " * 50})
    assert r.status_code == 413


def test_generate_503_while_loading(client):
    client.engine.state = "loading"
    r = client.post("/generate", json={"prompt": "hi"})
    assert r.status_code == 503
    assert "Retry-After" in r.headers


def test_generate_503_when_queue_full(client):
    client.engine._pending = 2  # capacity for queue_max=1 is 1 running + 1 queued
    r = client.post("/generate", json={"prompt": "hi"})
    assert r.status_code == 503
    assert "Retry-After" in r.headers
    assert client.engine.queue_depth == 2  # a rejected request must not leak a slot


def test_openai_completions_shape(client):
    r = client.post("/v1/completions", json={"prompt": "hi", "model": "alpaca", "max_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "alpaca"
    assert body["choices"][0] == {"index": 0, "text": "ok", "finish_reason": "stop"}
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_openai_models_lists_adapters(client):
    body = client.get("/v1/models").json()
    assert [m["id"] for m in body["data"]] == ["alpaca", "base"]


def test_seed_passthrough(client):
    client.post("/generate", json={"prompt": "hi", "seed": 42})
    assert client.engine.last_job["seed"] == 42


def test_stop_signal_reasons():
    from experts4bit_qlora.serve import _StopSignal
    import threading
    import time

    ev = threading.Event()
    sig = _StopSignal(ev, deadline=time.monotonic() + 3600)
    assert sig(None, None) is False and sig.reason is None
    ev.set()
    assert sig(None, None) is True and sig.reason == "cancelled"

    expired = _StopSignal(None, deadline=time.monotonic() - 1)
    assert expired(None, None) is True and expired.reason == "timeout"


def test_engine_release_never_negative():
    eng = Engine(ServeConfig())
    try:
        eng.release()
        assert eng.queue_depth == 0
    finally:
        eng.shutdown()
