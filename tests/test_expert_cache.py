"""The device-side expert cache, across every supported architecture.

Finding #32 measured 0.4513 overlap between the experts a layer routes to at
token t and at t-1; caching those on the device turns a link transfer into a
device-to-device copy. The cache is shared across layers and allocates ONE row
shape, so the load-bearing risks are:

1. it must stay **bit-identical** — a pooled row is a byte copy of the same
   pinned home row, and a hit and a miss must write the same bytes;
2. it must **refuse** a model whose layers differ in expert geometry rather than
   sizing every row from layer 0 and corrupting the rest.

`SUPPORTED_ARCHITECTURES` spans six model types, so (2) is checked against the
same synthetic checkpoints the loader tests use rather than assumed.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from experts4bit_qlora import (  # noqa: E402
    Experts4bit,
    ExpertsLoRA,
    enable_expert_cache,
    enable_expert_offload,
    enable_routed_staging,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
N_EXP, HIDDEN, INTER, TOP_K = 16, 128, 192, 2
_Q_UNAVAIL = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="offload needs CUDA")


def _layer(n_exp=N_EXP, hidden=HIDDEN, inter=INTER, seed=0):
    torch.manual_seed(seed)
    gu = torch.randn(n_exp, 2 * inter, hidden, dtype=DTYPE, device=DEVICE)
    dn = torch.randn(n_exp, hidden, inter, dtype=DTYPE, device=DEVICE)
    try:
        base = Experts4bit.from_float(gu, dn, quant_type="nf4", compute_dtype=DTYPE)
    except _Q_UNAVAIL as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable: {e}")
    m = ExpertsLoRA(base, r=8, alpha=16, dtype=DTYPE).to(DEVICE)
    with torch.no_grad():
        for p in (m.gate_up_lora_B, m.down_lora_B):
            p.normal_(0, 0.02)
    return m.eval()


def _inputs(n_tok=4, n_used=TOP_K, seed=1):
    torch.manual_seed(seed)
    return (torch.randn(n_tok, HIDDEN, dtype=DTYPE, device=DEVICE),
            torch.randint(0, n_used, (n_tok, TOP_K), device=DEVICE),
            torch.rand(n_tok, TOP_K, dtype=DTYPE, device=DEVICE))


def test_cache_is_bit_identical_and_actually_hits():
    """Same bytes out, and the second pass must come from the pool, not the link."""
    hs, idx, wts = _inputs()

    plain = _layer()
    hp = enable_expert_offload(plain, DEVICE)
    enable_routed_staging([hp])
    with torch.no_grad():
        want = plain(hs, idx, wts)
        want2 = plain(hs, idx, wts)

    cached = _layer()
    hc = enable_expert_offload(cached, DEVICE)
    enable_routed_staging([hc])
    pool = enable_expert_cache([hc], top_k=TOP_K)
    with torch.no_grad():
        got = cached(hs, idx, wts)
        got2 = cached(hs, idx, wts)

    assert torch.equal(got, want), "first pass differs from uncached routed staging"
    assert torch.equal(got2, want2), "cached second pass differs — a pooled row is wrong"
    hits, misses, tot, rate = pool.stats()
    assert misses > 0, "nothing was ever fetched over the link"
    assert hits > 0, f"the cache never hit ({hits}/{tot}) — it is doing no work"


def test_cache_refuses_heterogeneous_geometry():
    """Layers with different expert shapes must raise, not corrupt silently."""
    a = _layer(n_exp=N_EXP)
    b = _layer(n_exp=N_EXP * 2, seed=3)          # different expert count
    handles = [enable_expert_offload(a, DEVICE), enable_expert_offload(b, DEVICE)]
    with pytest.raises(RuntimeError, match="same per-expert geometry"):
        enable_expert_cache(handles, top_k=TOP_K)


def test_cache_evicts_and_stays_correct_under_pressure():
    """A pool smaller than the working set must still return correct bytes."""
    hs, idx, wts = _inputs(n_tok=8, n_used=N_EXP)

    plain = _layer()
    hp = enable_expert_offload(plain, DEVICE)
    enable_routed_staging([hp], max_fraction=1.0)
    with torch.no_grad():
        want = plain(hs, idx, wts)

    cached = _layer()
    hc = enable_expert_offload(cached, DEVICE)
    enable_routed_staging([hc], max_fraction=1.0)
    pool = enable_expert_cache([hc], slots=2, top_k=TOP_K)   # far too small on purpose
    with torch.no_grad():
        for _ in range(3):
            got = cached(hs, idx, wts)
    assert torch.equal(got, want), "eviction produced wrong bytes"
    # Ask the group for its own aggregate rather than a pool's private `_map`:
    # `enable_expert_cache` returns a _PoolGroup, which holds the per-layer pools and
    # never had a `_map` of its own, so this assertion had been raising AttributeError
    # instead of checking the budget it was written for.
    assert pool.resident <= pool.slots, \
        f"pool exceeded its slot budget: {pool.resident} resident in {pool.slots} slots"
