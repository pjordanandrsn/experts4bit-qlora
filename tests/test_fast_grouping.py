"""``engines.fast._group_by_expert``: the grouped kernels' host-side sizes and expert ids with ONE device read.

The old form (bincount -> nonzero -> two ``.tolist()``) synchronized five times per grouped forward, so a fused
MoE training layer pass paid 10 of its 13 host syncs on grouping and index transfers. The contract of the
replacement is the old lists exactly -- same values, same order, Python ints -- so every kernel downstream sees
the same launch, and one synchronizing call instead of five.
"""
import pytest

torch = pytest.importorskip("torch")

from experts4bit_qlora.engines import fast  # noqa: E402

CUDA = torch.cuda.is_available()


def _legacy(flat, E):
    counts = torch.bincount(flat, minlength=E)
    active = torch.nonzero(counts, as_tuple=False).view(-1)
    return counts[active].tolist(), active.to(torch.int32).tolist()


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
@pytest.mark.parametrize("E,tokens,k,seed", [(8, 5, 2, 0), (128, 512, 8, 1), (128, 3, 8, 2), (64, 1, 1, 3), (256, 1024, 8, 4)])
def test_same_lists_as_the_legacy_form(device, E, tokens, k, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(tokens)]).to(device)
    flat = idx.reshape(-1)
    sizes, eids = fast._group_by_expert(flat, E)
    want_sizes, want_eids = _legacy(flat, E)
    assert (sizes, eids) == (want_sizes, want_eids)
    assert all(type(v) is int for v in sizes + eids)
    assert sum(sizes) == flat.numel() and all(s > 0 for s in sizes)
    if tokens * k < E:
        assert len(eids) < E                       # some experts empty: dropped, as before


def test_legacy_knob_restores_the_old_form(monkeypatch):
    monkeypatch.setenv("E4B_GROUPING", "legacy")
    flat = torch.tensor([3, 1, 3, 0, 7])
    assert fast._group_by_expert(flat, 8) == ([1, 1, 2, 1], [0, 1, 3, 7])


def _count_syncs(fn):
    import warnings
    n = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        n = sum(1 for x in w if "called a synchronizing" in str(x.message))   # not the mode's own "prototype" notice
    return n


@pytest.mark.skipif(not CUDA, reason="sync counting needs CUDA")
def test_one_synchronizing_call_instead_of_five(monkeypatch):
    flat = torch.randint(0, 128, (4096,), device="cuda")
    fast._group_by_expert(flat, 128)                # warm
    torch.cuda.synchronize()
    new = _count_syncs(lambda: fast._group_by_expert(flat, 128))
    monkeypatch.setenv("E4B_GROUPING", "legacy")
    old = _count_syncs(lambda: fast._group_by_expert(flat, 128))
    assert new == 1, new
    assert old >= 4, old                            # the control: the instrument sees the old form's reads


@pytest.mark.skipif(not CUDA, reason="the fused training path needs CUDA")
def test_fused_training_is_bit_identical_under_either_grouping(monkeypatch):
    pytest.importorskip("nf4_qlora")
    from experts4bit_qlora import Experts4bit, ExpertsLoRA, enable_fast_train

    def run():
        torch.manual_seed(0)
        gu = torch.randn(16, 2 * 192, 128, device="cuda") * 0.1      # intermediate a multiple of the 64 blocksize
        dn = torch.randn(16, 128, 192, device="cuda") * 0.1
        base = Experts4bit.from_float(gu, dn, quant_type="nf4", compute_dtype=torch.bfloat16)
        mod = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32).to("cuda").train()
        with torch.no_grad():
            for p in (mod.gate_up_lora_B, mod.down_lora_B):
                p.normal_(0, 0.02)
        assert enable_fast_train(mod, dgrad=True) == 1
        torch.manual_seed(1)
        hs = torch.randn(96, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        idx = torch.randint(0, 16, (96, 2), device="cuda")
        wts = torch.rand(96, 2, dtype=torch.bfloat16, device="cuda")
        out = mod(hs, idx, wts)
        (out.float() * torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out)).sum().backward()
        return out.detach().clone(), hs.grad.clone(), {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}

    o1, g1, p1 = run()
    monkeypatch.setenv("E4B_GROUPING", "legacy")
    o2, g2, p2 = run()
    assert torch.equal(o1, o2) and torch.equal(g1, g2)
    assert p1.keys() == p2.keys() and len(p1) >= 4 and all(torch.equal(p1[k], p2[k]) for k in p1)
