"""SC1g A3's two side checks, their pure-torch parts (CPU): the attention reference sc1g_attn_check.py grades the fp8 decode
kernel against, and the routing / grouping sc1g_mtile_check.py feeds the NF4 M-tile. The GPU halves ran on the A2000
(bench/sc2/sc1g-a2000/a3_*) and ride box J on the 5090."""
import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
SC2 = pathlib.Path(__file__).resolve().parents[1] / "bench" / "sc2"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, SC2 / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _mod("sc1g_attn_check")
M = _mod("sc1g_mtile_check")


def _qkv(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(A.HQ, A.D, generator=g), torch.randn(T, A.HKV, A.D, generator=g),
            torch.randn(T, A.HKV, A.D, generator=g), torch.randn(A.HQ, generator=g))


def test_attend_with_zero_query_and_no_sinks_averages_the_values():
    q, K, V, s = _qkv(40)
    out = A.attend(torch.zeros_like(q), K, V, 0, s, A.D ** -0.5, use_sinks=False)
    want = V.mean(0).repeat_interleave(A.HQ // A.HKV, dim=0)
    assert torch.allclose(out, want, atol=1e-6)


def test_attend_window_reads_only_the_last_keys():
    q, K, V, s = _qkv(300)
    K2, V2 = K.clone(), V.clone()
    K2[:-128] = torch.randn_like(K2[:-128]) * 50          # anything outside the window must not matter
    V2[:-128] = 1e3
    sc = A.D ** -0.5
    assert torch.allclose(A.attend(q, K, V, 128, s, sc), A.attend(q, K2, V2, 128, s, sc), atol=1e-6)
    assert not torch.allclose(A.attend(q, K, V, 0, s, sc), A.attend(q, K2, V2, 0, s, sc), atol=1e-3)


def test_attend_sink_takes_mass_and_never_contributes_a_value():
    q, K, V, s = _qkv(64)
    sc = A.D ** -0.5
    huge = torch.full_like(s, 80.0)                        # a dominant sink absorbs nearly all the softmax mass
    assert A.attend(q, K, V, 0, huge, sc).abs().max() < 1e-6
    low = torch.full_like(s, -80.0)                        # a negligible sink == no sink
    assert torch.allclose(A.attend(q, K, V, 0, low, sc), A.attend(q, K, V, 0, s, sc, use_sinks=False), atol=1e-6)


def test_attend_maps_query_heads_to_kv_heads_in_blocks():
    q, K, V, s = _qkv(32)
    V2 = V.clone()
    V2[:, 1] += 7.0                                        # only kv head 1 changes: only q heads 8..15 may move
    sc = A.D ** -0.5
    d = (A.attend(q, K, V2, 0, s, sc) - A.attend(q, K, V, 0, s, sc)).abs().amax(-1)
    G = A.HQ // A.HKV
    assert torch.all(d[G:2 * G] > 1e-3) and torch.all(torch.cat([d[:G], d[2 * G:]]) < 1e-6)


@pytest.mark.parametrize("kind", ["uniform", "skewed"])
def test_route_gives_top4_distinct_experts_per_token(kind):
    ids = M.route(500, 32, kind, torch.Generator().manual_seed(1))
    assert ids.shape == (500, M.TOPK)
    assert all(len(set(r.tolist())) == M.TOPK for r in ids)
    top = torch.bincount(ids.reshape(-1), minlength=32).max().item()
    assert (top > 400) == (kind == "skewed")               # the skewed draw puts one expert on most tokens


def test_grouped_sorts_by_expert_and_order_inverts_it():
    g = torch.Generator().manual_seed(2)
    ids = M.route(50, 32, "uniform", g)
    x = torch.randn(50, 8, generator=g)
    xs, sizes, eids, order = M.grouped(x, ids)
    assert sum(sizes) == 50 * M.TOPK and eids == sorted(eids) and len(set(eids)) == len(eids)
    back = torch.empty_like(xs)
    back[order] = xs
    assert torch.equal(back, x.repeat_interleave(M.TOPK, dim=0))
    r0 = 0
    flat = ids.reshape(-1)
    for m, e in zip(sizes, eids):                          # each group's rows really are routed to that expert
        assert torch.all(flat[order[r0:r0 + m]] == e)
        r0 += m
