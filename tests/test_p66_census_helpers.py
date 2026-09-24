"""P66's census helpers that run without a GPU (bench/p66/p66_census.py).

The profiler walk is the load-bearing one: every count the lane reports comes from it. It is driven here
with a hand-built event tree shaped like torch.profiler's (``name``, ``cpu_children``, ``cpu_parent``,
``device_type``, ``kernels``), including the two traps a naive walk falls into -- the instrument's own
closing sync outside every token region, and the device-side twin of a record_function region."""
import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "p66"))
_spec = importlib.util.spec_from_file_location("p66_census", REPO / "bench" / "p66" / "p66_census.py")
cen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cen)


class K:
    def __init__(self, name):
        self.name = name


class Ev:
    def __init__(self, name, children=(), kernels=(), device_type="DeviceType.CPU"):
        self.name, self.device_type = name, device_type
        self.cpu_children = list(children)
        self.kernels = [K(n) for n in kernels]
        self.cpu_parent = None
        for c in self.cpu_children:
            c.cpu_parent = self


def _flatten(evs):
    out = []

    def go(e):
        out.append(e)
        for c in e.cpu_children:
            go(c)
    for e in evs:
        go(e)
    return out


def _token():
    # token -> layer -> fetch -> {launch, resolve -> {sync, H2D}} ; layer -> aten op -> launch (+ its kernel)
    resolve = Ev(cen.R_RESOLVE, [Ev("cudaStreamSynchronize"), Ev("cudaMemcpyAsync")])
    fetch = Ev(cen.R_FETCH, [Ev("cuLaunchKernel"), resolve])
    gemv = Ev("aten::mm", [Ev("cudaLaunchKernel")], kernels=["_gemv_nf4_dotpad"])
    layer = Ev(cen.R_LAYER, [fetch, gemv, Ev("cuLaunchKernel")])
    return Ev(cen.R_TOKEN, [Ev(cen.R_INPUT, [Ev("cudaMemcpyAsync")]), layer])


def test_region_walk_counts_inside_token_regions_only():
    toks = [_token(), _token()]
    closing = Ev("cudaDeviceSynchronize")                    # the instrument's own sync: outside
    twin = Ev(cen.R_TOKEN, device_type="DeviceType.CUDA")    # device-side annotation span: not a region
    rc = cen.region_counts(_flatten(toks) + [closing, twin])
    assert rc["token_regions"] == 2
    assert rc["api"] == {"cuLaunchKernel": 4, "cudaStreamSynchronize": 2, "cudaMemcpyAsync": 4,
                         "cudaLaunchKernel": 2}
    assert "cudaDeviceSynchronize" not in rc["api"]
    # inclusive: the resolve sync counts for resolve, fetch and layer alike
    assert rc["api_by_phase"][cen.R_FETCH] == {"cuLaunchKernel": 2, "cudaStreamSynchronize": 2,
                                               "cudaMemcpyAsync": 2}
    assert rc["api_by_phase"][cen.R_RESOLVE] == {"cudaStreamSynchronize": 2, "cudaMemcpyAsync": 2}
    assert rc["api_by_phase"][cen.R_LAYER]["cuLaunchKernel"] == 4
    assert rc["api_by_phase"][cen.R_INPUT] == {"cudaMemcpyAsync": 2}
    # nearest: each event once, under its innermost phase
    assert rc["api_by_phase_nearest"][cen.R_FETCH] == {"cuLaunchKernel": 2}
    assert rc["api_by_phase_nearest"][cen.R_LAYER] == {"cudaLaunchKernel": 2, "cuLaunchKernel": 2}
    assert rc["ops"] == {"aten::mm": 2}
    assert rc["kernels_by_phase"][cen.R_LAYER] == {"_gemv_nf4_dotpad": 2}


def test_api_global_sees_what_the_regions_miss():
    # a launch that failed to nest under a token region is still a CUDA API event in the window
    stray = Ev("cuLaunchKernel")
    rc = cen.region_counts(_flatten([_token()]) + [stray])
    g = cen.api_global(_flatten([_token()]) + [stray])
    assert g["cuLaunchKernel"] == rc["api"]["cuLaunchKernel"] + 1


def test_hot_sets_are_seeded_and_shared_across_paths():
    a = cen.hot_sets_for(4, 32, 8, 66)
    b = cen.hot_sets_for(4, 32, 8, 66)
    assert a == b and all(len(h) == 8 and len(set(h)) == 8 for h in a)
    assert cen.hot_sets_for(4, 32, 8, 67) != a
    assert cen.hot_sets_for(2, 32, 0, 66) == [[], []]


def test_routing_uniform_and_controlled():
    L, E, k = 3, 32, 4
    ids, w = cen.make_routing(10, L, E, k, seed=1)
    assert ids.shape == (10, L, k) and w.shape == (10, L, k) and w.dtype == torch.bfloat16
    assert all(len(set(ids[t, li].tolist())) == k for t in range(10) for li in range(L))
    hot = cen.hot_sets_for(L, E, 16, 66)
    for c in (0, 2, 4):
        ids, _ = cen.make_routing(10, L, E, k, hot_sets=hot, cold_lanes=c, seed=2)
        for t in range(10):
            for li in range(L):
                lane = ids[t, li].tolist()
                assert sum(e not in set(hot[li]) for e in lane) == c
                assert len(set(lane)) == k
    with pytest.raises(ValueError):
        cen.make_routing(2, L, E, k, hot_sets=cen.hot_sets_for(L, E, 2, 66), cold_lanes=0, seed=3)


def test_the_window_schedule_takes_fresh_tokens_for_every_pass():
    assert cen.tokens_needed(4, 8) == 4 + cen.PASSES_PER_WINDOW * 8 + 2


def test_predicted_transfer_sums_layers_and_averages_tokens():
    costs = {"b_link_gbs": 25.0, "b_vram_gbs": 1000.0, "b_dram_gbs": 50.0}
    rows = [[2, 0, 1], [0, 0, 0]]            # two tokens, three layers
    per_row = (1e6 / 1e9) * (1 / 25.0 + 1 / 1000.0) * 1e6
    assert cen.predicted_us(rows, 1_000_000, costs) == pytest.approx(3 * per_row / 2)


def test_manifest_places_every_expert_once():
    m = cen._manifest(2, 8, [0, 1], [[0, 1], [2]], [[3], [4, 5]])
    placed = sorted(tuple(p) for t in m["tiers"].values() for p in t)
    assert placed == sorted((li, e) for li in (0, 1) for e in range(8))
    assert m["masses"]["vram_frac"] == pytest.approx(3 / 16)
    assert m["masses"]["dram_frac"] == pytest.approx(3 / 16)
    assert cen._manifest(1, 4, [7], [[0]], None)["tiers"]["nvme"] == [[7, 1], [7, 2], [7, 3]]
