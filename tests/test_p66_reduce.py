"""P66's reducer counts what it says it counts (bench/p66/p66_reduce.py; bench/p66/P66-PREREG.md).

Synthetic kernel and API lists, no GPU: each rule the census depends on is driven by a case built to
break it -- the Triton driver-API launch name, a blocking copy that is a sync without the word, graph
replay where host launches and device kernels part ways, an attribution against the all-resident
reference, a count that is fixed across cold fraction and one that moves, and the pipelined have-skip."""
import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p66_reduce", REPO / "bench" / "p66" / "p66_reduce.py")
red = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(red)

FETCH = "e4b::p66.fetch"


def _mode(tokens, api, kernels, phases=None):
    return {"tokens": tokens, "api": api, "api_by_phase": phases or {}, "kernels": kernels}


# --------------------------------------------------------------------------- classification
def test_triton_driver_launches_count_as_launches():
    # a list holding only the runtime name would miss every Triton kernel -- the whole residency engine
    for name in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernelExC"):
        assert red.api_class(name) == "launch", name
    assert red.api_class("cudaGraphLaunch") == "graph"
    assert red.api_class("cudaStreamSynchronize") == "sync"
    assert red.api_class("cudaMemcpy") == "blocking"
    assert red.api_class("cudaMemcpyAsync") == "memcpy"
    assert red.api_class("cudaMemsetAsync") == "memset"
    assert red.api_class("cudaGetDevice") is None
    assert red.api_class("aten::index_select") is None


def test_kernel_classes():
    assert red.kernel_class("_gather_rows_addr") == "gather"
    assert red.kernel_class("_gather_rows_perm") == "gather"
    assert red.kernel_class("Memcpy HtoD (Pageable -> Device)") == "h2d"
    assert red.kernel_class("Memcpy DtoH (Device -> Pageable)") == "copy"
    assert red.kernel_class("Memset (Device)") == "copy"
    assert red.kernel_class("_gemv_nf4_dotpad") == "compute"


# --------------------------------------------------------------------------- per-token counts
def test_per_token_divides_by_tokens_and_keeps_blocking_copies_apart():
    m = _mode(4, {"cudaLaunchKernel": 40, "cuLaunchKernel": 20, "cudaStreamSynchronize": 8,
                  "cudaMemcpy": 4, "cudaMemcpyAsync": 12, "cudaGetDevice": 999},
              {"_gather_rows_addr": {"count": 4, "self_device_us": 400.0},
               "Memcpy HtoD (Pageable -> Device)": {"count": 8, "self_device_us": 80.0},
               "_gemv_nf4_dotpad": {"count": 8, "self_device_us": 160.0}})
    p = red.per_token(m)
    assert p["launches"] == 15.0                      # (40 + 20) / 4, the unclassified API ignored
    assert p["syncs_explicit"] == 2.0 and p["blocking_copies"] == 1.0
    assert p["syncs"] == 3.0                          # a blocking copy IS a host wait
    assert p["async_copies"] == 3.0
    assert p["submissions"] == 15.0 + 3.0 + 1.0       # launches + async copies + blocking copies
    assert p["kernels"] == 5.0                        # (4 + 8 + 8) / 4 device rows
    assert p["transfer_us"] == pytest.approx((400.0 + 80.0) / 4)
    assert p["device_us"] == pytest.approx(640.0 / 4)


def test_graph_replay_separates_host_launches_from_device_kernels():
    # one cudaGraphLaunch per token on the host; the device still runs every kernel
    m = _mode(8, {"cudaGraphLaunch": 8, "cudaMemcpyAsync": 16},
              {"_gemv_nf4_dotpad": {"count": 8 * 96, "self_device_us": 8 * 2469.0}})
    p = red.per_token(m)
    assert p["launches"] == 0.0 and p["graph_launches"] == 1.0
    assert p["kernels"] == 96.0 and p["syncs"] == 0.0


def test_zero_tokens_refuses():
    with pytest.raises(ValueError):
        red.per_token(_mode(0, {}, {}))


# --------------------------------------------------------------------------- attribution
def test_attribution_against_reference_by_name_and_phase():
    L = 2
    ref = _mode(2, {"cudaLaunchKernel": 20, "cuLaunchKernel": 8},
                {"_gemv_nf4_dotpad": {"count": 8, "self_device_us": 80.0}})
    arm = _mode(2, {"cudaLaunchKernel": 36, "cuLaunchKernel": 12},
                {"_gemv_nf4_dotpad": {"count": 8, "self_device_us": 88.0},
                 "_gather_rows_addr": {"count": 4, "self_device_us": 40.0}},
                phases={FETCH: {"cudaLaunchKernel": 16, "cuLaunchKernel": 4}})
    at = red.attribute(arm, ref, L, FETCH)
    assert at["delta_per_token"]["launches"] == pytest.approx((48 - 28) / 2)
    assert at["delta_per_layer"]["launches"] == pytest.approx(5.0)
    assert at["delta_per_token"]["syncs"] == 0.0
    assert at["fetch_per_token"]["launch"] == pytest.approx(10.0)
    assert at["fetch_per_layer"]["launch"] == pytest.approx(5.0)
    assert at["by_api"] == {"cudaLaunchKernel": 8.0, "cuLaunchKernel": 2.0}
    assert at["by_kernel"] == {"_gather_rows_addr": 2.0}      # the dotpad count did not move
    assert at["delta_per_token"]["device_us"] == pytest.approx((88 + 40 - 80) / 2)


def test_attribution_without_a_phase_reports_no_fetch():
    m = _mode(1, {"cudaLaunchKernel": 1}, {})
    assert red.attribute(m, m, 1)["fetch_per_token"] == {}


# --------------------------------------------------------------------------- cold fraction
def test_a_fixed_count_has_exactly_zero_spread():
    r = red.moves_with_cold([(0.0, 612.0), (0.5, 612.0), (1.0, 612.0)])
    assert r["fixed"] is True and r["spread"] == 0.0 and r["slope"] == pytest.approx(0.0)


def test_a_moving_count_carries_its_slope():
    r = red.moves_with_cold([(0.0, 100.0), (0.5, 150.0), (1.0, 200.0)])
    assert r["fixed"] is False and r["spread"] == 100.0 and r["slope"] == pytest.approx(100.0)


def test_one_point_cannot_test_movement():
    assert red.moves_with_cold([(0.3, 5.0)])["fixed"] is None


# --------------------------------------------------------------------------- the pipelined have-skip
def test_simulator_matches_the_engine_rule():
    k, hot = 3, {1, 2}
    # prime: every slot holds expert 0 (cold here)
    fetches = [[0, 1, 5],     # slot0 wants 0 == have -> 0 copies; slot1 hot; slot2 5 != 0 -> 1
               [0, 1, 5],     # everything already in place -> 0
               [5, 2, 0],     # slot0 5 != 0 -> 1; slot1 hot; slot2 0 != 5 -> 1
               [2, 2, 2]]     # all hot -> 0, and have is left alone
    assert red.simulate_pipelined_traffic(fetches, hot, k) == [1, 0, 2, 0]
    # the hot lanes of fetch 4 must not have disturbed have: re-asking fetch 3 copies nothing
    assert red.simulate_pipelined_traffic(fetches + [[5, 2, 0]], hot, k)[-1] == 0


def test_simulator_all_hot_copies_nothing_and_all_cold_copies_every_change():
    assert red.simulate_pipelined_traffic([[1, 2], [3, 4]], {1, 2, 3, 4}, 2) == [0, 0]
    assert red.simulate_pipelined_traffic([[1, 2], [3, 4], [3, 4]], set(), 2) == [2, 2, 0]


def test_simulator_refuses_a_wrong_width():
    with pytest.raises(ValueError):
        red.simulate_pipelined_traffic([[1, 2, 3]], set(), 2)


# --------------------------------------------------------------------------- the transfer model
def test_gpu_us_is_bytes_over_link_plus_bytes_over_vram():
    # 4 rows of 2.5 MB at 25 GB/s link + 1000 GB/s VRAM
    assert red.gpu_us(4, 2_500_000, 25.0, 1000.0) == pytest.approx((0.01 / 25 + 0.01 / 1000) * 1e6)
    assert red.gpu_us(0, 2_500_000, 25.0, 1000.0) == 0.0


def test_gpu_us_matches_cold_deadline():
    cd = pytest.importorskip("cold_deadline")
    c = cd.Costs(cpu_us_fixed=0.0, cpu_us_per_row=0.0, b_dram_gbs=50.0, b_vram_gbs=1569.4,
                 b_link_gbs=28.36, bytes_per_expert=2_654_208)
    for u in (1, 3, 8):
        assert red.gpu_us(u, 2_654_208, 28.36, 1569.4) == pytest.approx(cd.gpu_us(u, u, c))


def test_transfer_score_names_the_zero_case():
    assert red.transfer_score(10.0, 0.0)["ratio"] is None
    assert red.transfer_score(150.0, 100.0)["ratio"] == pytest.approx(1.5)


# --------------------------------------------------------------------------- the whole read
def _arm(name, path, ref, windows, hot_frac=1.0, L=2, phase=FETCH):
    return {"schema": "p66-census/1", "arm": name, "family": "synth", "path": path, "hot_frac": hot_frac,
            "n_hot": 4, "layers": L, "experts": 8, "k": 2, "row_bytes": 1000, "reference": ref,
            "residency_phase": phase, "windows": windows}


def _win(name, cold, launches, syncs=0, pred=None, gather_us=0.0, graph=None):
    w = {"name": name, "routing": {"cold_fraction": cold, "cold_rows_per_token": 2 * cold},
         "modes": {"eager": _mode(2, {"cudaLaunchKernel": 2 * launches, "cudaStreamSynchronize": 2 * syncs},
                                  {"_gather_rows_addr": {"count": 2, "self_device_us": 2 * gather_us}})}}
    if graph is not None:
        w["modes"]["graph"] = graph
    if pred is not None:
        w["predicted_transfer_us_per_token"] = pred
    return w


def test_census_deltas_cold_test_and_refusals(tmp_path):
    ref = _arm("ref", "hybrid", True, [_win("uniform", 0.0, 30)], phase=None)
    pipe = _arm("pipe-0.50", "pipelined", False,
                [_win("uniform", 0.5, 50, pred=100.0, gather_us=130.0),
                 _win("ctrl0", 0.0, 50), _win("ctrl2", 1.0, 50)], hot_frac=0.5)
    hyb = _arm("hyb-0.50", "hybrid", False,
               [_win("uniform", 0.5, 70, syncs=6, graph={"refused": "syncs by design"}),
                _win("ctrl0", 0.0, 60, syncs=2), _win("ctrl2", 1.0, 80, syncs=10)], hot_frac=0.5,
               phase="e4b::p66.cold")
    (tmp_path / "box.json").write_text(json.dumps({"gpu": "SYNTH", "host": {}, "probes": {}, "costs": {}}))
    for a in (ref, pipe, hyb):
        (tmp_path / f"rows_{a['arm']}.json").write_text(json.dumps(a))
    rep = red.census(*red.load_rows(tmp_path))
    fam = rep["families"]["synth"]
    assert fam["reference"] == "ref"
    pu = fam["arms"]["pipe-0.50"]["windows"]["uniform"]
    assert pu["modes"]["eager"]["vs_reference"]["delta_per_token"]["launches"] == pytest.approx(20.0)
    assert pu["transfer"]["ratio"] == pytest.approx(1.3)
    assert fam["arms"]["pipe-0.50"]["cold_test"]["eager"]["launches"]["fixed"] is True
    ht = fam["arms"]["hyb-0.50"]["cold_test"]["eager"]
    assert ht["launches"]["fixed"] is False and ht["launches"]["slope"] == pytest.approx(20.0)
    assert ht["syncs"]["slope"] == pytest.approx(8.0)
    assert "refused" in fam["arms"]["hyb-0.50"]["windows"]["uniform"]["modes"]["graph"]
    # the reference itself is never differenced against itself
    assert "vs_reference" not in fam["arms"]["ref"]["windows"]["uniform"]["modes"]["eager"]


# --------------------------------------------------------------------------- gates and verdicts
L2, T = 2, 4          # a two-layer synthetic family, four tokens per window


def _m(launch_pl, copies_pl=0, syncs_pl=0, fetch=None, extra_kernels=None, graph=False, instrument_syncs=2,
       stray_launches=0, gather_us=0.0, tokens=T):
    """A census mode record: per-LAYER counts turned into window totals the way the census writes them."""
    n = tokens * L2
    if graph:
        api = {"cudaGraphLaunch": tokens, "cudaMemcpyAsync": 2 * tokens}
    else:
        api = {"cudaLaunchKernel": launch_pl * n}
        if copies_pl:
            api["cudaMemcpyAsync"] = copies_pl * n
        if syncs_pl:
            api["cudaStreamSynchronize"] = syncs_pl * n
    glob = dict(api)
    glob["cudaDeviceSynchronize"] = instrument_syncs
    if stray_launches:
        glob["cudaLaunchKernel"] = glob.get("cudaLaunchKernel", 0) + stray_launches
    # device rows mirror the host submissions one for one, as they do on real CUDA (G6): the gather and any
    # named extra rows are carved OUT of the generic compute count, never added on top of it
    carved = (n if gather_us else 0) + sum(c * n for c, _us in (extra_kernels or {}).values())
    kern = {"k_compute": {"count": (launch_pl + copies_pl) * n - carved, "self_device_us": 10.0 * n}}
    if gather_us:
        kern["_gather_rows_addr"] = {"count": n, "self_device_us": gather_us * tokens}
    if graph:
        kern["Memcpy DtoD (Device -> Device)"] = {"count": 2 * tokens, "self_device_us": 1.0}
    for name, (c, us) in (extra_kernels or {}).items():
        kern[name] = {"count": c * n, "self_device_us": us * n}
    ph = {}
    if fetch:
        ph[FETCH] = {"cudaLaunchKernel": fetch[0] * n, "cudaMemcpyAsync": fetch[1] * n,
                     **({"cudaStreamSynchronize": fetch[2] * n} if len(fetch) > 2 and fetch[2] else {})}
    return {"tokens": tokens, "api": api, "api_global": glob, "api_by_phase": ph, "kernels": kern}


def _w(name, cold, eager, graph=None, pred=None, tc=True, gc=True, timing=None, routing=None):
    w = {"name": name, "routing": {"cold_fraction": cold, **(routing or {})}, "modes": {"eager": eager},
         "timing": timing or {}}
    if graph is not None:
        w["modes"]["graph"] = graph
        w["graph_check"] = {"bitwise": gc, "b_rel": 0.0 if gc else 0.5}
    if pred is not None:
        w["predicted_transfer_us_per_token"] = pred
    if tc is not None:
        w["traffic_check"] = {"match": tc}
    return w


def _a(name, path, windows, ref=False, hot=1.0, n_hot=8, fam="qwen", phase=FETCH):
    return {"arm": name, "family": fam, "path": path, "hot_frac": hot, "n_hot": n_hot, "layers": L2, "experts": 8,
            "k": 2, "row_bytes": 1000, "reference": ref, "residency_phase": phase, "capturable": True,
            "windows": windows}


def _family(pipe_hot_launch=18, pipe_cold_launch=15, moving=False, gather_ratio=1.0, bad_sim=False,
            bad_graph=False, stray=0):
    ref = _a("ref", "hybrid", [_w("uniform", 0.0, _m(8, stray_launches=stray), _m(8, graph=True), tc=None)],
             ref=True, phase=None)
    pw = []
    for nm, cf in (("uniform", 0.5), ("ctrl0", 0.0), ("ctrl2", 1.0)):
        launch = pipe_hot_launch + (3 if (moving and cf > 0.5) else 0)
        pw.append(_w(nm, cf, _m(launch, 2, fetch=(8, 2), gather_us=100.0 * gather_ratio),
                     _m(launch, 2, graph=True, gather_us=100.0 * gather_ratio), pred=100.0 if cf else None,
                     tc=not bad_sim, gc=not bad_graph,
                     timing={"graph": {"wall_ms_per_token": 3.0}, "eager": {"wall_ms_per_token": 9.0}}))
    p1 = _a("pipe-1.00", "pipelined", [_w("uniform", 0.0, _m(pipe_hot_launch, 2, fetch=(8, 2)),
                                          _m(pipe_hot_launch, 2, graph=True),
                                          timing={"graph": {"wall_ms_per_token": 2.5},
                                                  "eager": {"wall_ms_per_token": 8.0}})])
    ph = _a("pipe-0.50", "pipelined", pw, hot=0.5, n_hot=4)
    p0 = _a("pipe-0.00", "pipelined", [_w("uniform", 1.0, _m(pipe_cold_launch, 2, fetch=(5, 2),
                                                             gather_us=200.0 * gather_ratio),
                                          _m(pipe_cold_launch, 2, graph=True, gather_us=200.0 * gather_ratio),
                                          pred=200.0)], hot=0.0, n_hot=0)
    ref["windows"][0]["timing"] = {"graph": {"wall_ms_per_token": 2.0}, "eager": {"wall_ms_per_token": 6.0}}
    return [ref, p1, ph, p0]


def _mx():
    guard = {"compare": (1, 1.0), "reduce": (1, 1.0), "Memcpy DtoH (Device -> Pinned)": (1, 1.0)}
    # captured, the guard's 3 rows per layer (2 launches + 1 D2H copy) are not recorded: 38 + 2 - 3 = 35 + 2
    mref = _a("mref", "mxfp4-pinned", [_w("uniform", 0.0, _m(38, 2, syncs_pl=1, extra_kernels=guard),
                                          _m(35, 2, graph=True), tc=None,
                                          routing={"counters_hot_d2d_rows_per_token": 4})],
              ref=True, fam="gptoss")
    mpin = _a("mpin-0.50", "mxfp4-pinned", [_w(n, cf, _m(38, 2, syncs_pl=1, extra_kernels=guard),
                                              _m(35, 2, graph=True), tc=None)
                                           for n, cf in (("uniform", 0.5), ("ctrl1", 0.5))], hot=0.5, fam="gptoss")
    mn = _a("mnvme-0.50", "mxfp4-nvme", [_w(n, cf, _m(37, 5, syncs_pl=4, fetch=(15, 5, 3)), tc=None)
                                        for n, cf in (("uniform", 0.5), ("ctrl0", 0.0), ("ctrl2", 1.0))],
            hot=0.5, fam="gptoss")
    mn["capturable"] = False
    for w in mn["windows"]:
        w["modes"]["graph"] = {"refused": "by design"}
    return [mref, mpin, mn]


def _hyb():
    ref = _a("ref", "hybrid", [_w("uniform", 0.0, _m(8), tc=None)], ref=True, fam="hy", phase=None)
    h = _a("hyb-0.50", "hybrid", [_w("ctrl0", 0.0, _m(41, 0, syncs_pl=4), tc=None),
                                  _w("ctrl1", 0.5, _m(75, 0, syncs_pl=10), tc=None),
                                  _w("ctrl2", 1.0, _m(50, 0, syncs_pl=8), tc=None)],
           hot=0.5, fam="hy", phase="e4b::p66.cold")
    h["capturable"] = False
    return [ref, h]


BOX5090 = {"host": {"gpu": "NVIDIA GeForce RTX 5090, 32607 MiB"}}


def test_every_registered_branch_holds_on_a_conforming_read():
    v = red.verdicts(BOX5090, _family() + _mx() + _hyb())
    assert v["P0"]["ok"], v["P0"]
    for k in ("P1", "P2", "P4", "P5", "P6", "P7"):
        assert v[k]["status"] == "HOLDS", (k, v[k])
    assert v["P2"]["arms"]["pipe-0.00"]["delta_per_layer"] == {"launches": 7, "async_copies": 2, "syncs": 0}
    assert v["P4"]["arms"]["mnvme-0.50"]["syncs_per_layer"] == 4
    d = v["decision"]
    assert d.startswith("P1 ∧ P2") and "P4:" in d and "P6:" in d and "P7:" in d


def test_a_gate_failure_reads_nothing():
    for kw in ({"bad_sim": True}, {"bad_graph": True}, {"stray": 3}):
        v = red.verdicts(BOX5090, _family(**kw))
        assert not v["P0"]["ok"], kw
        assert v["decision"].startswith("¬P0"), kw


def test_the_mxfp4_eager_only_guard_is_parity_not_a_lost_kernel():
    ok = red.gates(_mx())
    assert all(x[1] for x in ok["graph_parity"]), ok["graph_parity"]
    # a graph that runs work the eager submissions did not (or loses some) fails G4 on a pipelined arm
    fam = _family()
    fam[1]["windows"][0]["modes"]["graph"]["kernels"]["compare"] = {"count": 3 * L2 * T, "self_device_us": 1.0}
    assert not all(x[1] for x in red.gates(fam)["graph_parity"])
    # device rows the host never submitted, in an EAGER window, fail G6 instead (never a "dropped record")
    fam = _family()
    fam[1]["windows"][0]["modes"]["eager"]["kernels"]["compare"] = {"count": 99, "self_device_us": 1.0}
    g = red.gates(fam)
    assert all(x[1] for x in g["graph_parity"]) and not all(x[1] for x in g["record_completeness"])


def test_p1_refuted_when_the_pipelined_count_moves():
    v = red.verdicts(BOX5090, _family(moving=True))
    assert v["P1"]["status"] == "REFUTED"
    assert "¬P1" in v["decision"]


def test_p2_refuted_on_a_different_count():
    v = red.verdicts(BOX5090, _family(pipe_hot_launch=19, pipe_cold_launch=16))
    assert v["P1"]["status"] == "HOLDS" and v["P2"]["status"] == "REFUTED"
    assert "¬P2" in v["decision"]


def test_p7_bands_and_the_registered_box():
    assert red.verdicts(BOX5090, _family(gather_ratio=1.4))["P7"]["status"] == "BETWEEN BANDS"
    assert red.verdicts(BOX5090, _family(gather_ratio=2.0))["P7"]["status"] == "REFUTED"
    assert red.verdicts(BOX5090, _family(gather_ratio=0.5))["P7"]["status"] == "REFUTED"
    other = red.verdicts({"host": {"gpu": "NVIDIA RTX A2000 12GB"}}, _family())
    assert other["P7"]["status"].startswith("RECORDED") and other["P3"]["status"].startswith("RECORDED")
    assert other["P1"]["status"] == "HOLDS"          # counts are structural: graded on any box


def test_p6_needs_movement_and_the_registered_syncs():
    fam = _hyb()
    fam[1]["windows"][1]["modes"]["eager"] = _m(75, 0, syncs_pl=9)
    assert red.verdicts(BOX5090, fam)["P6"]["status"] == "REFUTED"


def test_split_device_delta_separates_added_kernels_from_a_slower_shared_one():
    ref = {"tokens": 1, "kernels": {"gemv": {"count": 2, "self_device_us": 10.0}}}
    arm = {"tokens": 1, "kernels": {"gemv": {"count": 2, "self_device_us": 12.0},
                                    "_gather_rows_addr": {"count": 1, "self_device_us": 3.0}}}
    sp = red.split_device_delta(arm, ref)
    assert sp["count_changed_us"] == pytest.approx(3.0) and sp["count_equal_us"] == pytest.approx(2.0)
    assert sp["count_equal_by_kernel"] == {"gemv": pytest.approx(2.0)}


def test_nesting_gate_allows_only_the_instrument_syncs():
    assert red.nesting_gate(_m(8))["ok"]
    assert not red.nesting_gate(_m(8, instrument_syncs=3))["ok"]
    assert not red.nesting_gate(_m(8, stray_launches=1))["ok"]
    assert not red.nesting_gate({"api": {}})["ok"]           # no global record: cannot be checked, refuses


def test_render_md_carries_every_verdict():
    rep = red.census({"host": BOX5090["host"]}, _family() + _mx())
    md = red.render_md(rep, red.verdicts(BOX5090, _family() + _mx()))
    for k in ("P0 (gates)", "P1:", "P2:", "P3:", "P4:", "P5:", "P6:", "P7:", "Decision rule"):
        assert k in md, k


def _with_rfc_arm(graph_ms):
    fam = _family()
    rfc = _a("pipe-0.83", "pipelined", [_w("uniform", 22 / 128, _m(18, 2, fetch=(8, 2)), _m(18, 2, graph=True),
                                          timing={"graph": {"wall_ms_per_token": graph_ms},
                                                  "eager": {"wall_ms_per_token": 9.0}})],
             hot=106 / 128, n_hot=106)
    return fam + [rfc]


@pytest.mark.parametrize("graph_ms,phrase", [(3.0, "FIXED term"), (4.0, "FIXED term"), (5.0, "partial"),
                                             (7.0, "cannot explain")])
def test_p8_reads_the_rfc_matched_arm_only(graph_ms, phrase):
    # ref 2.0, all-hot 2.5 ms/token captured: the fixed tax is 0.5 ms; the RFC arm's residency cost is
    # graph_ms - 2.0, so the shares are 0.5, 0.25 (the edge, inclusive), 1/6 and 0.1 (the other edge, inclusive)
    v = red.verdicts(BOX5090, _with_rfc_arm(graph_ms))
    row = v["P8"]["rows"]["qwen/pipe-0.83/graph"]
    assert row["cold_fraction"] == pytest.approx(22 / 128)
    assert row["fixed_tax_ms"] == pytest.approx(0.5)
    assert phrase in v["decision"], v["decision"]


def test_p8_is_silent_without_the_rfc_matched_arm():
    assert "P8:" not in red.verdicts(BOX5090, _family())["decision"]


def test_record_completeness_bounds_lost_device_records():
    m = _m(8)                                       # 8 launches/layer x 2 layers x 4 tokens = 64 submissions, 64 rows
    assert red.record_completeness(m)["ok"] and red.record_completeness(m)["dropped_records"] == 0
    m["kernels"]["k_compute"]["count"] -= 1        # one lost record: within max(1, 0.2 %)
    rc = red.record_completeness(m)
    assert rc["ok"] and rc["dropped_records"] == 1
    m["kernels"]["k_compute"]["count"] -= 1        # two lost at 64 submissions: over the ceiling
    assert not red.record_completeness(m)["ok"]
    m = _m(8)
    m["kernels"]["k_compute"]["count"] += 1        # MORE device rows than the host submitted is never a drop
    assert not red.record_completeness(m)["ok"]
    big = _m(8, tokens=400)                         # 6400 submissions: the ceiling scales, 12 lost records pass
    big["kernels"]["k_compute"]["count"] -= 12
    assert red.record_completeness(big)["ok"]


def test_eager_fixedness_is_graded_on_host_counts_not_the_lossy_device_view():
    fam = _family()
    w = fam[2]["windows"][1]["modes"]["eager"]      # pipe-0.50 ctrl0: lose one device record, host unchanged
    w["kernels"]["k_compute"]["count"] -= 1
    v = red.verdicts(BOX5090, fam)
    assert v["P0"]["ok"] and v["P1"]["status"] == "HOLDS", v["P1"]


# The dump the A2000 rehearsal's probe printed (torch 2.8.0, CUDA 12.8): aten add, a Triton kernel, a D2D copy and an
# aten mul captured into one graph. The CUDA runtime read the same graph as KERNEL, KERNEL, MEMCPY, KERNEL.
REAL_DOT = r'''digraph dot {
subgraph cluster_1 {
label="graph_1" graph[style="dashed"];
"graph_1_node_0"[style="bold" shape="record" label="{KERNEL
| {ID | 0 (topoId: 3) | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_21CUDAFunctorOnSelf_addIfEESt5arrayIPcLm2EEEEviT0_T1_\<\<\<1,128,0\>\>\>}
| {{node handle | func handle} | {0x000055E7277A5C20 | 0x000055E7270A5A90}}
| {accessPolicyWindow | {base_ptr | num_bytes | hitRatio | hitProp | missProp} | {0x0000000000000000 | 0 | 0.000000 | N | N}}
| {cooperative | 0}
| {priority | 0}
}"];
"graph_1_node_1"[style="bold" shape="record" label="{KERNEL
| {ID | 1 (topoId: 2) | _k\<\<\<1,128,0\>\>\>}
| {{node handle | func handle} | {0x000055E7277A6370 | 0x000055E727662310}}
| {accessPolicyWindow | {base_ptr | num_bytes | hitRatio | hitProp | missProp} | {0x0000000000000000 | 0 | 0.000000 | N | N}}
| {cooperative | 0}
| {priority | 0}
}"];
"graph_1_node_2"[style="solid" shape="record" label="{
MEMCPY
| {{ID | node handle} | {2 (topoId: 1) | 0x000055E7277A6AC0}}
| {kind | DtoD (DEVICE to DEVICE)}
| {{srcPtr | dstPtr} | {pitch | ptr | xsize | ysize | pitch | ptr | xsize | ysize} | {0 | 0x0000000302000000 | 0 | 0 | 0 | 0x0000000302001200 | 0 | 0}}
| {{srcPos | {{x | 0} | {y | 0} | {z | 0}}} | {dstPos | {{x | 0} | {y | 0} | {z | 0}}} | {Extent | {{Width | 4096} | {Height | 1} | {Depth | 1}}}}
}"];
"graph_1_node_3"[style="bold" shape="record" label="{KERNEL
| {ID | 3 (topoId: 0) | _ZN2at6native29vectorized_elementwise_kernelILi4ENS0_13AUnaryFunctorIfffNS0_15binary_internal10MulFunctorIfEEEESt5arrayIPcLm2EEEEviT0_T1_\<\<\<1,128,0\>\>\>}
| {{node handle | func handle} | {0x000055E7277A7210 | 0x000055E727587C30}}
| {accessPolicyWindow | {base_ptr | num_bytes | hitRatio | hitProp | missProp} | {0x0000000000000000 | 0 | 0.000000 | N | N}}
| {cooperative | 0}
| {priority | 0}
}"];
"graph_1_node_0" -> "graph_1_node_1" [headlabel=0];
"graph_1_node_1" -> "graph_1_node_2" [headlabel=0];
"graph_1_node_2" -> "graph_1_node_3" [headlabel=0];
}
}
'''


def test_parse_graph_dot_reads_the_real_dump():
    n = red.parse_graph_dot(REAL_DOT)
    assert n["by_type"] == {"KERNEL": 3, "MEMCPY": 1}, n
    assert n["work"] == 4 and n["nodes"] == 4             # edges and the cluster header are not nodes
    # an unrecognised node is counted, not dropped, so G4 sees it
    odd = red.parse_graph_dot('"g_node_9"[shape="record" label="{SOMETHING_NEW | x}"];\n')
    assert odd["by_type"] == {"UNKNOWN": 1} and odd["work"] == 0


def _graph_nodes(work, dot=None, extra=None):
    bt = {"KERNEL": work - 2 * L2, "MEMCPY": 2 * L2, **(extra or {})}
    g = {"by_type": bt, "work": work, "nodes": sum(bt.values()), "source": "cudaGraphGetNodes"}
    if dot is not None:
        g["dot"] = dot
    return g


def test_g4_on_graph_nodes_needs_the_eager_submissions_and_two_agreeing_readers():
    fam = _family()
    for a in fam:
        for w in a["windows"]:
            if "graph" in w["modes"]:
                e = red.per_token(w["modes"]["eager"])
                sub = int(e["launches"] + e["async_copies"])
                w["graph_nodes"] = _graph_nodes(sub, dot={"by_type": _graph_nodes(sub)["by_type"]})
    g = red.gates(fam)
    assert all(x[1] for x in g["graph_parity"]), g["graph_parity"]
    # the two readers disagree -> G4 fails
    w = fam[1]["windows"][0]
    w["graph_nodes"]["dot"] = {"by_type": {"KERNEL": 1}}
    assert not all(x[1] for x in red.gates(fam)["graph_parity"])
    # a non-work node type (an EVENT_RECORD inside the token) fails even when the work count matches
    w["graph_nodes"] = _graph_nodes(int(red.per_token(w["modes"]["eager"])["launches"] + 2 * L2),
                                    extra={"EVENT_RECORD": 1})
    assert not all(x[1] for x in red.gates(fam)["graph_parity"])


def test_p1_captured_fixedness_reads_the_graph_nodes_not_the_lossy_rows():
    fam = _family()
    for a in fam:
        for w in a["windows"]:
            if "graph" in w["modes"]:
                e = red.per_token(w["modes"]["eager"])
                w["graph_nodes"] = _graph_nodes(int(e["launches"] + e["async_copies"]))
                w["modes"]["graph"]["kernels"]["k_compute"]["count"] -= 1   # a lost replay record: irrelevant now
    v = red.verdicts(BOX5090, fam)
    assert v["P1"]["status"] == "HOLDS", v["P1"]
    assert "graph_nodes.work" in v["P1"]["arms"]["pipe-0.50/graph"]["spreads"]
    fam[2]["windows"][1]["graph_nodes"]["work"] += 1                        # the GRAPH itself changed: refuted
    assert red.verdicts(BOX5090, fam)["P1"]["status"] == "REFUTED"
