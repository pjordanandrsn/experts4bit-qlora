"""bench/train-routing-trace/replay.py, per-layer budgets: three shipped residency rules on hand-computed traces.

Every expected count below was worked by hand from the cited source and agrees with the real code it re-states.
"""
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "bench" / "train-routing-trace"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load("replay")


def run(cls, k, reqs):
    tier = cls(k)
    return [tier.request(r) for r in reqs]


# -- lru_seeded (SGLang paged experts) ---------------------------------------------------------------------------
def test_seeded_lru_starts_holding_experts_0_to_k_minus_1():
    assert run(rp.SeededLRU, 2, [[0, 1], [0], [1]]) == [(2, 0, 0), (1, 0, 0), (1, 0, 0)]


def test_seeded_lru_device_rule_evicts_the_oldest_unstamped_slot_lowest_slot_on_ties():
    # seeded {0: s0, 1: s1, 2: s2}, all last use 0. [5] takes s0 (the lowest slot of the three-way tie), evicting 0;
    # [0] takes s1 (s1 and s2 tie at 0, s0 is at 1), evicting 1; [1] takes s2; [5] is still in s0.
    assert run(rp.SeededLRU, 3, [[5], [0], [1], [5]]) == [(0, 1, 1), (0, 1, 1), (0, 1, 1), (1, 0, 0)]


def test_seeded_lru_differs_from_the_empty_start_pager():
    reqs = [[0], [1], [0], [2], [1]]
    seeded = run(rp.SeededLRU, 2, reqs)
    pager = run(rp.PagerLRU, 2, reqs)
    assert [p for _h, _m, p in seeded] == [0, 0, 0, 1, 1]
    assert [p for _h, _m, p in pager] == [1, 1, 0, 1, 1]


def test_seeded_lru_wave_case_a_step_larger_than_k():
    # K=2, seeded {0, 1}. [0, 1, 2, 3] has 4 entries > K: host waves [[0, 1], [2, 3]] (residents first).
    # Wave 2 evicts 0 then 1; stored back as 2 (s0, newest-1) and 3 (s1, newest). Then device steps:
    # [2] hits; [0] evicts 3 (s1, the older stamp); [3] evicts 2 (s0).
    got = run(rp.SeededLRU, 2, [[0, 1, 2, 3], [2], [0], [3]])
    assert got == [(2, 2, 2), (1, 0, 0), (0, 1, 1), (0, 1, 1)]


def test_seeded_lru_wave_case_with_more_residents_than_one_wave():
    # K=2, seeded {0, 1}; [0, 1, 5] -> waves [[0, 1], [5]]; 5 evicts 0 (the LRU head after wave 1 refreshed 0 then 1).
    tier = rp.SeededLRU(2)
    assert tier.request([0, 1, 5]) == (2, 1, 1)
    assert sorted(tier.expert_slot) == [1, 5]
    assert tier.request([1]) == (1, 0, 0)


def test_seeded_lru_waves_put_the_residents_first():
    # [5] evicts 0 -> {1, 5}. [1, 2, 5] (3 > K): residents first gives waves [[1, 5], [2]], one page-in; in sorted
    # order [[1, 2], [5]] would page 2 in over 5 and then page 5 back.
    assert run(rp.SeededLRU, 2, [[5], [1, 2, 5]]) == [(0, 1, 1), (2, 1, 1)]


# -- lru (the vLLM expert pager's VRAM tier) ---------------------------------------------------------------------
def test_pager_lru_starts_empty_and_takes_the_oldest_slot():
    # step 1 [0] -> s0; [1] -> s1; [0] hit; [2] evicts 1 (s1, last use 2 < 3); [1] evicts 0 (s0).
    assert run(rp.PagerLRU, 2, [[0], [1], [0], [2], [1]]) == [(0, 1, 1), (0, 1, 1), (1, 0, 0), (0, 1, 1), (0, 1, 1)]


def test_pager_lru_never_evicts_a_slot_the_step_hits():
    tier = rp.PagerLRU(2)
    tier.request([0])
    tier.request([1])                             # 0 is now the oldest
    assert tier.request([0, 7]) == (1, 1, 1)      # 7 must take 1's slot, not the older 0's
    assert sorted(tier.slot_of) == [0, 7]


def test_pager_lru_refuses_more_entries_than_k():
    with pytest.raises(rp.RequestRefused):
        rp.PagerLRU(2).request([0, 1, 2])


# -- lfu_protect (e4b's ColdTier) --------------------------------------------------------------------------------
def test_coldtier_lfu_breaks_frequency_ties_by_least_recent_use():
    # [0] -> s1 (free list pops the highest slot), [1] -> s0; both freq 1. [2] evicts 0 (last use 1 < 2), so [1] hits.
    assert run(rp.ColdTierLFU, 2, [[0], [1], [2], [1]]) == [(0, 1, 1), (0, 1, 1), (0, 1, 1), (1, 0, 0)]


def test_coldtier_lfu_breaks_full_ties_by_slot_and_fills_the_highest_free_slot_first():
    # [0, 1] in one request: same frequency and last use. 0 takes s1, 1 takes s0; [2] evicts the lowest slot, 1.
    assert run(rp.ColdTierLFU, 2, [[0, 1], [2], [0], [1]]) == [(0, 2, 2), (0, 1, 1), (1, 0, 0), (0, 1, 1)]


def test_coldtier_counts_the_request_before_choosing_a_victim():
    # 6 (freq 2, older) and 5 (freq 2, newer). [4, 6] bumps 6 to 3 first, so 4 evicts 5, not 6.
    assert run(rp.ColdTierLFU, 2, [[6], [6], [5], [5], [4, 6]])[-1] == (1, 1, 1)


def test_coldtier_lfu_keeps_the_frequent_row():
    got = run(rp.ColdTierLFU, 2, [[9], [9], [9], [1], [2], [3], [9]])
    assert got[-1] == (1, 0, 0)


def test_coldtier_protection_case_a_request_never_evicts_a_row_it_already_resolved():
    # 9 (freq 3) fills s1. [0, 1]: 0 takes free s0; 1 must evict 9, NOT the just-placed 0 (freq 1 < 3). [0] hits.
    tier = rp.ColdTierLFU(2)
    assert [tier.request(r) for r in ([9], [9], [9])] == [(0, 1, 1), (1, 0, 0), (1, 0, 0)]
    assert tier.request([0, 1]) == (0, 2, 2)
    assert sorted(tier.slot_of) == [0, 1]
    assert tier.request([0]) == (1, 0, 0)


def test_coldtier_protection_covers_only_resolved_rows_a_resident_row_not_yet_reached_can_be_evicted():
    # 5 (freq 3) in s1, 6 (freq 1) in s0. [4, 6]: freqs -> 4:1, 6:2. 4 misses first and takes the lowest
    # (freq, last, slot) = 6 (2 < 3); then 6 misses and evicts 5. One miss at the start, two page-ins.
    tier = rp.ColdTierLFU(2)
    for r in ([5], [5], [5], [6]):
        tier.request(r)
    assert tier.request([4, 6]) == (1, 1, 2)
    assert sorted(tier.slot_of) == [4, 6]


def test_coldtier_refuses_more_unique_rows_than_k_and_counts_repeats_once():
    with pytest.raises(rp.RequestRefused):
        rp.ColdTierLFU(2).request([0, 1, 2])
    assert rp.ColdTierLFU(2).request([3, 3, 4]) == (0, 2, 2)


# -- the per-layer driver ------------------------------------------------------------------------------------------
def test_budgets_are_per_layer_not_a_shared_pool():
    # layer 0 cycles 3 experts through K=2 (LRU misses every time); layer 1 needs 1. A shared pool of 4 rows would
    # let layer 0 borrow layer 1's spare slot and hit on the second cycle.
    reqs = []
    for step, e in enumerate([0, 1, 2, 0, 1, 2]):
        reqs += [(step, 0, [e]), (step, 1, [9])]
    res = rp.simulate_per_layer(reqs, 2, "lru", fit_steps=0)
    assert res["layers"][0]["misses"] == 6 and res["layers"][0]["page_ins"] == 6
    assert res["layers"][1]["misses"] == 1
    assert res["total"]["misses"] == 7 and res["tokens"] == 6


def test_driver_runs_warm_through_the_fit_window_and_reports_per_token_rates():
    reqs = [(0, 0, [5]), (1, 0, [5]), (2, 0, [5, 6]), (3, 0, [7])]
    res = rp.simulate_per_layer(reqs, 2, "lru", fit_steps=2)
    t = res["total"]
    assert res["tokens"] == 2
    assert (t["requests"], t["routed"], t["hits"], t["misses"], t["page_ins"]) == (2, 3, 1, 2, 2)
    assert t["hit_rate"] == pytest.approx(1 / 3)
    assert t["per_token"] == {"hits": 0.5, "misses": 1.0, "page_ins": 1.0}
    assert res["layers"][0] == t


def test_driver_reports_page_ins_apart_from_misses():
    reqs = [(s, 0, r) for s, r in enumerate([[5], [5], [5], [6], [4, 6]])]
    t = rp.simulate_per_layer(reqs, 2, "lfu_protect", fit_steps=0)["total"]
    # per request (hits, misses, page-ins): (0,1,1) (1,0,0) (1,0,0) (0,1,1) (1,1,2): [4, 6] evicts 6 before reaching it
    assert (t["hits"], t["misses"], t["page_ins"]) == (3, 3, 4)


def test_replay_per_layer_refuses_k_below_top_k_and_prices_page_ins():
    reqs = [(0, 0, [0, 1]), (1, 0, [2, 3])]
    with pytest.raises(rp.RequestRefused):
        rp.replay_per_layer(reqs, [1], fit_steps=0, row_bytes=10, top_k=2)
    with pytest.raises(rp.RequestRefused):                             # refused up front, not by a request
        rp.replay_per_layer([(0, 0, [0])], [1], fit_steps=0, row_bytes=10, top_k=2)
    out = rp.replay_per_layer(reqs, [2], fit_steps=0, row_bytes=10, top_k=2)
    cells = {c["policy"]: c for c in out["cells"]}
    assert set(cells) == {"lru_seeded", "lru", "lfu_protect"}
    assert cells["lru_seeded"]["total"]["page_ins"] == 2               # seeded {0, 1}: only [2, 3] pages
    assert cells["lru"]["total"]["page_ins"] == 4
    assert cells["lru"]["page_in_bytes_per_token"] == 20


def _trace(tmp_path, steps):
    p = tmp_path / "t.jsonl"
    meta = {"meta": {"top_k": 2, "n_experts": 16, "layers": 11}}
    p.write_text("\n".join([json.dumps(meta)] + [json.dumps({"step": s, "routed": r}) for s, r in steps]) + "\n")
    return str(p)


def test_decode_requests_orders_layers_numerically(tmp_path):
    path = _trace(tmp_path, [(0, {"10": [3, 1], "9": [2, 0], "0": [5, 4]})])
    meta, reqs = rp.decode_requests(path)
    assert meta["top_k"] == 2
    assert reqs == [(0, 0, [4, 5]), (0, 9, [0, 2]), (0, 10, [1, 3])]


def test_cli_per_layer_mode(tmp_path, capsys):
    path = _trace(tmp_path, [(0, {"0": [0, 1]}), (1, {"0": [0, 2]})])
    out = tmp_path / "o.json"
    assert rp.main(["--decode", path, "--fit-steps", "0", "--per-layer-k", "2,3", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert [(c["k"], c["policy"]) for c in doc["cells"]] == [
        (2, "lru_seeded"), (2, "lru"), (2, "lfu_protect"), (3, "lru_seeded"), (3, "lru"), (3, "lfu_protect")]
    assert doc["top_k"] == 2 and doc["source"] == path
    with pytest.raises(SystemExit):
        rp.main(["--trace", path, "--expect-mb", "1", "--per-layer-k", "2", "--out", str(out)])
