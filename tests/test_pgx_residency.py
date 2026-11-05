"""PGX residency instrument (bench/pgx): hand-computed replays, the pager log reader, and the SGLang dump hook's
wiring on a stub package with SGLang's module names. CPU only, stdlib only."""
import importlib
import json
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGX = os.path.join(ROOT, "bench", "pgx")
sys.path.insert(0, PGX)
pr = importlib.import_module("pgx_residency")


def _row(p, call, layer, ids, decode=True, host=None):
    return {"pass": p, "call": call, "layer": layer, "tokens": len(ids), "ids": ids, "decode": decode,
            "host_loads": host}


# --------------------------------------------------------------- SGLang replay, hand-computed -----------------------
def test_device_rule_hand_computed():
    # K=2, slots seeded with experts 0 and 1. One layer, B=1 decode, top_k=1.
    # call 1 [0]: hit. call 2 [2]: miss; the victim is the least recently used slot not stamped this step: slot 1
    # (expert 1, last use 0) -> loads 1. call 3 [1]: miss; victim slot 0 (expert 0, last use 1 < slot 1's 2) -> 1.
    rows = [_row("w", 1, 0, [[0]]), _row("w", 2, 0, [[2]]), _row("w", 3, 0, [[1]])]
    s = pr.sglang_residency(rows, k=2, scored_pass="w")
    assert (s["hits"], s["misses"], s["page_ins"], s["tokens"]) == (1, 2, 2, 3)
    assert s["hit_rate"] == pytest.approx(1 / 3)


def test_wave_case_hand_computed():
    # K=2, one call with 3 distinct experts (entries 3 > K): the host wave plan. Residents first: [0, 1], then [2].
    # Wave 1 = [0, 1]: both resident, no load. Wave 2 = [2]: evicts the LRU head (expert 0) -> 1 load.
    rows = [_row("w", 1, 0, [[0, 1, 2]], decode=False)]
    s = pr.sglang_residency(rows, k=2, scored_pass="w", decode_only=False)
    assert (s["hits"], s["misses"], s["page_ins"]) == (2, 1, 1)


def test_unscored_pass_warms_the_state_but_is_not_tallied():
    rows = [_row("cold", 1, 0, [[5]]), _row("warm", 2, 0, [[5]])]
    s = pr.sglang_residency(rows, k=2, scored_pass="warm")
    assert (s["hits"], s["misses"], s["calls"]) == (1, 0, 1)


def test_prefill_calls_update_state_but_decode_only_tallies_decode():
    rows = [_row("w", 1, 0, [[7], [8], [9]], decode=False), _row("w", 2, 0, [[9]])]
    s = pr.sglang_residency(rows, k=2, scored_pass="w")
    assert (s["calls"], s["hits"], s["tokens"]) == (1, 1, 1)


def test_padding_ids_are_dropped_as_the_executor_drops_them():
    rows = [_row("w", 1, 0, [[0, -1]])]
    s = pr.sglang_residency(rows, k=2, scored_pass="w")
    assert (s["hits"], s["misses"]) == (1, 0)


def test_tokens_counted_once_per_step_whatever_layer_comes_first():
    rows = [_row("w", 1, 4, [[0]]), _row("w", 2, 3, [[0]]), _row("w", 3, 4, [[1]]), _row("w", 4, 3, [[1]])]
    assert pr.sglang_residency(rows, k=2, scored_pass="w")["tokens"] == 2


def test_host_path_check_flags_a_wrong_load_count():
    ok = [_row("w", 1, 0, [[2]], host=1), _row("w", 2, 0, [[2]], host=0)]
    assert pr.host_path_check(ok, k=2)["mismatched"] == 0
    bad = [_row("w", 1, 0, [[2]], host=0)]
    r = pr.host_path_check(bad, k=2)
    assert r["mismatched"] == 1 and r["first"]["replay"] == 1


def test_read_dump_refuses_a_malformed_row(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"pass": "w", "call": 1, "layer": 0, "ids": [1, 2]}) + "\n")
    with pytest.raises(ValueError, match="tokens x top_k"):
        pr.read_dump(str(p))


# --------------------------------------------------------------- the pager's log ------------------------------------
FMT = "vllm-expert-pager %s: vram hit %.1f%% (%d/%d), ram hit %.1f%% (%d/%d)"


def _line(layer, vh, vt, rh, rt):
    return FMT % (layer, 100.0 * vh / vt if vt else 0.0, vh, vt, 100.0 * rh / rt if rt else 0.0, rh, rt)


def test_pager_pass_counts_differences_the_last_snapshot_per_layer():
    before = "\n".join([_line("model.layers.0.mlp.experts", 1, 2, 2, 2),
                        _line("model.layers.0.mlp.experts", 3, 5, 5, 5),      # last for layer 0 wins
                        _line("model.layers.1.mlp.experts", 4, 4, 4, 4)])
    after = "\n".join([_line("model.layers.0.mlp.experts", 10, 15, 15, 15),
                       _line("model.layers.1.mlp.experts", 9, 12, 12, 12)])
    r = pr.pager_pass_counts(before, after)
    assert (r["vram_hits"], r["vram_misses"]) == (7 + 5, 3 + 3)
    assert r["hit_rate"] == pytest.approx(12 / 18)


def test_pager_pass_counts_fails_closed():
    a = _line("model.layers.0.mlp.experts", 3, 5, 5, 5)
    with pytest.raises(ValueError, match="layers differ"):
        pr.pager_pass_counts(a, _line("model.layers.1.mlp.experts", 3, 5, 5, 5))
    with pytest.raises(ValueError, match="went down"):
        pr.pager_pass_counts(a, _line("model.layers.0.mlp.experts", 2, 5, 5, 5))
    with pytest.raises(ValueError, match="layers differ"):
        pr.pager_pass_counts("", "")


def test_pager_giveups_reads_the_latest_line():
    log = ("vllm-expert-pager ssd: 3 fetches, 3 reads, 0.010 s, 3.33 ms/fetch, 0 re-publishes, 0 give-ups, 0 missed\n"
           "vllm-expert-pager ssd: 9 fetches, 9 reads, 0.030 s, 3.33 ms/fetch, 1 re-publishes, 2 give-ups, 0 missed\n")
    assert pr.pager_giveups(log) == 2 and pr.pager_giveups("") == 0


# --------------------------------------------------------------- the SGLang dump hook's wiring ----------------------
_STUB = {
    "sglang/__init__.py": "",
    "sglang/srt/__init__.py": "",
    "sglang/srt/layers/__init__.py": "",
    "sglang/srt/layers/moe/__init__.py": "",
    "sglang/srt/layers/moe/paged_experts/__init__.py": "",
    # the executor imports plan_waves BY NAME, as SGLang's does (executor.py:16-19)
    "sglang/srt/layers/moe/paged_experts/residency.py": """
        from collections import OrderedDict
        class Wave:
            def __init__(self, loads): self.loads = loads
        class Policy:
            def __init__(self, k): self.k, self.res = k, OrderedDict((e, e) for e in range(k))
        def plan_waves(policy, distinct):
            loads = []
            for e in distinct:
                if e in policy.res: policy.res.move_to_end(e)
            for e in distinct:
                if e not in policy.res:
                    _, s = policy.res.popitem(last=False); policy.res[e] = s; loads.append((e, s))
            return [Wave(loads)]
    """,
    "sglang/srt/layers/moe/paged_experts/executor.py": """
        from sglang.srt.layers.moe.paged_experts.residency import plan_waves
        def run(policy, ids):
            return plan_waves(policy, sorted({e for row in ids.rows for e in row if e >= 0}))
    """,
    "sglang/srt/layers/moe/paged_experts/method.py": """
        from sglang.srt.layers.moe.paged_experts import executor
        from sglang.srt.layers.moe.paged_experts.residency import Policy
        class PagedExpertsMoEMethod:
            def __init__(self, k): self.policy = Policy(k)
            def apply(self, layer, dispatch_output):
                ids = dispatch_output.topk_output.topk_ids
                if len(ids.rows) * len(ids.rows[0]) <= 1:   # 'device path': no host plan
                    return "device"
                executor.run(self.policy, ids)
                return "host"
    """,
}

_DRIVER = """
import json, os, sys, types
from sglang.srt.layers.moe.paged_experts.method import PagedExpertsMoEMethod
class Ids:
    def __init__(self, rows): self.rows = rows
    shape = property(lambda s: (len(s.rows), len(s.rows[0])))
    def reshape(self, *a): return self
    def tolist(self): return self.rows
def call(m, layer_id, rows):
    d = types.SimpleNamespace(topk_output=types.SimpleNamespace(topk_ids=Ids(rows)))
    return m.apply(types.SimpleNamespace(layer_id=layer_id), d)
m0, m1 = PagedExpertsMoEMethod(2), PagedExpertsMoEMethod(2)
open(os.path.join(os.environ["PGX_ROUTE_DUMP"], "PASS"), "w").write("warm")
print(call(m0, 0, [[2, 0]]), call(m1, 1, [[0]]), call(m0, 0, [[3, 2]]))
"""


def _stub_tree(tmp_path):
    for rel, src in _STUB.items():
        p = tmp_path / "site" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    return tmp_path / "site"


def test_hook_dumps_every_call_with_its_host_loads(tmp_path):
    site = _stub_tree(tmp_path)
    dump = tmp_path / "dump"
    dump.mkdir()
    env = dict(os.environ, PGX_ROUTE_DUMP=str(dump),
               PYTHONPATH=os.pathsep.join([os.path.join(PGX, "hook"), str(site)]))
    out = subprocess.run([sys.executable, "-c", "import usercustomize\n" + _DRIVER], env=env, capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ["host", "device", "host"]
    rows = []
    for name in os.listdir(dump):
        if name.startswith("route-"):
            rows += pr.read_dump(str(dump / name))
    assert [(r["layer"], r["ids"], r["host_loads"], r["pass"]) for r in rows] == [
        (0, [[2, 0]], 1, "warm"),       # expert 2 evicts the LRU head (expert 1)
        (1, [[0]], None, "warm"),       # the device path: no host plan, no host_loads
        (0, [[3, 2]], 1, "warm"),       # 2 resident, 3 evicts expert 0
    ]
    assert pr.host_path_check(rows, k=2)["mismatched"] == 0


def test_hook_is_inert_without_the_env(tmp_path):
    site = _stub_tree(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "PGX_ROUTE_DUMP"}
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(PGX, "hook"), str(site)])
    code = ("import usercustomize, sys\n"
            "from sglang.srt.layers.moe.paged_experts import executor\n"
            "print(executor.plan_waves.__module__)")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.strip() == "sglang.srt.layers.moe.paged_experts.residency"


def test_hook_fails_loud_when_the_method_class_is_missing(tmp_path):
    site = _stub_tree(tmp_path)
    (site / "sglang/srt/layers/moe/paged_experts/method.py").write_text("X = 1\n")
    env = dict(os.environ, PGX_ROUTE_DUMP=str(tmp_path / "d"),
               PYTHONPATH=os.pathsep.join([os.path.join(PGX, "hook"), str(site)]))
    r = subprocess.run([sys.executable, "-c", "import usercustomize\nimport sglang.srt.layers.moe.paged_experts.method"],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "PagedExpertsMoEMethod.apply not found" in r.stderr
