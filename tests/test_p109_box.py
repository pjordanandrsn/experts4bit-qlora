"""Lane P109's box on CPU, end to end, with the GPU construction stubbed (e4b#770).

No local card runs the fp8 paged KV, so the box's own code -- the default-server refusals, the D arm's grouping switch,
the two workloads, the receipt -- is exercised here over a real ``ContinuousScheduler`` and a scripted runner, and the
five receipts go through the registered reducer. Speeds are NOT asserted (sleep timing on a CI runner is not a
measurement); the receipt's shape, the engagement fields and the token identity are.
"""
import importlib.util
import json
import pathlib
import time

import pytest
import torch

from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines import hot_residency, paged_runner
from experts4bit_qlora.engines.scheduler import ContinuousScheduler

LANE = pathlib.Path(__file__).resolve().parents[1] / "bench" / "p109"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Runner:
    """Each request's tokens are a function of its prompt and position only, so every arm emits the same tokens."""

    def __init__(self):
        self.prompts, self.cursor = {}, {}
        self.graph_stats = {"1": {"replays": 0}}

    def bind(self, rid, slot, prompt):
        self.prompts[rid], self.cursor[rid] = tuple(prompt), 0

    def _next(self, rid):
        i = self.cursor[rid]
        self.cursor[rid] += 1
        return (sum(self.prompts[rid]) + 7 * i) % 997

    def run_prefill(self, chunks):
        return {rid: self._next(rid) for rid, start, n in chunks if start + n >= len(self.prompts[rid])}

    def run_decode(self, rids):
        time.sleep(0.002)                      # so LONG is measurably slower than SHORT; speeds are still not asserted
        return {r: self._next(r) for r in rids}

    def free_slot(self, rid):
        pass


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    def build_engine(cfg):
        runner = _Runner()
        sched = ContinuousScheduler(runner=runner, max_seqs=cfg.max_seqs, kv_slots=cfg.max_seqs,
                                    chunk_tokens=cfg.chunk_tokens, max_prefill_tokens_per_step=cfg.prefill_budget)
        grouping = serve_paged._batched_graph_grouping(cfg)
        # resolved through the class at call time, as build_engine's runner.enable_decode_graphs is, so the P arm's
        # capture=False wrapper (installed by the box on the class) is what answers
        status = paged_runner.PagedModelRunner.enable_decode_graphs(runner, cfg.buckets) if cfg.graphs else None
        return serve_paged.EngineParts(scheduler=sched, tokenizer=None, eos_ids=frozenset(), runner=runner,
                                       info={"graph_status": status, "grouping": grouping, "moe_layers": 1})
    monkeypatch.setattr(serve_paged, "build_engine", build_engine)
    monkeypatch.setattr(paged_runner.PagedModelRunner, "enable_decode_graphs",
                        lambda self, buckets=(1, 2, 4, 8, 16), *, capture=True, **kw:
                        {b: ("graph" if capture else "eager: capture=False") for b in buckets})
    for name, val in (("synchronize", lambda *a, **k: None), ("mem_get_info", lambda *a, **k: (0, 0)),
                      ("max_memory_allocated", lambda *a, **k: 0), ("memory_allocated", lambda *a, **k: 0),
                      ("memory_reserved", lambda *a, **k: 0)):
        monkeypatch.setattr(torch.cuda, name, val)
    monkeypatch.setattr(hot_residency, "DEVICE_GROUPING", [False])
    monkeypatch.setattr(hot_residency, "FORCE_SINGLETON_GROUPS", [False])
    box = _load("p109_box")
    rows = [[(k * 31 + i) % 500 + 1 for i in range(box.PROMPT)] for k in range(box.ROWS)]
    pf = tmp_path / "prompts.json"
    pf.write_text(json.dumps({"rows": rows, "prompts_sha256": box.digest(rows)}))
    for k, v in {"E4B_PAGED_MODEL": "Qwen/Qwen3-30B-A3B", "E4B_PAGED_REVISION": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
                 "E4B_PAGED_ARENA": "/a", "E4B_PAGED_CALIB": "/c", "E4B_SHA": "a" * 40,
                 "GNF4_SHA": "51a49166ae7bc1a0f84188b7b5d1f42ecbc37e00"}.items():
        monkeypatch.setenv(k, v)
    for k in ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_BUCKETS", "E4B_PAGED_PLACEMENT"):
        monkeypatch.delenv(k, raising=False)
    return box, pf, monkeypatch, tmp_path


def _arm(box, pf, mp, tmp, tag):
    arm = tag[0]
    mp.setenv("P109_ARM", arm)
    if arm in ("G", "P"):
        mp.setenv("E4B_PAGED_GRAPHS", "1")
    else:
        mp.delenv("E4B_PAGED_GRAPHS", raising=False)
    hot_residency.DEVICE_GROUPING[0] = False                    # each arm is a fresh process on the box
    hot_residency.FORCE_SINGLETON_GROUPS[0] = False
    out = tmp / f"arm_{tag}.json"
    assert box.main(["--prompts", str(pf), "--out", str(out), "--tag", tag, "--short", "2", "--long", "12", "--reps", "2"]) == 0
    return json.loads(out.read_text())


def test_the_six_arms_record_their_engagement_and_reduce(stubbed):
    box, pf, mp, tmp = stubbed
    recs = {t: _arm(box, pf, mp, tmp, t) for t in ("E1", "G1", "G2", "E2", "D1", "P1")}
    assert recs["G1"]["graph_status"] == {str(b): "graph" for b in (1, 2, 4, 8, 16)}
    assert recs["P1"]["graph_status"] == {str(b): "eager: capture=False" for b in (1, 2, 4, 8, 16)}   # Amendment 2
    assert recs["E1"]["graph_status"] is None and recs["D1"]["graph_status"] is None
    assert [recs[t]["grouping_flags_at_run"]["device_grouping"] for t in ("E1", "G1", "D1", "P1")] == [False, True, True, True]
    w16 = recs["E1"]["workloads"]["W16"]
    assert w16["batch"] == 16 and len(w16["tokens"]["12"]) == 16 and all(len(r) == 12 for r in w16["tokens"]["12"])
    assert len(recs["E1"]["workloads"]["W1"]["tokens"]["2"]) == 1 and len(w16["rep_digests"]["12"]) == 2
    red = _load("p109_reduce")
    v = red.reduce(recs, "a" * 40)
    # timing-dependent verdicts are not asserted; the one VOID a CI runner's clock can cause is a void slope
    assert v["verdict"] != "FUNCTION_FAIL" and (v["verdict"] != "VOID" or v["reasons"] == ["a decode slope is void"]), v
    if "report" in v:
        assert v["report"]["E1_eq_E2"] and v["report"]["E_vs_G_W16_rows_identical"] == 16 and v["report"]["G_eq_D"]
    assert all(recs[t]["workloads"][w]["tokens"] == recs["P1"]["workloads"][w]["tokens"] for t in ("G1", "G2", "D1") for w in ("W16", "W1"))


def test_the_box_refuses_anything_but_the_default_server(stubbed):
    box, pf, mp, tmp = stubbed
    mp.setenv("E4B_PAGED_MAX_SEQS", "8")
    mp.setenv("P109_ARM", "E")
    with pytest.raises(SystemExit, match="not the default server"):
        box.main(["--prompts", str(pf), "--out", str(tmp / "x.json")])


def test_the_box_refuses_an_arm_whose_switch_disagrees(stubbed):
    box, pf, mp, tmp = stubbed
    mp.setenv("P109_ARM", "E")
    mp.setenv("E4B_PAGED_GRAPHS", "1")
    with pytest.raises(SystemExit, match="arm E with E4B_PAGED_GRAPHS"):
        box.main(["--prompts", str(pf), "--out", str(tmp / "x.json")])


def test_the_box_refuses_prompts_that_do_not_match_their_digest(stubbed, tmp_path):
    box, pf, mp, tmp = stubbed
    d = json.loads(pf.read_text())
    d["rows"][0][0] += 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(d))
    mp.setenv("P109_ARM", "E")
    with pytest.raises(SystemExit, match="digest"):
        box.main(["--prompts", str(bad), "--out", str(tmp / "x.json")])


def test_an_unknown_arm_is_refused(stubbed):
    box, pf, mp, tmp = stubbed
    mp.setenv("P109_ARM", "X")
    with pytest.raises(SystemExit, match="P109_ARM"):
        box.main(["--prompts", str(pf), "--out", str(tmp / "x.json")])


def test_the_box_never_imports_torch_at_module_level():
    src = (LANE / "p109_box.py").read_text()
    head = src[:src.index("def digest")]
    assert "import torch" not in head
