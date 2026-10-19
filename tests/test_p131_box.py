# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P131's measurement (``bench/p131/p131_box.py``) on CPU, before any GPU is rented.

As P130's test, the RUNNERS are stood in:
- ``_PrefillStandIn`` has the prefill-graph surface the box drives. Each eager forward bumps the engine's own counters as
  the int4 K19 route would under the release's defaults: ``ROUTE_SEEN`` once a layer, ``K19_DISPATCH_SEEN`` once a layer
  with the table ``split`` iff ``E4B_K19_PREFILL_BIG_M`` is 32 or 64 AND the call routes more than 512 rows (a T-token
  prefill here routes 4,096; the box's decode check routes 512), dispatch ``lean`` (P130's default), and
  ``PREFILL_FOLD_SEEN`` by the fold table on prefill-sized calls (P1's default on ``qwen3_moe``);
- ``_StandIn`` is P117's decode runner, as P130's test stands it in.
What it checks is the box's bookkeeping: three captures with their own deltas, the decode check per arm, FUNCTION per
tall arm and the m0 digests, every timed replay in the rotated order, the eager forwards, the knob and grouping
restored, refused captures; and Phase B's arms and subjects in one process, scored against R held in memory.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p131", "p130", "p117", "p108", "p97")]
ENV = "E4B_K19_PREFILL_BIG_M"


def _load():
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("p131_box", LANES[0] / "p131_box.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
                         max_position_embeddings=256, decoder_sparse_step=1, mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


T, NW, DEC = 24, 3, 8


class _Graph:
    def __init__(self, runner, pg, arm):
        self.runner, self.pg, self.arm = runner, pg, arm

    def replay(self):                               # no Python counters: a replay runs none
        with torch.no_grad():
            o = type(self.runner.model).forward(self.runner.model, input_ids=self.pg["ids"])
        self.pg["logits"].copy_(o.logits)
        for lay, (k, v) in self.pg["staged"].items():
            k.copy_(o.logits[0, :, :4] * (lay + 1))
            v.copy_(o.logits[0, :, 4:8] + (1e-3 if (self.arm == self.runner.perturb and lay == 1) else 0.0))


class _PrefillStandIn:
    _PG_KEY = -1

    def __init__(self, model, perturb=None, refuse=(), split_decode=False):
        self.model, self.perturb, self.refuse, self.split_decode = model, perturb, refuse, split_decode
        self.ctx = types.SimpleNamespace(drop=lambda slot: None)
        self._prefill_graph = None
        self.layers = int(model.config.num_hidden_layers)

    def _arm(self):
        return {"0": "m0", "32": "m32", "64": "m64"}[os.environ.get(ENV, "0")]

    def _prefill_forward(self, ids, positions):
        from experts4bit_qlora.engines import glue_fuse
        from experts4bit_qlora.engines import hot_residency as hr
        rows = 4096 if ids.shape[1] == T else 512
        split = os.environ.get(ENV, "0") != "0" and (rows > 512 or self.split_decode)
        for _ in range(self.layers):
            hr._seen_route("int4_k19", rows)
            hr._seen_k19_dispatch("split" if split else "chained", "lean", rows)
            if ids.shape[1] == T:
                glue_fuse._seen_prefill("layer")
                glue_fuse._seen_prefill("attention")
                glue_fuse._seen_prefill("norm")
        if ids.shape[1] == T:
            glue_fuse._seen_prefill("norm")              # the final norm
        with torch.no_grad():
            return self.model(input_ids=ids)

    def _prefill_scope(self):
        return lambda: None

    def enable_prefill_graph(self, T_):
        from experts4bit_qlora.engines.paged_runner import PrefillGraphRefused
        arm = self._arm()
        ids = torch.randint(0, 256, (1, T_), generator=torch.Generator().manual_seed(1689))
        pos = torch.arange(T_)[None]
        if arm in self.refuse:
            for _ in range(3):
                self._prefill_forward(ids, pos)
            raise PrefillGraphRefused(f"the {T_}-token prefill forward did not capture (RuntimeError: a sync)")
        for _ in range(5):
            out = self._prefill_forward(ids, pos)
        staged = {lay: (torch.zeros(T_, 4), torch.zeros(T_, 4)) for lay in range(self.layers)}
        pg = {"T": T_, "ids": ids.clone(), "logits": out.logits.clone(), "staged": staged}
        pg["graph"] = _Graph(self, pg, arm)
        self._prefill_graph = pg
        return {"status": "on", "T": T_, "replays": 0, "pool_mib": 0}

    def disable_prefill_graph(self):
        self._prefill_graph = None


@pytest.fixture(scope="module")
def box():
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    return _load()


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("E4B_K19_PREFILL_BIG_M", "E4B_FUSE_PREFILL_GLUE", "E4B_PREFILL_LEAN_DISPATCH"):
        monkeypatch.delenv(k, raising=False)


def _prompts():
    g = torch.Generator().manual_seed(2)
    return [torch.randint(0, 256, (T + 5,), generator=g).tolist() for _ in range(NW)]


def _speed(box, runner, prompts=None):
    return box.speed(runner.model, prompts or _prompts(), T, "cpu", rounds=2, warm=1, eager_rounds=1, eager_windows=2,
                     decode_tokens=DEC, runner=runner)


def test_each_capture_records_its_own_table(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny()))
    L = 2
    folds = {k: 5 * v for k, v in {"norm": L + 1, "layer": L, "attention": L}.items()}
    for g, table in (("m0", "chained"), ("m32", "split"), ("m64", "split")):
        gr = rec["graphs"][g]
        assert gr["status"] == "on" and gr["T"] == T and gr["forwards"] == 5, gr
        assert gr["seen"] == {"route": {"int4_k19|gt256": 5 * L}, "k19": {f"{table}|lean|gt256": 5 * L}, "folds": folds}


def test_the_decode_check_reads_every_arm_on_the_chained_table(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny()))
    dc = rec["decode_check"]
    assert dc["tokens"] == DEC and set(dc["seen"]) == {"m0", "m32", "m64"}
    assert all(v["k19"] == {"chained|lean|gt256": 2} and v["folds"] == {} for v in dc["seen"].values())
    bad = _speed(box, _PrefillStandIn(_tiny(), split_decode=True))["decode_check"]["seen"]
    assert bad["m0"]["k19"] == {"chained|lean|gt256": 2} and bad["m64"]["k19"] == {"split|lean|gt256": 2}


def test_function_digests_and_every_timed_replay(box, clean_env):
    model = _tiny()
    rec = _speed(box, _PrefillStandIn(model))
    assert rec["function"] == {"m32": {"windows": NW, "differ": []}, "m64": {"windows": NW, "differ": []}}
    assert len(rec["digests_m0"]) == NW and len(set(rec["digests_m0"])) == NW
    assert {g: len(v) for g, v in rec["replay_ms"].items()} == {"m0": 2 * NW, "m32": 2 * NW, "m64": 2 * NW}
    assert set(rec["median_ms"]) == {"m0", "m32", "m64"} and all(v > 0 for v in rec["median_ms"].values())
    assert _speed(box, _PrefillStandIn(model))["digests_m0"] == rec["digests_m0"]     # the cross-process comparison


def test_the_capture_order_rotates_by_process_and_is_recorded(box, clean_env):
    for proc, want in ((1, ["m0", "m32", "m64"]), (2, ["m32", "m64", "m0"]), (3, ["m64", "m0", "m32"])):
        rec = box.speed(_tiny(), _prompts(), T, "cpu", rounds=1, warm=0, eager_rounds=0, eager_windows=0,
                        decode_tokens=DEC, runner=_PrefillStandIn(_tiny()), proc=proc)
        assert rec["capture_order"] == want and list(rec["graphs"]) == want, proc
    # over the reading's eight processes every arm is captured first, second and third
    assert all({box.capture_order(n).index(a) for n in range(1, 9)} == {0, 1, 2} for a in box.ARMS)


def test_the_order_rotates_through_every_arm(box):
    assert [box._rotated(k) for k in range(4)] == [("m0", "m32", "m64"), ("m32", "m64", "m0"), ("m64", "m0", "m32"),
                                                   ("m0", "m32", "m64")]


def test_the_eager_forwards_are_counted_and_every_arm_timed(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny()))
    e = rec["eager"]
    assert e["forwards"] == 1 * 2 * 3 and {g: len(v) for g, v in e["ms"].items()} == {"m0": 2, "m32": 2, "m64": 2}
    assert e["seen"]["k19"] == {"chained|lean|gt256": 2 * 2, "split|lean|gt256": 2 * 2 * 2}


def test_the_knob_and_the_grouping_are_restored(box, clean_env, monkeypatch):
    from experts4bit_qlora.engines import hot_residency as hr
    _speed(box, _PrefillStandIn(_tiny()))
    assert ENV not in os.environ and hr.DEVICE_GROUPING[0] is False
    monkeypatch.setenv(ENV, "0")
    _speed(box, _PrefillStandIn(_tiny()))
    assert os.environ[ENV] == "0"


def test_function_catches_a_kv_only_difference_on_one_arm(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny(), perturb="m64"))
    assert rec["function"]["m32"]["differ"] == []
    assert [d["window"] for d in rec["function"]["m64"]["differ"]] == list(range(NW))
    assert all(d["logits_equal"] and d["kv_layers_differ"] == [1] for d in rec["function"]["m64"]["differ"])


def test_a_refused_tall_capture_still_times_the_others(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny(), refuse=("m32",)))
    assert rec["graphs"]["m32"]["status"] == "refused" and "did not capture" in rec["graphs"]["m32"]["why"]
    assert set(rec["function"]) == {"m64"} and set(rec["replay_ms"]) == {"m0", "m64"} and len(rec["digests_m0"]) == NW


def test_a_refused_control_capture_times_nothing(box, clean_env):
    rec = _speed(box, _PrefillStandIn(_tiny(), refuse=("m0",)))
    assert rec["graphs"]["m0"]["status"] == "refused" and "replay_ms" not in rec and "function" not in rec
    assert set(rec["decode_check"]["seen"]) == {"m0", "m32", "m64"}          # the decode check still ran


# ---- Phase B: P117's stand-in runner, as P130's test stands it in

class _StandIn:
    def __init__(self, model, kv, device="cpu", **_):
        self.model, self.tokens, self.graph_stats, self.prompt_len = model, {}, {}, {}

    def enable_decode_graphs(self, buckets, *, capture=True, warmup=2, verbose=True):
        self.buckets, self.capture = tuple(sorted(int(b) for b in buckets)), capture
        self.graph_stats = {b: {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0} for b in self.buckets}
        return {b: ("graph" if capture else "eager: capture=False") for b in self.buckets}

    def bind(self, rid, slot, prompt):
        self.tokens[rid], self.prompt_len[rid] = list(prompt), len(prompt)

    def run_prefill(self, chunks):
        out = {}
        for rid, s, n in chunks:
            o = self.model(input_ids=torch.tensor([self.tokens[rid][:s + n]]))
            if s + n >= self.prompt_len[rid]:
                out[rid] = int(o.logits[0, -1].argmax())
                self.tokens[rid].append(out[rid])
        return out

    def run_decode(self, rids):
        from experts4bit_qlora.engines.paged_runner import bucket_for, chunk_rows
        got = {}
        for piece in chunk_rows(rids, self.buckets[-1]):
            b = bucket_for(len(piece), self.buckets)
            ids = torch.tensor([self.tokens[r] for r in piece] + [self.tokens[piece[0]]] * (b - len(piece)))
            st = self.graph_stats[b]
            logits = self.model(input_ids=ids).logits
            st["eager_steps"] += 1
            st["rows"] += len(piece)
            st["pad_rows"] += b - len(piece)
            for j, r in enumerate(piece):
                got[r] = int(logits[j, -1].argmax())
                self.tokens[r].append(got[r])
        return got


P, C, N = 12, 5, 20


@pytest.fixture(scope="module")
def quality(box):
    pytest.importorskip("fp8_kv", reason="Phase B's pool is grouped-nf4-gemm's fp8 KV")
    from experts4bit_qlora.engines import paged_runner
    real = paged_runner.PagedModelRunner
    paged_runner.PagedModelRunner = _StandIn
    saved = os.environ.pop(ENV, None)
    try:
        g = torch.Generator().manual_seed(1)
        ws = [torch.randint(0, 256, (P + C,), generator=g).tolist() for _ in range(N)]
        q = box.quality(_tiny(), ws, prompt=P, cont=C, chunk=P, floor_chunk=4, device="cpu", stand_in=True)
        after = os.environ.get(ENV)
    finally:
        paged_runner.PagedModelRunner = real
        if saved is not None:
            os.environ[ENV] = saved
    return q, after


def test_phase_b_scores_every_arm_and_both_subjects_in_one_process(quality, box):
    q, after = quality
    want = list(box.OFF_ARMS) + ["S32", "S64"]
    assert list(q["per_window"]) == want and all(len(v) == N for v in q["per_window"].values())
    assert q["phase"] == "all" and q["rep_identical"] is True and after is None
    assert q["subjects_bitwise"] == {"S32": True, "S64": True}              # the stand-in ignores the knob
    r = {x["window"]: x["nll"] for x in q["per_window"]["R"]}
    for s in ("S32", "S64"):
        assert all(abs(x["nll"] - r[x["window"]]) < 1e-6 and x["argmax_agree"] == 1.0 for x in q["per_window"][s])
    e = {a: q["engagement"][a][0] for a in want}
    assert e["R"]["graph_stats"]["16"]["eager_steps"] == C - 1 and e["half"]["graph_stats"]["8"]["eager_steps"] == 2 * (C - 1)
    assert e["chunk"]["chunk"] == 4 and all(set(e[a]["seen"]) == {"route", "k19", "folds"} for a in want)


# ---- Amendment 2: the GPU state recorded per process (reported only) ------------------------------------------------

def _fake_smi(tmp_path, monkeypatch, body):
    d = tmp_path / "bin"
    d.mkdir()
    f = d / "nvidia-smi"
    f.write_text("#!/bin/sh\n" + body)
    f.chmod(0o755)
    monkeypatch.setenv("PATH", str(d))


@pytest.mark.skipif(os.name == "nt", reason="a shell-script nvidia-smi stand-in")
def test_gpu_state_reads_clocks_power_temperature_and_the_event_reasons(box, tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, 'echo "2850, 14001, 512.30, 63, 0x0000000000000004"\n')
    g = box.gpu_state()
    assert {k: g[k] for k in box.GPU_FIELDS} == {"clocks.sm": "2850", "clocks.mem": "14001", "power.draw": "512.30",
                                                  "temperature.gpu": "63", "clocks_event_reasons.active": "0x0000000000000004"}
    assert "error" not in g and box._gpu_line(g).startswith("sm 2850 MHz mem 14001 MHz 512.30 W 63 C")


@pytest.mark.skipif(os.name == "nt", reason="a shell-script nvidia-smi stand-in")
def test_gpu_state_drops_the_event_reasons_on_a_driver_that_does_not_know_them(box, tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, 'case "$1" in *clocks_event_reasons*) echo "Field is not a valid field" >&2; exit 2;; esac\n'
                                     'echo "2850, 14001, 512.30, 63"\n')
    g = box.gpu_state()
    assert g["clocks.sm"] == "2850" and "clocks_event_reasons.active" not in g and "error" not in g


def test_gpu_state_records_an_unreadable_query_and_never_raises(box, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))            # no nvidia-smi at all: the CPU CI runner
    g = box.gpu_state()
    assert "error" in g and isinstance(g["utc"], float) and box._gpu_line(g).startswith("unread (")


def test_speed_samples_the_gpu_after_the_warm_rounds_and_after_the_timed_rounds(box, clean_env, monkeypatch):
    calls = []
    monkeypatch.setattr(box, "gpu_state", lambda: calls.append(len(calls) + 1) or {"utc": 0.0, "sample": len(calls)})
    rec = box.speed(_tiny(), _prompts(), T, "cpu", rounds=2, warm=1, eager_rounds=0, eager_windows=0,
                    decode_tokens=DEC, runner=_PrefillStandIn(_tiny()), proc=1)
    assert rec["gpu_after_warm"]["sample"] == 1 and rec["gpu_after_timed"]["sample"] == 2 and calls == [1, 2]
    assert all(len(v) == 2 * len(_prompts()) for v in rec["replay_ms"].values())     # timing unchanged: rounds x windows
