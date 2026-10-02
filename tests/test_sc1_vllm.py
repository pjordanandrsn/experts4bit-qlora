"""CPU tests for SC1's vLLM drivers (`bench/sc1/vllm/`), with `vllm` stubbed through `sys.modules` -- vLLM is never
installed on this machine, and the drivers' GPU paths are not exercised here (see the lane's report for what only a
GPU can verify). What IS pinned: the prompt-file contract (digest refusal, distinct rows, the SAMEPROMPT replication
and its receipt marks), the N-token assertion, the slope arithmetic, the exact LLM kwargs per arm, the capture-size
expectation, the NLL index sets in both modes (fake prompt_logprobs / logprobs structures, both container shapes vLLM
returns, with known values -> known means, ranks, suffix histogram, void and cache-miss handling), the K8 digest
convention, the TTFT reduction, and the engagement grep.
"""
import hashlib
import importlib.util
import json
import math
import pathlib
import struct
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
VDIR = REPO / "bench" / "sc1" / "vllm"


# ------------------------------------------------------------------------------------------------------- stubs
def _stub_vllm():
    mod = types.ModuleType("vllm")
    mod.__version__ = "0.30.0-stub"

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = dict(kw)
            self.__dict__.update(kw)

    class LLM:
        def __init__(self, **kw):
            self.kw = dict(kw)

    mod.SamplingParams, mod.LLM = SamplingParams, LLM
    return mod


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mods():
    """The common module plus the three drivers, imported with a stubbed `vllm`; the stub is removed afterwards so
    nothing else in the session sees it."""
    saved = sys.modules.get("vllm")
    sys.modules["vllm"] = _stub_vllm()
    try:
        common = _load("sc1_vllm_common_t", VDIR / "sc1_vllm_common.py")
        sys.modules["sc1_vllm_common"] = common          # the drivers `import sc1_vllm_common as C`
        arm = _load("sc1_vllm_arm_t", VDIR / "sc1_vllm_arm.py")
        ttft = _load("sc1_vllm_ttft_t", VDIR / "sc1_vllm_ttft.py")
        nll = _load("sc1_vllm_nll_t", VDIR / "sc1_vllm_nll.py")
    finally:
        if saved is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = saved
        sys.modules.pop("sc1_vllm_common", None)
    return types.SimpleNamespace(C=common, arm=arm, ttft=ttft, nll=nll)


def _prompt_file(tmp_path, rows, name="prompts.json", sha=None):
    pf = {"batch": len(rows), "prompt_len": len(rows[0]), "prompts": rows,
          "prompts_sha256": sha or hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
          "rows_sha256": [hashlib.sha256(json.dumps(r).encode()).hexdigest() for r in rows]}
    p = tmp_path / name
    p.write_text(json.dumps(pf))
    return p


def _rows(n, length=8, seed=1):
    return [[(seed * 1000 + i * 37 + j) % 151936 for j in range(length)] for i in range(n)]


# ------------------------------------------------------------------------------------------- prompt contract
def test_prompt_digest_refusal(mods, tmp_path):
    p = _prompt_file(tmp_path, _rows(2), sha="0" * 64)
    with pytest.raises(AssertionError, match="digest mismatch"):
        mods.C.load_prompts(p, 2, "graph_r1", prompt_len=8)


def test_prompt_row_count_and_length_refusals(mods, tmp_path):
    p = _prompt_file(tmp_path, _rows(2))
    with pytest.raises(AssertionError):
        mods.C.load_prompts(p, 3, "graph_r1", prompt_len=8)       # batch != rows
    with pytest.raises(AssertionError, match="prompt_len must be"):
        mods.C.load_prompts(p, 2, "graph_r1", prompt_len=512)     # rows are 8 long


def test_distinct_rows_required_on_matched_arms(mods, tmp_path):
    rows = _rows(1) * 2
    p = _prompt_file(tmp_path, rows)
    with pytest.raises(AssertionError, match="distinct"):
        mods.C.load_prompts(p, 2, "graph_r1", prompt_len=8)


def test_sameprompt_replicates_row0_and_marks_the_receipt(mods, tmp_path):
    rows = _rows(4)
    p = _prompt_file(tmp_path, rows)
    prompts, info = mods.C.load_prompts(p, 4, "sameprompt", prompt_len=8)
    assert prompts == [rows[0]] * 4
    assert info["rows_distinct_in_file"] is True
    sp = info["sameprompt"]
    assert sp["rows_identical"] is True and sp["source_row"] == 0 and sp["copies"] == 4
    file_sha = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    assert info["prompts_sha256"] == file_sha == sp["file_prompts_sha256"]           # the FILE's digest is carried
    assert sp["effective_prompts_sha256"] == hashlib.sha256(json.dumps(prompts).encode()).hexdigest() != file_sha
    assert sp["row_sha256"] == hashlib.sha256(json.dumps(rows[0]).encode()).hexdigest()


def test_sameprompt_refuses_batch_1(mods, tmp_path):
    p = _prompt_file(tmp_path, _rows(1))
    with pytest.raises(AssertionError, match="B>1"):
        mods.C.load_prompts(p, 1, "sameprompt", prompt_len=8)


def test_env_prefers_sc1_over_p37(mods, monkeypatch):
    monkeypatch.setenv("P37_ARM", "eager")
    assert mods.C.env("ARM") == "eager"
    monkeypatch.setenv("SC1_ARM", "graph_r1")
    assert mods.C.env("ARM") == "graph_r1"
    monkeypatch.delenv("SC1_ARM")
    monkeypatch.delenv("P37_ARM")
    with pytest.raises(SystemExit, match="SC1_ARM"):
        mods.C.env("ARM", required=True)


# ---------------------------------------------------------------------------------------- generation accounting
class _Comp:
    def __init__(self, token_ids, logprobs=None):
        self.token_ids, self.logprobs = list(token_ids), logprobs


class _Req:
    def __init__(self, comps, prompt_token_ids=None, num_cached_tokens=None, prompt_logprobs=None):
        self.outputs, self.prompt_token_ids = comps, prompt_token_ids
        self.num_cached_tokens, self.prompt_logprobs = num_cached_tokens, prompt_logprobs


def test_n_token_assertion(mods):
    ok = [_Req([_Comp(range(32))]) for _ in range(4)]
    assert mods.C.assert_generated(ok, 32, 4) == 128
    short = [_Req([_Comp(range(32))]) for _ in range(3)] + [_Req([_Comp(range(31))])]
    with pytest.raises(AssertionError, match="stopped early"):
        mods.C.assert_generated(short, 32, 4)
    swapped = [_Req([_Comp(range(33))]), _Req([_Comp(range(31))])]       # sum matches, rows do not
    with pytest.raises(AssertionError, match="exactly 32"):
        mods.C.assert_generated(swapped, 32, 2)
    assert mods.C.engine_prompt_tokens([_Req([_Comp([1])], prompt_token_ids=list(range(512)))]) == [512]


# ------------------------------------------------------------------------------------------------------ slope
def test_slope_arithmetic_matches_p37(mods):
    s = mods.C.slope([1.0, 1.1, 1.2], [2.0, 2.1, 2.2], 32, 128, 16)
    assert s["slope_extra_tokens"] == 96 * 16
    assert s["decode_tok_s"] == pytest.approx(1536.0) and s["decode_tok_s_median"] == pytest.approx(1536.0)
    assert s["decode_ms_per_step"] == pytest.approx(1000.0 / 96, abs=1e-3)
    assert s["decode_ms_per_step_median"] == pytest.approx(1000.0 / 96, abs=1e-3)
    assert s["end_to_end_tok_s_long"] == pytest.approx(128 * 16 / 2.0)
    assert s["wall_short_s"] == 1.0 and s["wall_long_s"] == 2.0
    assert s["walls_short_s"] == [1.0, 1.1, 1.2]


def test_slope_inverted_is_not_a_reading(mods):
    s = mods.C.slope([2.0, 2.0, 2.0], [1.9, 2.1, 2.2], 32, 128, 1)
    assert s["decode_tok_s"] is None and "inverted" in s["slope_error"]


def test_ttft_reduce(mods):
    r = mods.C.ttft_reduce([0.30, 0.20, 0.25], 4096)
    assert r["ttft_s_median"] == 0.25 and r["ttft_s_min"] == 0.2 and r["reps"] == 3
    assert r["prefill_tok_s_median"] == pytest.approx(4096 / 0.25)


# ------------------------------------------------------------------------------------------------- LLM kwargs
def test_llm_kwargs_per_arm(mods):
    C = mods.C
    g = C.build_llm_kwargs("graph_r1", 16)
    assert g["enable_prefix_caching"] is False and g["kv_cache_dtype"] == "auto" and g["enforce_eager"] is False
    assert g["attention_backend"] == "FLASH_ATTN" and g["moe_backend"] == "marlin" and g["seed"] == 0
    assert g["max_num_seqs"] == 16 and g["max_num_batched_tokens"] == 8192 and g["max_model_len"] == 2048
    assert g["gpu_memory_utilization"] == 0.90 and g["tensor_parallel_size"] == 1
    assert C.build_llm_kwargs("graph_r2", 16) == g == C.build_llm_kwargs("sameprompt", 16) == C.build_llm_kwargs("nodetok", 16)
    assert C.build_llm_kwargs("graph_r1", 1)["max_num_seqs"] == 1
    e = C.build_llm_kwargs("eager", 1)
    assert e["enforce_eager"] is True and {k: v for k, v in e.items() if k != "enforce_eager"} == \
        {k: v for k, v in C.build_llm_kwargs("graph_r1", 1).items() if k != "enforce_eager"}
    f = C.build_llm_kwargs("fp8kv", 16)
    assert f["kv_cache_dtype"] == "fp8" and f["attention_backend"] == "FLASHINFER" and f["moe_backend"] == "marlin"
    assert C.build_llm_kwargs("fp8kv", 16, attn="TRITON_ATTN")["attention_backend"] == "TRITON_ATTN"
    n = C.build_llm_kwargs("native", 16)
    assert n == {"model": C.MODEL_DEFAULT, "revision": C.REV_DEFAULT, "tokenizer_revision": C.REV_DEFAULT, "seed": 0}
    with pytest.raises(AssertionError, match="unknown arm"):
        C.build_llm_kwargs("turbo", 1)
    assert set(C.KWARG_SOURCES) >= {"attention_backend", "moe_backend", "max_logprobs", "prompt_logprobs", "num_cached_tokens"}


def test_sampling_kwargs_detokenize_rule(mods):
    """Registered text: the MATCHED arms keep vLLM's shipped detokenize=True (a comparator's loop is never trimmed);
    only the `nodetok` pair runs detokenize=False, so the detokeniser's cost is measured, not assumed."""
    C = mods.C
    for arm in ("graph_r1", "graph_r2", "eager", "fp8kv", "sameprompt", "native"):
        s = C.sampling_kwargs(arm, 128)
        assert s == {"temperature": 0.0, "max_tokens": 128, "ignore_eos": True, "min_tokens": 128, "detokenize": True}
    assert C.sampling_kwargs("nodetok", 32) == {"temperature": 0.0, "max_tokens": 32, "ignore_eos": True, "min_tokens": 32,
                                                "detokenize": False}


def test_registered_capacity_rule_on_matched_arms(mods):
    """max_num_seqs=B (16 on the B=16 arms, 1 on the B=1 arms) and max_model_len=2048 on every matched arm."""
    C = mods.C
    for arm in ("graph_r1", "graph_r2", "eager", "fp8kv", "sameprompt", "nodetok"):
        for b in (1, 16):
            kw = C.build_llm_kwargs(arm, b)
            assert kw["max_num_seqs"] == b and kw["max_model_len"] == 2048, (arm, b)
    assert "max_num_seqs" not in C.build_llm_kwargs("native", 16) and "max_model_len" not in C.build_llm_kwargs("native", 16)


def test_expected_capture_sizes(mods):
    assert mods.C.expected_capture_sizes(16) == [1, 2, 4, 8, 16, 24, 32]
    assert mods.C.expected_capture_sizes(1) == [1, 2]
    big = mods.C.expected_capture_sizes(256)
    assert big[-1] == 512 and 256 in big and all(b <= 512 for b in big)


def test_served_sampling_kwargs(mods):
    C = mods.C
    assert C.served_sampling_kwargs("full", 7) == {"max_tokens": 1, "temperature": 0.0, "detokenize": False, "flat_logprobs": True, "logprobs": -1}
    assert C.served_sampling_kwargs("token_ids", 7)["logprob_token_ids"] == [7]
    assert "logprobs" not in C.served_sampling_kwargs("token_ids", 7)
    assert C.served_sampling_kwargs("topk:50", 7)["logprobs"] == 50
    with pytest.raises(ValueError):
        C.served_sampling_kwargs("bogus", 7)


# ------------------------------------------------------------------------------------------------ K8 digest
def test_k8_text_sha_is_int64_little_endian_of_the_window(mods):
    ids = list(range(100, 100 + 40))
    P, S = 4, 10
    want = hashlib.sha256(struct.pack("<%dq" % (P + S + 1), *ids[:P + S + 1])).hexdigest()
    assert mods.C.k8_text_sha(ids, P, S) == want
    assert mods.C.k8_text_sha(ids + [999], P, S) == want            # ids beyond the window do not enter the digest


def test_window_loading_refuses_sha_mismatch(mods, tmp_path):
    ids = list(range(1, 60))
    P, S = 4, 10
    good = tmp_path / "k8_window_wikitext.json"
    good.write_text(json.dumps({"ids": ids, "text_sha": mods.C.k8_text_sha(ids, P, S)}))
    got, info = mods.C.load_window(good, P, S)
    assert got == ids and info["ppl_source"] == "wikitext" and info["text_sha_recomputed"] is True
    bad = tmp_path / "k8_window_c4val1.json"
    bad.write_text(json.dumps({"ids": ids, "text_sha": "f" * 64}))
    with pytest.raises(AssertionError, match="window digest mismatch"):
        mods.C.load_window(bad, P, S)
    with pytest.raises(AssertionError, match="K8 needs"):
        mods.C.load_window(good, P, 100)


def test_served_prompt_and_target_indices_are_k8s(mods):
    C = mods.C
    ids = list(range(3000))
    assert C.served_prompt(ids, 512, 0) == ids[:513] and C.served_target(ids, 512, 0) == 513
    assert C.served_prompt(ids, 512, 2047) == ids[:2560] and C.served_target(ids, 512, 2047) == 2560
    # K8: cont = ids[512:512+2048+1]; step t scores cont[t+1] = ids[513+t]
    cont = ids[512:512 + 2048 + 1]
    assert all(C.served_target(ids, 512, t) == cont[t + 1] for t in range(2048))


# ---------------------------------------------------------------------------------------------- NLL: prefill
class _LP:
    def __init__(self, logprob, rank):
        self.logprob, self.rank, self.decoded_token = logprob, rank, None


class _Flat:
    """vllm.logprobs.FlatLogprobs' surface: parallel lists + start/end indices, len() = positions."""

    def __init__(self):
        self.start_indices, self.end_indices, self.token_ids, self.logprobs, self.ranks = [], [], [], [], []

    def append(self, d):
        self.start_indices.append(len(self.logprobs))
        if d:
            for t, lp in d.items():
                self.token_ids.append(t)
                self.logprobs.append(lp.logprob)
                self.ranks.append(lp.rank)
        self.end_indices.append(len(self.logprobs))

    def __len__(self):
        return len(self.start_indices)


def _fake_prompt_logprobs(ids, P, S, flat):
    """slot 0 None; slot i holds {ids[i]: lp_i, other: lp} with lp_i = -(i * 0.01), rank 1 iff i even."""
    n = P + S + 1
    cont = _Flat() if flat else []
    cont.append(None)
    for i in range(1, n):
        lp = -(i * 0.01)
        cont.append({ids[i]: _LP(lp, 1 if i % 2 == 0 else 3), 151935: _LP(lp - 1.0, 2)})
    return cont


@pytest.mark.parametrize("flat", [False, True])
def test_prefill_nll_scores_slots_513_to_2560(mods, flat):
    P, S = 512, 2048
    ids = [(i * 7919) % 151900 for i in range(P + S + 1)]
    plp = _fake_prompt_logprobs(ids, P, S, flat)
    r = mods.C.prefill_nll(plp, ids, P, S)
    want = sum(i * 0.01 for i in range(P + 1, P + S + 1)) / S
    assert r["mean_nll"] == pytest.approx(want) and r["ppl"] == pytest.approx(math.exp(want))
    assert r["steps"] == 2048 == len(r["per_token_nll"]) and r["scored_index_range"] == [513, 2560]
    assert r["per_token_nll"][0] == pytest.approx(5.13) and r["per_token_nll"][-1] == pytest.approx(25.60)
    assert r["top1_agreement"] == pytest.approx(0.5)        # even slots rank 1


def test_prefill_nll_refuses_wrong_length_and_missing_token(mods):
    P, S = 4, 6
    ids = list(range(50, 50 + P + S + 2))
    with pytest.raises(ValueError, match="positions"):
        mods.C.prefill_nll(_fake_prompt_logprobs(ids, P, S + 1, False), ids, P, S)
    plp = _fake_prompt_logprobs(ids, P, S, False)
    plp[P + 2] = {999999: _LP(-0.5, 1)}                    # the actual token is absent at one scored slot
    with pytest.raises(ValueError, match="absent"):
        mods.C.prefill_nll(plp, ids, P, S)
    plp2 = _fake_prompt_logprobs(ids, P, S, False)
    plp2[0] = {ids[0]: _LP(-0.1, 1)}                       # slot 0 must be empty by the tag's convention
    with pytest.raises(ValueError, match="convention"):
        mods.C.prefill_nll(plp2, ids, P, S)


# ----------------------------------------------------------------------------------------------- NLL: served
def _served_outputs(ids, P, S, block=16, flat=False, nll_of=lambda t: 0.5 + t * 0.001, miss_at=(), nocache_at=()):
    """Request t: prompt ids[:P+1+t]; num_cached_tokens follows vLLM's block rule (0 on the cold first request, then
    block * floor((n-1)/block) because a full hit recomputes the last token); the generated token is the target on even t."""
    outs = []
    for t in range(S):
        n = P + 1 + t
        target = ids[n]
        cached = 0 if (t == 0 or t in nocache_at) else block * ((n - 1) // block)
        lp = -nll_of(t)
        d = {} if t in miss_at else {target: _LP(lp, 1)}
        d[42] = _LP(lp - 2.0, 2)
        cont = _Flat() if flat else []
        cont.append(d)
        gen = target if t % 2 == 0 else 42
        outs.append(_Req([_Comp([gen], cont)], prompt_token_ids=ids[:n], num_cached_tokens=cached))
    return outs


@pytest.mark.parametrize("flat", [False, True])
def test_served_rows_mean_histogram_and_predicate(mods, flat):
    C = mods.C
    P, S, block = 512, 64, 16
    ids = [(i * 7919) % 151900 for i in range(P + S + 2)]
    outs = _served_outputs(ids, P, S, block, flat)
    rows = [C.served_row(o, len(C.served_prompt(ids, P, t)), C.served_target(ids, P, t))[0] for t, o in enumerate(outs)]
    red = C.served_reduce(rows, block, S)
    assert red["verdict"] == "VALID" and red["void_rows"] == 0 and red["cache_predicate"] == "PASS"
    assert red["mean_nll"] == pytest.approx(sum(0.5 + t * 0.001 for t in range(S)) / S)
    assert red["top1_agreement"] == pytest.approx(0.5)
    assert red["mode_label"] == "served (partial-block, M<=16)"
    assert red["suffix_first"] == 513                                   # cold first request recomputes everything
    hist = {int(k): v for k, v in red["suffix_histogram"].items()}
    assert set(hist) - {513} == set(range(1, block + 1))                # the partial block cycles 1..16
    assert red["suffix_max_excl_first"] == block and 1 <= red["suffix_mean_excl_first"] <= block
    assert red["tokens_scored"] == S == len(red["per_token_nll"])
    assert rows[0]["n_entries"] == 2 and rows[0]["rank"] == 1


def test_served_missing_token_is_void_never_approximated(mods):
    C = mods.C
    P, S = 512, 20
    ids = [(i * 7919) % 151900 for i in range(P + S + 2)]
    outs = _served_outputs(ids, P, S, miss_at=(5,))
    rows = [C.served_row(o, P + 1 + t, C.served_target(ids, P, t))[0] for t, o in enumerate(outs)]
    red = C.served_reduce(rows, 16, S)
    assert rows[5]["nll"] is None and red["void_rows"] == 1 and red["verdict"] == "VOID"
    assert red["tokens_scored"] == S - 1
    assert red["mean_nll"] == pytest.approx(sum(0.5 + t * 0.001 for t in range(S) if t != 5) / (S - 1))


def test_served_cache_miss_relabels_prefill_shaped(mods):
    C = mods.C
    P, S = 512, 20
    ids = [(i * 7919) % 151900 for i in range(P + S + 2)]
    outs = _served_outputs(ids, P, S, nocache_at=(7, 9))
    rows = [C.served_row(o, P + 1 + t, C.served_target(ids, P, t))[0] for t, o in enumerate(outs)]
    red = C.served_reduce(rows, 16, S)
    assert red["cache_predicate"] == "FAIL" and red["n_cache_miss_rows"] == 2 and red["cache_miss_rows_head"] == [7, 9]
    assert red["mode_label"].startswith("prefill-shaped (no cache hit) on 2/19 rows")
    assert red["verdict"] == "VALID"                                    # relabelled, not VOID (F1)


def test_served_row_refuses_more_than_one_generated_token(mods):
    with pytest.raises(AssertionError, match="exactly 1"):
        mods.C.served_row(_Req([_Comp([1, 2], [])], num_cached_tokens=0), 513, 1)


def test_empty_logprob_container_reads_as_no_entries(mods):
    """The V1-runner shape of logprobs=-1 at the tag: an empty container -> n_entries 0, nll None (then the auto mode
    downgrades to logprob_token_ids; a forced full mode leaves the row VOID)."""
    row, entries = mods.C.served_row(_Req([_Comp([5], [])], num_cached_tokens=0), 513, 5)
    assert entries is None and row["n_entries"] == 0 and row["nll"] is None and row["top1"] == 1
    row2, _ = mods.C.served_row(_Req([_Comp([5], _Flat())], num_cached_tokens=0), 513, 5)
    assert row2["n_entries"] == 0


# ---------------------------------------------------------------------------------------------- engagement grep
LOG_GRAPH = """INFO 10-01 12:00:01 [auto_gptq.py:353] Using MarlinLinearKernel for AutoGPTQLinearMethod
INFO 10-01 12:00:02 [cuda.py:478] Using FLASH_ATTN backend.
INFO 10-01 12:00:03 [int_wna16.py:282] Using 'MARLIN' WNA16 MoE backend.
INFO 10-01 12:00:03 [int_wna16.py:436] Using MarlinExperts
INFO 10-01 12:00:04 [gpu_worker.py:441] Using V2 Model Runner
INFO 10-01 12:00:40 [gpu_model_runner.py:5417] Model loading took 16.12 GiB memory and 31.5 seconds
INFO 10-01 12:00:50 [gpu_worker.py:641] Available KV cache memory: 9.87 GiB
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████| 7/7 [00:03<00:00,  2.1it/s]
Capturing CUDA graphs (decode, FULL): 100%|██████████| 7/7 [00:02<00:00,  3.0it/s]
INFO 10-01 12:01:00 [gpu_model_runner.py:6930] Graph capturing finished in 6 secs, took 0.45 GiB
INFO 10-01 12:01:00 [gpu_worker.py:873] Actual usage is 28.1 GiB
"""


def test_engagement_grep_reads_the_marlin_and_capture_lines(mods, tmp_path):
    p = tmp_path / "engine.log"
    p.write_text(LOG_GRAPH)
    e = mods.C.grep_engagement(str(p))
    assert e["status"] == "grepped"
    assert e["marlin_moe"] is True and e["moe_backend_line"] == ["Using 'MARLIN' WNA16 MoE backend."]
    assert e["marlin_linear"] is True and e["linear_kernel_line"] == ["Using MarlinLinearKernel for AutoGPTQLinearMethod"]
    assert e["experts_cls_line"] == ["Using MarlinExperts"]
    assert e["cudagraphs_captured"] is True and len(e["cudagraph_capture_line"]) == 2
    assert e["graph_capture_finished_line"] == ["Graph capturing finished in 6 secs, took 0.45 GiB"]
    assert e["attention_backend_logged"] == ["FLASH_ATTN"] and e["v2_model_runner"] is True
    assert e["kv_cache_memory_line"] == ["Available KV cache memory: 9.87 GiB"]
    assert e["model_loading_line"] == ["Model loading took 16.12 GiB memory and 31.5 seconds"]
    assert e["kv_scale_warning_line"] == [] and e["actual_usage_line"] == ["Actual usage is 28.1 GiB"]


def test_engagement_grep_eager_and_humming_and_missing_log(mods, tmp_path):
    p = tmp_path / "eager.log"
    p.write_text("INFO Using 'HUMMING' WNA16 MoE backend.\nINFO Using HummingExperts\nWARNING Using KV cache scaling factor 1.0 for fp8_e4m3. "
                 "If this is unintended, verify that k/v_scale scaling factors are properly set in the checkpoint.\n")
    e = mods.C.grep_engagement(str(p))
    assert e["marlin_moe"] is False and e["moe_backend_line"] == ["Using 'HUMMING' WNA16 MoE backend."]
    assert e["cudagraphs_captured"] is False and e["marlin_linear"] is False
    assert e["kv_scale_warning_line"] == ["Using KV cache scaling factor 1.0 for fp8_e4m3"]
    none = mods.C.grep_engagement(str(tmp_path / "absent.log"))
    assert none["status"] == "no_log" and none["marlin_moe"] is None and none["cudagraphs_captured"] is None
    assert mods.C.grep_engagement(None)["status"] == "no_log"


# ------------------------------------------------------------------------------------------- drivers import clean
def test_drivers_import_without_side_effects_and_expose_main(mods):
    for m in (mods.arm, mods.ttft, mods.nll):
        assert callable(m.main)
    assert mods.ttft.TTFT_ARMS == ("graph_r1", "eager", "fp8kv") and mods.nll.MODES == ("prefill", "served")
    assert "calculate_kv_scales" in mods.arm.KV_SCALE_PROVENANCE
    assert set(mods.C.ARMS) == {"graph_r1", "graph_r2", "eager", "fp8kv", "sameprompt", "native", "nodetok"}
    assert "detok" not in mods.C.ARMS


def test_ttft_one_row_loader(mods, tmp_path):
    row = [list(range(4096))]
    p = _prompt_file(tmp_path, row, name="prompts_b1_4096.json")
    ids, info = mods.ttft.load_one_row(p)
    assert ids == row[0] and info["prompts_sha256"] == hashlib.sha256(json.dumps(row).encode()).hexdigest()
    with pytest.raises(AssertionError, match="one-row"):
        mods.ttft.load_one_row(_prompt_file(tmp_path, _rows(2), name="two.json"))


def test_receipt_is_json_serialisable(mods, tmp_path):
    rec = mods.C.base_receipt("arm", "graph_r1", mods.C.build_llm_kwargs("graph_r1", 16), {"x": {1, 2}})
    out = tmp_path / "r.json"
    mods.C.write_receipt(str(out), rec)
    back = json.loads(out.read_text())
    assert back["engine"] == "vllm" and back["vllm_tag_commit"] == mods.C.TAG_COMMIT and sorted(back["x"]) == [1, 2]
    assert back["llm_kwargs"]["attention_backend"] == "FLASH_ATTN" and "kwarg_sources" in back
