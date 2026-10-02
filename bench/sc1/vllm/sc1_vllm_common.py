"""sc1_vllm_common.py -- the helpers shared by SC1's three vLLM drivers (arm / ttft / nll; lane SC1, experts4bit-qlora#846).

Everything here is CPU-side and stdlib+numpy only, so `tests/test_sc1_vllm.py` can exercise it with `vllm` stubbed. The
drivers import it from their own directory (stage it NEXT TO them on the box -- it is one more file than P58's
single-file `p37_vllm.py` contract).

Every vLLM kwarg name below was read at tag v0.30.0 (commit ced6857afa0ea7b2e3f0846a62e1394e90f15607); the citations are
carried into every receipt as `kwarg_sources` so a reader does not have to trust this docstring:

  attention_backend  -> `LLM(**kwargs)` forwards unknown kwargs to `EngineArgs` (vllm/entrypoints/llm.py:224,290+);
                        `EngineArgs.attention_backend: AttentionBackendEnum | None` (vllm/engine/arg_utils.py:731);
                        strings are coerced by `AttentionConfig.validate_backend_before` ->
                        `AttentionBackendEnum[value.upper()]` (vllm/config/attention.py:156-167, arg_utils.py:2494-2505);
                        member names FLASH_ATTN / FLASHINFER / TRITON_ATTN (vllm/v1/attention/backends/registry.py:44-68).
  moe_backend        -> `EngineArgs.moe_backend: MoEBackend` (arg_utils.py:511), Literal incl. "marlin"
                        (vllm/config/kernel.py:125-148), applied at arg_utils.py:2560-2561 -> `kernel_config.moe_backend`.
  max_logprobs       -> `EngineArgs.max_logprobs` (arg_utils.py:562) -> `ModelConfig.max_logprobs` default 20, -1 = all
                        (vllm/config/model.py:254); `SamplingParams.logprobs/prompt_logprobs = -1` = full vocab
                        (vllm/sampling_params.py:285-295, validation 838-895).
  prompt_logprobs    -> position i (i >= 1) of `RequestOutput.prompt_logprobs` is the logprob of prompt token i given
                        tokens [0, i): the runner gathers target `prompt_token_ids[start_tok : ...]` with
                        start_tok = num_computed + 1 (vllm/v1/worker/gpu_model_runner.py:5684-5736) and the output
                        processor prepends None for position 0 (vllm/logprobs.py:167-172); the chosen token is ALWAYS in
                        the dict with its rank (vllm/v1/sample/sampler.py gather_logprobs; LogprobsProcessor
                        vllm/v1/engine/logprobs.py:121-190). Same convention in the V2 runner
                        (vllm/v1/worker/gpu/sample/prompt_logprob.py:166-170 "shift the pos by one").
  num_cached_tokens  -> `RequestOutput.num_cached_tokens` (vllm/outputs.py:126,168), written from
                        `prefill_stats.num_cached_tokens` (vllm/v1/engine/output_processor.py:682-686).
  prefix cache       -> block-granular; a full hit recomputes at least the last token/block
                        (vllm/v1/core/kv_cache_manager.py:289-296); requests with prompt_logprobs skip the lookup
                        (kv_cache_manager.py:282-287 + vllm/v1/request.py:226,307-318); default block size 16
                        (vllm/config/cache.py:71 DEFAULT_BLOCK_SIZE).
  logprobs=-1 output -> V2 model runner (the 0.30 default on CUDA with Triton; vllm/config/vllm.py:675-724): a real
                        top-vocab gather, dict keyed by token id (vllm/v1/worker/gpu/sample/states.py:59-60,
                        logprob.py compute_topk_scores). V1 runner (`VLLM_USE_V2_MODEL_RUNNER=0`): the sampler returns
                        EMPTY token-id/rank tensors (vllm/v1/sample/sampler.py:123-127) and the processor's zip yields
                        nothing (vllm/v1/engine/logprobs.py:84-86) -> an EMPTY container. The served scorer verifies
                        the entry count against the vocab size on the first request and downgrades EXPLICITLY.
  kv fp8 scales      -> no k_scale/v_scale in the checkpoint => both 1.0 (vllm/model_executor/layers/quantization/
                        kv_cache.py:123-129) with the warning "Using KV cache scaling factor 1.0 for fp8_e4m3"
                        (kv_cache.py:163-168); `calculate_kv_scales` does NOT exist at this tag (grep: only a comment in
                        vllm/models/kimi_k3/nvidia/mla.py:473).
  capture sizes      -> max = min(max_num_seqs * 2, 512); list = [1,2,4] + range(8,256,8) + range(256,max+1,16)
                        filtered <= max (vllm/config/vllm.py:2151-2200, compilation.py:699-716); recorded from
                        `compilation_config.cudagraph_capture_sizes` (compilation.py:655).
  collective_rpc     -> `LLM.collective_rpc(callable)` runs on the worker (llm.py:567-598); in multiprocess mode the
                        callable crosses the ZMQ boundary by cloudpickle ONLY under VLLM_ALLOW_INSECURE_SERIALIZATION=1
                        (vllm/v1/serial_utils.py:214-231).
"""
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter

TAG = "v0.30.0"
TAG_COMMIT = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
MODEL_DEFAULT = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
REV_DEFAULT = "9b534e4318b7ebc3c961a839f13eb18b1833f441"
ARMS = ("graph_r1", "graph_r2", "eager", "fp8kv", "sameprompt", "native", "nodetok")
PROMPT_LEN = 512
SHORT, LONG = 32, 128
DEFAULT_BLOCK_SIZE = 16          # vllm/config/cache.py:71

KWARG_SOURCES = {
    "attention_backend": "EngineArgs.attention_backend (arg_utils.py:731) <- LLM(**kwargs) (llm.py:224); string->enum via "
                         "AttentionConfig.validate_backend_before (config/attention.py:156-167; arg_utils.py:2494-2505); "
                         "names: v1/attention/backends/registry.py:44-68",
    "moe_backend": "EngineArgs.moe_backend (arg_utils.py:511); MoEBackend Literal 'marlin' (config/kernel.py:125-148); "
                   "-> kernel_config.moe_backend (arg_utils.py:2560-2561); oracle honours it at "
                   "fused_moe/oracle/int_wna16.py:290-300",
    "max_logprobs": "EngineArgs.max_logprobs (arg_utils.py:562) -> ModelConfig.max_logprobs default 20, -1 = all "
                    "(config/model.py:254); SamplingParams validation sampling_params.py:838-895",
    "prompt_logprobs": "SamplingParams.prompt_logprobs (sampling_params.py:293); gather target = next prompt token "
                       "(v1/worker/gpu_model_runner.py:5684-5736; V2: v1/worker/gpu/sample/prompt_logprob.py:166-170); "
                       "position 0 is None (vllm/logprobs.py:167-172)",
    "logprobs": "SamplingParams.logprobs (sampling_params.py:285); -1 = vocab (V2: v1/worker/gpu/sample/states.py:59-60; "
                "V1 sampler returns empty ids for -1: v1/sample/sampler.py:123-127)",
    "logprob_token_ids": "SamplingParams.logprob_token_ids (sampling_params.py:296-301); sampled-position gather "
                         "v1/sample/sampler.py:110-118 (gather_specific_token_logprobs)",
    "num_cached_tokens": "RequestOutput.num_cached_tokens (outputs.py:126,168) <- prefill_stats "
                         "(v1/engine/output_processor.py:682-686)",
    "enable_prefix_caching": "EngineArgs.enable_prefix_caching (arg_utils.py:538); CacheConfig default True "
                             "(config/cache.py:130); block-granular hit, last token recomputed "
                             "(v1/core/kv_cache_manager.py:289-296)",
    "kv_cache_dtype": "EngineArgs.kv_cache_dtype (arg_utils.py:459); choices config/cache.py:39-47; fp8 scales 1.0 when "
                      "absent (layers/quantization/kv_cache.py:123-129, warning :163-168); no calculate_kv_scales at tag",
    "max_num_seqs": "EngineArgs.max_num_seqs (arg_utils.py:559); drives capture sizes (config/vllm.py:2151-2200)",
    "max_num_batched_tokens": "EngineArgs.max_num_batched_tokens (arg_utils.py:556); LLM-class default 8192 on the 32 GB "
                              "tier (arg_utils.py:2718-2760); chunked prefill default True (config/scheduler.py:116)",
    "enforce_eager": "EngineArgs.enforce_eager (arg_utils.py:584); sets compile NONE + cudagraph NONE "
                     "(config/vllm.py:1546-1552, 1779-1784)",
    "cudagraph_capture_sizes": "CompilationConfig.cudagraph_capture_sizes (config/compilation.py:655)",
    "detokenize": "SamplingParams.detokenize (sampling_params.py:311); False -> tokenizer None in the LogprobsProcessor "
                  "(v1/engine/output_processor.py:234-242)",
    "collective_rpc": "LLM.collective_rpc(method | callable) (entrypoints/llm.py:567-598); cloudpickle only with "
                      "VLLM_ALLOW_INSECURE_SERIALIZATION=1 (v1/serial_utils.py:214-231)",
    "engagement_log_lines": "MoE: \"Using 'MARLIN' WNA16 MoE backend.\" (fused_moe/oracle/int_wna16.py:255-256, info_once "
                            "at :282/:325) + 'Using MarlinExperts' (:436); linear: 'Using MarlinLinearKernel for "
                            "AutoGPTQLinearMethod' (layers/quantization/auto_gptq.py:352-354); graphs: tqdm desc "
                            "'Capturing CUDA graphs (decode|mixed prefill-decode, MODE)' (v1/worker/gpu_model_runner.py"
                            ":7005) + 'Graph capturing finished in N secs, took X GiB' (:6930); attention: "
                            "'Using FLASH_ATTN backend.' (platforms/cuda.py:478); KV: 'Available KV cache memory: X GiB' "
                            "(v1/worker/gpu_worker.py:641); runner: 'Using V2 Model Runner' (gpu_worker.py:441)",
}

# Every pattern is anchored on a string read from the tag's source (see KWARG_SOURCES["engagement_log_lines"]).
ENGAGEMENT_PATTERNS = {
    "moe_backend_line": r"Using '([A-Z0-9_]+)' WNA16 MoE backend\.",
    "mxfp4_moe_backend_line": r"Using '([A-Z0-9_]+)' Mxfp4 MoE backend\.",
    "experts_cls_line": r"Using ([A-Za-z0-9_]+Experts)\b",
    "linear_kernel_line": r"Using ([A-Za-z0-9_]+LinearKernel) for ([A-Za-z0-9_]+)",
    "cudagraph_capture_line": r"Capturing CUDA graphs \(([^)]*)\)",
    "graph_capture_finished_line": r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB",
    "attention_backend_line": r"Using ([A-Z][A-Z0-9_]+) backend\.",
    "kv_cache_memory_line": r"Available KV cache memory: ([\d.]+) GiB",
    "model_loading_line": r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds",
    "actual_usage_line": r"Actual usage is ([\d.]+) GiB",
    "v2_runner_line": r"Using V2 Model Runner",
    "kv_scale_warning_line": r"Using KV cache scaling factor 1\.0 for fp8_e4m3",
    "engine_core_error_line": r"(EngineCore (?:failed|hit an exception|encountered)[^\n]{0,160})",
}

_TEE = None


# ----------------------------------------------------------------------------------------------------- env contract
def env(name, default=None, required=False):
    """`SC1_<name>` first, `P37_<name>` as the fallback (the P58 runner exports P37_* names)."""
    for key in (f"SC1_{name}", f"P37_{name}"):
        val = os.environ.get(key)
        if val is not None and val != "":
            return val
    if required:
        raise SystemExit(f"refusing: SC1_{name} (or P37_{name}) is unset")
    return default


def env_int(name, default):
    return int(env(name, str(default)))


def env_float(name, default):
    return float(env(name, str(default)))


def early_setup():
    """Run BEFORE `import vllm`: the logging level the engagement grep depends on, the in-process switch, the
    serialisation gate the worker memory RPC needs, and the log tee. The tee dup2's fd 1 and 2 of THIS process into
    `tee -a SC1_LOG`; the engine-core subprocess inherits those fds, so its logger lines and tqdm bars land in the same
    file the receipt is grepped from. Returns the log path (None when SC1_LOG is unset)."""
    global _TEE
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    if env("INPROC", "0") == "1":
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # needed only for LLM.collective_rpc(<callable>) across the engine-core process boundary (serial_utils.py:214-231);
    # our own driver process talks to our own engine, there is no third party on this socket.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    log = env("LOG")
    if log and _TEE is None:
        os.makedirs(os.path.dirname(os.path.abspath(log)) or ".", exist_ok=True)
        sys.stdout.flush()
        sys.stderr.flush()
        _TEE = subprocess.Popen(["tee", "-a", log], stdin=subprocess.PIPE)
        os.dup2(_TEE.stdin.fileno(), 1)
        os.dup2(_TEE.stdin.fileno(), 2)
    return log


def flush_log():
    """Give the tee a moment to drain before the log is grepped (the engine subprocess writes through the same pipe)."""
    sys.stdout.flush()
    sys.stderr.flush()
    if _TEE is not None:
        time.sleep(0.5)


# ------------------------------------------------------------------------------------------------- prompts & digests
def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()


def load_prompts(path, batch, arm, prompt_len=PROMPT_LEN):
    """P37's prompt-file contract, byte for byte: `batch`, `prompts` (lists of ids), `prompts_sha256` (sha of
    json.dumps(prompts)), `rows_sha256`. Refuses a digest mismatch, a wrong row count or length, and -- except on the
    `sameprompt` control -- non-distinct rows. On `sameprompt` the returned prompts are B copies of row 0 and the info
    dict says so (the FILE's digest is still asserted and carried)."""
    pf = json.load(open(path))
    prompts = pf["prompts"]
    assert pf["batch"] == batch and len(prompts) == batch, (pf["batch"], len(prompts), batch)
    assert all(len(p) == prompt_len for p in prompts), f"prompt_len must be {prompt_len}"
    prompts_sha = sha256_json(prompts)
    assert prompts_sha == pf["prompts_sha256"], "prompt file digest mismatch"
    distinct = len(set(tuple(p) for p in prompts)) == batch
    info = {"prompts_sha256": prompts_sha, "rows_sha256": pf.get("rows_sha256"), "prompt_tokens": [len(p) for p in prompts],
            "rows_distinct_in_file": distinct, "prompt_file": os.path.basename(path)}
    if arm == "sameprompt":
        assert batch > 1, "sameprompt is a B>1 control (B copies of row 0); at B=1 it is the distinct arm"
        row0 = list(prompts[0])
        prompts = [list(row0) for _ in range(batch)]
        info["sameprompt"] = {"source_row": 0, "rows_identical": True, "copies": batch,
                              "row_sha256": sha256_json(row0), "file_prompts_sha256": prompts_sha,
                              "effective_prompts_sha256": sha256_json(prompts),
                              "note": "routing-collapse control (F13): every row is row 0 of the prompt file; the file's "
                                      "own digest is asserted above; rows_distinct_in_file says what the file held"}
        info["prompt_tokens"] = [len(p) for p in prompts]
    else:
        assert distinct, "rows must be distinct prompts"
    return prompts, info


def k8_text_sha(ids, prompt_len, steps):
    """K8's digest: sha256 of ids[:prompt_len + steps + 1] as int64 little-endian bytes
    (step_decomp._k8_window: `ids[:a.prompt_len + a.ppl_steps + 1].numpy().tobytes()` on a torch int64 tensor)."""
    import numpy as np
    return hashlib.sha256(np.asarray(list(ids[:prompt_len + steps + 1]), dtype="<i8").tobytes()).hexdigest()


def load_window(path, prompt_len, steps):
    """`k8_window_<src>.json` = {"ids": [...], "text_sha": "..."} (+ optional "source"); recomputes and asserts the sha."""
    w = json.load(open(path))
    ids = [int(x) for x in w["ids"]]
    need = prompt_len + steps + 1
    assert len(ids) >= need, f"window holds {len(ids)} ids, K8 needs prompt_len + steps + 1 = {need}"
    sha = k8_text_sha(ids, prompt_len, steps)
    assert sha == w["text_sha"], f"window digest mismatch: file says {w['text_sha'][:12]}, ids hash to {sha[:12]}"
    return ids, {"text_sha": sha, "text_sha_recomputed": True, "window_file": os.path.basename(path), "window_len": len(ids),
                 "ppl_source": w.get("source") or w.get("ppl_source") or _source_from_name(path)}


def _source_from_name(path):
    m = re.search(r"k8_window_([A-Za-z0-9]+)\.json$", os.path.basename(path))
    return m.group(1) if m else None


# ------------------------------------------------------------------------------------------------------ LLM kwargs
def build_llm_kwargs(arm, batch, model=MODEL_DEFAULT, rev=REV_DEFAULT, gpu_util=0.90, max_len=2048, prompt_len=PROMPT_LEN,
                     attn=None, moe="marlin"):
    """The exact `LLM(**kw)` per arm. `native` = the shipped defaults (prefix caching ON, O2 graphs, async scheduling,
    max_num_seqs/max_num_batched_tokens by tier, gpu_memory_utilization 0.92) -- only the model identity and the seed
    are passed; every resolved knob is read back by `resolved_config`. All other arms pin: no prefix caching, seed 0,
    the REGISTERED CAPACITY RULE `max_num_seqs=B` (16 on the B=16 arms, 1 on the B=1 arms; capture list
    [1,2,4,8,16,24,32] at B=16) and `max_model_len=2048`, `max_num_batched_tokens` covering the prompts in one chunk,
    the attention backend (FLASH_ATTN for bf16/fp16 KV; FLASHINFER for fp8 KV -- FA2 on sm_120 cannot take fp8 KV,
    fa_utils.py:282-308), `moe_backend="marlin"`. `nodetok` takes the graph kwargs; it differs only in SamplingParams."""
    assert arm in ARMS, f"unknown arm {arm!r}; arms: {ARMS}"
    if arm == "native":
        return dict(model=model, revision=rev, tokenizer_revision=rev, seed=0)
    kv = "fp8" if arm == "fp8kv" else "auto"
    return dict(model=model, revision=rev, tokenizer_revision=rev, gpu_memory_utilization=gpu_util, max_model_len=max_len,
                enable_prefix_caching=False, seed=0, disable_log_stats=True, tensor_parallel_size=1,
                enforce_eager=(arm == "eager"), kv_cache_dtype=kv, max_num_seqs=batch,
                max_num_batched_tokens=max(8192, prompt_len * batch),
                attention_backend=(attn or ("FLASHINFER" if kv == "fp8" else "FLASH_ATTN")), moe_backend=moe)


def sampling_kwargs(arm, n_tokens):
    """Greedy, fixed length, no EOS stop. The MATCHED arms keep `detokenize=True` (vLLM's shipped default: a
    comparator's serving loop is never trimmed to e4b's omission -- registered text, F7); the `nodetok` pair at both
    batch sizes runs `detokenize=False` so the incremental detokeniser's cost is MEASURED, not assumed."""
    return dict(temperature=0.0, max_tokens=n_tokens, ignore_eos=True, min_tokens=n_tokens,
                detokenize=(arm != "nodetok"))


def expected_capture_sizes(max_num_seqs):
    """The capture list vLLM 0.30.0 derives from max_num_seqs with decode_query_len 1 (config/vllm.py:2151-2200,
    compilation.py:699-716): max = min(max_num_seqs * 2, 512); [1,2,4] + range(8,256,8) + range(256,max+1,16), <= max.
    An EXPECTATION read from source -- the receipt records the engine's own list beside it."""
    cap = min(max_num_seqs * 2, 512)
    sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, cap + 1, 16))
    return [s for s in sizes if s <= cap]


# ------------------------------------------------------------------------------------------- generation accounting
def count_generated(outputs):
    return sum(len(c.token_ids) for r in outputs for c in r.outputs)


def assert_generated(outputs, n_tokens, batch):
    """Every row generated exactly n_tokens (a row that stopped early is not the registered workload)."""
    got = count_generated(outputs)
    if got != n_tokens * batch:
        raise AssertionError(f"{got} != {n_tokens * batch} (a row stopped early: not the registered workload)")
    short = [i for i, r in enumerate(outputs) for c in r.outputs if len(c.token_ids) != n_tokens]
    if short:
        raise AssertionError(f"rows {short[:8]} did not generate exactly {n_tokens} tokens")
    return got


def engine_prompt_tokens(outputs):
    """The engine's OWN prompt-token count per row (F20: a prepended BOS or a re-tokenisation would show here)."""
    return [len(getattr(r, "prompt_token_ids", None) or []) for r in outputs]


def generated_ids(outputs):
    return {str(i): [int(t) for t in r.outputs[0].token_ids] for i, r in enumerate(outputs)}


def slope(w_short, w_long, short=SHORT, long_=LONG, batch=1):
    """P20/P37's estimator, keys byte-compatible with p37_vllm.py: extra tokens over extra wall, min-of-reps and
    median-of-reps both recorded."""
    extra = (long_ - short) * batch
    d_min = min(w_long) - min(w_short)
    d_med = statistics.median(w_long) - statistics.median(w_short)
    out = dict(walls_short_s=[round(w, 4) for w in w_short], walls_long_s=[round(w, 4) for w in w_long],
               wall_short_s=round(min(w_short), 4), wall_long_s=round(min(w_long), 4),
               end_to_end_tok_s_long=round(long_ * batch / min(w_long), 1), slope_extra_tokens=extra)
    if d_min <= 0 or d_med <= 0:
        out.update(decode_tok_s=None, decode_ms_per_step=None, decode_tok_s_median=None, decode_ms_per_step_median=None,
                   slope_error=f"inverted slope (d_min={d_min:.4f}s, d_med={d_med:.4f}s): not a reading")
        return out
    out.update(decode_tok_s=round(extra / d_min, 1), decode_ms_per_step=round(d_min / (long_ - short) * 1e3, 4),
               decode_tok_s_median=round(extra / d_med, 1), decode_ms_per_step_median=round(d_med / (long_ - short) * 1e3, 4))
    return out


def ttft_reduce(walls, prompt_len):
    med = statistics.median(walls)
    return {"walls_s": [round(w, 5) for w in walls], "ttft_s_median": round(med, 5), "ttft_s_min": round(min(walls), 5),
            "ttft_ms_median": round(med * 1e3, 2), "prefill_tok_s_median": round(prompt_len / med, 1), "reps": len(walls)}


# ------------------------------------------------------------------------------------------------ logprob readers
def n_positions(container):
    try:
        return len(container)
    except TypeError:
        return 0


def position_entries(container, pos):
    """(token_ids, logprobs, ranks) at `pos` for either shape vLLM returns: a FlatLogprobs (parallel lists +
    start/end indices, vllm/logprobs.py:31-160) or a list[dict[int, Logprob] | None]. None when the position holds
    nothing (prompt position 0, or an empty container)."""
    if container is None or n_positions(container) <= pos:
        return None
    if hasattr(container, "start_indices") and hasattr(container, "token_ids"):
        s, e = container.start_indices[pos], container.end_indices[pos]
        if e <= s:
            return None
        return list(container.token_ids[s:e]), list(container.logprobs[s:e]), list(container.ranks[s:e])
    d = container[pos]
    if not d:
        return None
    toks = list(d.keys())
    return toks, [float(d[t].logprob) for t in toks], [d[t].rank for t in toks]


def lookup(entries, token):
    """The entry for `token`, or None when the engine did not return it (then the row is VOID, never approximated)."""
    if entries is None:
        return None
    toks, lps, ranks = entries
    try:
        i = toks.index(token)
    except ValueError:
        return None
    return {"logprob": float(lps[i]), "rank": (int(ranks[i]) if ranks[i] is not None else None), "n_entries": len(toks)}


# ------------------------------------------------------------------------------------------------- NLL: prefill
def prefill_nll(prompt_logprobs, ids, prompt_len, steps):
    """K8's 2048 tokens from ONE prompt_logprobs=0 call on ids[:prompt_len+steps+1]. Index derivation: vLLM's
    prompt_logprobs list has one slot per prompt token; slot 0 is None; slot i (i>=1) holds the logprob of token
    ids[i] conditioned on ids[:i] (gather target prompt_token_ids[i], gpu_model_runner.py:5684-5736; None prepended by
    vllm/logprobs.py:167-172). K8 scores cont[t+1] = ids[513+t] for t in 0..2047, i.e. slots prompt_len+1 .. prompt_len+steps."""
    n = prompt_len + steps + 1
    got = n_positions(prompt_logprobs)
    if got != n:
        raise ValueError(f"prompt_logprobs holds {got} positions, expected {n} (one per prompt token, slot 0 None)")
    if position_entries(prompt_logprobs, 0) is not None:
        raise ValueError("prompt_logprobs[0] is not empty: the position convention differs from v0.30.0 (vllm/logprobs.py:167-172)")
    nll, top1 = [], 0
    for j in range(prompt_len + 1, n):
        hit = lookup(position_entries(prompt_logprobs, j), ids[j])
        if hit is None:
            raise ValueError(f"prompt slot {j}: actual token {ids[j]} absent (vLLM always gathers the chosen token: "
                             f"gpu_model_runner.py:5717-5736) -- the convention changed or the output is not a prompt_logprobs list")
        nll.append(-hit["logprob"])
        top1 += int(hit["rank"] == 1)
    mean = sum(nll) / steps
    return {"per_token_nll": nll, "mean_nll": mean, "ppl": math.exp(mean), "steps": steps, "tokens_scored": steps,
            "top1_agreement": top1 / steps, "scored_index_range": [prompt_len + 1, n - 1],
            "index_derivation": "slot i of RequestOutput.prompt_logprobs = log p(ids[i] | ids[:i]); scored slots "
                                f"{prompt_len + 1}..{n - 1} == K8's cont[t+1] = ids[{prompt_len + 1}+t], t in 0..{steps - 1}"}


# -------------------------------------------------------------------------------------------------- NLL: served
def served_prompt(ids, prompt_len, t):
    """Request t sends ids[:prompt_len+1+t] (length 513+t at P=512) and scores the GENERATED position (F1)."""
    return list(ids[:prompt_len + 1 + t])


def served_target(ids, prompt_len, t):
    return int(ids[prompt_len + 1 + t])


def served_sampling_kwargs(mode, target):
    """full: logprobs=-1 (needs LLM(max_logprobs=-1)); token_ids: logprob_token_ids=[target] (exact, always present);
    topk:K: logprobs=K (target present only if ranked <= K or sampled)."""
    base = dict(max_tokens=1, temperature=0.0, detokenize=False, flat_logprobs=True)
    if mode == "full":
        return dict(base, logprobs=-1)
    if mode == "token_ids":
        return dict(base, logprob_token_ids=[int(target)])
    if mode.startswith("topk:"):
        return dict(base, logprobs=int(mode.split(":", 1)[1]))
    raise ValueError(f"unknown served logprobs mode {mode!r}")


def served_row(req_out, prompt_len_t, target):
    """One request's reading: the generated token (greedy argmax -> top-1 agreement), the target's logprob from the
    generated position's distribution, num_cached_tokens and the recomputed suffix = prompt_len_t - num_cached_tokens."""
    comp = req_out.outputs[0]
    if len(comp.token_ids) != 1:
        raise AssertionError(f"expected exactly 1 generated token, got {len(comp.token_ids)}")
    n_cached = getattr(req_out, "num_cached_tokens", None)
    entries = position_entries(getattr(comp, "logprobs", None), 0)
    hit = lookup(entries, target)
    row = {"num_cached_tokens": n_cached, "suffix": prompt_len_t - (n_cached or 0), "prompt_len": prompt_len_t,
           "top1": int(int(comp.token_ids[0]) == int(target)), "n_entries": (len(entries[0]) if entries else 0),
           "nll": (-hit["logprob"] if hit else None), "rank": (hit["rank"] if hit else None)}
    return row, entries


def served_reduce(rows, block_size, steps):
    """The served-shape summary: suffix histogram, the cache-hit VALIDITY predicate (F1: every request after the
    first must have num_cached_tokens >= prompt_len - block_size, else the row is relabelled prefill-shaped), VOID
    rows (target absent), mean NLL over the scored rows, top-1 agreement."""
    hist = Counter(r["suffix"] for r in rows)
    valid = [r for r in rows if r["nll"] is not None]
    void = len(rows) - len(valid)
    miss = [i for i, r in enumerate(rows) if i >= 1 and (r["num_cached_tokens"] is None or r["suffix"] > block_size)]
    tail = [r["suffix"] for r in rows[1:]]
    mean = (sum(r["nll"] for r in valid) / len(valid)) if valid else None
    label = (f"served (partial-block, M<={block_size})" if not miss
             else f"prefill-shaped (no cache hit) on {len(miss)}/{max(len(rows) - 1, 1)} rows")
    return {"suffix_histogram": {str(k): v for k, v in sorted(hist.items())},
            "suffix_mean_excl_first": (sum(tail) / len(tail) if tail else None),
            "suffix_max_excl_first": (max(tail) if tail else None), "suffix_first": (rows[0]["suffix"] if rows else None),
            "block_size": block_size, "cache_predicate": ("PASS" if not miss else "FAIL"), "n_cache_miss_rows": len(miss),
            "cache_miss_rows_head": miss[:20], "void_rows": void, "rows": len(rows), "steps_requested": steps,
            "tokens_scored": len(valid), "mean_nll": mean, "ppl": (math.exp(mean) if mean is not None else None),
            "top1_agreement": (sum(r["top1"] for r in rows) / len(rows) if rows else None),
            "mode_label": label, "verdict": ("VALID" if (void == 0 and len(rows) == steps) else "VOID"),
            "per_token_nll": [r["nll"] for r in rows]}


# --------------------------------------------------------------------------------------------- resolved config / mem
def _jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    name = getattr(v, "name", None)
    if name is not None and not callable(name):
        return str(name)
    return str(v)


def resolved_config(llm, kw):
    """Every knob the engine actually resolved (p37's keys first, byte-compatible; SC1's additions after)."""
    out = {}
    try:
        cfg = llm.llm_engine.vllm_config
        out["vllm_config"] = str(cfg)[:6000]
        mc = getattr(cfg, "model_config", None)
        sc = getattr(cfg, "scheduler_config", None)
        cc = getattr(cfg, "cache_config", None)
        comp = getattr(cfg, "compilation_config", None)
        ac = getattr(cfg, "attention_config", None)
        kc = getattr(cfg, "kernel_config", None)
        r = {"dtype": str(getattr(mc, "dtype", None)), "quantization": getattr(mc, "quantization", None),
             "max_model_len": getattr(mc, "max_model_len", None), "max_num_seqs": getattr(sc, "max_num_seqs", None),
             "max_num_batched_tokens": getattr(sc, "max_num_batched_tokens", None),
             "chunked_prefill": getattr(sc, "enable_chunked_prefill", getattr(sc, "chunked_prefill_enabled", None)),
             "kv_cache_dtype": getattr(cc, "cache_dtype", None), "enable_prefix_caching": getattr(cc, "enable_prefix_caching", None),
             "cudagraph_mode": str(getattr(comp, "cudagraph_mode", None)),
             "enforce_eager": getattr(mc, "enforce_eager", kw.get("enforce_eager")),
             "speculative": str(getattr(cfg, "speculative_config", None)),
             # SC1 additions
             "attention_backend": _jsonable(getattr(ac, "backend", None)), "moe_backend": getattr(kc, "moe_backend", None),
             "linear_backend": getattr(kc, "linear_backend", None),
             "cudagraph_capture_sizes": _jsonable(getattr(comp, "cudagraph_capture_sizes", None)),
             "max_cudagraph_capture_size": getattr(comp, "max_cudagraph_capture_size", None),
             "compilation_mode": str(getattr(comp, "mode", None)), "optimization_level": str(getattr(cfg, "optimization_level", None)),
             "async_scheduling": getattr(sc, "async_scheduling", None), "policy": getattr(sc, "policy", None),
             "long_prefill_token_threshold": getattr(sc, "long_prefill_token_threshold", None),
             "block_size": getattr(cc, "block_size", None), "num_gpu_blocks": getattr(cc, "num_gpu_blocks", None),
             "gpu_memory_utilization": getattr(cc, "gpu_memory_utilization", None),
             "kv_cache_memory_bytes": getattr(cc, "kv_cache_memory_bytes", None),
             "max_logprobs": getattr(mc, "max_logprobs", None), "logprobs_mode": getattr(mc, "logprobs_mode", None),
             "seed": getattr(mc, "seed", None)}
        try:
            r["use_v2_model_runner"] = bool(cfg.use_v2_model_runner)
        except Exception as e:  # noqa: BLE001
            r["use_v2_model_runner"] = f"unreadable: {e!r}"[:120]
        try:
            r["vocab_size"] = int(mc.get_vocab_size())
        except Exception:  # noqa: BLE001
            r["vocab_size"] = None
        if r.get("num_gpu_blocks") and r.get("block_size"):
            r["kv_tokens_capacity"] = int(r["num_gpu_blocks"]) * int(r["block_size"])
        out["resolved"] = _jsonable(r)
    except Exception as e:  # noqa: BLE001  -- the receipt says what it could not read; the log has the engine's config line
        out["vllm_config_error"] = repr(e)[:300]
    return out


def mem_snapshot(llm, gpu_util=None):
    """F15: the allocator's peak (torch.cuda.max_memory_allocated) READ ON THE WORKER via collective_rpc -- in the
    default multiprocess mode the engine lives in a subprocess and this process's own allocator is empty -- and the
    policy reservation (gpu_memory_utilization x total) recorded SEPARATELY. Nested function so cloudpickle ships it
    by value (a module-level function would be pickled by reference to a module the worker cannot import)."""
    snap = {}
    try:
        import torch
        if torch.cuda.is_available():
            snap["driver_process"] = {"max_memory_allocated": int(torch.cuda.max_memory_allocated()),
                                      "memory_allocated": int(torch.cuda.memory_allocated())}
    except Exception as e:  # noqa: BLE001
        snap["driver_process_error"] = repr(e)[:200]

    def _worker_mem(worker_self):  # noqa: ARG001 -- vLLM passes the worker as the first arg
        import torch
        free, total = torch.cuda.mem_get_info()
        return {"max_memory_allocated": int(torch.cuda.max_memory_allocated()), "memory_allocated": int(torch.cuda.memory_allocated()),
                "memory_reserved": int(torch.cuda.memory_reserved()), "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
                "mem_get_info_free": int(free), "mem_get_info_total": int(total),
                "device_name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())}

    try:
        snap["worker"] = llm.collective_rpc(_worker_mem)[0]
    except Exception as e:  # noqa: BLE001
        snap["worker_error"] = repr(e)[:300]
    total = (snap.get("worker") or {}).get("mem_get_info_total")
    if total and gpu_util:
        snap["reservation"] = {"gpu_memory_utilization": gpu_util, "total_bytes": int(total), "requested_bytes": int(gpu_util * total),
                               "note": "policy reservation (weights + activations + KV budget), NOT a demand -- compare with "
                                       "worker.max_memory_allocated and the 'Available KV cache memory' log line"}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,power.limit,clocks.max.sm,driver_version,name",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        snap["nvidia_smi"] = q.stdout.strip()[:300]
    except Exception as e:  # noqa: BLE001
        snap["nvidia_smi_error"] = repr(e)[:120]
    return snap


# --------------------------------------------------------------------------------------------------- engagement grep
def grep_engagement(log_path, patterns=None, max_matches=4):
    """The engagement lines the reducer's VOID rule reads FROM THE RECEIPT (not from the log): every pattern is a
    string read from the tag's source. `status` says whether a log was there to grep at all."""
    patterns = patterns or ENGAGEMENT_PATTERNS
    out = {"log": log_path, "status": "no_log"}
    if not log_path or not os.path.exists(log_path):
        out["marlin_moe"] = out["marlin_linear"] = out["cudagraphs_captured"] = None
        return out
    text = open(log_path, errors="replace").read()
    out["status"] = "grepped"
    out["log_bytes"] = len(text)
    for key, pat in patterns.items():
        found = []
        for m in re.finditer(pat, text):
            s = m.group(0).strip()
            if s not in found:
                found.append(s)
            if len(found) >= max_matches:
                break
        out[key] = found
    out["marlin_moe"] = any("'MARLIN'" in s for s in out.get("moe_backend_line", []))
    out["marlin_linear"] = any(s.startswith("Using MarlinLinearKernel") for s in out.get("linear_kernel_line", []))
    out["cudagraphs_captured"] = bool(out.get("cudagraph_capture_line") or out.get("graph_capture_finished_line"))
    out["v2_model_runner"] = bool(out.get("v2_runner_line"))
    out["attention_backend_logged"] = [re.match(r"Using ([A-Z0-9_]+) backend", s).group(1) for s in out.get("attention_backend_line", [])]
    return out


def base_receipt(kind, arm, kw, extra=None):
    """The header every SC1 vLLM receipt carries."""
    rec = {"engine": "vllm", "driver": kind, "arm": arm, "vllm_tag": TAG, "vllm_tag_commit": TAG_COMMIT,
           "llm_kwargs": _jsonable(kw), "kwarg_sources": KWARG_SOURCES,
           "vast_instance_id": env("INSTANCE_ID"), "host_pid": os.getpid(),
           "env": {k: os.environ.get(k) for k in ("VLLM_LOGGING_LEVEL", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_V2_MODEL_RUNNER",
                                                   "VLLM_ALLOW_INSECURE_SERIALIZATION", "VLLM_ATTENTION_BACKEND", "CUDA_VISIBLE_DEVICES")}}
    try:
        import vllm
        rec["vllm_version"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        rec["vllm_version"] = f"unreadable: {e!r}"[:80]
    try:
        import torch
        rec["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        rec["torch"] = None
    if extra:
        rec.update(extra)
    return rec


def write_receipt(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    json.dump(_jsonable(rec), open(tmp, "w"), indent=1)
    os.replace(tmp, path)
