"""sc1_vllm_arm.py -- the vLLM timed arm of lane SC1 (experts4bit-qlora#846): `bench/h2h-20260905/p37/p37_vllm.py`
EXTENDED. Receipt keys, the slope method and the prompt-file contract are byte-compatible with P37/P58 (so
`p58_reduce.py`'s logic reads these receipts unchanged); everything SC1 adds is an extra key.

Arms (SC1_ARM; P37_ARM accepted):
  graph_r1 | graph_r2  default -O2 graphs (FULL_AND_PIECEWISE), kv_cache_dtype auto, prefix caching OFF, seed 0,
                       max_num_seqs=B (capture list [1,2,4,8,16,24,32] at B=16 -- recorded), max_num_batched_tokens
                       covering the 512-token prompts in one chunk, attention FLASH_ATTN + moe_backend marlin PINNED,
                       detokenize=False (F7)                                                   -- PRIMARY, two draws
  eager                enforce_eager=True (no graphs), otherwise as graph
  fp8kv                kv_cache_dtype="fp8", attention FLASHINFER pinned (FA2 on sm_120 cannot take fp8 KV); the receipt
                       carries the k_scale/v_scale provenance: 1.0 when the checkpoint ships none (F9)
  sameprompt           B copies of row 0 of the prompt file -- the routing-collapse control (F13); the receipt says the
                       rows are identical and carries the FILE's digest
  native               the shipped defaults: ONLY model/revision/seed are passed (prefix caching stays ON, 0.92 util,
                       tier defaults for max_num_seqs / max_num_batched_tokens, detokenizer on); every knob read back
  detok                the graph config with detokenize=True (F7: the incremental detokeniser's cost, measured)

Env (SC1_* first, P37_* fallback): ARM, BATCH, PROMPTS, OUT  [MODEL, REV, LOG, INSTANCE_ID, GPU_UTIL=0.90, MAX_LEN=2048,
SHORT=32, LONG=128, REPS=3, ATTN_BACKEND, MOE_BACKEND=marlin, INPROC=0]. With SC1_LOG set, fd 1/2 are tee'd into that
file BEFORE vllm is imported (the engine-core subprocess inherits them, VLLM_LOGGING_LEVEL=INFO is set), and the log is
grepped for the engagement lines into the receipt (`engagement`) so the reducer's VOID rule reads the receipt, not the log.
Memory (F15): the worker's `torch.cuda.max_memory_allocated` (via collective_rpc) and the `gpu_memory_utilization`
reservation are recorded separately, after load and after the timed runs.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import sc1_vllm_common as C  # noqa: E402

if __name__ == "__main__":
    C.early_setup()                    # BEFORE vllm is imported: log level, tee, in-process switch

import json  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

KV_SCALE_PROVENANCE = {
    "mechanism": "vLLM 0.30.0 fp8 KV without checkpoint scales stores K/V at k_scale = v_scale = 1.0 "
                 "(vllm/model_executor/layers/quantization/kv_cache.py:123-129; both scales load as -1.0 sentinels and "
                 "are replaced by 1.0) and logs 'Using KV cache scaling factor 1.0 for fp8_e4m3' (kv_cache.py:163-168)",
    "checkpoint": "Qwen/Qwen3-30B-A3B-GPTQ-Int4 ships no k_scale/v_scale tensors (GPTQ weight-only quantisation; "
                  "the upstream inspection and the P37/P58 fp8kv runs saw the 1.0 default) -- the warning line in "
                  "`engagement.kv_scale_warning_line` is the runtime proof for THIS run",
    "calculate_kv_scales": "NOT available at v0.30.0 (no such EngineArgs/CacheConfig field; the only mention is a comment "
                           "in vllm/models/kimi_k3/nvidia/mla.py:473) -- there is no on-the-fly calibration variant",
    "e4b_side": "e4b's Fp8PagedKV carries per-row absmax scales (k_groups 4); the two fp8 caches are NOT the same "
                "quantiser -- the headline pairing stays kv auto (F9)",
}


def main():
    arm = C.env("ARM", required=True)
    batch = int(C.env("BATCH", required=True))
    model = C.env("MODEL", C.MODEL_DEFAULT)
    rev = C.env("REV", C.REV_DEFAULT)
    out_path = C.env("OUT", required=True)
    log = C.env("LOG")
    short, long_, reps = C.env_int("SHORT", C.SHORT), C.env_int("LONG", C.LONG), C.env_int("REPS", 3)
    gpu_util, max_len = C.env_float("GPU_UTIL", 0.90), C.env_int("MAX_LEN", 2048)

    prompts, pinfo = C.load_prompts(C.env("PROMPTS", required=True), batch, arm)
    kw = C.build_llm_kwargs(arm, batch, model, rev, gpu_util=gpu_util, max_len=max_len,
                            attn=C.env("ATTN_BACKEND"), moe=C.env("MOE_BACKEND", "marlin"))
    rec = C.base_receipt("arm", arm, kw, {
        "model": model, "revision": rev, "batch": batch, "status": "ok",
        "method": "slope(32->128) isolates decode; min-of-3 (P20 estimator) and median-of-3 both recorded",
        "prompts": "identical token ids to the e4b arms (step_decomp._k8_window, wikitext-2 test, 512-token rows)",
        **pinfo,
        "generation": {"temperature": 0.0, "ignore_eos": True, "min_tokens": "= max_tokens", "greedy": True,
                       "detokenize": C.sampling_kwargs(arm, 1)["detokenize"]},
        "short": short, "long": long_, "reps": reps,
        "expected_cudagraph_capture_sizes": (None if arm in ("native", "eager") else C.expected_capture_sizes(batch)),
        "native_note": ("shipped defaults: only model/revision/tokenizer_revision/seed passed; see `resolved` for every knob "
                        "(prefix caching ON, O2 FULL_AND_PIECEWISE, async scheduling, tier max_num_seqs/batched_tokens, "
                        "gpu_memory_utilization 0.92, detokenize on); no speculative decoding (lane-wide)" if arm == "native" else None),
    })
    if arm == "fp8kv":
        rec["kv_scale_provenance"] = KV_SCALE_PROVENANCE
    try:
        _run(rec, arm, batch, prompts, kw, short, long_, reps, log, out_path)
    except BaseException as e:  # noqa: BLE001 -- the partial receipt names the failure; the alarm/driver rc still propagates
        rec["status"] = "harness_error"
        rec["error"] = repr(e)[:400]
        C.flush_log()
        rec["engagement"] = C.grep_engagement(log)
        C.write_receipt(out_path, rec)
        raise


def _run(rec, arm, batch, prompts, kw, short, long_, reps, log, out_path):
    t0 = time.perf_counter()
    llm = LLM(**kw)
    rec["load_s"] = round(time.perf_counter() - t0, 1)
    rec.update(C.resolved_config(llm, kw))
    C.flush_log()
    rec["engagement"] = C.grep_engagement(log)
    rec["mem_after_load"] = C.mem_snapshot(llm, kw.get("gpu_memory_utilization"))
    res = rec.get("resolved") or {}
    if rec.get("expected_cudagraph_capture_sizes") is not None and res.get("cudagraph_capture_sizes"):
        rec["capture_sizes_match_expectation"] = list(res["cudagraph_capture_sizes"]) == rec["expected_cudagraph_capture_sizes"]

    reqs = [{"prompt_token_ids": p} for p in prompts]

    def run(n_tokens):
        sp = SamplingParams(**C.sampling_kwargs(arm, n_tokens))
        o = llm.generate(reqs, sp, use_tqdm=False)                              # warm (untimed)
        C.assert_generated(o, n_tokens, batch)
        walls, windows, gen, ptoks = [], [], None, None
        for _ in range(reps):
            epoch0 = time.time()
            t = time.perf_counter()
            o = llm.generate(reqs, sp, use_tqdm=False)
            walls.append(time.perf_counter() - t)
            windows.append([round(epoch0, 3), round(time.time(), 3)])
            C.assert_generated(o, n_tokens, batch)
            gen, ptoks = C.generated_ids(o), C.engine_prompt_tokens(o)
        return walls, windows, gen, ptoks

    w_short, win_short, gen_short, pt_short = run(short)
    w_long, win_long, gen_long, pt_long = run(long_)
    rec.update(C.slope(w_short, w_long, short, long_, batch))
    rec.update(tokens=gen_long, tokens_short=gen_short, prompt_tokens_engine=pt_long,
               timed_windows_epoch={"short": win_short, "long": win_long,
                                    "note": "wall-clock bounds of each timed generate, for an external power sampler (F14/F16)"},
               ttft_note="not measured here: see sc1_vllm_ttft.py (F18 definition); informational only, no ratio (P37 fixture)")
    if any(n != C.PROMPT_LEN for n in pt_long + pt_short):
        rec["status"] = "void"
        rec["void_reason"] = f"engine prompt-token count {sorted(set(pt_long + pt_short))} != {C.PROMPT_LEN} (F20)"
    rec["mem_after_runs"] = C.mem_snapshot(llm, kw.get("gpu_memory_utilization"))
    C.flush_log()
    rec["engagement"] = C.grep_engagement(log)
    C.write_receipt(out_path, rec)
    summary = {k: rec.get(k) for k in ("arm", "batch", "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median",
                                       "end_to_end_tok_s_long", "vllm_version", "prompts_sha256", "status")}
    print("SC1VLLM " + json.dumps(summary), flush=True)
    print("P37VLLM " + json.dumps(summary), flush=True)        # the runner's summary grep (sc1_run.sh greps P37VLLM)


if __name__ == "__main__":
    main()
