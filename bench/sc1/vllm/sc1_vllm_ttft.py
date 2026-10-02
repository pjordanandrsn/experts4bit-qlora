"""sc1_vllm_ttft.py -- TTFT for the vLLM side of lane SC1 (experts4bit-qlora#846), per the design review's definition
(F18): the WALL of a `max_tokens=1` request at the engine's Python entry, warm engine, prompt NOT cached (prefix caching
OFF, so nothing is cached between repeats), median of 3; the engine's own prompt timer beside it where one exists (the
offline `LLM` exposes none -- stated in the receipt). The prefill chunk budget (`max_num_batched_tokens`) and
`max_model_len` are recorded because at 4096 tokens the comparison is chunking-as-configured.

Input: a ONE-row prompt file in P37's shape (`prompts_b1.json` = 512 tokens, or `prompts_b1_4096.json` = 4096 tokens;
`batch` 1, `prompts_sha256` asserted). The 512 and 4096 arms are separate engine starts (max_model_len >= prompt + 8).

Env (SC1_* first, P37_* fallback): PROMPTS, OUT  [ARM=graph_r1|eager|fp8kv, MODEL, REV, LOG, INSTANCE_ID, GPU_UTIL=0.90,
MAX_LEN=max(2048, prompt+8), MAX_BATCHED=max(8192, prompt), CHUNKED=1, REPS=3, ATTN_BACKEND, MOE_BACKEND=marlin, INPROC=0].
"""
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import sc1_vllm_common as C  # noqa: E402

if __name__ == "__main__":
    C.early_setup()

import json  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

TTFT_ARMS = ("graph_r1", "eager", "fp8kv")


def load_one_row(path):
    pf = json.load(open(path))
    prompts = pf["prompts"]
    assert pf["batch"] == 1 and len(prompts) == 1, f"TTFT takes a one-row prompt file (batch={pf['batch']}, rows={len(prompts)})"
    assert C.sha256_json(prompts) == pf["prompts_sha256"], "prompt file digest mismatch"
    return [int(t) for t in prompts[0]], {"prompts_sha256": pf["prompts_sha256"], "rows_sha256": pf.get("rows_sha256"),
                                           "prompt_file": os.path.basename(path)}


def main():
    out_path = C.env("OUT", required=True)
    log = C.env("LOG")
    arm = C.env("ARM", "graph_r1")
    assert arm in TTFT_ARMS, f"TTFT arm must be one of {TTFT_ARMS}"
    model, rev = C.env("MODEL", C.MODEL_DEFAULT), C.env("REV", C.REV_DEFAULT)
    reps, gpu_util = C.env_int("REPS", 3), C.env_float("GPU_UTIL", 0.90)
    row, pinfo = load_one_row(C.env("PROMPTS", required=True))
    plen = len(row)
    max_len = C.env_int("MAX_LEN", max(2048, plen + 8))
    assert max_len >= plen + 1, f"max_model_len {max_len} cannot hold prompt {plen} + 1"
    max_batched = C.env_int("MAX_BATCHED", C.capped_batched_tokens(1, max_len, max(8192, plen)))   # A5
    chunked = C.env("CHUNKED", "1") == "1"
    detok = C.env("DETOKENIZE", "1") == "1"          # shipped default kept: a comparator's loop is never trimmed (F7)

    kw = C.build_llm_kwargs(arm, 1, model, rev, gpu_util=gpu_util, max_len=max_len, prompt_len=plen,
                            attn=C.env("ATTN_BACKEND"), moe=C.env("MOE_BACKEND", "marlin"))
    kw["max_num_batched_tokens"] = max_batched
    kw["enable_chunked_prefill"] = chunked
    rec = C.base_receipt("ttft", arm, kw, {
        "model": model, "revision": rev, "batch": 1, "prompt_len": plen, "status": "ok", **pinfo,
        "method": "TTFT = wall of LLM.generate([{prompt_token_ids}], SamplingParams(max_tokens=1, temperature=0), "
                  "use_tqdm=False) at the Python entry; warm engine (one untimed call first); prefix caching OFF so "
                  "nothing is cached between repeats; median of REPS (F18)",
        "generation": {"temperature": 0.0, "max_tokens": 1, "greedy": True, "detokenize": detok,
                       "note": "detokenize stays at vLLM's shipped default (True) unless SC1_DETOKENIZE=0: the comparator's "
                               "loop is never trimmed; one token's detokenisation is inside the TTFT by design"},
        "engine_timer": "none: the offline LLM exposes no request-level TTFT (RequestOutput.metrics is None with "
                        "disable_log_stats=True); prefill tok/s below is derived from the wall, not an engine timer",
        "chunks_expected": math.ceil(plen / max_batched) if chunked else 1, "reps": reps,
    })
    try:
        _run(rec, row, plen, kw, reps, log, out_path, detok)
    except BaseException as e:  # noqa: BLE001
        rec["status"] = "harness_error"
        rec["error"] = repr(e)[:400]
        C.flush_log()
        rec["engagement"] = C.grep_engagement(log)
        C.write_receipt(out_path, rec)
        raise


def _run(rec, row, plen, kw, reps, log, out_path, detok=True):
    t0 = time.perf_counter()
    C.check_budget(kw)
    llm = LLM(**kw)
    rec["load_s"] = round(time.perf_counter() - t0, 1)
    rec.update(C.resolved_config(llm, kw))
    C.flush_log()
    rec["engagement"] = C.grep_engagement(log)
    rec["mem_after_load"] = C.mem_snapshot(llm, kw.get("gpu_memory_utilization"))

    sp = SamplingParams(max_tokens=1, temperature=0.0, detokenize=detok)
    req = [{"prompt_token_ids": row}]

    def one():
        o = llm.generate(req, sp, use_tqdm=False)
        C.assert_generated(o, 1, 1)
        return o[0]

    warm = one()                                                   # warm engine (untimed)
    walls, cached, first_ids, ptoks, windows = [], [], [], [], []
    for _ in range(reps):
        epoch0 = time.time()
        t = time.perf_counter()
        o = one()
        walls.append(time.perf_counter() - t)
        windows.append([round(epoch0, 3), round(time.time(), 3)])
        cached.append(getattr(o, "num_cached_tokens", None))
        first_ids.append(int(o.outputs[0].token_ids[0]))
        ptoks.append(len(getattr(o, "prompt_token_ids", None) or []))
    rec.update(C.ttft_reduce(walls, plen))
    rec.update(num_cached_tokens_per_rep=cached, first_token_ids=first_ids, prompt_tokens_engine=ptoks,
               warm_first_token_id=int(warm.outputs[0].token_ids[0]), timed_windows_epoch=windows)
    if any(c for c in cached if c):
        rec["status"] = "void"
        rec["void_reason"] = f"a repeat hit the prefix cache (num_cached_tokens={cached}); TTFT must prefill the whole prompt"
    if any(n != plen for n in ptoks):
        rec["status"] = "void"
        rec["void_reason"] = f"engine prompt-token count {sorted(set(ptoks))} != {plen} (F20)"
    rec["mem_after_runs"] = C.mem_snapshot(llm, kw.get("gpu_memory_utilization"))
    C.flush_log()
    rec["engagement"] = C.grep_engagement(log)
    C.write_receipt(out_path, rec)
    res = rec.get("resolved") or {}
    print("SC1VLLMTTFT " + json.dumps({"arm": rec["arm"], "prompt_len": plen, "ttft_ms_median": rec["ttft_ms_median"],
                                       "ttft_s_min": rec["ttft_s_min"], "prefill_tok_s_median": rec["prefill_tok_s_median"],
                                       "max_num_batched_tokens": res.get("max_num_batched_tokens"), "max_model_len": res.get("max_model_len"),
                                       "chunked_prefill": res.get("chunked_prefill"), "status": rec["status"]}), flush=True)


if __name__ == "__main__":
    main()
