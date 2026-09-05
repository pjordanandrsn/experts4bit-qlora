"""p37_vllm.py -- the vLLM arm of lane p37 (P37-VLLM-PREREG.md). Derived from P20's h2h_vllm_wt.py with two changes:
the prompts are the EXACT token ids step_decomp._k8_window produced for the e4b arms (prompts_b{B}.json, fed as
prompt_token_ids -- no tokenizer runs here), and every serving knob is recorded from the engine's own config.

Decode is isolated by the SLOPE method (P20's protocol, unchanged): generate 32 and 128 tokens from the same
prompts; extra tokens / extra wall. Prefill, load and the prompt's scheduling cancel; per-step scheduling and
detokenisation do not (vLLM's whole serving loop is inside its number; e4b's replay window has no scheduler).
Arms (P37_ARM): graph_r1 / graph_r2 = default CUDA graphs, kv auto (PRIMARY); eager = enforce_eager=True;
fp8kv = default graphs + kv_cache_dtype="fp8" (e4b's cache class). Env: P37_BATCH, P37_PROMPTS, P37_MODEL, P37_REV, P37_OUT.
"""
import hashlib
import json
import os
import statistics
import sys
import time

import vllm
import torch
from vllm import LLM, SamplingParams

ARM = os.environ["P37_ARM"]
BATCH = int(os.environ["P37_BATCH"])
MODEL = os.environ.get("P37_MODEL", "Qwen/Qwen3-30B-A3B-GPTQ-Int4")
REV = os.environ.get("P37_REV", "9b534e4318b7ebc3c961a839f13eb18b1833f441")
OUT = os.environ["P37_OUT"]
SHORT, LONG = 32, 128

pf = json.load(open(os.environ["P37_PROMPTS"]))
prompts = pf["prompts"]
assert pf["batch"] == BATCH and len(prompts) == BATCH, (pf["batch"], len(prompts), BATCH)
assert all(len(p) == 512 for p in prompts), "prompt_len must be 512"
assert len(set(tuple(p) for p in prompts)) == BATCH, "rows must be distinct prompts"
prompts_sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
assert prompts_sha == pf["prompts_sha256"], "prompt file digest mismatch"

kw = dict(model=MODEL, revision=REV, tokenizer_revision=REV, gpu_memory_utilization=0.90, max_model_len=2048,
          enable_prefix_caching=False, seed=0, disable_log_stats=True, tensor_parallel_size=1,
          enforce_eager=(ARM == "eager"), kv_cache_dtype=("fp8" if ARM == "fp8kv" else "auto"))
out = {"engine": "vllm", "arm": ARM, "vllm_version": vllm.__version__, "torch": torch.__version__, "model": MODEL, "revision": REV,
       "batch": BATCH, "method": "slope(32->128) isolates decode; min-of-3 (P20 estimator) and median-of-3 both recorded",
       "prompts": "identical token ids to the e4b arms (step_decomp._k8_window, wikitext-2 test, 512-token rows)",
       "prompts_sha256": prompts_sha, "rows_sha256": pf["rows_sha256"], "prompt_tokens": [len(p) for p in prompts],
       "llm_kwargs": {k: v for k, v in kw.items()}, "generation": {"temperature": 0.0, "ignore_eos": True, "min_tokens": "= max_tokens", "greedy": True},
       "vast_instance_id": os.environ.get("P37_INSTANCE_ID")}

t0 = time.perf_counter()
llm = LLM(**kw)
out["load_s"] = round(time.perf_counter() - t0, 1)
try:   # every knob the engine actually resolved (dtype, quantization, cudagraph mode, max_num_seqs, kv dtype, attention backend)
    cfg = llm.llm_engine.vllm_config
    out["vllm_config"] = str(cfg)[:6000]
    mc = getattr(cfg, "model_config", None); sc = getattr(cfg, "scheduler_config", None); cc = getattr(cfg, "cache_config", None); comp = getattr(cfg, "compilation_config", None)
    out["resolved"] = {"dtype": str(getattr(mc, "dtype", None)), "quantization": getattr(mc, "quantization", None), "max_model_len": getattr(mc, "max_model_len", None),
                       "max_num_seqs": getattr(sc, "max_num_seqs", None), "max_num_batched_tokens": getattr(sc, "max_num_batched_tokens", None),
                       "chunked_prefill": getattr(sc, "enable_chunked_prefill", getattr(sc, "chunked_prefill_enabled", None)),
                       "kv_cache_dtype": getattr(cc, "cache_dtype", None), "enable_prefix_caching": getattr(cc, "enable_prefix_caching", None),
                       "cudagraph_mode": str(getattr(comp, "cudagraph_mode", None)), "enforce_eager": getattr(mc, "enforce_eager", kw["enforce_eager"]),
                       "speculative": str(getattr(cfg, "speculative_config", None))}
except Exception as e:  # the receipt says what it could not read; the log carries the engine's own config line
    out["vllm_config_error"] = repr(e)[:300]

reqs = [{"prompt_token_ids": p} for p in prompts]


def run(n_tokens, reps=3):
    sp = SamplingParams(temperature=0.0, max_tokens=n_tokens, ignore_eos=True, min_tokens=n_tokens)
    llm.generate(reqs, sp, use_tqdm=False)                          # warm (untimed)
    walls, gen = [], None
    for _ in range(reps):
        t = time.perf_counter()
        o = llm.generate(reqs, sp, use_tqdm=False)
        walls.append(time.perf_counter() - t)
        got = sum(len(c.token_ids) for r in o for c in r.outputs)
        assert got == n_tokens * BATCH, f"{got} != {n_tokens * BATCH} (a row stopped early: not the registered workload)"
        gen = {str(i): list(map(int, r.outputs[0].token_ids)) for i, r in enumerate(o)}
    return walls, gen


w_short, _ = run(SHORT)
w_long, gen_long = run(LONG)
extra = (LONG - SHORT) * BATCH
d_min = min(w_long) - min(w_short)
d_med = statistics.median(w_long) - statistics.median(w_short)
out.update(
    walls_short_s=[round(w, 4) for w in w_short], walls_long_s=[round(w, 4) for w in w_long],
    wall_short_s=round(min(w_short), 4), wall_long_s=round(min(w_long), 4),
    decode_tok_s=round(extra / d_min, 1), decode_ms_per_step=round(d_min / (LONG - SHORT) * 1e3, 4),
    decode_tok_s_median=round(extra / d_med, 1), decode_ms_per_step_median=round(d_med / (LONG - SHORT) * 1e3, 4),
    end_to_end_tok_s_long=round(LONG * BATCH / min(w_long), 1),
    ttft_note="not measured: offline generate exposes no request-level TTFT; informational only, no ratio (P37 fixture)",
    tokens=gen_long,
)
json.dump(out, open(OUT, "w"), indent=1)
print("P37VLLM " + json.dumps({k: out[k] for k in ("arm", "batch", "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median",
                                                  "end_to_end_tok_s_long", "vllm_version", "prompts_sha256")}), flush=True)
