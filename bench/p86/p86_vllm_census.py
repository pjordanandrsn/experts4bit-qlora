"""p86_vllm_census.py -- the vLLM census arm of lane P86 (bench/p86/PREREG-p86.md; e4b#564).

Where does vLLM's decode step go, kernel by kernel, on the same box and prompts as e4b's census? P37's vLLM arm
(`bench/h2h-20260905/p37/p37_vllm.py`) times decode by the SLOPE method: generate 32 and 128 tokens from the same
prompts, and the extra tokens over the extra wall isolate decode. This arm applies the same slope to the GPU's own
kernel record:
- the engine runs IN-PROCESS (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), so the model runner's kernels launch in this process
  and `torch.profiler` (CUPTI) records them, including the kernels of CUDA-graph replays;
- each length is warmed once (untimed, unprofiled), then generated once under the profiler;
- per kernel name, (total in the 128-token run - total in the 32-token run) / 96 = its time per decode step. Prefill,
  load and the prompt's scheduling are identical in both runs and cancel.

The profiled runs are not timed: the profiler costs host time. The step time this census is read against comes from
P37's timing arm on the same box (P86's `vllm_graph` arms). Same LLM settings as P37's graph_r1 arm. Env: P86_BATCH,
P86_PROMPTS, P86_MODEL, P86_REV, P86_OUT (and P86_SHORT / P86_LONG, rehearsal only).
"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # before vllm is imported: the engine in this process

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import vllm  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

BATCH = int(os.environ["P86_BATCH"])
MODEL = os.environ.get("P86_MODEL", "Qwen/Qwen3-30B-A3B-GPTQ-Int4")
REV = os.environ.get("P86_REV", "9b534e4318b7ebc3c961a839f13eb18b1833f441")
OUT = os.environ["P86_OUT"]
SHORT, LONG = int(os.environ.get("P86_SHORT", "32")), int(os.environ.get("P86_LONG", "128"))
GPU_UTIL = float(os.environ.get("P86_GPU_UTIL", "0.90"))       # rehearsal only (a shared card); the lane uses 0.90
MAX_LEN = int(os.environ.get("P86_MAX_LEN", "2048"))

pf = json.load(open(os.environ["P86_PROMPTS"]))
prompts = pf["prompts"]
assert pf["batch"] == BATCH and len(prompts) == BATCH, (pf["batch"], len(prompts), BATCH)
assert len(set(tuple(p) for p in prompts)) == BATCH, "rows must be distinct prompts"
prompts_sha = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
assert prompts_sha == pf["prompts_sha256"], "prompt file digest mismatch"

kw = dict(model=MODEL, revision=REV, tokenizer_revision=REV, gpu_memory_utilization=GPU_UTIL, max_model_len=MAX_LEN,
          enable_prefix_caching=False, seed=0, disable_log_stats=True, tensor_parallel_size=1, enforce_eager=False,
          kv_cache_dtype="auto")
out = {"engine": "vllm", "arm": "census", "vllm_version": vllm.__version__, "torch": torch.__version__, "model": MODEL,
       "revision": REV, "batch": BATCH, "short": SHORT, "long": LONG, "decode_steps": LONG - SHORT,
       "in_process": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "prompts_sha256": prompts_sha,
       "llm_kwargs": dict(kw), "method": "per-kernel slope: (kernel total in the LONG run - in the SHORT run) / (LONG - SHORT)"}

t0 = time.perf_counter()
llm = LLM(**kw)
out["load_s"] = round(time.perf_counter() - t0, 1)
try:
    comp = getattr(llm.llm_engine.vllm_config, "compilation_config", None)
    out["cudagraph_mode"] = str(getattr(comp, "cudagraph_mode", None))
except Exception as e:      # recorded, not fatal: the timing arm's receipt carries the full resolved config
    out["cudagraph_mode_error"] = repr(e)[:200]
reqs = [{"prompt_token_ids": p} for p in prompts]


def _dev_us(evt):
    """The event's own device time in us (torch >= 2.4 names it self_device_time_total)."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def census(n_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=n_tokens, ignore_eos=True, min_tokens=n_tokens)
    llm.generate(reqs, sp, use_tqdm=False)                       # warm: graphs, allocator, compile caches
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        o = llm.generate(reqs, sp, use_tqdm=False)
        torch.cuda.synchronize()
    got = sum(len(c.token_ids) for r in o for c in r.outputs)
    assert got == n_tokens * BATCH, f"{got} != {n_tokens * BATCH} (a row stopped early: not the registered workload)"
    rows = {}
    for e in prof.key_averages():
        us = _dev_us(e)
        if us <= 0.0:
            continue
        r = rows.setdefault(e.key, {"name": e.key, "us": 0.0, "calls": 0})
        r["us"] += us
        r["calls"] += int(e.count)
    return rows


k_short = census(SHORT)
k_long = census(LONG)
steps = LONG - SHORT
per_step = []
for name in sorted(set(k_short) | set(k_long)):
    a, b = k_short.get(name, {"us": 0.0, "calls": 0}), k_long.get(name, {"us": 0.0, "calls": 0})
    per_step.append({"name": name, "us_per_step": (b["us"] - a["us"]) / steps, "calls_per_step": (b["calls"] - a["calls"]) / steps,
                     "us_short": a["us"], "us_long": b["us"], "calls_short": a["calls"], "calls_long": b["calls"]})
per_step.sort(key=lambda r: -r["us_per_step"])
out.update(kernels=per_step, kernel_ms_per_step=round(sum(r["us_per_step"] for r in per_step) / 1e3, 4),
           kernel_ms_short=round(sum(r["us"] for r in k_short.values()) / 1e3, 3),
           kernel_ms_long=round(sum(r["us"] for r in k_long.values()) / 1e3, 3))
json.dump(out, open(OUT, "w"), indent=1)
print("P86VLLMCENSUS " + json.dumps({k: out[k] for k in ("batch", "vllm_version", "decode_steps", "kernel_ms_per_step",
                                                        "kernel_ms_short", "kernel_ms_long", "in_process")}), flush=True)
for r in per_step[:12]:
    print(f"  {r['us_per_step']:9.1f} us/step  {r['calls_per_step']:6.2f} calls/step  {r['name'][:110]}", flush=True)
