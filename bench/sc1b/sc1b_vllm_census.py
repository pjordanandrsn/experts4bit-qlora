#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc1b_vllm_census.py -- lane SC1b (#846): bracket exactly N steady decode steps of vLLM 0.30.0's `gptq_graph` arm for an
Nsight Systems capture (`nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...`).

SC1's vLLM driver (bench/sc1/vllm/sc1_vllm_common.py) builds the exact `LLM(**kw)` of SC1's arm; this script adds only
`profiler_config={"profiler": "cuda", "delay_iterations": D, "max_iterations": N}` -- vLLM's CudaProfilerWrapper calls
cudaProfilerStart/Stop INSIDE the worker (the EngineCore child; VLLM_WORKER_MULTIPROC_METHOD=spawn, set by the box for
every vLLM pass), counted per execute_model, starting at the D-th call after start_profile (vllm/profiler/wrapper.py:
105-111). execute_model call 1 is the prefill and call k is decode step k-1, so D = K + 3 puts the first profiled step at decode
step K + 2 = 34: the same positions (34-97) e4b's bracket reads (its ramp ends on the first full decode step, then K more).

  env: SC1 vLLM arm env (SC1_ARM=graph_r1, SC1_BATCH, SC1_PROMPTS, SC1_MODEL, SC1_REV, SC1_OUT, SC1_LOG)
  sc1b_vllm_census.py [--skip 32 --steps 64 --tokens 160]
  sc1b_vllm_census.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SKIP, STEPS, TOKENS = 32, 64, 160


def profiler_config(skip: int, steps: int) -> dict:
    """execute_model call 1 is the prefill (one step: B x 512 <= max_num_batched_tokens) and call k is decode step k - 1;
    profiling starts at call delay = skip + 3, i.e. decode step skip + 2 (34 for K = 32), e4b's first bracketed step."""
    return {"profiler": "cuda", "delay_iterations": skip + 3, "max_iterations": steps}


def selftest():
    pc = profiler_config(32, 64)
    assert pc == {"profiler": "cuda", "delay_iterations": 35, "max_iterations": 64}, pc
    # the window ends at decode step 97, inside the 160-token generation
    assert SKIP + 2 + STEPS - 1 <= TOKENS - 1, (SKIP, STEPS, TOKENS)
    print("selftest OK (2 cases)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", type=int, default=SKIP)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tokens", type=int, default=TOKENS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm"))
    import sc1_vllm_common as C                                          # SC1's bytes, staged under $W/vllm/
    C.early_setup()
    from vllm import LLM, SamplingParams
    arm = C.env("ARM", required=True)
    batch = int(C.env("BATCH", required=True))
    model, rev = C.env("MODEL", C.MODEL_DEFAULT), C.env("REV", C.REV_DEFAULT)
    out_path = C.env("OUT", required=True)
    prompts, pinfo = C.load_prompts(C.env("PROMPTS", required=True), batch, arm)
    kw = C.build_llm_kwargs(arm, batch, model, rev, gpu_util=C.env_float("GPU_UTIL", 0.90), max_len=C.env_int("MAX_LEN", 2048),
                            attn=C.env("ATTN_BACKEND"), moe=C.env("MOE_BACKEND", "marlin"))
    kw_census = dict(kw, profiler_config=profiler_config(a.skip, a.steps))
    rec = {"engine": "vllm", "arm": arm, "batch": batch, "mode": "census", "skip": a.skip, "steps": a.steps, "tokens": a.tokens,
           "llm_kwargs": {k: v for k, v in kw.items() if k != "profiler_config"}, "profiler_config": kw_census["profiler_config"],
           "multiproc_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"), **pinfo}
    C.check_budget(kw)
    t0 = time.perf_counter()
    llm = LLM(**kw_census)
    rec["load_s"] = round(time.perf_counter() - t0, 1)
    reqs = [{"prompt_token_ids": p} for p in prompts]
    sp = SamplingParams(**C.sampling_kwargs(arm, a.tokens))
    C.assert_generated(llm.generate(reqs, sp, use_tqdm=False), a.tokens, batch)          # warm (untimed, unprofiled)
    llm.start_profile()
    t = time.perf_counter()
    out = llm.generate(reqs, sp, use_tqdm=False)
    rec["profiled_wall_s"] = round(time.perf_counter() - t, 4)
    llm.stop_profile()
    C.assert_generated(out, a.tokens, batch)
    rec["status"] = "ok"
    rec["note"] = "profiled wall: never a speed reading (SC1's unprofiled arm is the speed)"
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("SC1B_VLLM " + json.dumps({k: rec[k] for k in ("arm", "batch", "profiler_config", "profiled_wall_s")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
