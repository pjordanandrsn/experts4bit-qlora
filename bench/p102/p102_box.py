#!/usr/bin/env python3
"""Lane P102's instruments (bench/p102/PREREG-p102.md; e4b#916): the int4 store's prefill route A/B -- ``loop`` (today),
``batched`` (bit-identical), ``k19`` (K19 at prefill rows, the loop's operand precision) and ``mtile`` (the int8 M-tile)
-- on ONE engine / ONE model per mode, the route switched between requests through ``E4B_INT4_PREFILL`` (read at every
MoE call), so every route sees the same weights, the same graphs and the same box state.

  python p102_box.py ttft
      ``serve_paged.build_engine`` once (SC1's int4_sched stack, max_seqs 1, chunk 512, from the env). Each route is
      warmed once at both lengths (its kernels JIT-compile there). Then ROUNDS rounds; round r runs the routes in an
      order rotated by r, each one timed request at 512 tokens and one at 4096 (``sc1_e4b_sched.run_batch``,
      ``max_new_tokens=1``). Each request's first token is recorded. After the timing, one more 512-token request per
      route runs with ``p100_box.Census`` on (the dispatch census) and one under ``torch.profiler`` (the kernel names);
      then, for ``k19`` and ``mtile``, one 4096-token request under ``torch.profiler`` gives the device time by kernel
      (where a prefill's time goes once the loop is gone -- P100 left about 0.35 ms per token unattributed).
  python p102_box.py nll STEP_DECOMP_ARGS...
      ``bench/p39/step_decomp.py`` with ``--ppl-oracle eager`` (SC1's prefill-shaped NLL row: chunked teacher forcing
      through the model's own eager attention, T = 512 then T = 256 per MoE call), its ``_ppl_oracle_main`` replaced by a
      loop: for every registered window (c4val1 k = 9..16, wikitext k = 9..12; window k starts at token k * 4096, span
      2600, P96's), every route in an order rotated by the window's index, ``ppl_oracle_score`` once. One model build.
  python p102_box.py --self-test

Writes ``P102_OUT`` (JSON). The library is not edited.
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = ("loop", "batched", "k19", "mtile")
ROUNDS = 3
STRIDE, SPAN = 4096, 2600
WINDOWS = (("c4val1", (9, 10, 11, 12, 13, 14, 15, 16)), ("wikitext", (9, 10, 11, 12)))
#: the kernel each device-grouped route must launch (names as torch.profiler reports Triton kernels)
ROUTE_KERNEL = {"k19": "_gemm_int4_b32_grouped_smallm", "mtile": "_gemm_int4_b32_grouped"}


def rotated(seq, r: int):
    r %= len(seq)
    return tuple(seq[r:]) + tuple(seq[:r])


def kernel_names(events) -> dict:
    out = {}
    for e in events:
        if str(getattr(e, "device_type", "")).split(".")[-1] == "CUDA":
            out[e.name] = out.get(e.name, 0) + 1
    return out


def route_kernel_count(names: dict, route: str) -> int:
    """Launches of the route's own kernel. ``_gemm_int4_b32_grouped`` is a prefix of K19's name, so the M-tile's count
    excludes K19's."""
    k = ROUTE_KERNEL[route]
    if route == "mtile":
        return sum(c for n, c in names.items() if k in n and ROUTE_KERNEL["k19"] not in n)
    return sum(c for n, c in names.items() if k in n)


def top_device_time(avgs, n: int = 15) -> dict:
    """``key_averages()`` rows with device time: ``{device_ms_total, top: [[name, count, self_device_ms], ...]}``."""
    rows = []
    for a in avgs:
        t = getattr(a, "self_device_time_total", None)
        if t is None:
            t = getattr(a, "self_cuda_time_total", 0)
        if t:
            rows.append([str(a.key)[:120], int(a.count), round(t / 1e3, 3)])
    rows.sort(key=lambda r: -r[2])
    return {"device_ms_total": round(sum(r[2] for r in rows), 3), "top": rows[:n]}


def _ttft() -> int:
    sys.path.insert(0, HERE)
    import sc1_e4b_sched as s
    import torch
    from p100_box import Census

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    assert cfg.max_seqs == 1 and cfg.placement == "all-vram" and cfg.chunk_tokens == 512, cfg
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    rec = {"mode": "ttft", "load_s": round(time.perf_counter() - t0, 1), "census_build": parts.info,
           "chunk_tokens": cfg.chunk_tokens, "rounds": ROUNDS, "routes": list(ROUTES)}
    rows = {L: s.load_prompts(os.environ[f"P102_PROMPTS_{L}"], 1, prompt_len=None)["prompts"] for L in (512, 4096)}
    rec["prompts_sha256"] = {L: s.prompts_digest(rows[L]) for L in rows}

    def one(route, L):
        os.environ["E4B_INT4_PREFILL"] = route
        r = s.run_batch(parts, torch, rows[L], 1, prompt_len=len(rows[L][0]))
        return {"route": route, "L": L, "wall_s": round(r["wall"], 4), "sched_ttft_s": r["ttft"][0],
                "token": int(r["requests"][0].out[0])}

    rec["warm"] = [one(rt, L) for rt in ROUTES for L in (512, 4096)]
    draws = []
    for rnd in range(ROUNDS):
        for rt in rotated(ROUTES, rnd):
            for L in (512, 4096):
                d = one(rt, L)
                d["round"] = rnd
                draws.append(d)
    rec["draws"] = draws
    rec["ttft_s_median"] = {rt: {str(L): statistics.median(d["wall_s"] for d in draws if d["route"] == rt and d["L"] == L)
                                 for L in (512, 4096)} for rt in ROUTES}
    census = Census()
    census.install()
    rec["census"], rec["kernels"] = {}, {}
    from torch.profiler import ProfilerActivity, profile
    for rt in ROUTES:
        census.calls, census.dequant_total = [], 0
        census.active = True
        one(rt, 512)
        census.active = False
        rec["census"][rt] = census.record()["summary"]
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            one(rt, 512)
        names = kernel_names(prof.events())
        rec["kernels"][rt] = {"k19": route_kernel_count(names, "k19"), "mtile": route_kernel_count(names, "mtile"),
                              "device_kernels": sum(names.values())}
    rec["profile_4096"] = {}
    for rt in ("k19", "mtile"):                     # descriptive: not read by the rule
        try:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                d = one(rt, 4096)
            rec["profile_4096"][rt] = dict(top_device_time(prof.key_averages()), wall_s=d["wall_s"])
        except Exception as e:  # noqa: BLE001
            rec["profile_4096"][rt] = {"error": repr(e)[:300]}
    os.environ.pop("E4B_INT4_PREFILL", None)
    json.dump(rec, open(os.environ["P102_OUT"], "w"), indent=1, default=str)
    print("P102_TTFT " + json.dumps(rec["ttft_s_median"]), flush=True)
    return 0


def _nll(args) -> int:
    sys.path.insert(0, HERE)
    import step_decomp as sd
    import torch
    from transformers import AutoTokenizer
    out = {"mode": "nll", "argv": list(args), "routes": list(ROUTES), "windows": [], "stride": STRIDE, "span": SPAN}

    def multi(a, model, ppl_ids, ppl_sha):
        tok = AutoTokenizer.from_pretrained(a.model)
        model.config._attn_implementation = "eager"
        for m in model.modules():
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "eager"
        model.eval()
        i = 0
        for src, ks in WINDOWS:
            for k in ks:
                b = copy.copy(a)
                b.ppl_source, b.prompt_offset, b.prompt_span = src, k * STRIDE, SPAN
                _, _, _, ids, sha = sd._k8_window(b, tok)
                for rt in rotated(ROUTES, i):
                    os.environ["E4B_INT4_PREFILL"] = rt
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        nll = sd.ppl_oracle_score(model, ids, b.prompt_len, b.ppl_steps)
                    out["windows"].append({"source": src, "k": k, "route": rt, "mean_nll": nll,
                                           "ppl": float(torch.exp(torch.tensor(nll))), "text_sha": sha,
                                           "steps": b.ppl_steps, "prompt_len": b.prompt_len,
                                           "wall_s": round(time.perf_counter() - t0, 2)})
                    print(f"P102_NLL {src} k={k} route={rt} nll={nll:.6f}", flush=True)
                i += 1
        os.environ.pop("E4B_INT4_PREFILL", None)

    sd._ppl_oracle_main = multi
    sys.argv = [os.path.join(HERE, "step_decomp.py")] + list(args)
    sd.main()
    json.dump(out, open(os.environ["P102_OUT"], "w"), indent=1)
    return 0


class _Ev:
    def __init__(self, name, device_type="DeviceType.CUDA"):
        self.name, self.device_type = name, device_type


def selftest() -> int:
    assert rotated(ROUTES, 0) == ROUTES and rotated(ROUTES, 1) == ("batched", "k19", "mtile", "loop")
    assert rotated(ROUTES, 5) == rotated(ROUTES, 1)
    names = kernel_names([_Ev("_gemm_int4_b32_grouped_smallm"), _Ev("_gemm_int4_b32_grouped_smallm"),
                          _Ev("_gemm_int4_b32_grouped"), _Ev("aten::mm", "DeviceType.CPU")])
    assert route_kernel_count(names, "k19") == 2 and route_kernel_count(names, "mtile") == 1, names
    assert sum(len(ks) for _s, ks in WINDOWS) == 12 and min(k for _s, ks in WINDOWS for k in ks) >= 9

    class _Avg:
        def __init__(self, key, count, t):
            self.key, self.count, self.self_device_time_total = key, count, t
    top = top_device_time([_Avg("a", 2, 3000.0), _Avg("b", 1, 0), _Avg("c", 5, 1500.0)], n=1)
    assert top == {"device_ms_total": 4.5, "top": [["a", 2, 3.0]]}, top
    print("self-test OK (4 cases)")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--self-test":
        return selftest()
    if argv and argv[0] == "ttft" and len(argv) == 1:
        return _ttft()
    if argv and argv[0] == "nll":
        return _nll(argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
