"""Decompose a Phase-9 engine step into its cost buckets.

The fixbox run left ~84 ms/step unattributed between attention and engine
overhead (197 ms/step in the engine vs 113 ms bare forward, experts 66 ms).
This instrument splits a step without touching repo code:

  decode step = scheduler python
              + forward submission   (host time inside model(...))
                  |- attention host  (paged-attention calls, host ns)
                  |- dram experts    (synchronous CPU kernel wall)
                  |- other submission (router/norms/embed/lm_head launches)
              + drain                (argmax+tolist sync absorbing GPU tail)
              + bookkeeping

Device-side truth comes separately from CUDA events (attention kernels,
GPU expert kernels) — reported as device occupancy, never subtracted from
host buckets, because overlap makes subtraction a lie.

Methodology validated on the dev box (OLMoE); constants only bind on a
serving-class box (G6 tiny-model trap applies to MAGNITUDES, not shape).
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

PROF = {"attn_host_ns": 0, "attn_calls": 0, "attn_events": [],
        "mode": "decode"}
PER_MODE = {"prefill": {"attn_host_ns": 0, "attn_calls": 0},
            "decode": {"attn_host_ns": 0, "attn_calls": 0}}


def wrap_attention(impl_name):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS[impl_name]

    def timed(*a, **k):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter_ns()
        e0.record()
        out = orig(*a, **k)
        e1.record()
        dt = time.perf_counter_ns() - t0
        PROF["attn_host_ns"] += dt
        PROF["attn_calls"] += 1
        PROF["attn_events"].append((PROF["mode"], e0, e1))
        PER_MODE[PROF["mode"]]["attn_host_ns"] += dt
        PER_MODE[PROF["mode"]]["attn_calls"] += 1
        return out

    ALL_ATTENTION_FUNCTIONS[impl_name] = timed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--vram-gb", type=float, default=1.2)
    ap.add_argument("--dram-gb", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--hot-rows", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cpu-us-fixed", type=float, default=None)
    ap.add_argument("--cpu-us-per-row", type=float, default=None)
    ap.add_argument("--out", default="/workspace/g8out/step_decomp.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.hot_residency import target_modules
    from experts4bit_qlora.engines.hybrid import enable_hybrid_tier
    from experts4bit_qlora.engines.paged_attention import IMPL_NAME, register
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    from experts4bit_qlora.engines.placement import solve_placement
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler

    torch.manual_seed(1689)
    tok = AutoTokenizer.from_pretrained(a.model)
    model, _ = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16,
                                       r=8, alpha=16, quant_type="nf4",
                                       arena=a.arena)
    model.eval()
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    k = model.config.num_experts_per_tok
    idx = json.loads(Path(a.arena + ".index.json").read_text())
    bpe = 0
    for seg in idx["segments"]:
        n = 1
        for d in seg["shape_per_expert"]:
            n *= d
        bpe += n * (4 if seg["dtype"] == "F32" else 1)
    man = solve_placement(
        n_layers=L, n_experts=E, bytes_per_expert=bpe,
        vram_budget_bytes=int(a.vram_gb * 2**30),
        dram_budget_bytes=int(a.dram_gb * 2**30),
        calibration=json.loads(Path(a.calib).read_text()),
        batch=a.batch, top_k=k,
        cpu_us_fixed=a.cpu_us_fixed, cpu_us_per_row=a.cpu_us_per_row)
    n = enable_hybrid_tier(model, a.arena, man, hot_rows=a.hot_rows,
                           threads=a.threads, pool=True)
    assert n == L
    states = [m._hot_residency for m in mods]
    for st in states:
        st.arm_amortization(True)

    cfg = model.config
    hkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size
                                            // cfg.num_attention_heads)
    register(model)
    wrap_attention(IMPL_NAME)
    kv = Fp8PagedKV(L, hkv, hd, batch=a.batch,
                    max_tokens_per_seq=a.prompt_len + a.gen_tokens + 8,
                    k_groups=4, device="cuda")

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (ids.numel() - a.prompt_len) // max(1, a.batch))
    prompts = [ids[i * step:i * step + a.prompt_len].tolist()
               for i in range(a.batch)]

    # ------- timed runner: forward vs drain split, per-regime expert delta
    class TimedRunner(PagedModelRunner):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.decode_rows = []
            self.prefill_rows = []

        def _amort_snap(self):
            return (sum(st.amort["dram_ns"] for st in states),
                    sum(st.amort["gpu_ns"] for st in states))

        @torch.no_grad()
        def run_prefill(self, chunks):
            PROF["mode"] = "prefill"
            d0, g0 = self._amort_snap()
            t0 = time.perf_counter_ns()
            out = super().run_prefill(chunks)
            wall = time.perf_counter_ns() - t0
            d1, g1 = self._amort_snap()
            self.prefill_rows.append(
                {"chunks": len(chunks),
                 "tokens": sum(c[2] for c in chunks),
                 "wall_ns": wall, "dram_ns": d1 - d0, "gpu_ns": g1 - g0})
            PROF["mode"] = "decode"
            return out

        @torch.no_grad()
        def run_decode(self, rids):
            if not rids:
                return {}
            # duplicated from PagedModelRunner.run_decode by design: the
            # split being measured (forward submission vs drain) lives
            # INSIDE the method, so instrumentation must inline it
            self.ctx.mode = "decode"
            self.ctx.slots = [self.slot_of[r] for r in rids]
            ids_ = torch.tensor([[self.tokens[r][-1]] for r in rids],
                                dtype=torch.long, device=self.device)
            pos = torch.tensor([[self.pos_of[r] - 1] for r in rids],
                               dtype=torch.long, device=self.device)
            from experts4bit_qlora.engines.paged_attention import set_context
            ah0, ac0 = PROF["attn_host_ns"], PROF["attn_calls"]
            d0, g0 = self._amort_snap()
            prev = set_context(self.ctx)
            t0 = time.perf_counter_ns()
            try:
                out = self.model(input_ids=ids_, position_ids=pos,
                                 use_cache=False)
            finally:
                set_context(prev)
            t_fwd = time.perf_counter_ns() - t0
            t1 = time.perf_counter_ns()
            toks = out.logits[:, -1].argmax(-1).tolist()
            t_drain = time.perf_counter_ns() - t1
            d1, g1 = self._amort_snap()
            self.decode_rows.append(
                {"batch": len(rids), "fwd_ns": t_fwd, "drain_ns": t_drain,
                 "attn_host_ns": PROF["attn_host_ns"] - ah0,
                 "attn_calls": PROF["attn_calls"] - ac0,
                 "dram_ns": d1 - d0, "gpu_ns": g1 - g0})
            got = {}
            for rid, tk in zip(rids, toks):
                got[rid] = int(tk)
                self.tokens[rid].append(int(tk))
                self.pos_of[rid] += 1
            return got

    runner = TimedRunner(model, kv, device="cuda")
    sched = ContinuousScheduler(runner=runner, max_seqs=a.batch,
                                kv_slots=a.batch, chunk_tokens=a.chunk,
                                max_prefill_tokens_per_step=a.chunk)
    for p in prompts:
        sched.add_request(p, max_new_tokens=a.gen_tokens)

    step_walls = []            # decode-ONLY steps: a wall that included a
    while sched.active or sched.queue:   # prefill chunk would smear into
        pf0 = len(runner.prefill_rows)   # sched_py and mis-attribute
        dr0 = len(runner.decode_rows)
        t0 = time.perf_counter_ns()
        if sched.step().is_empty:
            break
        wall = time.perf_counter_ns() - t0
        if len(runner.prefill_rows) == pf0 and len(runner.decode_rows) > dr0:
            step_walls.append(wall)
    torch.cuda.synchronize()

    # device-side attention occupancy from the recorded events
    attn_dev = {"prefill": 0.0, "decode": 0.0}
    for mode, e0, e1 in PROF["attn_events"]:
        attn_dev[mode] += e0.elapsed_time(e1)

    dr = runner.decode_rows
    med = lambda key: statistics.median(r[key] for r in dr) / 1e6
    n_steps = len(dr)
    step_ms = statistics.median(step_walls[-n_steps:]) / 1e6 if dr else 0
    fwd, drain = med("fwd_ns"), med("drain_ns")
    attn_h, dram = med("attn_host_ns"), med("dram_ns")
    gpu_dev = med("gpu_ns")
    other_sub = fwd - attn_h - dram
    sched_py = step_ms - fwd - drain
    rep = {
        "model": a.model, "batch": a.batch, "layers": L,
        "decode_steps": n_steps,
        "decode_median_ms": {
            "step": step_ms, "forward_submission": fwd, "drain": drain,
            "attention_host": attn_h, "dram_experts_host": dram,
            "other_submission": other_sub,
            "scheduler_python_and_bookkeeping": sched_py,
        },
        "decode_device_ms": {
            "attention_kernels_per_step":
                attn_dev["decode"] / max(1, n_steps),
            "gpu_expert_kernels_per_step": gpu_dev,
        },
        "attn_calls_per_step": (statistics.median(r["attn_calls"]
                                                  for r in dr) if dr else 0),
        "attn_host_us_per_call": (attn_h * 1e3 / L if L else 0),
        "prefill": [{"tokens": r["tokens"],
                     "wall_ms": r["wall_ns"] / 1e6,
                     "dram_ms": r["dram_ns"] / 1e6,
                     "gpu_dev_ms": r["gpu_ns"] / 1e6}
                    for r in runner.prefill_rows],
        "prefill_attn_dev_total_ms": attn_dev["prefill"],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    d = rep["decode_median_ms"]
    print(f"DECOMP step={d['step']:.1f}ms  fwd_submit={d['forward_submission']:.1f} "
          f"(attn_host={d['attention_host']:.1f} dram={d['dram_experts_host']:.1f} "
          f"other={d['other_submission']:.1f})  drain={d['drain']:.1f}  "
          f"sched_py={d['scheduler_python_and_bookkeeping']:.1f}", flush=True)
    print(f"DEVICE attn={rep['decode_device_ms']['attention_kernels_per_step']:.2f}ms/step "
          f"gpu_experts={rep['decode_device_ms']['gpu_expert_kernels_per_step']:.2f}ms/step "
          f"attn_host_us_per_call={rep['attn_host_us_per_call']:.0f}",
          flush=True)
    for r in rep["prefill"][:4]:
        print(f"PREFILL tokens={r['tokens']} wall={r['wall_ms']:.0f}ms "
              f"dram={r['dram_ms']:.0f} gpu_dev={r['gpu_dev_ms']:.0f}",
              flush=True)
    print("STEP_DECOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
