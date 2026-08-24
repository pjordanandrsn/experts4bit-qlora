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
COMPILE_GRAPH_STEP = [False]


def _materialize_from_arena(mods, arena_path):
    """R1 mechanics (PREREG-b1): the streaming loader leaves module
    expert tensors as META stubs (the bytes live in the gnf4 arena), and
    the pipelined engine sources its pinned arena from MODULE tensors.
    Fill the modules from the SAME arena file -- identical packed bytes
    by construction, no requantization. Byte counts are asserted per
    segment, and the R0==R1 bitwise token gate downstream is the
    semantic backstop: wrong bytes cannot pass it."""
    import numpy as np
    import torch.nn as nn

    from nvme_arena import _seg_len, _seg_off, load_index, row_offset

    idx = load_index(arena_path)
    layer_ids = sorted({l for l, _e, _o in idx["rows"]})
    assert len(layer_ids) == len(mods), (len(layer_ids), len(mods))
    seg_map = (("gate_up_proj", "nf4.gate_up_blocks", True),
               ("gate_up_absmax", "nf4.gate_up_absmax", False),
               ("down_proj", "nf4.down_blocks", True),
               ("down_absmax", "nf4.down_absmax", False))
    mm = np.memmap(arena_path, dtype=np.uint8, mode="r")
    for mi, wrapped in enumerate(mods):
        base = getattr(wrapped, "base", wrapped)
        E = base.num_experts
        li = layer_ids[mi]
        for attr, suffix, is_param in seg_map:
            meta = getattr(base, attr)
            t = torch.empty(meta.shape, dtype=meta.dtype)
            flat = t.view(torch.uint8).reshape(E, -1) \
                if t.dtype != torch.uint8 else t.reshape(E, -1)
            off, ln = _seg_off(idx, suffix), _seg_len(idx, suffix)
            assert flat.shape[1] == ln, \
                (attr, tuple(meta.shape), flat.shape[1], ln)
            for e in range(E):
                lo = row_offset(idx, li, e) + off
                flat[e] = torch.from_numpy(
                    np.ascontiguousarray(mm[lo:lo + ln]))
            if is_param:
                setattr(base, attr,
                        nn.Parameter(t, requires_grad=False))
            else:
                setattr(base, attr, t)
    del mm
    print(f"materialized {len(mods)} modules from {arena_path} "
          f"(byte-exact, per-segment lengths asserted)", flush=True)


def _routed_topk(cfg):
    """The routed top-k under whatever name this family's config uses.
    Extend the alias list when onboarding a family, never hardcode a
    key at a call site (docs/hybrid/PORTABILITY.md)."""
    for key in ("num_experts_per_tok", "num_experts_per_token",
                "moe_top_k", "moe_topk", "top_k"):
        v = getattr(cfg, key, None)
        if isinstance(v, int) and v > 0:
            return v
    raise ValueError("cannot find the routed top-k in this config; add "
                     "its key to _routed_topk")
     # set when --compile-layers uses cudagraphs
PER_MODE = {"prefill": {"attn_host_ns": 0, "attn_calls": 0},
            "decode": {"attn_host_ns": 0, "attn_calls": 0}}


HOST_BRACKETS = [False]     # --host-brackets: region walls + record_function
REGION_PROF = {
    "moe": {"prefill": {"ns": 0, "calls": 0}, "decode": {"ns": 0, "calls": 0}},
    "moe_block": {"prefill": {"ns": 0, "calls": 0},
                  "decode": {"ns": 0, "calls": 0}},
    "lmhead": {"prefill": {"ns": 0, "calls": 0}, "decode": {"ns": 0, "calls": 0}},
}


def _wrap_region(fn, key, region_name):
    """Host-wall bracket + a profiler region around one call site
    (T5b Phase A). Region names feed the event-tree op counter; the
    wall feeds the per-step decomposition. Only installed when
    --host-brackets is set, so timing arms never carry the overhead."""
    def timed(*a, **k):
        t0 = time.perf_counter_ns()
        with torch.profiler.record_function(region_name):
            out = fn(*a, **k)
        d = REGION_PROF[key][PROF["mode"]]
        d["ns"] += time.perf_counter_ns() - t0
        d["calls"] += 1
        return out
    return timed


def wrap_attention(impl_name):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS[impl_name]

    def timed(*a, **k):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        if HOST_BRACKETS[0]:
            with torch.profiler.record_function("e4b::attn"):
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
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="start of the corpus slice the prompt windows are "
                         "cut from (disjoint-window generalization runs)")
    ap.add_argument("--prompt-span", type=int, default=0,
                    help="length of that corpus slice; 0 = to the end. "
                         "Without a bounded span, prompts spread over the "
                         "WHOLE remaining corpus and two offsets overlap")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--hot-rows", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cpu-us-fixed", type=float, default=None)
    ap.add_argument("--cpu-us-per-row", type=float, default=None)
    ap.add_argument("--profile", default=None,
                    help="expert_profile JSONL for a measured-routing placement")
    ap.add_argument("--profile-out", default=None,
                    help="write this run's decode routing hist as an "
                         "expert_profile JSONL (profile-pass mode)")
    ap.add_argument("--compile-layers", action="store_true",
                    help="torch.compile each decoder layer body; the "
                         "paged-attention fn and the MoE tier forward are "
                         "dynamo-disabled so they graph-break cleanly "
                         "(PREREG-t1-launchpath)")
    ap.add_argument("--layers-attr", default="model.layers",
                    help="dotted path to the decoder-layer list for "
                         "--compile-layers (latent/nested families "
                         "differ, e.g. model.language_model.layers)")
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    help="torch.compile mode for --compile-layers; drop "
                         "to 'default' if cudagraphs misbehave (recorded)")
    ap.add_argument("--kv-batched", action="store_true",
                    help="accepted for command compatibility -- batched "
                         "KV append is the DEFAULT since its cert "
                         "(PREREG-g9-kvappend); see --kv-per-seq")
    ap.add_argument("--kv-per-seq", action="store_true",
                    help="run the per-seq KV append path (the kvappend "
                         "cert A arm; the T5 cycle measured this point "
                         "by accident when batched was opt-in)")
    ap.add_argument("--engine", choices=["hybrid", "pipelined"],
                    default="hybrid",
                    help="PREREG-b1 R1 arm: 'pipelined' bypasses the "
                         "hybrid tier -- enable_pipelined_residency with "
                         "every expert hot (narrowest existing resident "
                         "grouped-NF4 path); no placement, no CPU tier")
    ap.add_argument("--placement-override", choices=["none", "all-vram"],
                    default="none",
                    help="PREREG-b1 R0 arm: after the solver runs, move "
                         "every expert into the VRAM tier, executor "
                         "machinery left intact -- isolates physical "
                         "heterogeneity from orchestration")
    ap.add_argument("--host-brackets", action="store_true",
                    help="T5b Phase A: host-wall brackets + profiler "
                         "regions around each MoE forward and lm_head. "
                         "Timing arms must NOT carry this")
    ap.add_argument("--region-ops-out", default=None,
                    help="JSON of per-region descendant op counts from "
                         "the profiler event tree (needs --host-brackets; "
                         "engages the torch profiler, stacks off)")
    ap.add_argument("--sync-attr-out", default=None,
                    help="JSON of op counts over the profiler's active "
                         "window, with aten::nonzero attributed to source "
                         "files via stack frames (T5 H1/H2 instrument). "
                         "Implies the torch profiler with with_stack=True "
                         "-- run timing arms WITHOUT this flag")
    ap.add_argument("--torch-profile-out", default=None,
                    help="capture ~12 decode steps under torch.profiler "
                         "and dump the CUDA kernel table (T2/T3 "
                         "attribution: which device kernels own the "
                         "attention and expert buckets, and how many "
                         "launches each)")
    ap.add_argument("--cprofile-out", default=None,
                    help="run the serving loop under cProfile and dump "
                         "the top functions by cumulative time (the G9 "
                         "host-bill attribution instrument)")
    ap.add_argument("--series-out", default=None,
                    help="write the per-step touched-expert series "
                         "(decode-only, all tiers) as gzipped JSON")
    ap.add_argument("--amort-out", default=None,
                    help="write decode-only per-tier unique/activation "
                         "accounting plus the manifest VRAM set")
    ap.add_argument("--dispatch-diet", action="store_true",
                    help="T5: enable the engine's dispatch-algebra diet "
                         "(one sync/layer, cached index algebra); arm B "
                         "of PREREG-t5-dispatch-diet")
    ap.add_argument("--amort", choices=["on", "off"], default="on",
                    help="off = production shape: no per-layer counters, "
                         "no per-layer event syncs; the T5 arms run off "
                         "(--profile-out/--series-out/--amort-out then "
                         "refuse, they have nothing to write)")
    ap.add_argument("--torch-threads", type=int, default=8,
                    help="torch intraop cap while the pool runs (serving "
                         "playbook: 8; the default thrashes pinned workers)")
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
    k = _routed_topk(model.config)
    idx = json.loads(Path(a.arena + ".index.json").read_text())
    bpe = 0
    for seg in idx["segments"]:
        n = 1
        for d in seg["shape_per_expert"]:
            n *= d
        bpe += n * (4 if seg["dtype"] == "F32" else 1)
    torch.set_num_threads(a.torch_threads)
    if a.engine == "pipelined":
        # R1 (PREREG-b1): the narrowest existing resident path. Every
        # expert hot, one code path, no hybrid tier anywhere in the
        # process -- no arena serving tier, no CPU pool, no placement.
        assert a.amort == "off", "--engine pipelined has no amort counters"
        assert a.placement_override == "none", \
            "--placement-override is a hybrid-arm knob"
        assert not a.dispatch_diet, "dispatch_diet is hybrid-only"
        from experts4bit_qlora.engines.pipelined import (
            enable_pipelined_residency)
        _materialize_from_arena(mods, a.arena)
        man = None
        n = enable_pipelined_residency(
            model, [list(range(E)) for _ in range(L)], device="cuda",
            k_slots=k)
        assert n == L, f"pipelined patched {n}/{L} modules"
    else:
        man = solve_placement(
            n_layers=L, n_experts=E, bytes_per_expert=bpe,
            vram_budget_bytes=int(a.vram_gb * 2**30),
            dram_budget_bytes=int(a.dram_gb * 2**30),
            calibration=json.loads(Path(a.calib).read_text()),
            profile_path=a.profile,
            batch=a.batch, top_k=k,
            cpu_us_fixed=a.cpu_us_fixed, cpu_us_per_row=a.cpu_us_per_row)
        if a.placement_override == "all-vram":
            # R0 (PREREG-b1): physical heterogeneity removed, executor
            # machinery intact -- the solver ran, then every expert is
            # moved into the VRAM tier
            pairs = sorted(tuple(pp) for t in ("vram", "dram", "nvme")
                           for pp in man["tiers"][t])
            man["tiers"] = {"vram": [list(pp) for pp in pairs],
                            "dram": [], "nvme": []}
            man["masses"] = {"vram_frac": 1.0, "dram_frac": 0.0,
                             "nvme_frac": 0.0}
        n = enable_hybrid_tier(model, a.arena, man, hot_rows=a.hot_rows,
                               threads=a.threads, pool=True,
                               dispatch_diet=a.dispatch_diet)
        assert n == L
    assert not (a.kv_batched and a.kv_per_seq),         "--kv-batched and --kv-per-seq are contradictory"
    states = ([] if a.engine == "pipelined"
               else [m._hot_residency for m in mods])
    amort_on = a.amort == "on"
    if not amort_on:
        for flag in ("profile_out", "series_out", "amort_out"):
            assert not getattr(a, flag), \
                f"--{flag.replace('_', '-')} needs --amort on"
    for st in states:
        st.arm_amortization(amort_on)

    cfg = model.config
    hkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size
                                            // cfg.num_attention_heads)
    register(model)
    wrap_attention(IMPL_NAME)
    if a.region_ops_out and not a.host_brackets:
        raise SystemExit("--region-ops-out needs --host-brackets")
    if a.host_brackets:
        HOST_BRACKETS[0] = True
        for m in mods:
            m.forward = _wrap_region(m.forward, "moe", "e4b::moe")
        # the sparse-MoE BLOCK (router + experts) as its own region, so
        # router/top-k host cost = moe_block - moe (PREREG-b1)
        _ll = model
        for _part in a.layers_attr.split("."):
            _ll = getattr(_ll, _part)
        n_blk = 0
        for _lyr in _ll:
            _blk = getattr(_lyr, "mlp", None)
            if _blk is not None and hasattr(_blk, "experts"):
                _blk.forward = _wrap_region(_blk.forward, "moe_block",
                                            "e4b::moe_block")
                n_blk += 1
        assert n_blk == len(mods), \
            f"moe_block bracket found {n_blk} blocks for {len(mods)} " \
            f"expert modules -- the region would silently under-count"
        model.lm_head.forward = _wrap_region(
            model.lm_head.forward, "lmhead", "e4b::lmhead")
    if a.compile_layers:
        import torch._dynamo as dynamo
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        # clean graph breaks: the paged-attention shim (host-bound KV
        # paging) and the hybrid MoE forward (CPU tier dispatch) must
        # never be traced -- compile owns only the dense layer body
        ALL_ATTENTION_FUNCTIONS[IMPL_NAME] = dynamo.disable(
            ALL_ATTENTION_FUNCTIONS[IMPL_NAME])
        for m in mods:
            m.forward = dynamo.disable(m.forward)
        n_c = 0
        layer_list = model
        for part in a.layers_attr.split("."):
            layer_list = getattr(layer_list, part)
        for lyr in layer_list:
            lyr.forward = torch.compile(lyr.forward, mode=a.compile_mode,
                                        dynamic=False)
            n_c += 1
        if "reduce-overhead" in a.compile_mode:
            COMPILE_GRAPH_STEP[0] = True
        print(f"compiled {n_c} layer bodies (mode={a.compile_mode}); "
              f"paged attention + MoE tier dynamo-disabled; "
              f"graph step marking={COMPILE_GRAPH_STEP[0]}", flush=True)
    kv = Fp8PagedKV(L, hkv, hd, batch=a.batch,
                    max_tokens_per_seq=a.prompt_len + a.gen_tokens + 8,
                    k_groups=4, batched_append=not a.kv_per_seq,
                    device="cuda")

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    if a.prompt_offset or a.prompt_span:
        end = (a.prompt_offset + a.prompt_span) if a.prompt_span \
            else ids.numel()
        assert a.prompt_offset + a.batch * a.prompt_len < end <= ids.numel(), \
            "prompt slice leaves too little corpus for the windows"
        ids = ids[a.prompt_offset:end]
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
            if not states or states[0].amort is None:   # off / pipelined
                return (0, 0)
            return (sum(st.amort["dram_ns"] for st in states),
                    sum(st.amort["gpu_ns"] for st in states))

        @torch.no_grad()
        def run_prefill(self, chunks):
            if COMPILE_GRAPH_STEP[0]:
                torch.compiler.cudagraph_mark_step_begin()
            PROF["mode"] = "prefill"
            d0, g0 = self._amort_snap()
            # decode-only accounting by construction: capture the tier
            # counters, let the prefill run (its own dram/gpu deltas are
            # recorded below, before the rollback), then restore — so
            # prefill chunks interleaved by the scheduler at ANY later
            # step never leak into --amort-out / --profile-out
            # (Bugbot, e4b#189).
            saved = []
            for st in states:
                am = st.amort
                saved.append(None if am is None else
                             {k2: (v.clone() if torch.is_tensor(v)
                                   else list(v) if isinstance(v, list)
                                   else v)
                              for k2, v in am.items()})
            t0 = time.perf_counter_ns()
            out = super().run_prefill(chunks)
            wall = time.perf_counter_ns() - t0
            d1, g1 = self._amort_snap()
            for st, am in zip(states, saved):
                if am is not None and st.amort is not None:
                    cur = st.amort
                    for k2, v in am.items():
                        if torch.is_tensor(v):
                            cur[k2].copy_(v)
                        else:
                            # lists were snapshot-copied above, so plain
                            # assignment drops any prefill appends
                            cur[k2] = v
            self.prefill_rows.append(
                {"chunks": len(chunks),
                 "tokens": sum(c[2] for c in chunks),
                 "wall_ns": wall, "dram_ns": d1 - d0, "gpu_ns": g1 - g0})
            PROF["mode"] = "decode"
            return out

        @torch.no_grad()
        def run_decode(self, rids):
            if COMPILE_GRAPH_STEP[0]:
                # the documented remedy for cudagraph replay reuse across
                # steps (T1's crash): declare the step boundary so outputs
                # of the previous replay are not read after overwrite
                torch.compiler.cudagraph_mark_step_begin()
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
            rm0 = REGION_PROF["moe"]["decode"]["ns"]
            rb0 = REGION_PROF["moe_block"]["decode"]["ns"]
            rl0 = REGION_PROF["lmhead"]["decode"]["ns"]
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
                 "moe_ns": REGION_PROF["moe"]["decode"]["ns"] - rm0,
                 "moe_block_ns":
                     REGION_PROF["moe_block"]["decode"]["ns"] - rb0,
                 "lmhead_ns": REGION_PROF["lmhead"]["decode"]["ns"] - rl0,
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

    tprof = None
    tprof_steps = [0]
    if a.torch_profile_out or a.sync_attr_out or a.region_ops_out:
        from torch.profiler import (ProfilerActivity, profile, schedule)
        tprof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(skip_first=24, wait=0, warmup=2, active=12,
                              repeat=1),
            record_shapes=True,
            with_stack=bool(a.sync_attr_out))
        tprof.__enter__()
    prof = None
    if a.cprofile_out:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
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
            if tprof is not None:
                tprof.step()
                tprof_steps[0] += 1
    torch.cuda.synchronize()
    if tprof is not None:
        tprof.__exit__(None, None, None)
        tbl = tprof.key_averages().table(sort_by="cuda_time_total",
                                         row_limit=80)
        # the schedule fills its active window only after skip_first(24)
        # + warmup(2) decode steps; label the receipt with the ACTUAL
        # window so a short run cannot masquerade as a full attribution
        active = max(0, min(12, tprof_steps[0] - 24 - 2))
        hdr = (f"profiled decode steps: {tprof_steps[0]} "
               f"(active window: {active}/12)\n")
        if active < 12:
            hdr += ("WARNING: active window INCOMPLETE -- this table "
                    "under-samples and must not be cited as the "
                    "attribution\n")
        if a.torch_profile_out:
            Path(a.torch_profile_out).write_text(hdr + tbl)
            print(f"TORCH_PROFILE_OUT {a.torch_profile_out} "
                  f"active={active}/12", flush=True)
        if a.sync_attr_out:
            # counts over the ACTIVE window; the verdict divides by
            # `active_steps` -- never assume the window filled
            counts = {}
            for evt in tprof.key_averages():
                counts[evt.key] = counts.get(evt.key, 0) + evt.count
            nz = {"engine": 0, "other": 0, "frames": {}}
            for evt in tprof.key_averages(group_by_stack_n=24):
                if evt.key != "aten::nonzero" or not evt.count:
                    continue
                frames = [f for f in (evt.stack or []) if ".py" in f]
                site = next((f for f in frames
                             if "hot_residency.py" in f or "hybrid.py" in f
                             or "nvme_experts.py" in f), None)
                bucket = "engine" if site else "other"
                nz[bucket] += evt.count
                label = site or (frames[0] if frames else "<no-py-frame>")
                nz["frames"][label] = nz["frames"].get(label, 0) + evt.count
            Path(a.sync_attr_out).write_text(json.dumps({
                "active_steps": active,
                "dispatch_diet": bool(a.dispatch_diet),
                "op_counts": {kk: counts.get(kk, 0) for kk in
                              ("aten::nonzero", "aten::copy_", "aten::to",
                               "aten::_to_copy",
                               "aten::index_select", "aten::index_put_",
                               "aten::unique2", "aten::sort",
                               "aten::arange", "aten::item",
                               "aten::_local_scalar_dense",
                               "cudaLaunchKernel", "cudaMemcpyAsync",
                               "cudaStreamSynchronize",
                               "cudaDeviceSynchronize")},
                "nonzero_attr": nz,
            }, indent=1))
            print(f"SYNC_ATTR_OUT {a.sync_attr_out} active={active}/12",
                  flush=True)
        if a.region_ops_out:
            evs = tprof.profiler.function_events
            regions = {"e4b::moe": {}, "e4b::moe_block": {},
                       "e4b::attn": {}, "e4b::lmhead": {}}
            rcounts = dict.fromkeys(regions, 0)
            rtime = dict.fromkeys(regions, 0.0)

            def _walk(ev, bag):
                for c in ev.cpu_children:
                    if c.name.startswith("aten::"):
                        bag[c.name] = bag.get(c.name, 0) + 1
                    _walk(c, bag)

            for ev in evs:
                if ev.name in regions:
                    rcounts[ev.name] += 1
                    rtime[ev.name] += ev.cpu_time_total
                    _walk(ev, regions[ev.name])
            # a silent no-match must fail loudly, never read as zero
            # (PREREG-t5b): every region must appear at its call rate
            assert rcounts["e4b::moe"] >= L * max(1, active), rcounts
            assert rcounts["e4b::moe_block"] >= L * max(1, active), rcounts
            assert rcounts["e4b::attn"] >= L * max(1, active), rcounts
            assert rcounts["e4b::lmhead"] >= max(1, active), rcounts
            Path(a.region_ops_out).write_text(json.dumps({
                "active_steps": active,
                "layers": L,
                "region_calls": rcounts,
                "region_cpu_ms_total": {k: v / 1e3
                                        for k, v in rtime.items()},
                "region_ops": regions,
            }, indent=1))
            print(f"REGION_OPS_OUT {a.region_ops_out} "
                  f"moe={rcounts['e4b::moe']} attn={rcounts['e4b::attn']} "
                  f"lmhead={rcounts['e4b::lmhead']}", flush=True)
    if prof is not None:
        prof.disable()
        import io
        import pstats
        buf = io.StringIO()
        st_ = pstats.Stats(prof, stream=buf)
        st_.sort_stats("cumulative").print_stats(60)
        Path(a.cprofile_out).write_text(buf.getvalue())
        print(f"CPROFILE_OUT {a.cprofile_out}", flush=True)

    # device-side attention occupancy from the recorded events
    attn_dev = {"prefill": 0.0, "decode": 0.0}
    for mode, e0, e1 in PROF["attn_events"]:
        attn_dev[mode] += e0.elapsed_time(e1)

    dr = runner.decode_rows
    n_full = len(dr)
    n_warm = 4 if a.compile_layers else 0
    n_dropped = 0
    if n_warm and len(dr) > 2 * n_warm:
        dr = dr[n_warm:]
        n_dropped = n_warm
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
        "compile_layers": bool(a.compile_layers),
        "compile_mode": a.compile_mode if a.compile_layers else None,
        "warmup_rows_dropped": n_dropped,
        # the cross-arm void gate: greedy continuations must be
        # token-identical between eager and compiled arms
        "generated_tokens": {str(r): list(map(int, t))
                             for r, t in sorted(runner.tokens.items())},
        "decode_steps": n_steps,
        "decode_median_ms": {
            "step": step_ms, "forward_submission": fwd, "drain": drain,
            "attention_host": attn_h, "dram_experts_host": dram,
            "other_submission": other_sub,
            "scheduler_python_and_bookkeeping": sched_py,
        },
        "decode_device_ms": {
            "attention_kernels_per_step":
                attn_dev["decode"] / max(1, n_full),
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
    if a.profile_out:
        # mass semantics match load_routing_mass: tokens_routed accumulates
        # raw selection counts; routing_probabilities divides by the layer
        # total and multiplies by top_k, so p_e = count_e / decode_tokens.
        with open(a.profile_out, "w") as f:
            for li, st in enumerate(states):
                hist = st.amort["hist"].cpu().tolist()
                for e, c in enumerate(hist):
                    if c:
                        f.write(json.dumps({"row": "expert", "layer_id": li,
                                            "expert_id": e,
                                            "tokens_routed": int(c)}) + "\n")
        print(f"PROFILE_OUT {a.profile_out}", flush=True)
    if a.series_out:
        import gzip
        ser = []
        for st in states:
            ser.append([u.cpu().tolist() for u in st.amort["series"]])
        n_steps_series = {len(x) for x in ser}
        assert len(n_steps_series) == 1, \
            f"layers disagree on series length: {n_steps_series}"
        with gzip.open(a.series_out, "wt") as f:
            json.dump({"per_layer_series": ser}, f)
        print(f"SERIES_OUT {a.series_out} steps={n_steps_series.pop()}",
              flush=True)
    if a.amort_out:
        per_layer = []
        for st in states:
            am = st.amort
            row = {k2: int(am[k2]) for k2 in
                   ("steps", "acts", "uniq_vram", "uniq_dram",
                    "uniq_nvme", "acts_vram", "acts_dram", "acts_nvme",
                    "dram_steps")}
            row["touch"] = am["touch"].cpu().tolist()
            row["hist"] = am["hist"].cpu().tolist()
            per_layer.append(row)
        Path(a.amort_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.amort_out).write_text(json.dumps({
            "vram_gb": a.vram_gb, "batch": a.batch, "top_k": k,
            "geometry": man["geometry"],
            "manifest_counts": {t: len(p) for t, p in man["tiers"].items()},
            "manifest_vram": man["tiers"]["vram"],
            "decode_steps": n_steps,
            "decode_median_ms": rep["decode_median_ms"],
            "per_layer": per_layer,
        }, indent=1))
        print(f"AMORT_OUT {a.amort_out}", flush=True)
    rep["engine"] = a.engine
    rep["placement_override"] = a.placement_override
    rep["manifest_counts"] = (None if man is None else
                              {t: len(pp) for t, pp in man["tiers"].items()})
    if a.host_brackets:
        moe_h, lmh_h = med("moe_ns"), med("lmhead_ns")
        blk_h = med("moe_block_ns")
        rep["decode_median_ms"]["moe_host"] = moe_h
        rep["decode_median_ms"]["moe_block_host"] = blk_h
        rep["decode_median_ms"]["router_topk_host"] = blk_h - moe_h
        rep["decode_median_ms"]["lmhead_host"] = lmh_h
        # dram is INSIDE the moe bracket -- never sum them; the residual
        # is what no region owns
        rep["decode_median_ms"]["host_residual"] = (
            fwd - attn_h - moe_h - lmh_h)
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
