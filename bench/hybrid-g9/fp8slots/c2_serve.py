"""C2 serving driver (PREREG-c2.md): the frozen C1 controller rule running
IN-ENGINE over a sequential multi-window workload, against the same
deployed-static baseline, with runtime expert swaps via
_HybridTier.swap_expert.

  python c2_serve.py --model ... --arena ... --calib ... \
      --prior-dir bench/hybrid-g9/fp8slots/receipts-online \
      [--controller] [--swap-selftest] --out run.json

One process serves ten 28k-token windows back to back (base 14500,
stride 28400) at B=16, gen 128 each. The controller (when on) runs
every 8 decode steps: per-layer gain-gated swaps, trailing-32 estimates
with a 0.25-prior floor -- the C1 rule under the per-layer engine
geometry, receipt-checked at 17.5% raw / 15.2% adjusted on the C1 data.
"""
import argparse
import gzip
import json
import pathlib
import statistics
import time

import torch

def load_prior(prior_dir, L, E):
    """Pooled per-(layer,expert) touch rates + selection masses from the
    committed design-set receipts."""
    idx = json.load(open(pathlib.Path(prior_dir) / "windows.json"))
    touch = [[0] * E for _ in range(L)]
    hist = [[0] * E for _ in range(L)]
    steps = 0
    for tag in idx["windows"]:
        am = json.load(open(pathlib.Path(prior_dir) / f"{tag}.amort.json"))
        assert am["geometry"]["n_layers"] == L
        assert am["geometry"]["n_experts"] == E
        steps += am["per_layer"][0]["steps"]
        for l, pl in enumerate(am["per_layer"]):
            for e in range(E):
                touch[l][e] += pl["touch"][e]
                hist[l][e] += pl["hist"][e]
    prior = [[touch[l][e] / steps for e in range(E)] for l in range(L)]
    return prior, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--prior-dir", required=True)
    ap.add_argument("--controller", action="store_true")
    ap.add_argument("--controller-cp", action="store_true",
                    help="engine controller with change-point reset")
    ap.add_argument("--swap-selftest", action="store_true")
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--window-base", type=int, default=14500)
    ap.add_argument("--window-stride", type=int, default=28400)
    ap.add_argument("--window-span", type=int, default=28000)
    ap.add_argument("--vram-gb", type=float, default=10.0)
    ap.add_argument("--dram-gb", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.hot_residency import target_modules
    from experts4bit_qlora.engines.hybrid import enable_hybrid_tier
    from experts4bit_qlora.engines.paged_attention import register
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner
    from experts4bit_qlora.engines.placement import solve_placement
    from experts4bit_qlora.engines.scheduler import ContinuousScheduler

    torch.manual_seed(1689)
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(a.model)
    model, _ = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16,
                                       r=8, alpha=16, quant_type="nf4",
                                       arena=a.arena)
    model.eval()
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    k = model.config.num_experts_per_tok
    prior, hist = load_prior(a.prior_dir, L, E)

    # deployed static placement: the solver over the design-set masses
    # (the engine's own shipped path), written as a profile JSONL
    prof = pathlib.Path(a.out + ".prior_profile.jsonl")
    with open(prof, "w") as f:
        for l in range(L):
            for e in range(E):
                if hist[l][e]:
                    f.write(json.dumps({"row": "expert", "layer_id": l,
                                        "expert_id": e,
                                        "tokens_routed": hist[l][e]}) + "\n")
    idx = json.loads(pathlib.Path(a.arena + ".index.json").read_text())
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
        calibration=json.loads(pathlib.Path(a.calib).read_text()),
        profile_path=str(prof), batch=a.batch, top_k=k,
        cpu_us_fixed=55, cpu_us_per_row=2)
    n = enable_hybrid_tier(model, a.arena, man, hot_rows=64,
                           threads=a.threads, pool=True,
                           swappable=a.controller or a.controller_cp or a.swap_selftest)
    assert n == L
    states = [m._hot_residency for m in mods]
    for st in states:
        st.arm_amortization(True)

    cfg = model.config
    hkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size
                                            // cfg.num_attention_heads)
    if a.swap_selftest:
        st0 = states[0]
        assert st0.swappable
        snap = {n2: getattr(st0, n2).clone()
                for n2 in ("h_gu_p", "h_gu_a", "h_dn_p", "h_dn_a")}
        hot0 = set(st0.hot_ids.tolist())
        promote = next(e for e in range(E) if e not in hot0)
        demote = int(st0.hot_ids[-1])
        ids = tok("The capital of France is",
                  return_tensors="pt").input_ids.cuda()
        def cont():
            out = model(input_ids=ids, use_cache=False)
            return int(out.logits[0, -1].argmax())
        t0 = cont()
        st0.swap_expert(promote, demote)
        slot = int(st0.g2h[promote])
        di = int(st0.g2d_cpu[promote])
        assert torch.equal(st0.h_gu_p[slot].cpu(), st0.d_gu_p[di])
        assert torch.equal(st0.h_gu_a[slot].cpu(), st0.d_gu_a[di])
        assert torch.equal(st0.h_dn_p[slot].cpu(), st0.d_dn_p[di])
        assert torch.equal(st0.h_dn_a[slot].cpu(), st0.d_dn_a[di])
        st0.swap_expert(demote, promote)
        for n2, t in snap.items():
            assert torch.equal(getattr(st0, n2), t), f"{n2} not restored"
        assert cont() == t0, "decode diverged after swap round-trip"
        print("SWAP-SELFTEST-OK", flush=True)
        return

    # paged attention registers only for the serving arms -- the selftest
    # above calls the bare model, which must use the stock attention path
    register(model)


    class Runner(PagedModelRunner):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.step_rows = []

        def _snap(self):
            return (sum(st.amort["dram_ns"] for st in states),
                    sum(st.amort["uniq_dram"] for st in states),
                    sum(st.amort["gpu_ns"] for st in states))

        @torch.no_grad()
        def run_prefill(self, chunks):
            # decode-only counters by construction (the step_decomp
            # rollback, ported): capture the tier counters, let the
            # prefill run, restore -- so prefill uniques never reach
            # uniq_dram or the boundary first-32 metric (Bugbot,
            # e4b#205). Lists are snapshot-copied like tensors.
            saved = []
            for st in states:
                am = st.amort
                saved.append(None if am is None else
                             {k2: (v.clone() if torch.is_tensor(v)
                                   else list(v) if isinstance(v, list)
                                   else v)
                              for k2, v in am.items()})
            out = super().run_prefill(chunks)
            for st, am in zip(states, saved):
                if am is not None and st.amort is not None:
                    cur = st.amort
                    for k2, v in am.items():
                        if torch.is_tensor(v):
                            cur[k2].copy_(v)
                        else:
                            cur[k2] = v
            return out

        @torch.no_grad()
        def run_decode(self, rids):
            t0 = time.perf_counter_ns()
            d0, _, g0 = self._snap()
            out = super().run_decode(rids)   # engine hook fires in here
            if not rids:
                return out
            d1, u1, g1 = self._snap()
            self.step_rows.append(
                {"wall_ns": time.perf_counter_ns() - t0,
                 "dram_ns": d1 - d0, "gpu_ns": g1 - g0, "uniq_cum": u1})
            return out

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    all_ids = tok(text, return_tensors="pt").input_ids[0]

    ctrl = None
    if a.controller or a.controller_cp:
        from experts4bit_qlora.engines.slot_controller import SlotController
        ctrl = SlotController(states, prior, cp=a.controller_cp,
                              trim_series=True)

    win_rows = []
    all_step_rows = []
    for w in range(a.windows):
        off = a.window_base + w * a.window_stride
        span_ids = all_ids[off:off + a.window_span]
        step = max(1, (span_ids.numel() - a.prompt_len) // a.batch)
        prompts = [span_ids[i * step:i * step + a.prompt_len].tolist()
                   for i in range(a.batch)]
        # fresh KV + runner per window; the TIER (and its swap state)
        # lives on the modules and persists -- that is the experiment
        kv_w = Fp8PagedKV(L, hkv, hd, batch=a.batch,
                          max_tokens_per_seq=a.prompt_len + a.gen_tokens + 8,
                          k_groups=4, device="cuda")
        runner = Runner(model, kv_w, device="cuda")
        if ctrl is not None:
            runner.slot_controller = ctrl
        sched = ContinuousScheduler(runner=runner, max_seqs=a.batch,
                                    kv_slots=a.batch, chunk_tokens=a.chunk,
                                    max_prefill_tokens_per_step=a.chunk)
        for p in prompts:
            sched.add_request(p, max_new_tokens=a.gen_tokens)
        s0 = ctrl.swaps if ctrl else 0
        u0 = sum(st.amort["uniq_dram"] for st in states)
        while sched.active or sched.queue:
            if sched.step().is_empty:
                break
        torch.cuda.synchronize()
        rows = runner.step_rows
        all_step_rows.extend(rows)
        win_rows.append({
            "window": w, "decode_steps": len(rows),
            "dram_ms_total": sum(r["dram_ns"] for r in rows) / 1e6,
            "wall_ms_total": sum(r["wall_ns"] for r in rows) / 1e6,
            "dram_ms_median": statistics.median(
                r["dram_ns"] for r in rows) / 1e6 if rows else 0,
            "gpu_ms_median": statistics.median(
                r["gpu_ns"] for r in rows) / 1e6 if rows else 0,
            "uniq_dram_end": rows[-1]["uniq_cum"] if rows else 0,
            "uniq_first32": (rows[31]["uniq_cum"] - u0
                             if len(rows) >= 32 else None),
            "swaps": (ctrl.swaps - s0) if ctrl else 0,
        })
        print("window %d: steps %d dram %.1f ms wall %.1f ms swaps %d"
              % (w, len(rows), win_rows[-1]["dram_ms_total"],
                 win_rows[-1]["wall_ms_total"], win_rows[-1]["swaps"]),
              flush=True)

    nv = sum(st.amort["uniq_nvme"] for st in states)
    assert nv == 0, f"NVMe touched: {nv}"
    dram_med = statistics.median(r["dram_ns"] for r in all_step_rows) / 1e6
    gpu_med = statistics.median(r["gpu_ns"] for r in all_step_rows) / 1e6
    balance = (min(dram_med, gpu_med) / max(dram_med, gpu_med)
               if max(dram_med, gpu_med) > 0 else 0.0)
    rep = {
        "controller": bool(a.controller or a.controller_cp),
        "windows": win_rows,
        "uniq_dram_total": int(sum(st.amort["uniq_dram"] for st in states)),
        "dram_ms_grand": sum(x["dram_ms_total"] for x in win_rows),
        "wall_ms_grand": sum(x["wall_ms_total"] for x in win_rows),
        "swaps_total": ctrl.swaps if ctrl else 0,
        "controller_ms": (ctrl.ns / 1e6) if ctrl else 0.0,
        "cp_resets": ctrl.cp_resets if ctrl else 0,
        "controller_cp": bool(a.controller_cp),
        "dram_ms_median": dram_med,
        "gpu_ms_median": gpu_med,
        "balance_ratio": balance,
        "decode_steps": sum(x["decode_steps"] for x in win_rows),
    }
    pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
    print("C2_RUN_DONE", json.dumps({k2: rep[k2] for k2 in
          ("controller", "uniq_dram_total", "dram_ms_grand",
           "dram_ms_median", "gpu_ms_median", "balance_ratio",
           "swaps_total", "controller_ms", "decode_steps")}), flush=True)


if __name__ == "__main__":
    main()
