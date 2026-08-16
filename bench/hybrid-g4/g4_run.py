# G4 gate runner — speculative-prefetch A/B at Qwen3-235B, cold-started arms.
# Each arm rebuilds the tier from scratch (enable -> warm -> timed decode ->
# disable) so arm order cannot leak residency; B1/C/B2 sandwich detects any
# page-cache drift. Stall metric = demand tier misses inside the timed
# window (each one is a synchronous NVMe fetch on the critical path).
# Needs the banked route_profile.jsonl + calib.json from the formal G3 run.
import argparse
import json
import os
import time
from pathlib import Path

ARENA = os.environ.get("G3_ARENA", "/root/q235.arena")
SNAP = os.environ.get("G3_SNAP", "/root/q235")
OUT = Path("/root/out")
PROFILE = str(OUT / "route_profile.jsonl")
CALIB = str(OUT / "calib.json")
PROMPT = "The three tiers of a memory hierarchy, from fastest to slowest, are"


def main(threads: int, dram_gb: int, n_new: int, arms: str):
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    import torch
    from nvme_arena import load_index
    from transformers import AutoTokenizer

    from experts4bit_qlora import (load_moe_4bit_streaming, save_manifest,
                                   solve_placement)
    from experts4bit_qlora.engines import hybrid as hy
    from experts4bit_qlora.engines.hot_residency import target_modules

    tok = AutoTokenizer.from_pretrained(SNAP)
    model, _cfg = load_moe_4bit_streaming(
        SNAP, "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
        arena=ARENA)
    model.eval()
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    idx = load_index(ARENA)
    print(f"L={L} E={E} row_stride={idx['row_stride']}", flush=True)

    free_b, _total = torch.cuda.mem_get_info()
    vram_budget = max(0, free_b - 4 * (1 << 30))
    m = solve_placement(
        n_layers=L, n_experts=E, bytes_per_expert=idx["row_stride"],
        vram_budget_bytes=vram_budget,
        dram_budget_bytes=dram_gb * (1 << 30),
        calibration=CALIB, profile_path=PROFILE)
    man_path = str(OUT / f"placement_g4_dram{dram_gb}.json")
    sha = save_manifest(m, man_path)
    print(f"PLACEMENT dram_gb={dram_gb} sha={sha[:16]} "
          f"masses={m['masses']}", flush=True)

    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()

    def run_arm(name, prefetch_on, sample=False):
        n = hy.enable_hybrid_tier(model, ARENA, man_path,
                                  hot_rows=max(E, 128), threads=threads,
                                  prefetch=True)
        assert n == L, f"{name}: engaged {n}/{L}"
        hy.set_prefetch(model, prefetch_on)
        st = mods[0]._hot_residency

        # in-process sampling profiler (py-spy is ptrace-blocked here):
        # a 50 ms poll of sys._current_frames over the timed decode gives a
        # line-level histogram of where the main thread actually is
        import collections
        import sys as _sys
        import threading as _th
        hist_main: collections.Counter = collections.Counter()
        hist_rest: collections.Counter = collections.Counter()
        stop_s = _th.Event()
        main_id = _th.get_ident()

        def _tag(f):
            return (f"{f.f_code.co_filename.rsplit('/', 1)[-1]}:{f.f_lineno}"
                    f" {f.f_code.co_name}")

        def sampler():
            while not stop_s.wait(0.05):
                for tid, f in _sys._current_frames().items():
                    if tid == main_id:
                        hist_main[_tag(f)] += 1
                    elif "hybrid" in f.f_code.co_filename or \
                            "nvme" in f.f_code.co_filename:
                        hist_rest[_tag(f)] += 1

        with torch.no_grad():
            model.generate(ids, max_new_tokens=4, do_sample=False)
            torch.cuda.synchronize()
            s0 = dict(st.tier_stats())
            p0 = dict(hy.prefetch_stats(model))
            smp = None
            if sample:
                smp = _th.Thread(target=sampler, daemon=True)
                smp.start()
            t0 = time.perf_counter()
            out = model.generate(ids, max_new_tokens=n_new, do_sample=False)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if smp is not None:
                stop_s.set()
                smp.join()
            s1 = dict(st.tier_stats())
            p1 = dict(hy.prefetch_stats(model))
        # byte-verify a sample of resident rows against direct disk reads,
        # and check the residency maps are a consistent bijection — rules
        # out (or in) corruption from concurrent publishes before teardown.
        # QUIESCE the prefetch worker first: with it live, a speculative
        # ensure can reassign a snapshotted slot between the map copy and
        # the byte read. That race can only manufacture FALSE corruption
        # (a reassigned slot cannot match the old key's reference bytes),
        # never hide real damage — but the check should not need that
        # argument to be trusted.
        verify = None
        if prefetch_on:
            from nvme_reader import alloc_landing
            hy.set_prefetch(model, False)
            if hy._PF_POOL is not None:
                hy._PF_POOL.submit(lambda: None).result()
            t = mods[0]._e4b_cold_tier
            with t._lock:
                res = sorted(t._slot_of.items())
                incons = sum(1 for k, s in res if t._key_of[s] != k)
            sample = res[:: max(1, len(res) // 16)][:16]
            bad = 0
            for (lay, e), slot in sample:
                live = bytes(t._slot_view(slot)[: t.row_bytes])
                ref, keep = alloc_landing(t.row_stride, pinned=False)
                t.reader.read_row_sync(lay, e, ref)
                if live != bytes(ref[: t.row_bytes]):
                    bad += 1
                del keep
            verify = {"rows_checked": len(sample), "rows_corrupt": bad,
                      "map_inconsistencies": incons}
        hy.disable_hybrid_tier(model)
        rep = {"arm": name, "prefetch": prefetch_on, "toks": n_new / dt,
               "misses_timed": s1.get("misses", 0) - s0.get("misses", 0),
               "demand_misses_timed": (s1.get("demand_misses", 0)
                                       - s0.get("demand_misses", 0)),
               "spec_misses_timed": (s1.get("spec_misses", 0)
                                     - s0.get("spec_misses", 0)),
               "demand_waits_timed": (s1.get("demand_waits", 0)
                                      - s0.get("demand_waits", 0)),
               "demand_fill_s": (s1.get("demand_fill_ns", 0)
                                 - s0.get("demand_fill_ns", 0)) / 1e9,
               "demand_wait_s": (s1.get("demand_wait_ns", 0)
                                 - s0.get("demand_wait_ns", 0)) / 1e9,
               "spec_fill_s": (s1.get("spec_fill_ns", 0)
                               - s0.get("spec_fill_ns", 0)) / 1e9,
               "decode_wall_s": dt, "verify": verify,
               "main_thread_frames": hist_main.most_common(14),
               "other_thread_frames": hist_rest.most_common(8),
               "tier_before": s0, "tier_after": s1,
               "pf_delta": {k: p1.get(k, 0) - p0.get(k, 0) for k in p1},
               "tail": tok.decode(out[0][-12:])}
        print(f"ARM {json.dumps(rep)}", flush=True)
        return rep

    wanted = [a.strip().lower() for a in arms.split(",")]
    got = {}
    for a in wanted:
        got[a] = run_arm({"b1": "B1_prefetch_off", "c": "C_prefetch_on",
                          "b2": "B2_prefetch_off_recheck"}[a], a == "c",
                         sample=os.environ.get("G4_SAMPLE") == "1")
    if set(wanted) != {"b1", "c", "b2"}:
        print("PARTIAL_ARMS_DONE", json.dumps(got), flush=True)
        return
    b1, c, b2 = got["b1"], got["c"], got["b2"]

    # The gate metric is DEMAND misses: synchronous NVMe fetches on the
    # serving thread's critical path. Speculative fetches are background
    # warming and counted separately (the first G4 attempt conflated them).
    base = (b1["demand_misses_timed"] + b2["demand_misses_timed"]) / 2
    red = 1.0 - (c["demand_misses_timed"] / base) if base else float("nan")
    nvme_frac = m["masses"]["nvme_frac"]
    rep = {"dram_gb": dram_gb, "n_new": n_new, "masses": m["masses"],
           "manifest_sha": sha, "vram_budget_gb": vram_budget / 2**30,
           "threads": threads, "arms": [b1, c, b2],
           "stall_baseline_misses": base, "stall_reduction": red,
           "tails_identical": b1["tail"] == c["tail"] == b2["tail"],
           "gate_g4": {"nvme_frac": nvme_frac,
                       "band_ok": 0.10 <= nvme_frac <= 0.15,
                       "reduction_ok": base > 0 and red >= 0.50}}
    p = OUT / f"g4_dram{dram_gb}.json"
    p.write_text(json.dumps(rep, indent=2))
    print("G4_REPORT " + json.dumps(rep), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--dram-gb", type=int, default=15)
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--arms", default="b1,c,b2")
    a = ap.parse_args()
    main(a.threads, a.dram_gb, a.n_new, a.arms)
