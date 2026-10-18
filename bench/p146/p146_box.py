#!/usr/bin/env python3
"""p146_box.py -- lane P146's BOX script (bench/p146/PREREG-p146.md). The CLI trainer, one fresh process per arm.

The subject is ``python -m experts4bit_qlora.train`` at its own recipe (its defaults: SEQ 192, TOKEN_BUDGET 2048,
GRAD_ACCUM 4, r 8 / alpha 16, attention LoRA on, torch AdamW at lr 2e-4 with cosine warm-up, alpaca train[:2000],
SEED 0), on Qwen3-30B-A3B with its experts offloaded. The two arms differ only in LAYER_MAJOR:
  A0  OFFLOAD_EXPERTS=1 FAST_TRAIN=1               (the stock accumulation loop on the fused path)
  L1  OFFLOAD_EXPERTS=1 LAYER_MAJOR=1              (layer_major_step; LAYER_MAJOR=1 implies FAST_TRAIN=1)
Both run STEPS = WARM + TIMED = 34 with DO_GEN=0. Each arm is its own process, as a user runs it; ABBA order.

Nothing in train.py is changed. This script calls ``experts4bit_qlora.train.main()`` after wrapping, from outside:
  * datasets.load_dataset("tatsu-lab/alpaca", ...) -> the staged parquet of that dataset at its pinned revision
    (P137 Amendment 1: no dataset download on the box), with the same split string;
  * census.PhaseClock.step_end (TR1_CENSUS=1, train.py's own step clock: one CUDA sync per step, wall from before
    zero_grad to after sched.step) -> the step's host-to-device stagings (E4B_OFFLOAD_STATS's copy count) and peak
    allocated memory are added to the step's row, then both counters are reset;
  * engines.fast.enable_fast_train -> its patched count, the offloaded layer count and the load time are recorded;
  * torch.optim.AdamW -> the sha256 of the adapters it is built over (every arm must start from the same adapters);
  * train.eval_loss -> the BEFORE and AFTER held-out losses.

    python p146_box.py --mode cli-abba --model DIR --parquet alpaca.parquet --outdir DIR
    python p146_box.py --arm TAG --model DIR --parquet alpaca.parquet --outdir DIR      (one arm; the parent runs these)
    python p146_box.py --self-test
"""
import argparse
import json
import os
import subprocess
import sys
import time

WARM, TIMED = 2, 32             # step 0 (the equality step) is the first warm step; P137 Amendment 3's n
STEPS = WARM + TIMED
ARM_ENV = {"A0": {"OFFLOAD_EXPERTS": "1", "FAST_TRAIN": "1"},
           "L1": {"OFFLOAD_EXPERTS": "1", "LAYER_MAJOR": "1"}}
LOOPS = {"A0": "stock", "L1": "layer_major"}
TAGS = ("A0a", "L1a", "L1b", "A0b")             # ABBA, one fresh process each
ALPACA = "tatsu-lab/alpaca"
ARM_TIMEOUT_S = 1500
# Every lever train.py or the engines read starts unset in an arm's environment; the arm sets its own.
LEVERS = ("LAYER_MAJOR", "FAST_TRAIN", "OFFLOAD_EXPERTS", "OFFLOAD_PIN", "E4B_OFFLOAD_STATS", "E4B_OFFLOAD_ARENA",
          "E4B_OFFLOAD_PIN_SLAB", "E4B_CKPT_OFFLOAD", "E4B_CHUNKED_LM_LOSS", "E4B_CHUNKED_EVAL_LOSS", "E4B_ABSMAX_DQ",
          "E4B_MOE_KEEP_LAYERS", "TRAIN_ARENA", "TRAIN_VRAM_FRAC", "TRAIN_ATTN_4BIT", "TRAIN_ROUTER", "TRAIN_EXPERTS",
          "TRAIN_ATTENTION", "SEQ", "STEPS", "GRAD_ACCUM", "TOKEN_BUDGET", "LR", "R", "ALPHA", "N_TRAIN", "EVAL_EVERY",
          "SEED", "QUANT_TYPE", "DO_GEN", "OUT", "MODEL", "TR1_CENSUS", "TR1_CENSUS_OUT", "TR1_PROFILE_OUT",
          "E4B_EXPERT_PROFILE", "TRITON_INTERPRET")


def arm_env(tag, model, outdir, base=None):
    """The environment one arm's process runs under: the levers cleared, then the CLI's switches for this arm."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k not in LEVERS}
    env.update(ARM_ENV[tag[:2]])
    env.update({"MODEL": model, "STEPS": str(STEPS), "DO_GEN": "0", "TR1_CENSUS": "1",
                "TR1_CENSUS_OUT": os.path.join(outdir, f"census_{tag}.json"), "E4B_OFFLOAD_STATS": "1",
                "OUT": os.path.join(outdir, f"out_{tag}")})
    return env


def summarize(steps, warm=WARM):
    """Per-arm summary from the per-step rows: the median timed wall, and the steady stagings (every step after 0)."""
    timed = [s["wall_s"] for s in steps[warm:]]
    srt = sorted(timed)
    mid = len(srt) // 2
    med = (srt[mid] if len(srt) % 2 else (srt[mid - 1] + srt[mid]) / 2) if srt else None
    return {"median_s": med, "timed_walls": timed, "stagings_steady": sorted({s["stagings"] for s in steps[1:]}),
            "stagings_step0": steps[0]["stagings"] if steps else None,
            "peak_alloc": max((s["peak_alloc"] for s in steps[warm:]), default=0),
            "peak_alloc_all": max((s["peak_alloc"] for s in steps), default=0)}


def rows_from_census(census):
    """train.py's census rows -> the lane's per-step records (wall in seconds)."""
    return [{"step": i, "wall_s": r["step_wall_ms"] / 1e3, "loss": r["loss"], "stagings": r["stagings"],
             "peak_alloc": r["peak_alloc"]} for i, r in enumerate(census["steps"])]


def adapter_hash(params):
    """sha256 over every trainable tensor's bytes, in order."""
    import hashlib

    import torch

    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().to("cpu").contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run_arm(a):
    """One arm: wrap the instruments, then train.py's own main(). Run in a fresh process by run_cli_abba."""
    t_start = time.perf_counter()
    import torch

    import datasets

    from experts4bit_qlora import census
    from experts4bit_qlora.engines import ckpt_offload, fast
    from experts4bit_qlora.engines import offload as off

    got = {"dataset_calls": [], "evals": []}
    _load_dataset = datasets.load_dataset

    def load_dataset(path, *args, split=None, **kw):
        if path != ALPACA or args or kw:
            raise SystemExit(f"P146: unexpected dataset request {path!r} {args} {kw}")
        got["dataset_calls"].append(split)
        return _load_dataset("parquet", data_files={"train": a.parquet}, split=split)

    datasets.load_dataset = load_dataset

    _step_end = census.PhaseClock.step_end

    def step_end(self, extra=None):
        _step_end(self, extra)
        row = self.steps[-1]
        row["stagings"] = len(off._stats().copies)
        row["peak_alloc"] = torch.cuda.max_memory_allocated()
        off.reset_offload_stats()
        torch.cuda.reset_peak_memory_stats()

    census.PhaseClock.step_end = step_end

    _enable = fast.enable_fast_train

    def enable_fast_train(model, *args, **kw):
        n = _enable(model, *args, **kw)
        got.update(n_fast=n, n_offloaded=len(off.offload_handles(model)),
                   load_s=round(time.perf_counter() - t_start, 1))
        return n

    fast.enable_fast_train = enable_fast_train

    class AdamW(torch.optim.AdamW):
        def __init__(self, params, *args, **kw):
            params = list(params)
            flat = [p for g in params for p in (g["params"] if isinstance(g, dict) else [g])]
            got["adapter_sha_at_start"] = adapter_hash(flat)
            got["n_trainable"] = sum(p.numel() for p in flat)
            super().__init__(params, *args, **kw)

    torch.optim.AdamW = AdamW

    from experts4bit_qlora import train as t

    _eval = t.eval_loss

    def eval_loss(model, eval_data):
        v = _eval(model, eval_data)
        got["evals"].append(v)
        return v

    t.eval_loss = eval_loss
    t.main()

    cen = json.load(open(os.environ["TR1_CENSUS_OUT"]))
    steps = rows_from_census(cen)
    import importlib.metadata as md

    import transformers

    tag = a.arm
    out = {"tag": tag, "loop": LOOPS[tag[:2]], "model": a.model, "steps": steps,
           "step0_loss_hex": steps[0]["loss"].hex() if steps else None, "summary": summarize(steps),
           "eval_before_hex": got["evals"][0].hex() if got["evals"] else None,
           "heldout": got["evals"][-1] if len(got["evals"]) >= 2 else None,
           "n_fast": got.get("n_fast"), "n_offloaded": got.get("n_offloaded"), "load_s": got.get("load_s"),
           "ckpt": dict(ckpt_offload.CKPT_OFFLOAD_STATS), "adapter_sha_at_start": got.get("adapter_sha_at_start"),
           "n_trainable": got.get("n_trainable"), "dataset_calls": got["dataset_calls"],
           "token_budget_effective": cen["meta"].get("token_budget_effective"),
           "process": "fresh process per arm (python -m experts4bit_qlora.train's main)",
           "switches": {k: os.environ.get(k) for k in ("OFFLOAD_EXPERTS", "FAST_TRAIN", "LAYER_MAJOR")},
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "bitsandbytes": md.version("bitsandbytes"), "gnf4": md.version("grouped-nf4-gemm"),
                        "e4b_sha": os.environ.get("E4B_SHA"), "gpu": torch.cuda.get_device_name(0)},
           "recipe": {"seq": t.SEQ, "token_budget": t.TOKEN_BUDGET, "grad_accum": t.GRAD_ACCUM, "steps": t.STEPS,
                      "warm": WARM, "timed": TIMED, "lr": t.LR, "r": t.R, "alpha": t.ALPHA, "n_train": t.N_TRAIN,
                      "eval_every": t.EVAL_EVERY, "seed": int(os.environ.get("SEED", "0")),
                      "train_attention": t.TRAIN_ATTENTION, "train_experts": t.TRAIN_EXPERTS,
                      "fast_train": t.FAST_TRAIN, "layer_major": t.LAYER_MAJOR, "offload": t.OFFLOAD_EXPERTS}}
    json.dump(out, open(os.path.join(a.outdir, f"arm_{tag}.json"), "w"), indent=1)
    print("P146_ARM " + json.dumps({k: out[k] for k in ("tag", "loop", "summary", "heldout", "step0_loss_hex",
                                                         "load_s", "token_budget_effective")}), flush=True)
    return 0


def run_cli_abba(a):
    """A0a L1a L1b A0b, each `python p146_box.py --arm TAG` in a fresh process; one log per arm."""
    os.makedirs(os.path.join(a.outdir, "logs"), exist_ok=True)
    rc_all = 0
    for tag in TAGS:
        log = os.path.join(a.outdir, "logs", f"arm_{tag}.log")
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", tag, "--model", a.model, "--parquet", a.parquet,
               "--outdir", a.outdir]
        t0 = time.perf_counter()
        with open(log, "w") as f:
            try:
                rc = subprocess.run(cmd, env=arm_env(tag, a.model, a.outdir), stdout=f, stderr=subprocess.STDOUT,
                                    timeout=ARM_TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        print("P146_PROC " + json.dumps({"tag": tag, "rc": rc, "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
        for line in open(log, errors="replace"):
            if line.startswith("P146_ARM ") or "host CPUs:" in line or "host-side fuse" in line:
                print(line.rstrip()[:900], flush=True)
        rc_all = rc_all or rc
    return rc_all


def self_test():
    n = 0
    base = {"PATH": "/bin", "LAYER_MAJOR": "1", "FAST_TRAIN": "0", "SEQ": "2048", "E4B_CHUNKED_LM_LOSS": "1", "HOME": "/r"}
    e = arm_env("A0a", "/m", "/o", base=base)
    assert e["OFFLOAD_EXPERTS"] == "1" and e["FAST_TRAIN"] == "1" and "LAYER_MAJOR" not in e and "SEQ" not in e
    assert "E4B_CHUNKED_LM_LOSS" not in e and e["PATH"] == "/bin" and e["HOME"] == "/r"
    n += 1
    e = arm_env("L1b", "/m", "/o", base=base)
    assert e["LAYER_MAJOR"] == "1" and "FAST_TRAIN" not in e and e["OFFLOAD_EXPERTS"] == "1"   # LAYER_MAJOR implies it
    n += 1
    assert e["STEPS"] == "34" and e["DO_GEN"] == "0" and e["TR1_CENSUS"] == "1" and e["E4B_OFFLOAD_STATS"] == "1"
    assert e["TR1_CENSUS_OUT"].endswith("census_L1b.json") and e["MODEL"] == "/m"
    n += 1
    cen = {"steps": [{"step_wall_ms": w, "loss": 1.0, "stagings": st, "peak_alloc": p} for w, st, p in
                     [(9000.0, 381, 1), (5000.0, 380, 2), (4000.0, 380, 3), (6000.0, 380, 2), (5500.0, 380, 1)]]}
    steps = rows_from_census(cen)
    assert steps[1]["wall_s"] == 5.0 and steps[4]["step"] == 4
    n += 1
    sm = summarize(steps, warm=2)
    assert sm["median_s"] == 5.5 and sm["stagings_steady"] == [380] and sm["stagings_step0"] == 381
    n += 1
    assert sm["peak_alloc"] == 3 and sm["peak_alloc_all"] == 3
    n += 1
    steps[0]["peak_alloc"] = 9
    assert summarize(steps, warm=2)["peak_alloc"] == 3 and summarize(steps, warm=2)["peak_alloc_all"] == 9
    n += 1
    assert summarize(steps[:4], warm=2)["median_s"] == 5.0
    n += 1
    assert summarize([])["median_s"] is None and summarize([])["stagings_step0"] is None
    n += 1
    assert set(LOOPS) == {"A0", "L1"} and TAGS == ("A0a", "L1a", "L1b", "A0b") and (WARM, TIMED) == (2, 32)
    n += 1
    # the stock and layer-major step losses agree bit for bit: A0 sums x/4 in order, L1 divides the sum by 4, and a
    # power-of-two scale commutes with round-to-nearest
    xs = [0.741409182548523, 1.2829464673995972, 1.4187248945236206, 1.5992339849472046]
    acc = 0.0
    for x in xs:
        acc += x / 4
    assert acc.hex() == (sum(xs) / 4).hex()
    n += 1
    print(f"p146_box self-test OK ({n}/11 cases)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--mode", choices=["cli-abba"])
    p.add_argument("--arm", choices=TAGS)
    p.add_argument("--model")
    p.add_argument("--parquet", help="tatsu-lab/alpaca's train parquet at the pinned revision (the staged file)")
    p.add_argument("--outdir")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.arm:
        return run_arm(a)
    return run_cli_abba(a)


if __name__ == "__main__":
    sys.exit(main())
