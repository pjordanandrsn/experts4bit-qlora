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
    python p146_box.py --probe --outdir DIR --tag TAG                                     (one pinned H2D probe; A1)
    python p146_box.py --preflight                         (A1: every guard signal readable, before anything is paid for)
    python p146_box.py --self-test

Amendment A1 (after p146-4090-5, VOID on the self-pairs alone while the host got noisy mid-box): a host-quiet guard,
recorded and never acted on here; the reducer reads it.
  * CPU, per arm, over its timed window (from the end of warm step 2 to the end of the last step): the box cgroup's
    throttled time (cpu.stat: v2 throttled_usec, v1 throttled_time) as a fraction of the window's wall, and /proc/stat
    steal as a fraction of all CPU time on the "cpu" line;
  * link: one pinned host-to-device probe (engines.offload.report_offload_environment, 256 MB x 20) in its own process
    before the first arm and after every arm, appended to probes.json.
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


def read_throttled_usec(root="/sys/fs/cgroup"):
    """The cgroup's cumulative throttled time in microseconds: v2 cpu.stat throttled_usec, v1 cpu/cpu.stat
    throttled_time (ns). None when neither is readable."""
    for path, key, scale in ((os.path.join(root, "cpu.stat"), "throttled_usec", 1.0),
                             (os.path.join(root, "cpu", "cpu.stat"), "throttled_time", 1e-3)):
        try:
            for line in open(path):
                k, _, v = line.partition(" ")
                if k == key:
                    return float(v) * scale
        except OSError:
            continue
    return None


def read_proc_stat(path="/proc/stat"):
    """(steal, total) jiffies from the aggregate "cpu" line: total is user..steal. None when unreadable."""
    try:
        f = open(path).readline().split()
        if f[0] != "cpu" or len(f) < 9:
            return None
        v = [int(x) for x in f[1:9]]
        return v[7], sum(v)
    except (OSError, ValueError, IndexError):
        return None


def host_snapshot(root="/sys/fs/cgroup", stat="/proc/stat"):
    return {"t": time.monotonic(), "throttled_usec": read_throttled_usec(root), "proc_stat": read_proc_stat(stat)}


def host_window(a, b):
    """The timed window's guard signals from two snapshots; a signal that could not be read is None."""
    wall = b["t"] - a["t"]
    thr = None
    if a["throttled_usec"] is not None and b["throttled_usec"] is not None and wall > 0:
        thr = (b["throttled_usec"] - a["throttled_usec"]) / (wall * 1e6)
    steal = None
    if a["proc_stat"] and b["proc_stat"] and b["proc_stat"][1] > a["proc_stat"][1]:
        steal = (b["proc_stat"][0] - a["proc_stat"][0]) / (b["proc_stat"][1] - a["proc_stat"][1])
    return {"wall_s": wall, "throttled_frac": thr, "steal_frac": steal}


def h2d_pinned_gbps(mb=256, iters=20):
    """A pinned host-to-device copy rate in plain torch (report_offload_environment's micro-bench, without e4b)."""
    import torch

    n = mb * 1024 * 1024
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    src = torch.empty(n, dtype=torch.uint8).pin_memory()
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    e.record()
    torch.cuda.synchronize()
    return iters * n / 1e9 / (s.elapsed_time(e) / 1e3)


def preflight_ok(snap, gbps):
    """A1 pre-flight: every signal the guard reads must be readable on this host."""
    bad = []
    if snap["throttled_usec"] is None:
        bad.append("cgroup cpu.stat throttled time unreadable")
    if snap["proc_stat"] is None:
        bad.append("/proc/stat steal unreadable")
    if not gbps or gbps <= 0:
        bad.append("pinned H2D probe failed")
    return bad


def run_preflight():
    """Read cpu.stat and /proc/stat once and run one pinned H2D probe; rc 2 if any of them is unreadable."""
    snap = host_snapshot()
    try:
        gbps = h2d_pinned_gbps()
    except Exception as e:                       # noqa: BLE001 -- any failure is "unreadable"
        print(f"P146_PREFLIGHT_PROBE_ERROR {type(e).__name__}: {e}", flush=True)
        gbps = None
    bad = preflight_ok(snap, gbps)
    print("P146_PREFLIGHT " + json.dumps({"throttled_usec": snap["throttled_usec"], "proc_stat": snap["proc_stat"],
                                          "pinned_gbps": gbps, "bad": bad}), flush=True)
    return 2 if bad else 0


def run_probe(a):
    """One pinned H2D probe in this (short) process, appended to probes.json with its tag."""
    from experts4bit_qlora.engines.offload import report_offload_environment

    out = report_offload_environment("cuda", lambda s: None) or {}
    path = os.path.join(a.outdir, "probes.json")
    probes = json.load(open(path)) if os.path.exists(path) else []
    probes.append({"tag": a.tag, "pinned_gbps": out.get("ceiling_pinned_gbps"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    json.dump(probes, open(path, "w"), indent=1)
    print("P146_PROBE " + json.dumps(probes[-1]), flush=True)
    return 0


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

    got = {"dataset_calls": [], "evals": [], "host": {}}
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
        if len(self.steps) == WARM:              # A1: the timed window opens after the warm steps
            got["host"]["start"] = host_snapshot()
        elif len(self.steps) == STEPS:           # ... and closes after the last step
            got["host"]["end"] = host_snapshot()

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
           "host_guard": (host_window(got["host"]["start"], got["host"]["end"])
                          if "start" in got["host"] and "end" in got["host"] else None),
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
    probe(a, "before_A0a")
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
        probe(a, f"after_{tag}")
    return rc_all


def probe(a, tag):
    """A1: the link probe in its own process (the arms hold no CUDA context while it runs)."""
    cmd = [sys.executable, os.path.abspath(__file__), "--probe", "--outdir", a.outdir, "--tag", tag]
    try:
        r = subprocess.run(cmd, env=arm_env("A0a", a.model, a.outdir), capture_output=True, text=True, timeout=300)
        line = [x for x in r.stdout.splitlines() if x.startswith("P146_PROBE ")]
        print(line[-1] if line else f"P146_PROBE_FAILED {tag} rc={r.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"P146_PROBE_FAILED {tag} timeout", flush=True)


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
    import tempfile
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "cpu"))
    open(os.path.join(d, "cpu.stat"), "w").write("usage_usec 10\nnr_throttled 3\nthrottled_usec 2500000\n")
    assert read_throttled_usec(d) == 2500000.0
    os.remove(os.path.join(d, "cpu.stat"))
    open(os.path.join(d, "cpu", "cpu.stat"), "w").write("nr_periods 9\nthrottled_time 4000000000\n")
    assert read_throttled_usec(d) == 4000000.0                          # v1: ns -> us
    os.remove(os.path.join(d, "cpu", "cpu.stat"))
    assert read_throttled_usec(d) is None
    n += 1
    ps = os.path.join(d, "stat")
    open(ps, "w").write("cpu  100 0 50 800 10 0 0 40 0 0\ncpu0 1 2 3\n")
    assert read_proc_stat(ps) == (40, 1000)
    open(ps, "w").write("intr 1 2\n")
    assert read_proc_stat(ps) is None
    n += 1
    w = host_window({"t": 0.0, "throttled_usec": 1e6, "proc_stat": (40, 1000)},
                    {"t": 100.0, "throttled_usec": 6e6, "proc_stat": (90, 2000)})
    assert abs(w["throttled_frac"] - 0.05) < 1e-12 and abs(w["steal_frac"] - 0.05) < 1e-12 and w["wall_s"] == 100.0
    w = host_window({"t": 0.0, "throttled_usec": None, "proc_stat": None}, {"t": 1.0, "throttled_usec": 5.0, "proc_stat": (1, 2)})
    assert w["throttled_frac"] is None and w["steal_frac"] is None
    n += 1
    ok = {"t": 0.0, "throttled_usec": 0.0, "proc_stat": (0, 10)}
    assert preflight_ok(ok, 20.0) == []
    assert len(preflight_ok({**ok, "throttled_usec": None}, 20.0)) == 1
    assert len(preflight_ok({**ok, "proc_stat": None}, 20.0)) == 1
    assert len(preflight_ok(ok, None)) == 1 and len(preflight_ok(ok, 0.0)) == 1
    assert len(preflight_ok({"t": 0.0, "throttled_usec": None, "proc_stat": None}, None)) == 3
    n += 1
    print(f"p146_box self-test OK ({n}/15 cases)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--mode", choices=["cli-abba"])
    p.add_argument("--arm", choices=TAGS)
    p.add_argument("--model")
    p.add_argument("--parquet", help="tatsu-lab/alpaca's train parquet at the pinned revision (the staged file)")
    p.add_argument("--outdir")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--tag")
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.preflight:
        return run_preflight()
    if a.probe:
        return run_probe(a)
    if a.arm:
        return run_arm(a)
    return run_cli_abba(a)


if __name__ == "__main__":
    sys.exit(main())
