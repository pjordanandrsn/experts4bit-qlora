"""One arm of the host-RAM ceiling experiment.

The question is narrow: with only N GB of host RAM available, does QLoRA training
of an MoE start at all? Two arms differ in exactly one thing -- where the frozen
4-bit expert weights live.

  host   experts fused, quantized and pinned in host DRAM  (~3.6 GB, unswappable)
  arena  experts read from a baked arena on NVMe           (~0.2 GB of slots)

Everything else -- model, dataset, batch shape, seed, step count -- is identical.

WHY THIS CONTAINER AND NOT A RENTED ONE: the verdict has to be "did not fit",
and that is only meaningful if there was nowhere to hide. A rented container's
cgroup is read-only, and the kernels seen there had no memsw accounting, so an
over-limit process pages out to swap and survives -- which is a different
outcome from fitting, and reports as success. Here docker sets BOTH
memory.limit_in_bytes and memory.memsw.limit_in_bytes, so the cap covers swap
too and there is no escape. The cap was positive-controlled in both directions
before any of this ran (512m kills a 900 MB allocation, 2g does not).

Death by cap is SIGKILL, so this process prints nothing when it loses. That is
expected: the authoritative verdict is docker's own State.OOMKilled plus the
exit code, read from outside. What is printed here is only for the runs that
survive.
"""
import argparse
import json
import os
import resource
import sys
import threading
import time
import torch

MODEL = os.environ.get("E4B_MODEL", "allenai/OLMoE-1B-7B-0924")
ARENA = os.environ.get("E4B_ARENA", "/work/arena/olmoe-nf4.arena")


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _status():
    d = {}
    try:
        for line in open("/proc/self/status"):
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip()
    except Exception:
        pass
    return d


def _smaps():
    d = {}
    try:
        for line in open("/proc/self/smaps_rollup"):
            k, _, v = line.partition(":")
            v = v.strip()
            if v.endswith("kB"):
                d[k.strip()] = int(v.split()[0])
    except Exception:
        pass
    return d


def _kb(key):
    try:
        return int(_status().get(key, "0 kB").split()[0])
    except Exception:
        return 0


def _read_int(p):
    try:
        s = open(p).read().strip()
        return int(s) if s.isdigit() else s
    except Exception:
        return None


def cgroup_info():
    """Read the cap from inside, so the run records the ceiling it actually had.

    memsw is the field that matters: without it the cap bounds resident memory
    only and swap is a free escape hatch.
    """
    out = {}
    v2, v1 = "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    if os.path.exists(v2):
        out.update(version=2, limit=_read_int(v2), swap_max=_read_int("/sys/fs/cgroup/memory.swap.max"),
                   peak=_read_int("/sys/fs/cgroup/memory.peak"), current=_read_int("/sys/fs/cgroup/memory.current"))
    elif os.path.exists(v1):
        g = "/sys/fs/cgroup/memory/"
        out.update(version=1, limit=_read_int(g + "memory.limit_in_bytes"),
                   memsw_limit=_read_int(g + "memory.memsw.limit_in_bytes"),
                   peak=_read_int(g + "memory.max_usage_in_bytes"),
                   memsw_peak=_read_int(g + "memory.memsw.max_usage_in_bytes"),
                   current=_read_int(g + "memory.usage_in_bytes"),
                   failcnt=_read_int(g + "memory.failcnt"))
        st = {}
        try:
            for line in open(g + "memory.stat"):
                k, _, v = line.partition(" ")
                st[k] = int(v)
        except Exception:
            pass
        out["stat"] = {k: st.get(k) for k in ("rss", "cache", "mapped_file", "swap", "unevictable")}
    else:
        out["version"] = None
    return out


class Sampler(threading.Thread):
    """Track peak RSS and, separately, peak ANONYMOUS.

    They do not coincide, and only one of them behaves like a requirement. RSS
    peaks while the safetensors are mmap'd -- clean, file-backed, reclaimable
    pages that a tight cgroup simply drops. Anonymous peaks later, once the
    pinned expert buffers exist. Under a cap it is the unreclaimable half that
    decides whether the job lives.
    """
    daemon = True

    def __init__(self, interval=0.05):
        super().__init__()
        self.interval = interval
        self.stop = threading.Event()
        self.peak_kb = self.peak_anon_kb = 0
        self.peak_smaps = {}
        self.t0 = time.time()

    def run(self):
        while not self.stop.is_set():
            sm = _smaps()
            r, a = sm.get("Rss", 0), sm.get("Anonymous", 0)
            if r > self.peak_kb:
                self.peak_kb, self.peak_smaps = r, sm
            if a > self.peak_anon_kb:
                self.peak_anon_kb = a
            self.stop.wait(self.interval)


MARKS = []
T0 = time.time()


def mark(name):
    """Timestamped progress. If the cap kills the run, the last mark reached is
    the evidence for WHERE it died -- loading, quantizing, or training."""
    m = dict(mark=name, t=round(time.time() - T0, 1),
             rss_gb=round(_kb("VmRSS") / 1e6, 3), hwm_gb=round(_kb("VmHWM") / 1e6, 3))
    MARKS.append(m)
    print("MARK " + json.dumps(m), flush=True)


def versions():
    import importlib.metadata as md
    out = {"python": sys.version.split()[0], "torch": torch.__version__}
    for p in ("transformers", "experts4bit-qlora", "grouped-nf4-gemm", "accelerate",
              "bitsandbytes", "peft", "datasets", "safetensors", "huggingface-hub", "numpy"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


def build(hot):
    from experts4bit_qlora import enable_fast_train, load_moe_4bit_streaming
    torch.manual_seed(1234)
    if hot is None:
        m, _ = load_moe_4bit_streaming(MODEL, "cuda", torch.bfloat16, r=8, alpha=16,
                                       quant_type="nf4", offload=True)
    else:
        from experts4bit_qlora import enable_nvme_train_residency
        m, _ = load_moe_4bit_streaming(MODEL, "cuda", torch.bfloat16, r=8, alpha=16,
                                       quant_type="nf4", offload=False, arena=ARENA,
                                       arena_train=True)
        assert enable_nvme_train_residency(m, ARENA, hot_rows=hot, device="cuda") > 0
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    for n, p in m.named_parameters():
        p.requires_grad_("lora" in n and "experts" in n)
    assert enable_fast_train(m) > 0
    return m


def run(hot, steps):
    mark("imports")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    bs = []
    for r in ds.select(range(steps * 3)):
        ins = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
        t = f"### Instruction:\n{ins}\n\n### Response:\n{r['output']}"
        ids = tok(t, truncation=True, max_length=384, return_tensors="pt")["input_ids"]
        if ids.shape[1] >= 16:
            bs.append(ids.to("cuda"))
        if len(bs) == steps:
            break
    mark("data_ready")
    m = build(hot)
    mark("model_built")
    losses = []
    for ids in bs:
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()
        m.zero_grad(set_to_none=True)
        losses.append(float(out.loss.detach()))
    torch.cuda.synchronize()
    mark("trained")
    return dict(losses=losses, arena_gb=(os.path.getsize(ARENA) / 1e9) if hot else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("host", "arena"))
    ap.add_argument("--hot", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    hot = None if a.arm == "host" else a.hot

    samp = Sampler()
    samp.start()
    try:
        r = run(hot, a.steps)
        r.update(ok=True)
    except BaseException as exc:
        # A caught error means the process chose to stop -- e.g. a pinned
        # allocation was refused. That is a DIFFERENT outcome from SIGKILL by
        # the cgroup, and the two are worth telling apart in the write-up.
        r = dict(ok=False, error=f"{type(exc).__name__}: {exc}"[:400])
    samp.stop.set()
    samp.join(timeout=2)

    sm = samp.peak_smaps
    rss_kb, anon_kb = sm.get("Rss", 0), sm.get("Anonymous", 0)
    r.update(arm=a.arm, tag=a.tag, hot=hot, steps=a.steps)
    r["instruments"] = dict(
        ru_maxrss_gb=round(peak_rss_gb(), 3),
        vmhwm_gb=round(_kb("VmHWM") / 1e6, 3),
        peak_rss_gb=round(rss_kb / 1e6, 3),
        peak_anonymous_at_rss_peak_gb=round(anon_kb / 1e6, 3),
        peak_file_backed_gb=round((rss_kb - anon_kb) / 1e6, 3),
        peak_anon_gb=round(samp.peak_anon_kb / 1e6, 3),
    )
    r["marks"] = MARKS
    r["cgroup"] = cgroup_info()
    r["versions"] = versions()
    r["nproc"] = os.cpu_count()
    print("RESULT " + json.dumps(r), flush=True)
