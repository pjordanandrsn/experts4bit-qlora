"""Observation: what a run actually used, kept distinct from what the plan estimated.

* device, allocator view: ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` (measured);
* device, driver view: this process's ``used_memory`` from ``nvidia-smi --query-compute-apps``, sampled on a thread
  (measured; the difference to the allocator's reserved peak is the CUDA context and library workspaces);
* host: ``VmHWM`` (peak resident set) and ``VmRSS`` from ``/proc/self/status`` (measured);
* provenance: git commits of the source trees actually imported, distribution versions, the command line.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time


class DriverMemorySampler:
    """Peak driver-reported device memory of THIS process, and its peak anonymous / file-backed host RSS, sampled
    every ``interval`` seconds. VmHWM alone overstates what a run NEEDS from the host: memory-mapped checkpoint shards
    count in it as file-backed pages the kernel can reclaim. What cannot be reclaimed is ``RssAnon`` plus ``RssShmem``,
    which is where pinned (cudaHostAlloc) memory is accounted -- measured: a 512 MiB pinned tensor adds 511 MiB of
    RssShmem and nothing to RssAnon."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.peak, self.samples = interval, 0, 0
        self.anon_peak, self.file_peak, self.shmem_peak, self.required_peak = 0, 0, 0, 0
        #: the anonymous host peak since the last :meth:`mark` (a phase's own peak, e.g. serving after a load)
        self.phase_anon_peak = 0
        self._stop = threading.Event()
        self._smi = shutil.which("nvidia-smi")
        self._t = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        out = subprocess.run([self._smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        pid = str(os.getpid())
        for line in out.splitlines():
            p, _, mib = line.partition(",")
            if p.strip() == pid and mib.strip().isdigit():
                return int(mib.strip()) << 20
        return None

    def _run(self):
        while not self._stop.is_set():
            try:
                v = self._read() if self._smi else None
            except (subprocess.SubprocessError, OSError):
                v = None
            if v is not None:
                self.peak, self.samples = max(self.peak, v), self.samples + 1
            st = proc_status()
            self.anon_peak = max(self.anon_peak, st.get("RssAnon", 0))
            self.phase_anon_peak = max(self.phase_anon_peak, st.get("RssAnon", 0))
            self.file_peak = max(self.file_peak, st.get("RssFile", 0))
            self.shmem_peak = max(self.shmem_peak, st.get("RssShmem", 0))
            # pinned host memory (cudaHostAlloc) is accounted as RssShmem; mapped checkpoint shards as RssFile
            self.required_peak = max(self.required_peak, st.get("RssAnon", 0) + st.get("RssShmem", 0))
            self._stop.wait(self.interval)

    def mark(self, anon_now: int = 0):
        """Start a new phase: :attr:`phase_anon_peak` restarts from ``anon_now``. A load can peak higher than the
        serving that follows it (a repack's host buffers, handed back before serving), so serving's own growth needs
        its own peak."""
        self.phase_anon_peak = anon_now

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=15)


def proc_status() -> dict:
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"):
                    out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


def git_commit(path) -> dict:
    """``{"commit", "branch", "dirty"}`` for the git checkout that TRACKS ``path``, or ``{}`` (an installed wheel, or a
    file that merely sits inside some enclosing repository -- reporting that repository's commit would be false)."""
    d = os.path.dirname(os.path.abspath(path))
    try:
        tracked = subprocess.run(["git", "-C", d, "ls-files", "--error-unmatch", os.path.abspath(path)],
                                 capture_output=True, text=True, timeout=10)
        if tracked.returncode:
            return {}
        sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if sha.returncode:
            return {}
        dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        branch = subprocess.run(["git", "-C", d, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
        return {"commit": sha.stdout.strip(), "branch": branch, "dirty": bool(dirty)}
    except (subprocess.SubprocessError, OSError):
        return {}


def _source_file(name):
    """Where ``name`` WOULD be imported from, without importing it (so provenance can be taken before any import)."""
    import importlib.util

    mod = sys.modules.get(name)
    if mod is not None:
        return getattr(mod, "__file__", None)
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return spec.origin if spec else None


def provenance(modules=("experts4bit_qlora", "nf4_grouped", __package__.split(".")[0])) -> dict:
    """Commits, versions and argv. Call it at PROCESS START, before the measured code is imported: a commit read later
    can name code that changed on disk after the process loaded it. :func:`changed_since` checks for that."""
    out = {"argv": list(sys.argv), "python": sys.version.split()[0], "sources": {}}
    for name in modules:
        f = _source_file(name)
        if f:
            out["sources"][name] = {"file": f, **git_commit(f)}
    from importlib.metadata import PackageNotFoundError, version

    for dist in ("torch", "triton", "transformers", "bitsandbytes", "accelerate", "experts4bit-qlora",
                 "grouped-nf4-gemm"):
        try:
            out.setdefault("versions", {})[dist] = version(dist)
        except PackageNotFoundError:
            pass
    out["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def changed_since(prov: dict) -> list:
    """Source trees whose commit or dirty state differs now from ``prov``: the run's code may not be what it names."""
    out = []
    for name, src in prov.get("sources", {}).items():
        now = git_commit(src["file"]) if src.get("file") else {}
        if (now.get("commit"), now.get("dirty")) != (src.get("commit"), src.get("dirty")):
            out.append(f"{name}: {src.get('commit', 'untracked')[:10]}{'+dirty' if src.get('dirty') else ''} at start, "
                       f"{now.get('commit', 'untracked')[:10]}{'+dirty' if now.get('dirty') else ''} at end")
    return out


def decode_step_b1(rows) -> dict | None:
    """The single-stream decode step from an experts4bit-qlora step trace (``E4B_PAGED_STEP_TRACE``, one row per
    engine step; ``experts4bit_qlora.engines.step_trace``): the steps that decoded exactly one row and ran no prefill.
    ``step_ms`` is the step's whole host time (inputs, replay or eager step, the token sync, bookkeeping), so a decode
    step and its tokens per second are what one sequence sees. ``device_ms`` is the decode's device time
    (``gpu.dec_issue - gpu.dec_prep``) where the trace resolved its events. None when the trace has no such step."""
    steps = [r for r in rows if r.get("decode_rows") == 1 and not r.get("prefill_tokens")
             and isinstance(r.get("step_ms"), (int, float))]
    if not steps:
        return None

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    dev = [r["gpu"]["dec_issue"] - r["gpu"]["dec_prep"] for r in steps
           if isinstance(r.get("gpu"), dict) and {"dec_issue", "dec_prep"} <= set(r["gpu"])]
    return {"decode_step_ms_b1": round(median([r["step_ms"] for r in steps]), 4),
            "decode_device_ms_b1": round(median(dev), 4) if dev else None,
            "decode_steps_b1": len(steps),
            "decode_buckets_b1": sorted({r.get("bucket") for r in steps}, key=lambda b: (b is None, b))}
