# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""HO3's driver-peak probe (bench/ho3/PREREG-ho3.md, Amendment 2): can this host give HO2's rule its input?

HO2's rule reads each row's sampled driver peak, which Loggetta 0.6.2 takes from ``DriverMemorySampler`` (its
``measure.py``): this process's ``used_memory`` in ``nvidia-smi --query-compute-apps``. A host that lists no per-process
memory there, or lists the process under another pid (a container's pid namespace), gives every row zero samples, and the
box reads VOID after its whole run. The probe runs THAT sampler, not a re-implementation: ``loggetta_measure.py`` is
Loggetta 0.6.2's ``loggetta/measure.py`` byte for byte (staged-ho3.sha256 pins it, and the runner's tripwire checks the
installed module against it), on a process holding 256 MiB on the GPU. It refuses (rc 19, before the install) when the
sampler takes no sample or reads a peak below the allocation. So a probe pass and a row's samples cannot disagree.

    python ho3_drvprobe.py              # the probe (rc 0 pass, rc 19 refuse)
    python ho3_drvprobe.py --self-test  # the decision, against fixture nvidia-smi outputs, through the same sampler
"""
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import loggetta_measure  # noqa: E402  -- Loggetta 0.6.2's own measure.py, staged

ALLOC_BYTES = 256 << 20
SAMPLE_S = 5.0


def decide(samples: int, peak: int) -> tuple[bool, str]:
    """Pass only when the sampler took a sample AND its peak covers what this process holds."""
    if samples == 0:
        return False, "the sampler took no sample: nvidia-smi does not list this process's memory"
    if peak < ALLOC_BYTES:
        return False, f"the sampled peak {peak >> 20} MiB is below the {ALLOC_BYTES >> 20} MiB this process holds"
    return True, f"samples={samples} peak_mib={peak >> 20}"


def sample(seconds: float = SAMPLE_S, interval: float = 0.25) -> tuple[int, int]:
    with loggetta_measure.DriverMemorySampler(interval) as s:
        time.sleep(seconds)
    return s.samples, s.peak


def probe() -> int:
    try:
        import torch
        x = torch.empty(ALLOC_BYTES, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        samples, peak = sample()
        del x
        ok, why = decide(samples, peak)
    except Exception as e:  # a host that cannot hold the allocation cannot give a row its peak either
        ok, why = False, f"probe failed: {type(e).__name__}: {e}"
    print(f"DRIVER_PEAK_PROBE {'ok' if ok else 'none'} {why}")
    return 0 if ok else 19


def _with_smi(stdout: str):
    """Run the staged sampler against a fixture ``nvidia-smi`` output: its own ``_read`` parses it."""
    real_run, real_which = subprocess.run, loggetta_measure.shutil.which
    subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, stdout=stdout, stderr="")
    loggetta_measure.shutil.which = lambda name: "/usr/bin/nvidia-smi"
    try:
        return sample(seconds=0.6, interval=0.05)
    finally:
        subprocess.run, loggetta_measure.shutil.which = real_run, real_which


#: (name, the fixture nvidia-smi output, whether the probe must pass). ``{pid}`` is this process's pid.
FIXTURES = (
    ("no per-process rows (ho3-4090-r1's host)", "", False),
    ("another process only", "4242, 900\n", False),
    ("this process under the HOST pid (a pid namespace)", "{other}, 300\n", False),
    ("memory not reported", "{pid}, [N/A]\n", False),
    ("listed, below the allocation", "{pid}, 200\n", False),
    ("listed with the allocation and its context", "{pid}, 520\n", True),
    ("listed among others", "4242, 9000\n{pid}, 400\n", True),
)


def self_test() -> int:
    pid, other = os.getpid(), os.getpid() + 7919
    for name, out, want in FIXTURES:
        samples, peak = _with_smi(out.format(pid=pid, other=other))
        got, why = decide(samples, peak)
        assert got is want, f"{name}: probe {'passed' if got else 'refused'} ({why}), want {'pass' if want else 'refuse'}"
    print(f"drvprobe self-test OK ({len(FIXTURES)} fixtures)")
    return 0


if __name__ == "__main__":
    sys.exit(self_test() if sys.argv[1:] == ["--self-test"] else probe())
