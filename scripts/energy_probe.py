#!/usr/bin/env python3
"""CPU (RAPL) + GPU energy around a workload, for honest J/token numbers.

Why this exists: a GPU-only energy measurement silently flatters whichever engine does
its work on the CPU. Comparing experts4bit (GPU-heavy, experts streamed from NVMe) against
llama.cpp (`-ot exps=CPU`, tens of cores busy) on GPU watts alone is not a comparison, it
is a category error.

**THIS NEEDS BARE METAL.** Measured 2026-08-01 on both RunPod and a Vast *datacenter*
host: containers block BOTH RAPL interfaces. `/dev/cpu/*/msr` is absent, and
`/sys/class/powercap/` is the nastier one -- it LISTS `intel-rapl`, `intel-rapl:0`,
`intel-rapl:0:0`, so a glance says RAPL is available, but they are dangling symlinks and
`intel-rapl:0/energy_uj` does not exist. Resolve the leaf file, never trust the listing.
`capsh --print` explains it: root in-container but `!cap_sys_rawio`, `!cap_sys_admin`,
`!cap_perfmon`. There is no container workaround; rent a bare-metal host.

Both interfaces are supported because which one exists is not predictable: a QNAP
Xeon W-1250 has `/dev/cpu/0/msr` and NO powercap, while a typical distro server has
powercap and often no `msr` module loaded. Trying only one is how a paid hour gets wasted.
On AMD the powercap driver still registers under the name `intel-rapl` -- that is kernel
naming, not a bug.

Two things that make naive RAPL readings wrong, both handled here:

* **The counter is 32-bit and wraps.** At Intel's usual 15.3 uJ unit that is ~65.7 kJ, so
  a 100 W package wraps in ~11 minutes; AMD's finer unit wraps in under 3 minutes at
  200 W. A single before/after read is therefore not merely imprecise, it can be
  NEGATIVE or silently short by a full period. This samples on an interval and accumulates
  deltas, treating any decrease as one wrap.
* **Energy is per PACKAGE, not per core.** Reading cpu0 on a two-socket box measures half
  the machine. Cores are grouped by `physical_package_id` and one MSR is read per package.

Usage::

    sudo ./energy_probe.py --label ours -- python3 bench.py       # wrap a command
    sudo ./energy_probe.py --idle 10                              # idle baseline

Reports CPU package joules, GPU joules, wall time, and mean watts for each. Divide by the
token count your workload printed to get J/token; subtract an idle run to get the
above-idle figure, which is the fairer one when comparing engines on a shared host.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time

# --- Intel ---
INTEL_UNIT, INTEL_PKG = 0x606, 0x611
# --- AMD (Zen and later); note these are NOT the Intel numbers ---
AMD_UNIT, AMD_PKG = 0xC0010299, 0xC001029B

WRAP = 1 << 32


def wrap_delta(prev: int, cur: int) -> int:
    """Ticks elapsed between two 32-bit RAPL reads, treating a decrease as one wrap.

    Extracted so it is testable: the wrap window is ~73 min at this box's 61 uJ tick and
    60 W, so a bug here cannot be caught by running the probe for a plausible time -- it
    just silently loses a full 262 kJ period, or reports a negative delta.

    Single-wrap only, which is the honest limit: two wraps between samples is
    indistinguishable from none, so sample far faster than the wrap period (the default
    200 ms interval is ~20000x margin).
    """
    return (cur - prev) if cur >= prev else (cur + WRAP - prev)


def _vendor() -> str:
    try:
        txt = open("/proc/cpuinfo").read()
    except OSError:
        return "unknown"
    if "AuthenticAMD" in txt:
        return "amd"
    if "GenuineIntel" in txt:
        return "intel"
    return "unknown"


def _packages():
    """{package_id: representative_cpu_index} — one MSR read per socket, not per core."""
    out = {}
    for entry in sorted(os.listdir("/sys/devices/system/cpu")):
        m = re.fullmatch(r"cpu(\d+)", entry)
        if not m:
            continue
        cpu = int(m.group(1))
        path = f"/sys/devices/system/cpu/{entry}/topology/physical_package_id"
        try:
            pkg = int(open(path).read().strip())
        except OSError:
            pkg = 0
        out.setdefault(pkg, cpu)
    return out or {0: 0}


def _rdmsr(cpu: int, reg: int) -> int:
    with open(f"/dev/cpu/{cpu}/msr", "rb") as f:
        f.seek(reg)
        return struct.unpack("<Q", f.read(8))[0]


class PowercapEnergy:
    """Wrap-safe package energy from `/sys/class/powercap` (microjoules).

    Preferred when present: it needs no `msr` module and no vendor-specific register
    numbers. Each domain wraps at its own `max_energy_range_uj`, which is read per domain
    rather than assumed -- the 32-bit MSR constant does NOT apply here.
    """

    backend = "powercap"

    def __init__(self):
        self.vendor = _vendor()
        self.domains = []
        base = "/sys/class/powercap"
        for entry in sorted(os.listdir(base)):
            # package domains only: "intel-rapl:0", not subzones "intel-rapl:0:0"
            if not re.fullmatch(r"intel-rapl:\d+", entry):
                continue
            path = os.path.join(base, entry, "energy_uj")
            rng = os.path.join(base, entry, "max_energy_range_uj")
            try:
                cur = int(open(path).read().strip())
                mx = int(open(rng).read().strip())
            except OSError:
                continue          # dangling symlink (containers) -- not a usable domain
            self.domains.append({"path": path, "wrap": mx + 1, "last": cur})
        if not self.domains:
            raise OSError(f"{base}: no readable package domains")
        self.pkgs = {i: i for i in range(len(self.domains))}
        self.joules_per_tick = 1e-6
        self.ticks = 0

    def poll(self):
        for d in self.domains:
            try:
                cur = int(open(d["path"]).read().strip())
            except OSError:
                continue
            prev = d["last"]
            self.ticks += (cur - prev) if cur >= prev else (cur + d["wrap"] - prev)
            d["last"] = cur

    @property
    def joules(self) -> float:
        return self.ticks * self.joules_per_tick


class MsrEnergy:
    """Wrap-safe RAPL package-energy accumulator, read straight from the MSRs."""

    backend = "msr"

    def __init__(self):
        self.vendor = _vendor()
        self.pkgs = _packages()
        if self.vendor == "amd":
            self.unit_reg, self.pkg_reg = AMD_UNIT, AMD_PKG
        else:
            self.unit_reg, self.pkg_reg = INTEL_UNIT, INTEL_PKG
        raw = _rdmsr(next(iter(self.pkgs.values())), self.unit_reg)
        # energy unit exponent lives in bits 12:8 on both vendors; joules = 1 / 2**exp
        self.joules_per_tick = 1.0 / (1 << ((raw >> 8) & 0x1F))
        self._last = {p: _rdmsr(c, self.pkg_reg) & 0xFFFFFFFF for p, c in self.pkgs.items()}
        self.ticks = 0

    def poll(self):
        for pkg, cpu in self.pkgs.items():
            cur = _rdmsr(cpu, self.pkg_reg) & 0xFFFFFFFF
            prev = self._last[pkg]
            self.ticks += wrap_delta(prev, cur)
            self._last[pkg] = cur

    @property
    def joules(self) -> float:
        return self.ticks * self.joules_per_tick


def open_cpu_energy():
    """Powercap first, MSR second; raise with BOTH reasons if neither works.

    Reporting only the second failure would hide the usual container symptom (powercap
    present-but-dangling) behind a confusing "no such file: /dev/cpu/0/msr".
    """
    errs = []
    for cls in (PowercapEnergy, MsrEnergy):
        try:
            return cls()
        except (OSError, PermissionError) as e:
            errs.append(f"{cls.backend}: {e}")
    raise OSError("no readable CPU energy interface -- " + " | ".join(errs)
                  + ". Containers block both (see module docstring); this needs bare metal.")


class _GpuReader:
    """GPU power, read as cheaply as possible.

    This matters more than it looks. Shelling out to `nvidia-smi` costs tens of
    milliseconds of CPU per call, so polling it at the CPU sampler's 200 ms cadence
    would burn a noticeable fraction of a core -- and that cost lands in the very RAPL
    counter this tool attributes to the workload. The probe would be measuring itself,
    and worse, it would inflate the CPU-heavy engine's number by more than the GPU-heavy
    one's, which is exactly the comparison at stake.

    So: NVML in-process when available (a library call, microseconds), and when it is
    not, `nvidia-smi` at a deliberately slower cadence with the sampling error taken
    over the self-measurement error.
    """

    def __init__(self):
        self.handles, self.nvml = [], None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                            for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            self.nvml = None

    @property
    def cheap(self) -> bool:
        return self.nvml is not None

    def watts(self):
        if self.nvml is not None:
            try:
                return sum(self.nvml.nvmlDeviceGetPowerUsage(h) / 1000.0 for h in self.handles)
            except Exception:
                return None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout
            return sum(float(x) for x in out.split() if x.replace(".", "", 1).isdigit())
        except Exception:
            return None


class Sampler(threading.Thread):
    """CPU polled fast (a file read); GPU polled at whatever its reader can afford."""

    def __init__(self, cpu, interval=0.2, gpu_interval=None):
        super().__init__(daemon=True)
        self.cpu, self.interval = cpu, interval
        self.gpu = _GpuReader()
        # a subprocess-based reader gets a slow cadence so it does not pollute the CPU
        # counter; NVML is cheap enough to ride the CPU interval
        self.gpu_interval = gpu_interval or (interval if self.gpu.cheap else 1.0)
        self.gpu_j, self.gpu_samples, self.stop_flag = 0.0, 0, threading.Event()

    def run(self):
        last = time.monotonic()
        gpu_last = last
        while not self.stop_flag.is_set():
            time.sleep(self.interval)
            now = time.monotonic()
            last = now
            if self.cpu:
                self.cpu.poll()
            if now - gpu_last >= self.gpu_interval:
                w = self.gpu.watts()
                if w is not None:
                    # integrate over the GPU interval actually elapsed, not the CPU one
                    self.gpu_j += w * (now - gpu_last)
                    self.gpu_samples += 1
                gpu_last = now


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="run")
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--idle", type=float, default=0.0, help="measure N seconds of idle instead")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    try:
        cpu = open_cpu_energy()
    except (OSError, PermissionError) as e:
        print(f"CPU energy unavailable -- {e}\nReporting GPU only, which is NOT a fair "
              f"comparison for any CPU-heavy engine.", file=sys.stderr)
        cpu = None

    s = Sampler(cpu, args.interval)
    t0 = time.monotonic()
    s.start()
    rc = 0
    if args.idle:
        time.sleep(args.idle)
    else:
        cmd = [c for c in args.cmd if c != "--"]
        if not cmd:
            ap.error("give a command after -- , or use --idle N")
        rc = subprocess.run(cmd).returncode
    s.stop_flag.set()
    s.join(timeout=5)
    if cpu:
        cpu.poll()
    wall = time.monotonic() - t0

    res = {
        "label": args.label, "rc": rc, "wall_s": round(wall, 2),
        "cpu_backend": cpu.backend if cpu else None,
        "cpu_vendor": cpu.vendor if cpu else None,
        "cpu_packages": len(cpu.pkgs) if cpu else 0,
        "cpu_joules": round(cpu.joules, 1) if cpu else None,
        "cpu_mean_W": round(cpu.joules / wall, 1) if cpu and wall else None,
        "gpu_joules": round(s.gpu_j, 1) if s.gpu_samples else None,
        "gpu_mean_W": round(s.gpu_j / wall, 1) if s.gpu_samples and wall else None,
        "gpu_samples": s.gpu_samples,
        "gpu_reader": "nvml" if s.gpu.cheap else "nvidia-smi",
        "gpu_interval_s": s.gpu_interval,
    }
    if res["cpu_joules"] is not None and res["gpu_joules"] is not None:
        res["total_joules"] = round(res["cpu_joules"] + res["gpu_joules"], 1)
    print(json.dumps(res) if args.json else
          "  ".join(f"{k}={v}" for k, v in res.items() if v is not None))
    return rc


if __name__ == "__main__":
    sys.exit(main())
