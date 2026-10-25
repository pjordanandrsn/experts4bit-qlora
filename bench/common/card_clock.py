#!/usr/bin/env python3
"""card_clock.py -- the card's loaded SM clock and its clock-event reasons: a host floor a rented box must clear before install, and the
same check on a run's per-second clock log.

TC1 amendment 79's box (e4b#2129) held its RTX 5090 at a flat 2400 MHz under load, with NVML's *Reliability* reason set on every sample,
the 1st percentile of TC1's 76 RTX 5090 boxes. Device time doesn't travel between such a card and a normal one, so a lane that compares
device time across boxes, or reads a ratio on one box and quotes it as the release's, screens the card first. ONE implementation:
TC1 amendment 80 calls it with the RTX 5090's floor, and other lanes call it with their own registered floor and calibration.

Two uses:
    python3 card_clock.py burn --floor MHZ [--seconds 20] [--min-clean 0.95] [--out card_clock.json]
        A bf16 matmul burn on the current CUDA device, sampled by nvidia-smi (clocks.sm, clocks.mem, temperature.gpu,
        clocks_event_reasons.active: the fields TC1's per-second sampler logs, tc1_run.sh:725). It passes when all three hold:
        - the busy SM median is >= MHZ;
        - no HARD reason (every refusing bit but SwPowerCap) is set on ANY sample;
        - SwPowerCap is clear on at least --min-clean of the busy samples. A healthy 575 W card may touch it briefly under a
          full-power burn, so it is held to a fraction, not to every sample.
        It exits 0 on a pass and 18 otherwise (TC1's host-floor code: the launcher excludes the machine on its next draw). A probe
        that can't read the card refuses too: an unscreened card is never passed. The JSON's ``refusal`` says which it was:
        ``below_floor`` (read, and failed) or ``probe_error`` (not read: no torch, no CUDA, no nvidia-smi).
        One GPU: the burn runs on cuda:0 and samples ``nvidia-smi -i 0``. A multi-GPU lane screens each card itself.
    python3 card_clock.py check-log GPUCLK_TXT --floor MHZ [--min-clean 0.95]
        The same test on a run's ``gpuclk_<arm>.txt``. It prints the summary as JSON and exits 0 on a pass, 1 otherwise. A reducer
        imports ``summarize_log`` and ``passes`` instead.

The reason bits are NVML's. The table below is copied from NVIDIA's binding, ``nvidia_ml_py-13.615.71`` (PyPI; wheel sha256
959bf4adf6fe1308e4bd739e722236b0d1ec8392e2cefad33ff70c311380b9b6), ``pynvml.py`` lines 2190-2199 and 2215-2226. That version
matches the driver branch (615.71) of the box that showed 0x400. Only the names are claimed; NVML's binding describes them no further.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

REASON_BITS = {
    0x0000000000000001: "GpuIdle",
    0x0000000000000002: "ApplicationsClocksSetting",
    0x0000000000000004: "SwPowerCap",
    0x0000000000000008: "HwSlowdown",
    0x0000000000000010: "SyncBoost",
    0x0000000000000020: "SwThermalSlowdown",
    0x0000000000000040: "HwThermalSlowdown",
    0x0000000000000080: "HwPowerBrakeSlowdown",
    0x0000000000000100: "DisplayClockSetting",
    0x0000000000000200: "BoardLimit",
    0x0000000000000400: "Reliability",
}
#: The bits that refuse a card (TC1 amendment 80): every slowdown and cap. GpuIdle, ApplicationsClocksSetting, SyncBoost and
#: DisplayClockSetting do not. At the burn, SwPowerCap (SOFT_BITS) is held to a fraction of busy samples and the rest (HARD_BITS) to
#: every sample. On a run's log, all of them are held to a fraction of busy samples.
REFUSE_BITS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80 | 0x200 | 0x400
SOFT_BITS = 0x4
HARD_BITS = REFUSE_BITS & ~SOFT_BITS
#: A sample counts as busy above this SM clock: the idle clock on these cards reads 180-405 MHz.
BUSY_MHZ = 1000
HOST_FLOOR_RC = 18
SMI_QUERY = "clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active"


def decode(mask: int) -> list:
    """NVML's names for the bits set in ``mask``; an unknown bit is named by its value."""
    out = [name for bit, name in REASON_BITS.items() if mask & bit]
    rest = mask & ~sum(REASON_BITS)
    if rest:
        out.append(hex(rest))
    return out


def parse_sample(fields: str):
    """``sm, mem, temp, reasons`` (nvidia-smi csv, noheader, nounits) -> (sm, mem, temp, mask), or None if unreadable."""
    try:
        a = [x.strip() for x in fields.split(",")]
        return int(float(a[0])), int(float(a[1])), int(float(a[2])), int(a[3], 16)
    except (ValueError, IndexError):
        return None


def parse_gpuclk_line(line: str):
    """One line of TC1's ``gpuclk_<arm>.txt``: ``EPOCH sm, mem, temp, reasons | loadavg | /proc/stat``. Returns (epoch, sm, mem, temp,
    mask), or None for a line the driver could not fill."""
    head = line.split("|", 1)[0].strip()
    if " " not in head:
        return None
    epoch, rest = head.split(None, 1)
    s = parse_sample(rest)
    if s is None:
        return None
    try:
        return (int(epoch),) + s
    except ValueError:
        return None


def summarize(samples, busy_mhz: int = BUSY_MHZ) -> dict:
    """The busy samples' SM median, p10 and max, the reason bits seen, and the share of busy samples with no refusing bit."""
    busy = [s for s in samples if s[0] > busy_mhz]
    sm = sorted(s[0] for s in busy)
    seen = 0
    for s in samples:
        seen |= s[3]
    clean = sum(1 for s in busy if not s[3] & REFUSE_BITS)
    soft_clean = sum(1 for s in busy if not s[3] & SOFT_BITS)
    hard_seen = 0
    for s in samples:
        hard_seen |= s[3] & HARD_BITS
    return {
        "samples": len(samples), "busy_samples": len(busy),
        "sm_median": statistics.median(sm) if sm else None, "sm_p10": sm[len(sm) // 10] if sm else None, "sm_max": sm[-1] if sm else None,
        "mem_median": statistics.median(s[1] for s in busy) if busy else None,
        "reasons_seen": decode(seen), "reasons_seen_mask": hex(seen),
        "busy_clean_fraction": round(clean / len(busy), 4) if busy else None,
        "busy_soft_clean_fraction": round(soft_clean / len(busy), 4) if busy else None,
        "hard_reasons_any_sample": decode(hard_seen),
    }


def summarize_log(path: str, busy_mhz: int = BUSY_MHZ) -> dict:
    rows = [r for r in (parse_gpuclk_line(ln) for ln in open(path, encoding="utf-8", errors="replace")) if r is not None]
    return summarize([r[1:] for r in rows], busy_mhz)


def passes(summary: dict, floor_mhz: float, min_clean: float, min_busy: int = 10, burn: bool = False):
    """(ok, why): the busy SM median is >= ``floor_mhz``, and then, depending on the mode:
    - a run's log (``burn`` False): every refusing bit is clear on >= ``min_clean`` of the busy samples;
    - the burn (``burn`` True): no HARD bit is set on any sample, and SwPowerCap is clear on >= ``min_clean`` of the busy samples."""
    why = []
    if (summary.get("busy_samples") or 0) < min_busy:
        why.append(f"{summary.get('busy_samples') or 0} busy samples < {min_busy} (the card was not read under load)")
    else:
        if summary["sm_median"] < floor_mhz:
            why.append(f"busy SM median {summary['sm_median']} MHz < floor {floor_mhz}")
        if burn:
            if summary["hard_reasons_any_sample"]:
                why.append(f"a hard slowdown or cap reason was set during the burn: {summary['hard_reasons_any_sample']}")
            if summary["busy_soft_clean_fraction"] < min_clean:
                why.append(f"SwPowerCap on {1 - summary['busy_soft_clean_fraction']:.1%} of busy samples "
                           f"(clean {summary['busy_soft_clean_fraction']:.1%} < {min_clean:.0%})")
        elif summary["busy_clean_fraction"] < min_clean:
            why.append(f"refusing reason bits on {1 - summary['busy_clean_fraction']:.1%} of busy samples "
                       f"(clean {summary['busy_clean_fraction']:.1%} < {min_clean:.0%}; seen {summary['reasons_seen']})")
    return (not why), "; ".join(why)


def _sample_loop(out: list, stop: threading.Event, period: float):
    while not stop.is_set():
        try:
            line = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv,noheader,nounits", "-i", "0"],
                                  capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            s = parse_sample(line)
            if s is not None:
                out.append(s)
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
        stop.wait(period)


def burn(seconds: float = 20.0, n: int = 8192, period: float = 0.25) -> dict:
    """A bf16 ``n x n`` matmul loop on cuda:0 for ``seconds``, sampled by nvidia-smi every ``period`` s. Returns ``summarize``'s dict."""
    import torch
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):                                   # warm: the first launches carry cuBLAS setup
        a @ b
    torch.cuda.synchronize()
    samples, stop = [], threading.Event()
    t = threading.Thread(target=_sample_loop, args=(samples, stop, period), daemon=True)
    t.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        for _ in range(8):
            a @ b
        torch.cuda.synchronize()
    stop.set()
    t.join(timeout=15)
    out = summarize(samples)
    out.update({"burn_seconds": seconds, "matmul_n": n, "dtype": "bfloat16", "sample_period_s": period})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("burn")
    b.add_argument("--floor", type=float, required=True)
    b.add_argument("--seconds", type=float, default=20.0)
    b.add_argument("--min-clean", type=float, default=0.95)
    b.add_argument("--out", default=None)
    c = sub.add_parser("check-log")
    c.add_argument("path")
    c.add_argument("--floor", type=float, required=True)
    c.add_argument("--min-clean", type=float, default=0.95)
    a = ap.parse_args(argv)
    if a.cmd == "burn":
        try:
            summ = burn(a.seconds)
        except Exception as e:                           # no torch, no CUDA, no nvidia-smi: the card is unscreened, so it refuses
            summ = {"error": f"{type(e).__name__}: {e}"}
        ok, why = passes(summ, a.floor, a.min_clean, burn=True) if "error" not in summ else (False, summ["error"])
        summ.update({"floor_mhz": a.floor, "min_clean": a.min_clean, "hard_bits_every_sample": True, "pass": ok, "why": why,
                     "refusal": None if ok else ("probe_error" if "error" in summ else "below_floor")})
        if a.out:
            with open(a.out, "w") as f:
                json.dump(summ, f, indent=1)
        print("CARD_CLOCK " + json.dumps(summ, sort_keys=True))
        return 0 if ok else HOST_FLOOR_RC
    summ = summarize_log(a.path)
    ok, why = passes(summ, a.floor, a.min_clean)
    summ.update({"floor_mhz": a.floor, "min_clean": a.min_clean, "pass": ok, "why": why})
    print(json.dumps(summ, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
