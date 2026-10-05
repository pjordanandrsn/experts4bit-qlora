"""bench/dq5/dq5_h2d_probe.py -- lane DQ5, BOX side: a DESCRIPTIVE record of this host's pinned host-to-device link.
Never a gate (bench/dq5/DQ5-PREREG.md): a slow-but-gen-4 host is in band, and two gen-4 5090 hosts have read 0.64 and
0.87-0.96 single-copy/back-to-back efficiency (memory: finding_two_link_probes_disagree_on_gen4), so which probe "is the
link" is itself a reading, not a definition.

Records, each a median over repeats, both the patterns that have disagreed:
  single_64mb      one 64 MiB pinned copy, synchronized before and after (the per-copy rate)
  b2b_64mb         40 back-to-back non-blocking 64 MiB copies, CUDA-event timed (the sustained DMA rate)
  single_layer     one copy of DQ3's per-layer streamed bytes (243,793,920 B), synchronized (what the prototype issues)
plus nvidia-smi's link max/current generation and width read DURING a copy (an idle link reads gen 1 current), and lspci's
LnkSta when lspci exists. Writes JSON to argv[1]; exit 0 unless CUDA is absent (2). Nothing here refuses a host.
"""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import threading
import time

LAYER_BYTES = 243_793_920
MB64 = 64 * 2**20


def _smi(fields: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unreadable: {exc!r}"


def main(out: str) -> int:
    import torch

    if not torch.cuda.is_available():
        print("H2D_PROBE_ERROR: no CUDA device")
        return 2
    dev = torch.device("cuda")
    rec = {"schema": "dq5-h2d/1", "device": torch.cuda.get_device_name(),
           "link_max": _smi("pcie.link.gen.max,pcie.link.width.max"), "link_idle_current": _smi("pcie.link.gen.current,pcie.link.width.current")}

    def single(nbytes: int, reps: int = 10) -> float:
        src = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
        dst = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        rates = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()
            rates.append(nbytes / (time.perf_counter() - t0) / 1e9)
        return statistics.median(rates)

    def b2b(nbytes: int, n: int = 40, reps: int = 5) -> float:
        src = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
        dst = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        rates = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            a.record()
            for _ in range(n):
                dst.copy_(src, non_blocking=True)
            b.record()
            torch.cuda.synchronize()
            rates.append(n * nbytes / (a.elapsed_time(b) / 1e3) / 1e9)
        return statistics.median(rates)

    loaded = {}

    def read_under_load():
        time.sleep(0.3)
        loaded["link_loaded_current"] = _smi("pcie.link.gen.current,pcie.link.width.current")

    t = threading.Thread(target=read_under_load)
    t.start()
    rec["b2b_64mb_gbs"] = b2b(MB64)
    t.join()
    rec.update(loaded)
    rec["single_64mb_gbs"] = single(MB64)
    rec["single_layer_gbs"] = single(LAYER_BYTES)
    rec["link_eff_single_over_b2b"] = rec["single_64mb_gbs"] / rec["b2b_64mb_gbs"]
    if shutil.which("lspci"):
        try:
            txt = subprocess.run(["lspci", "-vv", "-d", "10de:"], capture_output=True, text=True, timeout=20).stdout
            rec["lspci_lnk"] = [ln.strip() for ln in txt.splitlines() if "LnkSta:" in ln or "LnkCap:" in ln][:6]
        except (OSError, subprocess.SubprocessError) as exc:
            rec["lspci_lnk"] = f"unreadable: {exc!r}"
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"H2D_PROBE single_64mb {rec['single_64mb_gbs']:.2f} b2b_64mb {rec['b2b_64mb_gbs']:.2f} "
          f"single_layer {rec['single_layer_gbs']:.2f} GB/s; link max {rec['link_max']} loaded {rec.get('link_loaded_current')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
