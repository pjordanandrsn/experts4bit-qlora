"""bench/dq5/dq5_link_gate.py -- lane DQ5, BOX side: is the GPU's link actually running x16? (bench/dq5/DQ5-PREREG.md)

``nvidia-smi``'s ``pcie.link.width.max`` is the width the GPU and slot CAN negotiate, not the width they did: on the RTX
A2000 rehearsal it read 16 while the link ran x8 under load. So the generation is gated on ``pcie.link.gen.max`` (the
current generation idles at 1, power management) and the WIDTH on ``pcie.link.width.current`` read while a pinned copy is in
flight. Exit 0 = width 16 under load; 3 = narrower (the runner refuses at rc 19, out of band -- not machine evidence);
2 = could not tell (no CUDA, unreadable nvidia-smi: the runner treats it as a harness error, rc 9). This is a LINK gate; the
H2D bandwidth stays descriptive (dq5_h2d_probe.py).
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

WANT_WIDTH = 16


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("LINK_GATE_ERROR: no CUDA device")
        return 2
    src = torch.empty(256 * 2**20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    stop = threading.Event()

    def load():
        while not stop.is_set():
            dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()

    t = threading.Thread(target=load, daemon=True)
    t.start()
    time.sleep(0.5)
    widths = []
    try:
        for _ in range(3):
            out = subprocess.run(["nvidia-smi", "--query-gpu=pcie.link.width.current,pcie.link.gen.current",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            widths.append(out.strip())
            time.sleep(0.2)
    except (OSError, subprocess.SubprocessError) as exc:
        stop.set()
        print(f"LINK_GATE_ERROR: nvidia-smi unreadable: {exc!r}")
        return 2
    stop.set()
    t.join(timeout=5)
    try:
        w = max(int(s.split(",")[0]) for s in widths)
    except (ValueError, IndexError):
        print(f"LINK_GATE_ERROR: unparseable width/gen under load {widths}")
        return 2
    line = f"width/gen under load: {widths}"
    if w != WANT_WIDTH:
        print(f"LINK_GATE_OUT_OF_BAND {line} (want x{WANT_WIDTH})")
        return 3
    print(f"LINK_GATE_OK {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
