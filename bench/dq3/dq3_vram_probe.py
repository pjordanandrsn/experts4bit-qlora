"""bench/dq3/dq3_vram_probe.py -- lane DQ3, BOX side: refuse a host whose GPU will not hand out the memory the subject
needs, before anything is installed. Exit 0 = fits; exit 1 = refuse (dq3_run.sh turns it into the registered host-floor
refusal, rc 18, which rent.py accepts as machine evidence for --exclude-vast-lane-receipt).

Why it exists: dq3-5090-1 (host 564677, Ryzen 9 9950X, driver 595.84) died in arm 1 on the subject's FIRST allocation, a
2.90 GiB embedding, raising CUDA OOM with 30.85 GiB free and 0 bytes allocated by PyTorch. The same allocation succeeded on
DQ2's 5090 and on the A2000 rehearsal, so the host refused it. An arm failure (rc 11) cannot name the machine, so a
relaunch could buy it again; this probe can.

One block larger than the subject's largest tensor (3.5 GiB > the 2.90 GiB fp32 embedding), then 2 GiB blocks to 28 GiB,
above the resident arm's expected peak. Each block is written so the allocation is real.
"""
from __future__ import annotations

import sys

FIRST_GIB = 3.5
TOTAL_GIB = 28.0
STEP_GIB = 2.0


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("VRAM_PROBE_FAIL: no CUDA device")
        return 1
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM_PROBE device={torch.cuda.get_device_name()} free={free / 2**30:.2f} GiB total={total / 2**30:.2f} GiB")
    blocks, got = [], 0.0
    try:
        size = FIRST_GIB
        while got < TOTAL_GIB:
            b = torch.empty(int(size * 2**30), dtype=torch.uint8, device="cuda")
            b.fill_(1)
            blocks.append(b)
            got += size
            size = STEP_GIB
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as exc:
        print(f"VRAM_PROBE_FAIL after {got:.1f} GiB, asking {size:.1f} GiB: {str(exc)[:400]}")
        return 1
    print(f"VRAM_PROBE_OK {got:.1f} GiB in {len(blocks)} blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
