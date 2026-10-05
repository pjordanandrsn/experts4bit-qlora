### DQ3 (#1083): the box refuses a host whose GitHub egress cannot carry the install, at rc 14 before any install (#1173)

- **What happened.** `dq3-5090-2` ($0.203) passed the link and VRAM checks, then cloned from GitHub at about 33 KB/s.
  The host was machine 147454 in Shanghai. The 115 MB experts4bit-qlora clone could never finish inside pip's
  30-minute alarm.
- **The probe.** `bench/dq3/dq3_egress_probe.py` reads the pinned grouped-nf4-gemm codeload tarball for at most 30 s.
  - Below 1 MB/s, it exits 4, which the runner turns into rc 14, the registered egress refusal and machine evidence for
    a relaunch's exclusion.
  - No transfer at all, or a probe crash, goes to rc 9 and excludes nothing.
- **Tests:** the runner refuses at 14 before any install, maps 1 and 2 to rc 9, and runs the VRAM probe first.
