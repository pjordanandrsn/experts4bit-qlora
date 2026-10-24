# `p236-5090-2`: grouped-nf4-gemm 0.51.0's decode GEMV vs bitsandbytes 0.50.2, op level (READ)

One RTX 5090 (cc 12.0, 170 SMs), run 2026-10-24 through the guarded launcher. Cost: $0.010.
- **Code:** e4b `d3b518b3` (P236 Amendment 2's merge, e4b#2124); grouped-nf4-gemm v0.51.0 `30adb1bf`.
- **Stack:** torch 2.8.0+cu128 (the image's), bitsandbytes 0.50.2, its published wheel proven by its bytes
  (`release_proof.json`: RECORD equal, imports from the install, fresh bytecode prefix).
- **Id:** `-2`, the lane's last started box, after `p236-5090-1` read VOID (Amendment 2).
- **Run:** 20 reps of 50 calls per arm per cell, after 10 warm-up calls; 30 cells in 10.1 s.

**Verdict: READ** (`verdict_p236.json`): "every cell correct, engaged as registered and timed; descriptive, no bar". The per-cell table is in
[`RESULTS-p236.md`](../../RESULTS-p236.md).

Files here: `bench_p236.json` (the bench's record), `verdict_p236.json`, `summary.txt`, `versions.txt`,
`release_proof.json`, `logs/` (the bench log and the two install logs). The box did not write the `SHA256SUMS` its
registration lists; the one here was computed when these files were committed, over every file in this directory but
itself. The launcher's
receipt and records, `forensics.txt` and `outer.log` stay in the private receipt store and are named, with their
sha256, in `STORE-ONLY.sha256`.
