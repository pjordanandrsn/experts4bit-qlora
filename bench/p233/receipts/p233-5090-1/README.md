# `p233-5090-1`: serving at solver against all-vram, teacher-forced quality (SOLVER_SANE, TRIPWIRE fired)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.84, a 575 W power limit recorded on the box and never ruled) on an AMD
EPYC 7532 host (64 CPUs, one NUMA node, AVX2 only), 2026-10-24T12:53:21Z to 14:31:51Z, through the guarded launcher.
Cost $1.45.
- **Code:** e4b `2a0a9afc` (the registration with Amendments 1-4, e4b#2110's merge), 0.57.0; grouped-nf4-gemm v0.50.0
  `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2.
- **Model:** Qwen/Qwen3-30B-A3B @ad44e777, its NF4 arena baked on the box.

**Before the phases:**
- The download probe (75.9 MB/s against the 40 MB/s floor) and the guard fit passed (11,587 s left against the 11,157 s
  needed).
- The tripwire passed, and so did the self-tests (`p233_reduce.py` 114 cases, `p233_box.py` 69, `p115_quality.py` 11)
  and the premise, exactly as registered: on the card 37 passed (`logs/premise_gpu.log`), and on the CPU 6 passed
  (`logs/premise_cpu.log`).

**The run:** three quality phases, each its own process, in the registered order: `quality_A.json` (all-vram, the
reference, with its floor and mutant arms), `quality_S.json` (solver at 8.0 / 3.0 GiB) and `quality_C.json` (solver at
8.0 / 0). Each reads group T = 1 and 8, 24 wikitext windows x 128 positions after a 512-token prompt. Every process
rc 0.

**Verdict: `SOLVER_SANE`, TRIPWIRE fired at both T** (`verdict_p233.json`):

| T | NLL gap (S - A) | argmax agreement | KL (S against A) | rules |
|---|---|---|---|---|
| 1 | +0.002077 | 2993 / 3072 | 0.003925 | SANE |
| 8 | -0.000047 | 2993 / 3072 | 0.003729 | SANE |

Bars: |NLL gap| <= 0.02 and argmax >= 0.95. The tripwire's argmax agreement (0.974284 < 0.98) and KL (> 0.002)
conditions fired at both T. The full read and the written look the tripwire asks for are in
[../../RESULTS-p233.md](../../RESULTS-p233.md).

**Files:**
- **What `SHA256SUMS` covers:** every file here.
- **What stays only in the private receipt store** (`STORE-ONLY.sha256`, by path within the run's receipt): the
  launcher's teardown proof and instance record, which name the provider's instance. The launcher's receipt, its ledger
  row, the host facts and the guard files stay there too.
- **Not copied:** the staged scripts, tests and calibration file, each byte-identical to the file `staged-p233.sha256`
  pins at `2a0a9afc`; the nonce markers, the run id, the start marker, the instance id file and the pip constraints file.
- **No identifiers on the box:** no file here names an instance, an offer, a host name, an endpoint or a GPU UUID. The
  runner does not collect the UUID, so `forensics.txt` and `outer.log` are public here.
- **Byte-identical, carriage returns kept:** the quality and fetch logs carry the bare carriage returns of their
  progress bars (no CRLF line ending anywhere). They are kept as the box wrote them, so `SHA256SUMS` matches the store.
