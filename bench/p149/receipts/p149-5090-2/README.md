# `p149-5090-2`: P149 box Q+S, quality at decode and the served read (FLIP, routing qualifier VOID)

One RTX 5090 (cc 12.0, 170 SMs, 32,607 MiB, a **575 W** power limit read on the box before any install, a 3,090 MHz
maximum SM clock) on an AMD Ryzen 7 9800X3D host (8 cores, 16 CPUs). The lane ran 2026-10-20T19:03Z to 20:38Z. Cost
$1.574.
- **Code:** e4b `835874d1` (the pre-read band's merge, #1893; the staged pieces are the kit #1858's, byte for byte, per
  `bench/p149/staged-p149.sha256`); grouped-nf4-gemm 0.49.0 at `a83f6bb3`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2 (`versions.txt`).
- **Model:** `mistralai/Mixtral-8x7B-Instruct-v0.1` at `eba92302`, NF4 experts baked on the box (`work/bake.json`).

**Order** (`phases.jsonl` stamps each phase's UTC start and end):
1. Self-tests and the premise on the card.
2. Fetch and bake, then P147's windows, sha-checked.
3. S1w and its early gates (`gate_S1w.json`).
4. S0a and S1a: speed, then Q in the same process.
5. The Q reduce (`q_verdict.json`).
6. T1, then S1b, S0b, S0c, S1c, then T2, then S1d and S0d.
7. Both SANE phases, then the reduce.

**Verdict: FLIP** (`verdict_p149.json`).
- g1 1.160795, the minimum of the four S1/S0 W1 pairs; g16 1.018108.
- Q GO at both gates: W1 upper bound 0.002311 against 0.011537; W16 0.002894 against 0.011313.
- Self-pairs inside the NOISY band.
- The routing qualifier read **QUALIFIER_VOID**: S0, fed its own stream, differed from it at 5 of 1,100 steps in every A block (`tf_T1.json`, `tf_T2.json`). So the default-flip PR is held (Amendment 4).
- The verdict re-derives byte-identical from these files:

```
python bench/p149/p149_reduce.py --dir bench/p149/receipts/p149-5090-2 --out v.json \
  --e4b-sha 835874d1449314fc569442228a3e46e7a6602748
```

**Files:**
- `arm_S{0,1}{a,b,c,d}.json`: the eight timed processes.
- `tf_T1.json`, `tf_T2.json`: the token-pinned processes.
- `gate_S1w.json`, `q_verdict.json`, `sane_S0.json`, `sane_S1.json`, `verdict_p149.json`.
- `summary.txt`: the runner's summary.
- `phases.jsonl`, `prompts.json`, `versions.txt`, `work/bake.json`.
- `logs/`.

`SHA256SUMS` covers them and this README. `STORE-ONLY.sha256` names what stays in the private receipt store.
