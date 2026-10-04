# P112 — results: **closed VOID**. Neither reading produced a verdict, and the question moves to a new lane. Seen but not read: on a quiet host, `GNF4_PDL=1` ran the int4 serving step's B=1 decode 3.7 % faster and its B=16 decode 1.5 % slower, with identical tokens

Registration: `bench/p112/PREREG-p112.md` (#1023, `d621e46`); Amendment 1 (#1025, `282070a`); Amendment 2 (this
change). Issue: #1015. The switch: grouped-nf4-gemm#448. The lane it follows: grouped-nf4-gemm K28 (LEVER,
grouped-nf4-gemm#451).

Code under test: e4b at the launch commits, grouped-nf4-gemm at `951a97f` (0.36.0 with the switch), torch 2.8.0+cu128,
transformers 5.17.0. The subject is `Qwen/Qwen3-30B-A3B` @ `ad44e77` on the default graph server with SC1's int4_sched
levers.

## The runs

| run | host | verdict | why | cost |
|---|---|---|---|---:|
| `p112-5090-1` | Vast 54118102, machine 37958, Ryzen 9 7950X | **VOID** | engagement: "6907 of 6927 switched launches carried PDL", the lane's own accounting (Amendment 1) | $0.3210 |
| `p112-5090-2` | Vast 54122612, machine 45511, EPYC 7C13 (shared: 762 of 1,007 GB of host RAM in use) | **VOID** | "a decode slope is void": W16 prefill jitter of about 1 s a pass, against the slope's 1.2 s of decode (Amendment 2) | $1.0210 |

The lane cost **$1.3420** by the receipts. Run 2's figure comes from the provider's invoice and includes $0.649 for
downloading the 62.3 GB checkpoint. Run 1's is rate × runtime and may leave out its own download.

## What held in both runs

- **The premise passed on the card:** `test_decode_graph_buckets.py` 7 passed, and grouped-nf4-gemm's `test_pdl.py`
  23 passed, none skipped.
- **SC1's int4 configuration was in force in every arm:** int4 stores on all 48 MoE layers, 96 int4 attention
  projections, and the fusions (qkv 48, T1 glue 193, R2 48 / 48, router epilogue 48). Every bucket was captured.
- **Tokens were identical** in every arm, on every row of both workloads, at both lengths.
- **The default-server census** read the same twice. e4b's default NF4 server's build launches `_swiglu_rows` and
  `_combine_rows` (1,440 launches) and no other switched kernel. So "the default decode step" that K28's consequence
  named reaches 2 of the 12, as the registration said.

## Run 1: VOID on the lane's own instrument

Triton 3.4 reads `kernel.function` for the launch hook before `kernel.run` initialises the handle. The launch that
compiles a variant therefore reached the hook with no handle, and the box counted it as PDL-less. Amendment 1 changed the
accounting to count compile launches and compiled variants separately. **Run 2's accounting confirms the cause
directly.**

| kernel | launches | compile launches | compiled variants | P1: variants / handled launches with PDL |
|---|---:|---:|---:|---|
| `_rmsnorm_rows` | 879 | 4 | 4 | 4 / 875 |
| `_gemv_int4_b32` | 576 | 4 | 4 | 4 / 572 |
| `_quant_x_rows`, `_reduce_partials` | 576 each | 3 | 3 | 3 / 573 |
| `_rope_norm_heads` | 1,440 | 2 | 2 | 2 / 1,438 |
| `_rmsnorm_resid_rows`, `_router_epilogue`, `_swiglu_rows`, `_combine_rows` | 720 each | 1 | 1 | 1 / 719 |

In P0 nothing carried PDL.

**Run 1's arms, not a reading.** Self-pairs were 1.0023–1.0047.

| arm | W1 tok/s | W1 ms/step | W16 tok/s | W16 ms/step |
|---|---:|---:|---:|---:|
| P0a | 225.05 | 4.4435 | 1763.88 | 9.0709 |
| P1a | 233.89 | 4.2755 | 1737.58 | 9.2082 |
| P1b | 233.82 | 4.2768 | 1745.73 | 9.1652 |
| P0b | 225.57 | 4.4333 | 1771.60 | 9.0314 |

Pair ratios P1/P0 were **W1 1.0393 / 1.0366** and **W16 0.9851 / 0.9854**. Read by the rule's later steps, which a VOID
does not reach, these would be SLOWER (g1 1.0366, g16 0.9851).

## Run 2: VOID on the timing method, on a noisy host

P109's slope differences whole-pass walls, and each pass begins by prefilling every row. On this host one arm's W16
32-token passes took 6.02, 7.44 and 6.14 s, and its 160-token passes 7.28, 9.47 and 8.32 s. That jitter is larger than
the 128 decode steps the slope isolates (about 1.2 s). One W16 slope came out void, and another read an impossible
6,459 tok/s. W1's walls moved by up to 0.17 s against about 0.55 s of decode. P111 used the same method without trouble,
but on the NF4 store and a quiet host.

SC1 met the same problem and timed decode only, subtracting each pass's last time-to-first-token (its A10). P112's
registered instrument does not.

## Why the lane closes instead of running a third time

- **The ceiling.** The registration's $2.00 ceiling was priced at the hourly rate alone ($0.75 a run). The provider also
  bills the checkpoint download, about $0.65 a run, and the launcher's own estimate is $1.95. A third run would exceed
  the registered ceiling.
- **The method.** A third run would need a new instrument (decode-only timing), and its question would change. Run 1
  shows PDL helping one request and costing 16. The candidate default is therefore PDL at small row counts only, which
  is a different switch.

**The consequence.** A VOID registers none. `GNF4_PDL` stays opt-in, as it is.

**The next lane** reads, on one RTX 5090, under decode-only timing:
- `GNF4_PDL` off;
- PDL on every switched launch (this lane's arm, to replicate run 1);
- PDL only on launches of at most a few rows (a switch grouped-nf4-gemm would add).

## Receipts

`receipts/p112-5090-1/` and `receipts/p112-5090-2/` hold, each with `SHA256SUMS`:
- the four arm receipts, `verdict.json`, `census_default.json`;
- `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- the logs (force-added) and the teardown proof.

The launcher's receipts and ledger rows are in the receipt store: adertha-receipts `a88df20` (run 1; `551651d` removed
the clone its fetch picked up) and `19fb230` (run 2).
