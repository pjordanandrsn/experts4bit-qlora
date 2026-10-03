# P98 — Hybrid decode under CUDA graphs through the serving stack: Qwen3.6-35B-A3B's bucketed graph replays against its padded eager step, and the first hybrid decode speed, on one RTX 5090 (registered 2026-10-02, before any run)

Issue: experts4bit-qlora#564. Code under test: #907 (hybrid decode graphs: the per-slot state through the bucket
selector, the pool warmed and frozen) and #908 (the pool's index cache), on top of #889 / #897, read in P97.

**Why.**
- P97 (#906) read hybrid paged serving SUPPORTED on the card, eagerly: 830 ms per 4-row decode step on Qwen3.6, a
  correctness reading, not a serving speed.
- #907 lets `enable_decode_graphs` capture a hybrid model, and #908 removes the capture's last host copy.
- On a GPU they have been read only on small models:
  - an all-linear dense Qwen3.5 under real capture on the A2000 (#908: both mechanisms bit for bit, each mutation
    caught);
  - the fp8 bucket path on a dense hybrid, in `tests/test_hybrid_decode_graphs_gpu.py`, which has never run, since it
    needs sm_89+.
- **This lane** asks, on a 5090, through the production serving stack (`serve_paged.build_engine`):
  - does Qwen3.6-35B-A3B capture every decode bucket and replay exactly as its padded eager step?
  - how fast does it decode under graphs, against plain eager?

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

- **Box and software.** One RTX 5090.
  - e4b at the launch commit; grouped-nf4-gemm at `34da93d` (v0.34.1, e4b CI's pin).
  - transformers 5.17.0; bitsandbytes 0.50.2.
  - No `fla` or `causal_conv1d`: the Gated DeltaNet layers take transformers' torch path, as in P97.
  - Every serving lever unset.
- **Model.** `Qwen/Qwen3.6-35B-A3B` at `995ad96`, the checkpoint P97 read.
- **The engine** (`serve_paged.build_engine`):
  - its NF4 arena, baked on the box from the pinned checkpoint (`p98_bake.py`, `bench/p39/k8_bake.py`'s steps with the
    revision pinned);
  - the hybrid expert tier, all-VRAM placement, P39's placement calibration;
  - the fp8 KV on a compact 10-layer pool, the per-slot linear state;
  - 16 slots, buckets 1, 2, 4, 8, 16, with the batched lane's sync-free device grouping when graphs are on.
- **Three arms, each a fresh engine in its own process** (`build_engine` sets module-global grouping switches):
  - **g:** bucketed decode graphs, captured (`E4B_PAGED_GRAPHS=1`);
  - **e:** the same buckets and grouping, every step run eagerly on the same padded layout (`capture=False`), the
    bitwise oracle for g's replays;
  - **p:** plain eager decode, the serving default.
- **Two workloads per arm**, through the engine's own continuous scheduler, greedy, no stop set:
  - **W16:** 16 requests at once, 256-token prompts (wikitext-2 test windows k × 4096, k = 0–15), 48 + 4i new tokens.
    The sequences finish one by one, so the active set walks every bucket;
  - **W1:** one request (window 16), 96 new tokens.
- **Timing.** Every `run_decode` call is CUDA-synchronised and timed with its row count. Throughput is rows over time
  after each workload's first 3 calls.
- **The premise, on the card, before anything is fetched** (rc 25). The three hybrid GPU test files must PASS, all four
  tests, none skipped:
  - `test_linear_state_gpu.py`: the hybrid path through the real kernel;
  - `test_hybrid_decode_graphs_gpu.py`: hybrid bucket graphs under the scheduler against the padded eager step;
  - `test_linear_state_graph_gpu.py`: the linear state under real capture.
- **The order:** fetch; bake; arms g, e, p; reduce.

**The reducer** (`p98_reduce.py`, 19-case self-test).
- **VOID** if any of these holds:
  - a record is missing, or ran a rehearsal knob (stand-in attention; a placement other than all-vram);
  - a record is not the registered shape or model / commit, or a request produced other than its budget;
  - a layout is not Qwen3.6's on a compact pool (10 pool layers, 10 attention, 30 linear, the state allocated; frozen
    in g and e);
  - the oracle is not one (e captured or replayed; p ran graphs);
  - the W16 trace never stepped some bucket.
- **NOT_SUPPORTED** if, in arm g, any of these holds:
  - a bucket did not capture;
  - a step ran eagerly;
  - a captured bucket never replayed;
  - W16's or W1's tokens differ from arm e's.
- **SUPPORTED** otherwise.
- **Reported, not gated:**
  - decode throughput, W16 and W1, in every arm;
  - graphs over plain eager;
  - g's token agreement with plain eager (a different row count per GEMM, so free-running greedy can diverge);
  - per-row-count step times.

**The registered consequence.**
- **SUPPORTED:** `docs/SERVING.md`'s hybrid paragraph cites P98: hybrid decode graphs read on the card, with their
  speed.
- **NOT_SUPPORTED:** the failing bucket or step is recorded on an issue before any change; graphs for hybrids stay
  documented as unread.
- **VOID:** nothing moves.

## Predictions (written before the data)

- **SUPPORTED.** Every bucket captures and replays, with no eager step, and arm g's tokens equal arm e's on all 17
  requests.
- **Graphs are faster:** at least 2× plain eager on W1 and at least 1.5× on W16. Decode at Qwen3.6's 3B active
  parameters is dominated by per-op launch and Python overhead: the torch Gated DeltaNet path issues many small kernels
  in each of 30 layers. A graph removes that overhead, and the torch path's arithmetic stays.
- **Plain eager W1** lands between 2 and 20 tok/s. P97's eager harness read 830 ms per 4-row step, outside the serving
  stack.
- **Peak GPU memory** stays at or below 28 GB in every arm. P97 read 23.05 GB resident.
- **Graph-vs-plain token agreement:** no prediction. Free-running greedy decode diverges for good after one flipped
  argmax, and different row counts per GEMM can flip one.

## Box and cost

- **`p98-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P98_PROVE=1` runs:
  - the refusals;
  - the install with its tripwire;
  - the reducer's self-test;
  - the premise (the four GPU tests);
  - an HF CDN egress probe.

  It loads no model.
- **`p98-5090-<n>`:** one RTX 5090, after a passing proof, with ≥ 200 GB of disk (the runner refuses below 170 GB).
  **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).** Estimated:
  - install and premise ~6 min;
  - fetch ~6;
  - bake ~10;
  - three arms ~8 each (build ~3, workloads ~5).
- **Lane ceiling $2.50; hard stop $3.50.**

## Rehearsal

Run on the NAS RTX A2000 (sm_86, 12 GB) from e4b `adf516f` (main, with #907 and #908), grouped-nf4-gemm at the pin,
with the local checkpoint. The card has no native e4m3 and cannot hold Qwen3.6's experts, so the rehearsal could not
run arms g and e, the fp8 kernel, or all-VRAM placement. It ran the bake and arm p with the stand-in attention, solver
placement and a shortened workload (4 requests of 64 + 8–14 tokens, W1 of 8). No time is quoted.

- **The bake (`p98_bake.py --offload`), run 1, FAILED:** "shape [256, 1024, 1024] is invalid for input of size 0".
  An offloaded expert module holds 0-element placeholders; its NF4 tensors live in the offload handle's pinned host
  copies. The bake now reads those copies when a layer is offloaded. The registered run loads resident and reads the
  modules either way.
- **The bake, run 2, OK:**
  - 40 layers × 256 experts, a 16.88 GiB NF4 snapshot, and an 18.1 GB arena of 10,240 rows, baked in 57 s;
  - the load took 1,056 s, reading 68 GB off the NAS disks.
- **`serve_paged.build_engine` on the arena works on the hybrid:**
  - 40 MoE layers, 256 experts, top-8, `qwen3_5_moe_text`, KV 2 heads × 256;
  - a 10-layer pool on attention layers 3, 7, …, 39, 30 linear layers, the state allocated.

  So the hybrid expert tier and the arena serve a `qwen3_5_moe` checkpoint.
- **Arm p, run 1, FAILED:** "request of 80 unique rows exceeds hot_rows=64". Under solver placement, experts stream
  from the arena, and a prefill chunk routes more unique experts than the default staging buffer holds. With
  `E4B_PAGED_HOT_ROWS=256` (rehearsal only), **arm p ran end to end**: build 57 s, every request's token budget met,
  the timing records written, and the record's layout as registered. All-VRAM placement, the registered setting, keeps
  every expert resident and never takes that cold path.
- **Not rehearsed, and read for the first time on the 5090:**
  - arm g's capture of Qwen3.6, and arm e's padded eager steps on the fused fp8 append;
  - all-VRAM placement with the batched lane's device grouping;
  - the premise's two fp8 test files.

  The premise runs those test files on the proving rental, before any model is fetched.

Amendments, dated, go below this line before any data is read.
