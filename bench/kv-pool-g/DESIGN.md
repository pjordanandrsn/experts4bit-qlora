# Design packet ($0): slots that share the KV pool, blocks claimed on demand (serving roadmap lever G)

**Status:** design only, for one non-author review before any code. No reading, no default change, no claim.

## Why

On the paged server, `E4B_PAGED_MAX_SEQS=auto` picks the widest slot count whose KV pool fits at
`E4B_PAGED_MAX_TOKENS_PER_SEQ` tokens per slot. At a long per-slot ceiling, fewer slots fit than the decode buckets
could serve, and requests queue even when most of them are short.

The pool is not the constraint. The constraint is how it is claimed:

- `Fp8PagedKV` already hands blocks out of **one free list per layer**, lowest row first
  (`engines/fp8_paged_kv.py`, `_ensure_blocks` and `claim_blocks`). A slot holds only the rows it was given.
- But the pool is **built** with `batch * blocks_per_seq + n_scratch` rows. `PagedModelRunner` then **claims every block a
  slot can reach** the first time that slot runs under the decode graphs (`_flush_bulk` and `_ensure_graph_ready`,
  `kv.claim_blocks(slot, blocks_per_seq - 1, ...)`). The claim-all exists because a graph step "addresses KV blocks from
  the device table and never allocates".

So slots x per-slot ceiling is a hard device floor today. Lever G removes it in three steps. Each step keeps today's
behaviour as the default until a GPU smoke shows the contract below.

- **G1:** graphed decode claims blocks incrementally.
- **G2:** the scheduler admits by reserved blocks.
- **G3:** `auto` sizes the pool from free memory, and the slots from the widest captured bucket.

## The contract

Every token, sequence and finish reason is identical to today's claim-all path **on the same schedule** (same prompts,
lengths and arrival order). Block **ids** may differ; block **values** may not. Attention reads K/V through the table, its
split plan depends on lengths, not on row ids, and the quantizer is per token and head, so moving a block to another
row changes no arithmetic.

## G1: incremental claims under the decode graphs

**What changes.** A slot owns the blocks for the positions it has written, plus the block its next write lands in.

- At the prompt flush (`_flush_bulk`), claim `ceil((prompt_len + 1) / 16)` blocks instead of `blocks_per_seq`. That
  covers the prompt (positions `0 .. prompt_len - 1`) and the first decode token at position `prompt_len`, including at
  a block boundary: a 32-token prompt claims 3 blocks, so the first decode's write at position 32 lands in block 2,
  which it owns.
- Before each decode step, for every slot in the step whose next write position `p` (the host's `pos_of[rid] - 1`) sits
  in a block it does not own, claim block `p // 16`.
- All of a step's new claims go in **one** table write: a new `Fp8PagedKV.claim_next(slots_blocks)`, one staged H2D
  copy into `_bt_all` through the existing two-buffer pinned ring (`_claim_staging`).
- `_graph_ready` stops meaning "every block claimed". It becomes "the flush claim happened". `free_slot` still releases
  every row through `reset` / `reset_all_layers`.

**(1) Graph safety: which stream writes the table, and why the write is ordered before the replay.**

- The captured decode graphs read block ids from the **same `_bt_all` tensor**, never a copy that could move. With
  `E4B_KV_STEP_SELECT` the gather `index_select(_bt_all, 1, slot_l, out=st["tbl"])` runs outside the graph in
  `graph_bucket_load`; without it the graph indexes `_bt_all` itself. `_bt_all` is allocated once in `__init__` (the per-layer
  `block_table` entries are views of it) and is never reallocated. G1 writes **into** it (`_bt_all[:, seq, blk].copy_(..., non_blocking=True)`); it never rebinds it.
- The write is issued on the **current stream** of the engine thread. `_run_decode_bucketed` issues the step on that
  same stream, in this order:
  1. the claim (`claim_next`);
  2. `buf["ids"]` / `buf["pos"]` copies;
  3. `graph_bucket_load`;
  4. `g.replay()`.
  Stream order alone places the H2D table write before both the gather and the replay. No event or sync is needed, and
  none is added. The write is a real host-to-device copy: `_bt_all` is a device tensor, and `claim_blocks` (and
  `claim_next`) fill a pinned host buffer, then `_bt_all[:, seq, lo:lo + n].copy_(src, non_blocking=True)` on the
  current stream. No CPU-side tensor is written that the graph never sees.
- **Capture is unchanged.** Buckets are captured on scratch slots only (`_capture_bucket`), each backed by one block
  claimed at build. No live slot's table is touched during a capture.
- **The decode lookahead** (`E4B_PAGED_DECODE_LOOKAHEAD`) queues step t+1 while step t is in flight. Step t+1's claim is
  computed from host positions, which advance deterministically by one per issued step, not from tokens read back. Its
  H2D write is queued after step t's replay and before step t+1's replay, on the same stream, so the copy cannot
  execute while step t's replay runs. It also writes a table entry step t never reads: a new claim is block
  `p_{t+1} // 16` only when that block was not owned at step t, and step t's attention reads entries only up to its
  own length. Under `E4B_KV_STEP_SELECT` step t's gather into the bucket's `st["tbl"]` is queued before step t's replay,
  and step t+1's gather after the claim. The staging ring's event guard already keeps a queued copy's host buffer from
  being rewritten.

**(2) Out of blocks at a boundary.**

- G2's reservation makes this impossible: a slot is admitted only with `ceil((prompt_len + max_new_tokens) / 16)` blocks
  reserved, and every claim lies inside that bound.
- `claim_next` keeps today's check (`len(self._free[layer]) < n`) and raises. G1 adds the reason
  (`"claim past the slot's reservation"`).
- In production the raise happens on the engine thread, inside the step. It takes the step-failure path, so every
  open stream gets an `error` item and the server reports `state: error`. The step is not run, and no table entry points
  at a row another slot owns. It refuses; it never corrupts.
- A unit test drives the scheduler past a deliberately wrong reservation and asserts exactly this.

**(3) Alias and scratch slots, and prefill staging.**

- **Scratch slots:** unchanged. One block each, claimed at build; padding rows write position 0 of their own scratch
  block.
- **Alias slots** (speculative verify, `alias_bind`): an alias owns no row and copies the speculating slot's table row.
  A verify over `k + 1` rows writes positions `p .. p + k`. So before a verify step the slot claims through block
  `(p + k) // 16`. `decode_budgets` already keeps `p + k` inside the request's length, so the claim stays inside the
  reservation. `alias_bind` keeps its precondition ("the slot must already own every block the verify can reach"),
  which G1 now satisfies per step instead of up front.
- **Prefill staging:** unchanged. Chunks stage bf16 K/V by slot in Python; the flush quantizes into the pool and is where
  G1's first claim happens. A first-chunk prefill graph reads no history and touches no table.

**(4) The two KV paths.** The serving engine builds only `Fp8PagedKV` (`serve_paged.build_engine`). The bf16
`engines/paged_kv.py` (`TieredPagedKV`) holds one sequence and no block tables shared across slots; the server never
builds it. It is out of scope, and G1 adds a test that `build_engine` refuses the incremental mode with any other KV class.

## G2: admission by reserved blocks

- `ContinuousScheduler` gains `kv_blocks` (rows per layer available to live slots: the pool's rows minus scratch) and a
  `blocks_of(request)`: `ceil((prompt_len + max_new_tokens) / 16)`.
- `_admit` takes a request only if a slot is free **and** the reserved total plus its need fits `kv_blocks`. `_retire`
  releases the reservation when the slot is freed.
- FIFO is unchanged. A request that does not fit waits; nothing behind it jumps ahead. There is no preemption and no
  eviction.
- With `kv_blocks = None` (today), admission is unchanged.
- The server already refuses `prompt_len + max_tokens > E4B_PAGED_MAX_TOKENS_PER_SEQ`, so no single request can need
  more than one slot's ceiling.

## G3: sizing (lands last, opt-in)

`E4B_PAGED_KV_POOL=claim-all` (the default, today's behaviour) or `shared`:

- the pool's rows come from free device memory after weights, with the existing prefill-graph reserve and margin;
- the slots are the widest captured decode bucket that `max_seqs=auto` allows;
- `E4B_PAGED_MAX_TOKENS_PER_SEQ` becomes the per-slot ceiling, not the divisor.

`/health` reports the pool rows, the reserved total and the derivation. `serve_recipe.choose_max_seqs` and Loggetta's
serve plan price the same pool, so the planner and the server agree by construction. `shared` stays opt-in until a GPU
smoke shows the contract.

## (5) Tests

**CPU (fake runner and a CPU `Fp8PagedKV` where it constructs):**

1. **Bit-identity vs claim-all on the same schedule.** One workload (mixed prompt lengths, staggered arrivals, a stop
   mid-run, an abort) through claim-all and through incremental claims. Assert every emitted token, finish reason and
   per-slot `seq_lens` are equal. Assert the K/V **values** read back through each slot's table are equal block by
   block, while the row ids are allowed to differ.
2. **Pool smaller than slots x blocks_per_seq.** For example, 4 slots and room for 2.5 sequences' blocks. Admission waits
   (the third long request queues while two short ones run alongside it), nothing raises, and every request finishes
   with its tokens.
3. Claims land one step ahead under the lookahead, and the H2D staging ring is never reused while queued.
4. A verify over `k + 1` rows claims through `(p + k) // 16` before its step.
5. A wrong reservation raises inside the step and takes the step-failure path (every stream gets `error`, state
   `error`), with no table entry pointing at another slot's row.

**GPU (one explore box, before G3 is offered):** the same workload served graphed under claim-all and under `shared`,
with identical tokens.

## Risks and what is not changed

- **Host cost:** one small H2D table write per step in which any slot crosses a 16-token boundary, against today's one
  large claim per request. The step trace's `dec_ready` segment shows both. If the per-step write costs more than the
  claim-all saved, G1 batches claims a few blocks ahead.
- **Not changed:** quantization, attention kernels, graph capture, bucket selection, the prefill graph, the free-list
  order (lowest row first), and every default until G3's smoke.
