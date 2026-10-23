# HC: does `serve_paged` compose with the hybrid tier? (a design note; USD 0, no box)

The capacity frontier's first open fact: `serve_paged` serves a model whose experts do not all fit in VRAM only if the
paged runner, its bucketed decode graphs and the hybrid tier compose. Every serving number so far was read at the
all-resident point (`E4B_PAGED_PLACEMENT=all-vram`). The ruling for this item: a USD 0 CPU test and this note first, and at
most one explore box after it. Nothing here was measured on a GPU. Every claim below is read from the code at this note's
base and cited by file and line.

## 1. What composes today, by placement

`build_engine` (`serve_paged.py`) always runs the placement solver, then either moves every expert into the VRAM tier
(`all-vram`) or keeps the solver's three tiers (`solver`), then calls `enable_hybrid_tier` either way.

| piece | `all-vram` | `solver` (some experts in DRAM or on NVMe) |
|---|---|---|
| MoE decode forward | the all-resident collapse (`hot_residency.py`, `_forward_collapsed`, gated on `_all_hot()`) | the baseline tiered path on every layer that is not all-hot: VRAM rows on the GPU, DRAM rows computed on the host by the CPU kernels, NVMe rows streamed to the GPU (`hybrid.py::_cold_contrib`) |
| decode graphs, `E4B_PAGED_GRAPHS` unset or `auto` | captured on CUDA at sm_89+ | off: eager decode (`serve_paged.py::_graphs_env`) |
| decode graphs, `E4B_PAGED_GRAPHS=1`, more than one sequence and a bucket above 1 | captured | refused at startup in words (`serve_paged.py::_batched_graph_grouping`) |
| decode graphs, `E4B_PAGED_GRAPHS=1`, `max_seqs=1` or buckets `(1,)` | captured | accepted at startup; every capture raises and that bucket runs eagerly (section 2) |
| prefill graph (`E4B_PAGED_PREFILL_GRAPH`) | per its own rule | refused: device grouping is off (`paged_runner.py`, the prefill-graph refusal) |
| prefill | GPU | GPU-only by default (`paged_runner.py`, `gpu_only_prefill=True`): DRAM experts are copied host to device once per unique expert per chunk (`hybrid.py::_dram_on_gpu`); NVMe experts stream as at decode |
| int4 expert lever (`E4B_SERVE_EXP_INT4`) | applies | refused per tiered layer (`int4_experts.py`: "layer ... is tiered") |
| `E4B_NF4_SMALLM_BW` | applies | refused (`hot_residency.py`: "runs on the all-resident collapse only") |
| the pipelined engine (`engines/pipelined.py`) | not reachable from `serve_paged` (wired only in `serve.py`) | not reachable |

So at `solver` the server builds and decodes eagerly through all three tiers. Its graphs, its prefill graph and two of its
speed levers stand down or refuse by name.

## 2. Why a tiered layer cannot be captured, and why that is safe today

A CUDA graph replays fixed kernels on fixed addresses with no host work. The tiered forward needs host work on every call:

- **Before any branch that depends on routing:** `hot_row.nonzero()` (`hot_residency.py`, the baseline forward's hot
  branch). Its output shape depends on the routing, so it synchronizes, and a capture that reaches it raises.
- **On the hot branch:** host grouping (`unique_consecutive`, then `counts.tolist()`); the code itself calls this sync
  "structurally illegal inside CUDA-graph capture".
- **On the DRAM branch:** a boolean-mask split, a device-to-host copy of the activations, `tolist()` of the expert ids and
  counts, the CPU kernels on a thread pool, then a host-to-device copy of the result.
- **On the NVMe branch:** `.cpu()` of the ids, a host `unique`, an NVMe read into the tier's view, and fresh device tensors
  every call.

The first sync runs on every tiered layer whatever the tokens route to. `_capture_bucket` warms and captures each bucket
on token 0 at position 0 (`paged_runner.py`), and the capture of a tiered layer raises at that first `nonzero` no matter
which experts token 0 picks. `enable_decode_graphs` records the exception as `eager: ...` for the bucket and runs its padded
step eagerly. **So the failure a reviewer fears, a capture that succeeds while routing hits only resident experts and
then replays without the cold experts, is not reachable in this code.** It would become reachable only if the tiered
forward were made sync-free without also moving the cold-expert ids and the staging addresses into device memory.

The CPU test (`tests/test_serve_hybrid_compose.py`) pins the two facts this depends on: that the collapse runs only when
every expert is resident and in identity order (`_all_hot`, driven on CPU), and that the baseline forward reaches the
`nonzero` before any branch on the routing. It also pins the startup matrix in the table above.

## 3. What is not known

- **No GPU has ever run `serve_paged` at `solver`.** No test drives `PagedModelRunner` with `enable_hybrid_tier` at a
  tiered placement. The engine tests that cover the tier need CUDA (`hybrid_available()`), and the CPU kernels it needs are
  not available on every host. Whether the server builds, answers and decodes at `solver` is therefore unread.
- **That a forced capture raises rather than hangs or corrupts** follows from the code, but was never observed.
- **Tokens across placements are not bitwise.** The DRAM tier dequantizes in fp32 on the host while the GPU paths round
  through the compute dtype (`hybrid.py` module docstring: "same placement => bit-identical runs; moving an expert between
  tiers changes its rounding path within the documented cross-placement tolerance"). A comparison against `all-vram` reads
  the first divergence, never identity.
- **Speed at `solver` is unread.** Decode there is eager, and the DRAM rows run on the host per step, so the CPU family
  matters as it did for training.

## 4. Options for graphs at `solver` (for the record; none is proposed for code now)

1. **Eager at `solver` (today).** Correct by construction. The only change worth making soon is that
   `E4B_PAGED_GRAPHS=1` at `solver` should refuse at startup, in words, instead of capturing nothing bucket by bucket. That
   is a separate one-line change with its own test, and only if the maintainer wants it.
2. **Piecewise capture.** Capture attention, norms and the dense layers between MoE layers, and run each tiered MoE layer
   eagerly between graph segments. This keeps all three tiers, including the DRAM tier, which can never be captured
   because its compute is on the host. Cost: a graph segment per layer boundary, and a launch-floor saving bounded by the
   share of the step outside the MoE layers. A design of its own, after the box below says the eager path is worth
   speeding up.
3. **Per-hot-set capture.** Make the tiered forward sync-free: device-built split of hot and cold rows, cold experts
   streamed into fixed staging slots, one graph per hot set. This fits NVMe streaming only, not the DRAM tier, and it is
   exactly the change that would make the hazard in section 2 reachable, so it would need a replay check against eager on
   every hot-set change. Not before options 1 and 2 are read.

## 5. The one explore box (after this note merges; explore tier, not a reading)

**Purpose:** turn section 3's unknowns into facts. One RTX 5090, Qwen3-30B-A3B on the NF4 arena, the default stack
otherwise, `E4B_PAGED_PLACEMENT=solver` at a VRAM budget that leaves some experts in DRAM and some on NVMe (the solver's
own split, recorded from `/health`).

**What it records, in this order, each a stop if it fails:**

1. The server builds at `solver` and `/health` reports the placement, the tier masses and `graphs: false`.
2. Ten greedy completions at one request complete. Their tokens are compared with the same prompts at `all-vram` on the
   same box, eager, reporting the first diverging position per prompt (not a pass/fail: section 3).
3. `E4B_PAGED_GRAPHS=1` with `max_seqs=1` at `solver`: every bucket's status reads `eager: ...`, and the completions match
   step 2's `solver` tokens exactly (same placement, so identity is expected).
4. Decode tokens per second at one request: `solver` eager, `all-vram` eager and `all-vram` graphs, same prompts, as
   explore numbers only.

**Cost and caps:** about 40 minutes of a 5090 at the explore tier's per-box cap, a guard of 1 h, the explore tier's bus
line at create and at teardown. **No result from it is a reading or a public number.** Anything it shows that is worth
stating publicly gets a registration first.

## 6. What this note asks for

- Review of sections 1 and 2 (the composition table and the capture argument) against the code.
- A ruling on option 1's startup refusal for `E4B_PAGED_GRAPHS=1` at `solver`: wanted now, or not.
- Leave to run the section 5 box at the explore tier once this merges.
