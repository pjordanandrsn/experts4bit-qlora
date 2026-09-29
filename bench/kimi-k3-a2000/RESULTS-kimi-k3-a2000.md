# Kimi-K3 at full depth on one 12 GB RTX A2000 — released stack

**Measured 2026-09-28** on the shipped packages: `experts4bit-qlora` 0.37.5 and
`grouped-nf4-gemm` 0.33.4, both from PyPI. All 93 layers run on real weights. The
experts stream from an SSD arena, the dense side is served from safetensors byte
offsets, and no dense byte stays resident in host RAM. Register row
`e4b.offload.kimi-k3.full-depth.a2000.2026-09-28`.

This re-runs the 2026-07-30 driver on released code. The July run
([`receipts/2026-07-30/`](receipts/2026-07-30/)) used a pre-0.9.0 e4b whose commit
was never recorded and a loose, non-git copy of the kernel files, so its numbers
had no reproducible stack. The comparison table below puts them side by side.

## Results

| | 2026-07-30 (stack unrecorded) | **2026-09-28 (e4b 0.37.5, gnf4 0.33.4)** |
|---|---|---|
| greedy completion of `The capital city of France is` | `' Paris. It is'` | **`' Paris. It is'`** |
| token ids | [17374, 13, 1344, 387] | **[17374, 13, 1344, 387]** |
| p(`' Paris'`) / p(`'.'`) / p(`' It'`) / p(`' is'`) | 71.20 / 16.52 / 28.24 / 86.91 % | **68.90 / 17.22 / 26.17 / 86.76 %** |
| prefill, 6 tokens | 178.3 s, 6,130 expert rows | 270.5 s, 6,115 expert rows |
| decode step, median of 3 | 93.5 s | **92.4 s** (97.0, 92.4, 92.1) |
| per decode step | 108.8 GB dense + 1,467 expert rows, 92 slot claims | 108.8 GB dense + 1,466–1,467 expert rows, 92 slot claims |
| cached decode vs fresh prefill, last position | cos 0.999546, rel 1.86e-2, argmax agrees | cos 0.999408, rel 2.42e-2, argmax agrees |
| VRAM peak (`torch.cuda.max_memory_allocated`) | 4.07 GB | 4.32 GB |
| dense bytes pinned in host RAM | 0 | **0** |
| perplexity, 90-token window (89 predictions) | 3.191 (NLL 1.1603) | **3.181** (NLL 1.1573) |
| perplexity pass | 785 s, 29,893 expert rows, VRAM 3.67 GB | 777 s, 29,866 expert rows, VRAM 3.67 GB |
| routing, cached decode vs fresh prefill | 47 of 92 layers differ, 60 of 1,472 slots | 45 of 92 layers differ, 50 of 1,472 slots |

Every row of the September column comes from the JSON and log in
[`receipts/2026-09-28/`](receipts/2026-09-28/). The per-token probabilities and
the prefill time are printed in `k3_gen.log`, not stored in the JSON.

## What changed between the two runs

- **The argmax sequence is identical, and the probabilities moved by up to 2.3
  points.** The two runs share weights, prompt, driver logic and third-party
  stack. They differ in e4b (pre-0.9.0 → 0.37.5) and in the kernel (a July
  working copy → gnf4 0.33.4). The prefill read 6,115 distinct expert rows
  against July's 6,130, so the two builds route at least some tokens to
  different experts. At top-16 of 896 in bf16 a near-tie can flip, and that
  alone moves the output distribution (the routing row shows it). Neither run
  is a reference for the other. **Corrected the same day:** the gap is not
  attributable to the build. Later runs of this same build gave p(' Paris')
  71.20 % with 6,130 rows, exactly July's, and others did not; the forward
  drifts from run to run ([follow-up](#follow-up-the-same-day)).
- **The prefill took 270.5 s against 178.3 s.** Found the same day: one-time
  Triton compilation in a fresh cache, about 91 s ([follow-up](#follow-up-the-same-day)).
  Decode steps agree within 1.2 %, and each one reads the same 108.8 GB of dense
  weights, so the gap is not in the dense path's steady state.
- **VRAM peak rose 0.25 GB** on the generation pass (4.07 → 4.32 GB) and did
  not move on the perplexity or routing passes. Warm runs peak at 4.07 GB, so
  the extra 0.25 GB belongs to the first run in a fresh Triton cache.

## The cache check prints a warning that is not a finding

`k3_gen.log` ends its cache check with `*** CACHE PATH DISAGREES WITH A FRESH
PREFILL ***`. The driver's gate is `cos < 0.9999`, and that threshold was never
calibrated. The routing pass explains the gap. The cached decode step and a fresh
prefill of the same sequence select a different expert **set** in 45 of 92 MoE
layers, 50 substitutions of 1,472 slots. Each substitution swaps a whole expert,
so the divergence is discrete. The July runs showed the same pattern at one and
at three decode steps: cos 0.999575 after one step, 0.999546 after three. The
gap does not grow with steps, so it is not accumulating. Both paths agree on the
argmax. Bit-identity across prefill and decode is therefore not claimed for K3.
A cosine bound is the wrong gate here. A correct gate compares routing first and
compares logits only where routing agrees.

## How it ran

- **Model:** `moonshotai/Kimi-K3` at revision `f831ab66814297da540d832a5235f8e904f29d06`,
  the revision the local archive manifest records for this snapshot. `config.json`
  sha256 is `9710e121…`. Text path only. The one tensor the checkpoint does not
  ship, `vision_tower.patch_embed.pos_emb.time_weight`, is a non-persistent buffer
  in the vision tower. It stays on `meta`, and a text forward never reaches it.
- **Experts:** the 1,446,456,066,048-byte MXFP4 arena (92 layers × 896 experts,
  17,547,264-byte rows) was baked 2026-07-30 and is not re-baked here. It is read
  by gnf4's `Mxfp4NvmeResidencyK3`: 92 engines, `k_slots = 16`, one shared
  `SlotStore` of 281 MB (25.8 GB if each engine had its own) and one pinned
  `ColdTier` of 48 hot rows.
- **Dense side:** 108.76 GB across 1,251 tensors in 93 layers, served by
  `enable_dense_offload(pin=False, source=DenseDiskSource(...))`. The driver
  asserts that every placeholder became a disk home and that no disk-policy layer
  fell back to a host copy. `lm_head` (2.35 GB) runs on the host CPU.
  `loader._fit` narrows 69 KDA `A_log` tensors from 128 to 96 values.
- **Decoding:** greedy. One prefill of the 6-token prompt, then 3 cached decode
  steps (the prefill yields the first token). Eager attention.
- **Host:** the NAS's RTX A2000 12 GB (sm_86, driver 575.64.05) in a PCIe 3.0 x8
  slot, Xeon W-1250, 128 GB RAM. The arena and the dense extract are on a
  SATA-SSD ZFS pool. The card and pool were shared with the NAS's own services
  during the run. `sdxl-sidecar` was stopped for its duration to free VRAM.
- **Software:** Python 3.10.12, torch 2.8.0+cu128, triton 3.4.0, transformers
  4.57.6, fla-core 0.5.2 (K3's KDA kernels), tiktoken 0.13.0, bitsandbytes 0.50.2.
  These pin the third-party stack to the July run's, so the two columns differ
  only in e4b and gnf4. Full `pip freeze` in `versions.txt`.
- **Inputs:** sha256 of the dense index, arena index and manifest, `config.json`,
  tokenizer files and the remote modelling code are in `inputs.sha256`.

## Files

| file | what |
|---|---|
| [`k3_run_rel.py`](k3_run_rel.py) | the driver as run. It is July's `k3_run.py` with the imports moved to the released module paths, the output path, and a `provenance` block in every JSON. Its sha256 (`55f5c33e…`) is recorded in each JSON's provenance. |
| [`run_all.sh`](run_all.sh) | runs `gen` (4 tokens), `ppl` and `route` in sequence |
| `receipts/2026-09-28/k3_{gen_n4,ppl_n0,route_n0}_pin0.json` | results with provenance |
| `receipts/2026-09-28/k3_{gen,ppl,route}.log` | full stdout and stderr of each pass |
| `receipts/2026-09-28/{status.log,versions.txt,gpu.txt,host.txt,inputs.sha256}` | timings, environment, input hashes |
| `receipts/2026-07-30/` | the July driver (`k3_run.py`), its logs and JSONs. `k3_gen_pin0.json` there is the **1-step** run, which overwrote the 4-token run's JSON. The 4-token run survives in `k3_gen_base.log`. |

## What this does not establish

- It is not a quality claim. No reference implementation of K3 was run against
  it: Moonshot's API refuses `logprobs` for `kimi-k3`, and nothing else on this
  host can hold the model. The perplexity is one 90-token paragraph, a sanity
  band and not a benchmark. (A reference ran the same day: Fireworks, see the
  [follow-up](#follow-up-the-same-day).)
- It is not a speed claim beyond this host. Dense bytes from the SSD dominate
  each decode step, so the step time belongs to the pool and not to the kernels.
- It is not an e4b load path. The driver builds the model on `meta` and wires the
  engines itself. `load_moe_4bit_streaming` is not involved.

## Follow-up the same day

Four questions the first run left open, answered on the same host and stack, from
2026-09-28T20:28Z. Each receipt set carries the driver that produced it, and every
JSON's `provenance.driver_sha256` matches the driver file next to it.

| receipts | driver | what |
|---|---|---|
| [`receipts/2026-09-28-gate/`](receipts/2026-09-28-gate/SHA256SUMS) | `k3_run_rel.py` there (sha256 `30b837b0…`) | routing-replayed cache gate, warm prefill repeat, negative control |
| [`receipts/2026-09-28-reference/`](receipts/2026-09-28-reference/SHA256SUMS) | `k3_run_rel.py` there (`ea727400…`; adds per-token NLL to `ppl`) | perplexity with per-token NLL, Fireworks reference, per-token comparison |
| [`receipts/2026-09-28-determinism/`](receipts/2026-09-28-determinism/SHA256SUMS) | the reference set's driver | two prefill-only processes with Triton autotune caching on |

### 1. A cache gate that can fail: routing first, then logits

The gate now records every MoE call's (expert ids, weights) along the cached path
and **replays** them into the fresh, cache-free prefill, so routing is identical by
construction and the logits comparison measures only the kernels and the cache. The
threshold, cos ≥ 0.9999 with argmax agreeing, was fixed in the driver before any
replayed number existed.

| run | free routing | **replayed routing** | gate |
|---|---|---|---|
| correct cache, 4 tokens | cos 0.998699, 90 of 92 layers route differently somewhere in 9 positions (386 substitutions) | **cos 0.999966**, rel 4.36e-3, argmax agrees | **PASS** |
| negative control: all 69 KDA recurrent states zeroed before the last step | — | **cos 0.877523**, rel 0.433, argmax agrees | **FAIL** |

The control is what makes the gate evidence: a broken cache that still picks the
right next token is caught by the replayed cosine, which an argmax check would pass.
Under replay, 94 local near-tie flips were overridden (62 of 92 layers), the flips
that made the free comparison unusable as a gate.

### 2. The slow prefill was one-time Triton compilation

The first run was the first process in a fresh venv. Triton's cache gained 183
compiled kernels at 19:01–19:02Z, inside that run's prefill, and 28 more at 19:05Z,
inside its first decode step (97.0 s against 92.1–92.4 s for the others). With the
cache warm the prefill takes **179.0 s**, and a second prefill in the same process
**176.9 s**, against July's 178.3 s. The extra 0.25 GB of VRAM belonged to the same
cold run: warm runs peak at 4.07 GB.

### 3. A reference: Fireworks' kimi-k3, per token

[`fireworks_ref.py`](receipts/2026-09-28-reference/fireworks_ref.py) scores the same
raw text (no chat template) on Fireworks' serverless `kimi-k3`, which serves the
native MXFP4 expert weights with MXFP8 activations, through its completions endpoint
with `echo` for prompt logprobs. Both tokenize the paragraph into the same 90 tokens
(checked by string), and position 0 has no prediction on either side, so 89 positions
compare.

| | A2000, e4b 0.37.5 (bf16 activations) | Fireworks (MXFP8 activations) |
|---|---|---|
| greedy completion | `' Paris. It is'` | `' Paris. It is'` |
| p(`' Paris'`) | 68.90–71.59 % over 5 processes (below) | 69.26 % |
| perplexity, 89 predictions | 3.181 and 3.196 (two processes) | 3.150 |
| per-token NLL vs the reference | **Pearson r = 0.9965**, median \|Δ\| 0.038 nats, mean \|Δ\| 0.082, 52 of 89 within 0.05, 76 within 0.2 | — |
| largest per-token gap | `' sits'`, 5.66 vs 6.28 nats | |

Fireworks returns its logprobs at reduced precision (values like -6.28125), and its
MXFP8 activations make it a second implementation, not ground truth. What the table
supports: two independent MXFP4 implementations of K3 agree token by token, and
agree on the greedy text. It is still one paragraph.

### 4. The forward drifts from run to run, and autotuning is not why

Five processes on this build ran the same 6-token prefill:

| process | p(`' Paris'`) | prefill expert rows |
|---|---|---|
| first run (fresh Triton cache) | 68.90 % | 6,115 |
| gate run | 71.59 % | 6,126 |
| negative control (its prefill precedes the zeroing) | 71.29 % | 6,131 |
| determinism A (`TRITON_CACHE_AUTOTUNING=1`) | 71.20 % | 6,130 |
| determinism B (same) | 71.20 % | 6,130 |

Both July processes also gave 71.20 % and 6,130 rows. So the output is not a
function of the build alone. **Autotuning is ruled out** as the source after the
first run: the path's ten autotuned kernels (fla's KDA, gated-delta, short-conv
and norm kernels) persisted their choices to Triton's cache during that run (19:01–19:05Z),
and run A, with `TRITON_PRINT_AUTOTUNING=1`, benchmarked nothing. The prime suspect
is not confirmed: grouped-nf4-gemm's MXFP4 prefill combine folds each token's 16
expert outputs with `out.index_add_(0, rows, ...)` (`mxfp4_pipelined.py`), which
on CUDA accumulates floats with atomics when destination rows repeat, so its
summation order can change between runs. At top-16 of 896 in bf16, one bit is
enough to flip a near-tie downstream. Confirming it needs the same A/B under
`torch.use_deterministic_algorithms(True)`. Until then, every probability above
is one draw from this spread, and the register quotes the range.

*Answered the same evening; see [§5](#5-the-drift-is-the-mxfp4-prefill-combine-and-a-fixed-order-removes-it).*

### 5. The drift is the MXFP4 prefill combine, and a fixed order removes it

Nine prefill-only processes ran on the same box and build, from 21:41 to 22:35Z. Each ran the
reference set's driver, unmodified (sha256 `ea727400…`, copied into the receipt set), through a
wrapper, `det_wrap.py`. The wrapper does three things. It sets torch's deterministic-algorithms
switch from the environment. It can put a candidate `mxfp4_pipelined.py` ahead of the installed
one on `sys.path`. It hashes every MoE engine call's input `x`, router ids, router weights and
output into `moe_trace.json`, 92 calls per process. It only reads tensors; it computes nothing
the forward uses. Every process ran with `TRITON_CACHE_AUTOTUNING=1` and
`TRITON_PRINT_AUTOTUNING=1`, and no log shows a benchmark. The first six processes alternated
det, plain, det, plain, det, plain; the three fix processes came last.

| arm | processes | p(`' Paris'`) | prefill expert rows | MoE traces, process against process |
|---|---|---|---|---|
| shipped combine, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 3 | 0.7144126892089844, all three | 6,118 | identical at all 92 calls |
| shipped combine, default | 3 | 0.7119670510292053, all three | 6,130 | every pair differs, at 8, 5 and 3 of 92 calls |
| [grouped-nf4-gemm#410](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/410)'s `mxfp4_pipelined.py`, default | 3 | 0.7144126892089844, all three | 6,118 | identical at all 92 calls, and identical to the deterministic arm |

What the traces show:

- **In the default arm, the engine returns different bits for identical inputs.** Every
  differing call has the same `x`, ids and weights, and a different output
  ([`calls_detail.txt`](receipts/2026-09-28-det-ab/calls_detail.txt)). No call inherited a
  difference from upstream: each jitter was absorbed by the next bf16 residual add. That is why
  all three default processes landed on the same p. It is the common draw: July's two processes
  and determinism A and B (§4) gave the same 71.20 % and 6,130 rows. The warm-cache 71.59 % and
  71.29 % in §4 are draws where a jitter was not absorbed and a top-16 near-tie flipped.
- **Deterministic mode raised on no op** (`warn_only` off), so nothing in this forward lacks a
  deterministic implementation. Against the default arm, its first difference is always an
  engine output with identical inputs, at layer 9 or 11.
- **The fixed combine alone, with deterministic mode off, reproduces the deterministic arm bit
  for bit** at every call of every process. So in these nine processes the only run-to-run
  difference was the combine's `index_add_`. Nothing else in the 93 layers moved: the fla KDA
  kernels, the eager MLA attention, the grouped GEMMs, and the host-side `lm_head` all gave the
  same bits.

The kernel-level read is
[grouped-nf4-gemm#408](https://github.com/pjordanandrsn/grouped-nf4-gemm/issues/408). The
combine was replayed at K3 geometry on this card: 50 identical calls gave 50 different fp32
outputs. The fix, #410, gives each `index_add_` call unique rows, so each token's 16 terms add
in ascending expert id, bitwise the sequential loop. In that replay, torch's deterministic
`index_add_` equals the same sequential sum, which is why the two arms agree.

**What this changes.** 71.44 % is not a more correct answer than 71.20 %. It is the result of
one fixed summation order. Released-package runs keep drawing from the spread until a
grouped-nf4-gemm release carrying #410 is installed. With it, a K3 probability on this path
reproduces to the last bit on this box and build. Register row
`e4b.parity.kimi-k3.prefill-drift-is-the-combine.a2000.2026-09-28`.

**Receipts:** [`receipts/2026-09-28-det-ab/`](receipts/2026-09-28-det-ab/SHA256SUMS).

- The nine run directories: `k3_gen_n1_pin0.json`, `k3_gen_prefill.log`, `moe_trace.json`.
- `det_wrap.py`.
- `run_v5.sh` and `phase2.sh`: the runner inside the container.
- `orch_v5.sh`: the host side. It stops the sdxl sidecar to free the ~4.5 GiB of VRAM a run
  needs, and starts it again after (`host.log`: down 21:40:58 to 22:35:36Z).
- `compare_v5.py`, which writes `compare_all.txt`.
- `calls_v5.py`, which writes `calls_detail.txt`.
- `shadow/mxfp4_pipelined.py`: the file the fix processes imported (grouped-nf4-gemm
  `f180045`, sha256 `a601d7d1…`).
- The driver.

`phase2.sh` also ran #410's GPU tests, a must-fail control and the kernel replay; those receipts
are with grouped-nf4-gemm#410.

Read two things with care:

- **The `rc=` values in `status.log` are `date`'s exit status, not the runs'.** The runner
  expands `$(date …)` after the command, in the same string, and that resets `$?`. The evidence
  that a run completed is its own JSON and its `saved ->` line, present for all nine.
- **The fix processes' JSON `provenance.packages` still says grouped-nf4-gemm 0.33.4**, because
  that is the installed distribution. The module that actually ran is the one `moe_trace.json`
  names by path and sha256.

*On the release, the next day: see [§6](#6-on-the-released-grouped-nf4-gemm-0336-the-forward-reproduces).*

### 6. On the released grouped-nf4-gemm 0.33.6, the forward reproduces

grouped-nf4-gemm 0.33.6, published to PyPI on 2026-09-29, ships #410. `venv-k3rel` was upgraded
to it with `pip --no-deps` at 01:25Z. The freeze diff shows that as the only change
([`venv-k3rel.freeze.diff`](receipts/2026-09-29-rel0336/venv-k3rel.freeze.diff)), and the
installed `mxfp4_pipelined.py` is byte-identical to the `v0.33.6` tag's (sha256 `05c760e4…`).

Three prefill-only processes then ran, from 01:33 to 01:51Z, with deterministic mode off and
nothing shadowed. They used the same wrapper and driver as §5; both files are byte-identical to
that set's.

| processes | p(`' Paris'`) | prefill expert rows | MoE traces |
|---|---|---|---|
| rel1, rel2, rel3 | 0.7144126892089844, all three | 6,118 | identical at all 92 calls, to each other and to §5's deterministic and fix runs |

A released-package run on this box and build now gives one answer. The spread in §4 was the
0.33.4 combine. Register row `e4b.parity.kimi-k3.reproducible-on-gnf4-0.33.6.a2000.2026-09-29`.

**Receipts:** [`receipts/2026-09-29-rel0336/`](receipts/2026-09-29-rel0336/SHA256SUMS).

- The three run directories.
- `run_v6.sh`. This runner captures `rc` before it logs, so the `rc=` values in its `status.log`
  are the runs' own.
- The host side. `orch_v6.sh` waited for the sdxl sidecar to go idle: it was serving image
  generations in bursts. `orch_v6_now.sh` replaced it when the wait was called off. `host.log`
  shows the sidecar down from 01:33:24 to 01:51:32Z.
- `compare_v6.txt`.
- The freeze and its diff.
- The wrapper, the comparison script and the driver.
