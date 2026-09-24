# Results — P66: what residency adds per token in launches and syncs, against the all-resident step, by cold fraction (RTX 5090, 2026-09-24)

Pre-registration: [`P66-PREREG.md`](P66-PREREG.md) (#731, Amendment 1 in #732; both merged before any data).
Record: [#711](https://github.com/pjordanandrsn/experts4bit-qlora/issues/711). Receipts: [`receipts/`](receipts/).

**How the numbers were produced.** The reducer ([`p66_reduce.py`](p66_reduce.py)) ran on the box and wrote
[`receipts/RESULTS-p66-generated.md`](receipts/RESULTS-p66-generated.md) and [`receipts/p66_read.json.gz`](receipts/).
Run again locally over the committed rows, it reproduces the generated markdown byte for byte except one float's last
digit (see [`receipts/README.md`](receipts/README.md)). Every number below is from those files, or is the arithmetic
named beside it.

## Runs

| run | what | outcome | cost | receipt (adertha-receipts) |
|---|---|---|---|---|
| `p66-prove-1` | proof | refused at the provider: cheapest verified 5090 $0.7237/h, above the registered $0.65/h | $0 | `426446b` |
| `p66-prove-2` | proof | refused: $0.6604/h above $0.65/h | $0 | `04b97f5` |
| — | **Amendment 1** (#732): $0.75/h, 0.2 h proof guard | | | |
| `p66-prove-3` | proof | refused: $0.8141/h above $0.75/h | $0 | `1e58bd8` |
| `p66-prove-4` | proof | NOT_RUN: the launcher's pre-flight hit an SSL handshake timeout on the Vast API; nothing ran | $0.0195 | `827686c` |
| `p66-prove-5` | proof | refused at validation: `p66-prove-4` was named as a machine exclusion, but its failure was the API, not ssh readiness; the run id is burned | $0 | `8d63453` |
| `p66-prove-6` | **proof** | **PASS** rc 0 with `P66_PROVED`; install, tripwire, pin; 8-stream egress 80.1 MB/s recorded | $0.0674 | `864e837` |
| `p66-5090-1` | reading | refused by the runner (rc 14): 8-stream egress 68.5 MB/s against the 80 MB/s floor, before any data | $0.0664 | `b2ddbfe` |
| **`p66-5090-2`** | **the reading** | **rc 0**, `P66_SUCCESS` | **$0.3752** | `f9903cc` |

- **Lane total:** $0.5285 against the $2 ceiling. Every rental's teardown is proven.
- **Box.** One RTX 5090 (32,607 MiB, driver 580.95.05, 600 W, 3,135 MHz, 170 SMs; PCIe gen 4 × 16 — `nvidia-smi` read
  gen 1 *current* at the pre-run forensics, the idle link state) on an AMD EPYC 7C13 (256 CPUs, 2 NUMA nodes, 818 GB
  RAM; cgroup 377 GB, 49 cores). Vast verified/secure instance 52436541. 8-stream egress 224.0 MB/s.
  - Calibration (`bench/calibrate.py`): device triad 1,568.3 GB/s; link H2D 23.07 / D2H 28.65 GB/s; NVMe sequential
    5.69 GB/s.
  - Unit host costs: aten launch 6.42 µs, Triton launch 13.17 µs, idle `cudaStreamSynchronize` 4.57 µs, `.item()`
    round trip 8.90 µs.
- **Software.** e4b 0.37.3 at `c85145f`, grouped-nf4-gemm 0.33.2 at `68a1250`, torch 2.8.0+cu128, Triton 3.4.0,
  transformers 5.16.1, Python 3.11, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- **Wall time.** 35.5 min from guard to teardown: install and egress 2.5, calibrate 2, Qwen3 fetch 5, NF4 bake 1.8,
  Level L with the gpt-oss fetch 16.1 (`ref` 0.6, `pipe` 3.3, `hyb` 7.1, gpt-oss fetch and bake 1.5, `mref` 0.5,
  `mpin` 1.0, `mnvme` 2.1), Level M 7.5, reduce under 1.
- **Authorization.** The owner, relayed at
  [#711](https://github.com/pjordanandrsn/experts4bit-qlora/issues/711#issuecomment-5815871057).

## Gates (P0): all six hold

| gate | read |
|---|---|
| G1 nesting | 39 / 39 windows |
| G2 simulator | 14 / 14 pipelined windows: the have-skip simulation equals the engine's counters (0 to 3,056 cold rows per window) |
| G3 graph correctness | 21 / 21 captured windows within `b_rel ≤ 1.5e-2`; eager and replay output hashes recorded |
| G4 device parity | 21 / 21: the captured token's work nodes equal the eager token's host submissions |
| G5 budget coverage | 39 / 39 tables at ≥ 90 % of Self CUDA |
| G6 record completeness | 39 / 39 under the 2 % bound; 162 device records dropped over the whole run |

The two paths registered as refusing capture refused, and the probe named the op: the hybrid tier at
`hot_residency.py:738` (`hot_row.nonzero`), the MXFP4 NVMe engine at `mxfp4_pipelined.py:430` (the padded-index
guard's `bool(...any())`), then its own `_resolve_src` refusal.

## Verdicts: P1–P6 held, P7 refuted, P8 and P9 informational

Per token, Qwen3-30B-A3B has 48 MoE layers and gpt-oss-20b 24. "Launches" are host kernel launches
(`cudaLaunchKernel` + `cuLaunchKernelEx`), "copies" `cudaMemcpyAsync`, "syncs" `cudaStreamSynchronize`.

### P1 and P2: the pipelined engine's count is fixed, and it is the registered count — HELD

| arm | windows | eager launches / copies / syncs per token | Δ vs `ref` per layer | captured (graph work nodes) |
|---|---|---|---|---|
| `ref` (all resident) | uniform | 384 / 0 / 0 | — | 384 kernel, 0 memcpy |
| `pipe` at h = 0.25, 0.5, 0.83, 1.0 | uniform, ctrl0, ctrl4, ctrl8 | **864 / 96 / 0** in every window | **+10 / +2 / 0** | 864 kernel + 96 memcpy; host side 1 graph launch + 2 staging copies |
| `pipe` at h = 0 | uniform | **720 / 96 / 0** | **+7 / +2 / 0** | 720 kernel + 96 memcpy |

- **P1.** Within every pipelined arm, the spread across windows is exactly 0 in every class, eager (launches, copies,
  syncs, memsets) and captured (work, kernel, memcpy, memset nodes). The windows span realized cold fraction 0, 0.5
  and 1.0 at a fixed hot set (`ctrl0/4/8`) and the uniform draw. Launches are identical across every `n_hot > 0` arm.
- **P2.** The delta is the registered one, and it sits where the registration put it: the `_fetch` region holds 8
  launches and 2 copies per layer (5 and 2 at `n_hot = 0`, where the row-index dispatch is skipped); the other 2 are
  the engine's unfused epilogue and combine chain against the reference's fused `_swiglu_rows` / `_combine_rows`.

### P3: the fixed tax, all hot and captured — HELD (band 0.5–1.5 ms/token)

| mode | device time added, kernels residency adds | shared-kernel shift (`_gemv_nf4_dotpad`, reported, not folded in) | submissions added | wall tax per token |
|---|---|---|---|---|
| captured | **1.184 ms** (576 rows, 2.06 µs each) | +0.011 ms | 0 | 1.401 ms (4.754 against 3.353) |
| eager | 1.233 ms | +0.005 ms | 576 | **8.054 ms** (26.301 against 18.247) |

- The eager step is host-bound: submit time equals wall (26.29 of 26.30 ms), so residency's 576 extra submissions cost
  the eager token six times what their device time is. The A2000 rehearsal read 0.49 ms of added kernels on 16 layers
  (192 rows, 2.6 µs each); the per-row cost on this box is 2.06 µs.

### P4, P5, P6: the MXFP4 engines and the hybrid tier read as registered — HELD

| path | per layer, per token (eager) | vs its all-resident sibling | fixed across windows and hot fractions? |
|---|---|---|---|
| `mref` (MXFP4 all resident) | 38 launches, 3 copies, 1 sync | — | — |
| `mpin` (pinned all-E arena) at h = 0, 0.5 | 38 / 3 / 1 | **Δ 0 / 0 / 0** (P5) | yes |
| `mnvme` (NVMe → pinned tier) at h = 0, 0.25, 0.5, 1 | 37 / 6 / **4** | **−1 / +3 / +3** (P4) | yes, spread 0 in 9 windows |
| `hyb` (hot in VRAM, rest NVMe) at h = 0.25, 0.5 | ctrl0 41 / 7 / **4** · ctrl4 (mixed) 75 / 20 / **10** · ctrl8 (cold only) 50 / 15 / **8** | — | **no** (P6): launch spread 1,632 per token |

- **P4.** The four syncs per layer are one each at `mxfp4_pipelined.py:430` (the padded-index guard),
  `mxfp4_residency.py:598` and `:644` (`_resolve_src`) and `:660` (`_invalidate`), 192 per window each.
  - The decision rule licenses a pinned-staging follow-up only if the two pageable H2D copies' measured host cost is
    ≥ 10 % of the token's wall. This instrument records counts and sync sites, not the host time of those copies, so
    **the follow-up is not licensed by this read**.
- **P6.** The tier's syncs follow the layer's composition exactly as registered (4 / 10 / 8). `cold_dest="deadline"`
  prices bytes only, so its GPU side has no term for the mixed-layer dispatch; recorded against `cold_deadline`
  in the follow-up below.

### P7: cold_deadline's bytes model against the measured transfer — REFUTED

The measure is the device time of the link-crossing work per token (`_gather_rows_*` UVA reads plus `Memcpy HtoD`);
the prediction is `cold_deadline.gpu_us` on exactly the rows copied with the box's calibration blob
(`b_link` 23.07, `b_vram` 1,568.3 GB/s). The MXFP4 engines' hot-row D2D re-copies are taken off their measure.

| path | windows | measured / predicted | implied rate over the rows copied |
|---|---|---|---|
| **pipelined, uniform (graded)** | h = 0 · 0.25 · 0.5 · 0.83 | **1.574 · 1.734 · 1.766 · 2.018** | 14.4 · 13.1 · 12.9 · 11.3 GB/s |
| pipelined, controlled | ctrl4, ctrl8 at h = 0.25, 0.5, 0.83 | 1.395–1.764 | 12.9–16.3 GB/s |
| MXFP4 pinned (recorded) | h = 0 uniform; h = 0.5 uniform, ctrl2, ctrl4 | 1.163; 1.551, 1.684, 1.284 | 19.6 GB/s at h = 0 |
| MXFP4 NVMe (recorded) | 7 windows | 1.445–1.655 | 13.6–15.6 GB/s |
| hybrid tier (recorded; pageable H2D from the tier) | 6 windows | 5.5–16.5 | 1.4–4.1 GB/s |

- Every graded ratio is above the 1.50 refutation line, so **P7 is refuted as registered**: bytes over the link plus
  bytes over VRAM under-predicts the pipelined gather by 1.57–2.02× on this box. The ratio grows as the cold fraction
  falls (fewer rows per layer).
- **What the receipt also holds, stated without a cause.** The box measured its link twice. The calibration blob's
  `b_link` (23.07 GB/s) is 40 back-to-back 64 MB pinned copies timed with CUDA events. The census's own probe, a
  single 64 MB pinned copy synchronized on each side (median of ten), read **14.72 GB/s**. The pipelined gather's
  implied rate at h = 0 is 14.44 GB/s. On the A2000 rehearsal (PCIe gen 3 × 8) the two probes agreed (6.14 against
  6.12 GB/s) and the same ratios read 0.97–1.02. So on this gen 4 × 16 link the gather runs at the single-copy
  rate, not the back-to-back rate the model was given. Which constant belongs in `cold_deadline`, and why the two
  differ here and not there, is not decided by this read; both numbers go to the follow-up.

### P8 (informational): the RFC-matched reading, 17.19 % cold (`pipe-0.83`)

| mode | residency's whole cost (`pipe-0.83 − ref`, wall ms/token) | fixed tax (`pipe-1.00 − ref`) | tax share |
|---|---|---|---|
| captured | 15.76 | 1.40 | **8.9 %** |
| eager | 6.92 | 8.05 | 116 % |

- Captured, the share is under the registered 10 % line: **in this engine, launches cannot explain an RFC-size (2–4×)
  gap; the gap would be link-side** — which is where P7 put it.
- Eager, the all-hot arm (26.30 ms) is slower than the 17 %-cold arm (25.17 ms): the step is host-submission-bound,
  and the cold gather's device time hides under it. A two-point eager comparison would book more than residency's
  whole cost as a fixed term.
- Either way this measures *this* engine, which adds 10 launches per layer; the RFC's two-bank forward adds about
  one.

### P9 (informational): Level M agrees with Level L exactly

| served arm (`step_decomp`, 12 profiled steps) | device ms/step | Δ per step vs `M_ref`: `cudaLaunchKernel` / `cudaMemcpyAsync` / `cudaStreamSynchronize` | Level L per token, same names |
|---|---|---|---|
| `M_ref` (all VRAM, collapsed) | 10.07 | — | — |
| `M_pipe-1.00` | 11.20 | +528 / +96 / 0 | +528 / +96 / 0 |
| `M_pipe-0.00` | 79.72 | +384 / +96 / 0 | +384 / +96 / 0 |
| `M_hyb-0.50` | 78.80 | +1,032 / +360 / +192 (96 `nonzero`, 24 `.item()`) | not run at Level L at this composition |

Coverage ≥ 99.2 % of Self CUDA on every table. The served all-hot step's +1.13 ms/step of device time is P3's tax
seen from the serving loop.

## What the decision rule did

- **P1 ∧ P2 → register the measured claim:**
  `e4b.serve.p66.qwen3.pipelined-residency-fixed-count.5090.2026-09-24`. The cost model gets a **per-step fixed
  term** — P3's device time when captured (`e4b.serve.p66.qwen3.pipelined-fixed-tax.captured.5090.2026-09-24`), and
  submissions × the host's unit cost when eager — and never a per-cold-expert launch term. Filed as a
  grouped-nf4-gemm follow-up on `kernel/cold_deadline.py` with these values; its docstring's exclusion list is wrong
  about kernel compute only in the fixed part.
- **P4 → structural.** No pinned-staging follow-up: the licensing measurement was not taken.
- **P6 → recorded** beside the P2 follow-up: `cold_dest="deadline"` omits the mixed-layer dispatch term. Registered
  with P4 and P5 as `e4b.serve.p66.gptoss.mxfp4-and-hybrid-sync-census.5090.2026-09-24`.
- **P7 refuted →** the transfer itself departs from bytes over link. **The model needs a measured efficiency factor
  (UVA-read efficiency, #105 candidate 4) before any RFC comparison is made.** The measured ratios are registered as
  `e4b.serve.p66.qwen3.pipelined-gather-over-cold-deadline.5090.2026-09-24` and go to the same follow-up, with the
  two link probes.
- **P8 ≤ 10 % captured →** link-side.
- Nothing changes a default.

## What this read does not say

- **Why the gather runs at the single-copy rate.** The link's idle state, NUMA placement of the pinned arena and
  the gather kernel's own access pattern are all candidates; none was measured.
- **PCIe contention.** A batch-1 token issues its transfers one layer at a time; P7 is the transfer in isolation.
- **Anything about vLLM's forward.** The RFC's 5.1 ms/token was not re-measured; this is the count and the transfer
  of this engine on one box.
- **Time bands on any other box.** Counts are structural and hold anywhere; P3 and P7 are graded on the 5090 only.
