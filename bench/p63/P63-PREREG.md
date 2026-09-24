# P63 — does a token's output depend on how many rows share its forward? (registered 2026-09-23, before the run)

Record: [#708](https://github.com/pjordanandrsn/experts4bit-qlora/issues/708). Hand-off: grouped-nf4-gemm#393 /
lane B393 (`kernel/PREREG-b393-combine-reduce-bitwise.md`, read in grouped-nf4-gemm#397), whose outcome B sends the
end-to-end sizing of `combine_rows` here with `E4B_FUSE_COMBINE=0` as the control. Authorization: the work and one
rental within the standing caps, as relayed to this lane by the coordinating session on 2026-09-23; the owner's own
words belong in the PR that merges this registration.

## Question

A token decoded alone (T = 1) and the same token inside a multi-row forward (a speculative verify, a prefill, a
batched step) should get the same output if the kernels' arithmetic does not depend on the row count. llama.cpp lost
exactly this when it fused its MoE weight-and-sum (ggml-org/llama.cpp#29168: greedy speculative decoding stopped being
byte-identical). Here the question is wider, because several e4b routes pick a different kernel by row count on
purpose. Which routes are exact, which differ only in summation order (REORDER), and which compute a different
function at T = 1 than at T > 1 (PRECISION)? And how large is the difference at the logits?

This is not a speed lane and moves no default. It answers #708's ask and #359's open item (2): which batch-shape
variance is the kernels' and which is the model's.

## The routes, verified against the code

Read at e4b `94842a2` (main at registration; #708 cites `c2637e2`, one commit earlier with the same code) and
grouped-nf4-gemm `66d41c8` (main), whose probed kernels (`int4_b32.py`, `nf4_grouped.py`, `int4_smallm.py`,
`int4_pack_ref.py`) are byte-identical to v0.33.0, the e4b CI pin. Line numbers are main's.

| route | T = 1 | T > 1 | selected by | operand models (T = 1 / T > 1) |
|---|---|---|---|---|
| int4 expert store, default | `singleton_groups` (`hot_residency.py:730`) → `gemv_int4_b32` on `quant_x_rows` int8 activations (`:363-374`) | host-grouped "prefill / verify" branch (`:376-396`): `dequant_int4_ref`, `.to(bf16)`, bf16 matmul per expert | T | int8 x · exact int4 grid / bf16 x · bf16-rounded grid |
| int4 store, `DEVICE_GROUPING=1` | as above | `_int4_gemv_decode` (`:161-162`): the same GEMV for ≤ 256 routed rows; the captured grouped int4 GEMM above (`:329-362`) | `hot_residency.DEVICE_GROUPING[0]` (module flag; harnesses set it) | int8 · exact at both; the GEMV's split-K is planned from its rows (`int4_b32._plan`, gnf4 `:181`), with the row term only on parts with ≤ 64 SMs (`SPLITK_R_TERM_MAX_SMS`, `:162`) |
| int4 store, `FORCE_SINGLETON_GROUPS=1` | as above | the singleton GEMV at `T * top_k` rows | `hot_residency.FORCE_SINGLETON_GROUPS[0]` (the S2 verify default, `--moe-grouping singleton`) | int8 · exact at both — **and a latent memory-safety defect, below** |
| NF4 experts, default | decode GEMV (`gnf4 nf4_grouped.py:1237`): dot-pad bf16 MMA at the Qwen3 census shapes on ≥ 160-SM parts (`:1254`), else the scalar fp32 GEMV | M-tile `tl.dot` on fp32 operands = TF32 (`:822`) whenever some expert gets ≥ 2 rows | T | bf16 x · bf16(lut·absmax) or exact / bf16 x · TF32(lut·absmax) |
| NF4, `FORCE_SINGLETON_GROUPS=1` | decode GEMV | the decode GEMV at `T * top_k` rows; the scalar path's split-K is planned from its row count (`_decode_plan`, `:1018`; the sm ≥ 160 K1 winners are keyed on T = 8, `:1052`) | module flag (+ `GNF4_GEMV_DOTPAD=0` for the scalar path) | same model; split-K can differ |
| NF4, `DEVICE_GROUPING=1` | decode GEMV | captured M-tile (`gemm_4bit_grouped_captured`) | module flag | as default |
| int4 attention (`Int4Linear`) | GEMV, int8 activations (`int4_attn.py:181`, `GEMV_ROWS_MAX = 1`) | 2–16 rows: K16 small-M GEMM, bf16 MMA on in-register-dequantised weights (`:189`); > 16 rows: cuBLAS on the cached bf16 weight (`:197`) | rows; `E4B_ATTN_INT4_SMALLM` (auto = on with gnf4 ≥ 0.32) | int8 · exact / bf16 · bf16-rounded (K16 and cuBLAS share one model) |
| MoE combine | `combine_rows` (`hot_residency.py:744`; gnf4 `int4_b32.py:744`): one program per token, slot order | same kernel, same program per token | `E4B_FUSE_COMBINE` (default 1) | fp32 products, fp32 sum at every T; B393 read it is not bitwise the torch chain |
| decode folds (not in #708's list) | `rmsnorm_rows`, `rmsnorm_resid_rows`, `rope_norm_heads`, `router_epilogue` | the same kernels up to **64 rows** (`glue_fuse.py:96`, `glue_r2.py:49`, `router_epilogue.py:35`); the upstream torch chains above | rows; `E4B_FUSE_T1_GLUE[_R2]`, `E4B_FUSE_ROUTER_EPI` | a 17-row verify stays on the kernels; a 160-row prefill does not. (Round 1's per-head q/k norm counts heads × T rows and would leave its kernel at T = 3 for Qwen3's 32 q heads, but on the `int4` stack round 2's rope-norm fold serves the attention up to 64 rows first) |
| router epilogue output dtype (not in #708's list; found by the rehearsal) | `router_epilogue` returns **fp32** routing weights (gnf4 `int4_b32.py:1033`; e4b `router_epilogue.py`'s `softmax_topk` branch passes them through) | above 64 rows the upstream router: `router_top_value.to(router_logits.dtype)` = **bf16** | rows; `E4B_FUSE_ROUTER_EPI` | the combine weights each expert output by fp32 weights at ≤ 64 rows and by bf16-rounded weights above: PRECISION. The indices are the same function (fp32 softmax + top-k on the same bf16 logits) |
| attention core (not in #708's list) | HF: SDPA, one query row; paged (the serving path, `paged_attention.py:163`): the fp8 decode kernel | HF: SDPA with a mask; paged verify (`:211`): the SAME fp8 decode kernel, staggered lengths; paged prefill (`:242`): SDPA over the staged **bf16** K/V | backend | paged decode/verify read fp8 K/V, paged prefill reads bf16: a PRECISION difference by design. The fp8 kernel plans `n_split` from its row count (gnf4 `fp8_paged_attn.py:1407-1425`), capped by the block table — at this fixture's short table the cap binds at both row counts |
| dense bf16 GEMMs (not in #708's list) | cuBLAS at M = 1 | cuBLAS at M = T | — | the router gate (`F.linear`), the lm_head, and the NF4 stack's bf16 attention: same operands, cuBLAS picks its kernel by M |

**What says otherwise.** `hot_residency.py:233-236` says the singleton path's "per-row arithmetic is identical to the
grouped path, so the outputs are bitwise-equal -- pinned in CI". That is true at T = 1, where every group has one row
and both take the decode GEMV. `:238-240` extends it to T > 1 ("the M dimension does not participate in any
reduction"): true of the math; in arithmetic the group size picks the kernel. `tests/test_singleton_groups.py` pins it
"through a mocked GEMM so CI needs no CUDA", so a kernel switch is invisible to it.

**A defect found while mapping (fixed on this branch, separately reviewable).** `enable_serve_experts_int4` sizes each
store's split-K buffer `st["part"]` for one token's `top_k` rows; the singleton branch handed it to the GEMV at every T.
Under `FORCE_SINGLETON_GROUPS` at T > 1 — the S2 verify default on the int4 store — the GEMV indexes `sk * T * top_k`
rows of it. On the NAS A2000 (sm_86, gnf4 `66d41c8`), with the buffer a view into a larger sentinel-filled one, T = 17
wrote 16,384, 12,288 and 180,224 fp32 elements past it at OLMoE's gate_up (2048, 2048) and Qwen3's gate_up
(1536, 2048) and down (2048, 768) shapes; T = 1 and T = 2 wrote nothing past it
([`rehearsal-a2000/part_oob.json`](rehearsal-a2000/part_oob.json)). On a > 64-SM part the plan keeps its N-only split
count at every row count, so by the plan's own arithmetic a 17-token call there indexes 17x the rows the buffer
holds (derived, not measured). The fix (`hot_residency._int4_part_or_none`) keeps the
buffer when it has the rows the call's own plan needs and passes None otherwise, so the wrapper allocates (through the
graph pool under capture). T = 1 is unchanged. `tests/test_int4_singleton_part_fits.py` pins it on CPU. The probe
refuses the int4 singleton sub-arm on a cut without the fix.

## Instrument

[`p63_probe.py`](p63_probe.py) builds ONE stack per process and runs its sub-arms. [`p63_compare.py`](p63_compare.py)
holds all the comparison arithmetic (CPU-tested: `tests/test_p63_compare.py`); [`p63_reduce.py`](p63_reduce.py)
applies this registration (`tests/test_p63_reduce.py`); the capture plumbing is tested on a tiny random Qwen3-MoE on
CPU (`tests/test_p63_probe_plumbing.py`).

- **Stacks** (built as `step_decomp` builds the shipped stack: NF4 through the arena, all-VRAM hybrid tier with
  `collapse_resident`, then the lanes; the P42 hook is NOT used, the probe applies the lanes itself and refuses if the
  hook is active):
  - `nf4` — NF4 experts, bf16 attention, no folds (P59's NF4 control).
  - `int4` — the shipped int4 lane: RTN int4 experts packed from the source checkpoint, RTN int4 attention (K16
    route auto), folds r1/r2/router epilogue on, q/k/v fused on Qwen3-MoE (the current default, P59).
  - `int4nf` — the same with the folds off, so a prefill's switch from the fold kernels to the torch chains at 64
    rows can be told apart from the rest.
- **Sub-arms** (route selection by configuration only: the two module flags, `E4B_FUSE_COMBINE`, `GNF4_GEMV_DOTPAD`, the
  attention backend):

  | stack | sub-arms |
  |---|---|
  | nf4 | `hf.default`, `hf.singleton`, `hf.device`, `hf.combine0`, `hf.singleton.dotpad0`, `paged.default` |
  | int4 | `hf.default`, `hf.device`, `hf.singleton`, `hf.combine0`, `paged.default`, `paged.device` |
  | int4nf | `hf.default`, `hf.device`, `paged.device` |

  `hf` = HF `DynamicCache` + SDPA (the K8/KL oracle's attention); `paged` = the serving path's `PagedAttentionContext`
  over one `Fp8PagedKV` slot, the model run with `use_cache=False` and explicit positions (`paged_runner.py`'s
  contract): decode mode for T = 1, verify mode for the windows, prefill mode for the prefill.
- **Fixture.** [`fixture.txt`](fixture.txt), an original passage written for this lane (sha256 in every receipt),
  tokenised by the model's own tokenizer; the first **L = 160** tokens. Token ids and their sha256 are recorded.
- **The control** is incremental eager decode: tokens 0..159 each through their own T = 1 forward over the carried
  cache — never a full-sequence forward.
- **Modes.** Verify: for p0 ∈ {128, 96, 64, 32} (descending, so on the paged backend no verify overwrites a later
  window's prefix) and W ∈ {17, 16}, the control's cache cut to p0 (a deep-copied `DynamicCache` snapshot; a paged
  `rewind`), one forward over tokens p0..p0+W−1. W = 17 is #708's verify and above every 16-row bucket; W = 16 is the
  K16 bucket. Prefill: one 160-row forward from an empty cache (above the 16-row, 64-row and 256-routed-row lines).
- **Reading 1 — end to end.** Per token, per decoder layer, six sites in forward order (`attn_in`, `attn_core` = the
  o_proj input, `attn_out`, `mlp_in`, `mlp_out`, `layer_out`), the final norm and the logits: bit equality, max |d|,
  relative L2, max bf16 ULP over the significant elements, logits argmax and KL(P_control ‖ P_test) in fp64 over the full
  vocabulary, and the control's top-1 margin at each flip. The first differing (layer, site) of each token localises
  where the row count first changed a bit. Row 0 of each window sees exactly the control's cache, so its first
  difference is purely per-op.
- **Reading 2 — module replay ("same bytes in").** During the control each target module's T = 1 inputs and outputs
  are recorded; the first n ∈ {1, 16, 17, 160} are then stacked into ONE n-row call. Row i must be token i's T = 1
  output bit for bit, or the module's arithmetic depends on the row count. Targets: every layer's experts module (the
  DISPATCHED one), and in each stack's first sub-arm also its router, input norm, attention projections and the lm_head.
  Single-GEMM modules at layers 0, mid and last and the lm_head are also scored against fp64 (below). n = 1 must
  reproduce the recording: that is the replay's own fidelity check.
- **Reading 3 — kernel census.** Layer 0's gate_up GEMM on the control's own layer-0 rows and routing (row t·k + j =
  token t against its j-th expert: exactly what the module computes), through every kernel path the dispatch can take:
  int4 GEMV, int4 dequant + bf16 matmul, int4 grouped GEMM; NF4 decode GEMV (whichever the box's dispatch runs, read off
  gnf4's own `dispatch_counts`), NF4 M-tile, the scalar GEMV with `GNF4_GEMV_DOTPAD=0`. Each path at T = 1 per token and
  at T ∈ {16, 17, 160} tokens, and each against the exact fp64 product of its logical operands. Plus `combine_rows` and
  its torch chain on random rows at the family's (top_k, hidden), T ∈ {1, 2, 16, 17, 64, 160}.
- **Classes.** EXACT: every row bit-equal to its T = 1 call. REORDER: not bit-equal, and both paths apply the same
  operand roundings (only the summation order differs). PRECISION: the paths round operands differently (int8 vs bf16
  activations, exact vs bf16 vs TF32 weights) — different functions. The operand model of every path is read from the
  kernel source (`p63_compare.OPERAND_MODELS`, the table above) and registered here.
- **Accuracy is the DEFECT line, not a classifier.** Each path's error against fp64 is normalised by the worst-case bound
  its declared model allows: `2^-8 |y| + (1 + 2^-8)[(K + 2) u_acc S + u_w S + (1 + u_w) E]`, S = Σ|x·w|, E the
  activation model's per-element error summed against |w|, u_acc = 2^-23 (tensor-core adds are not guaranteed
  round-to-nearest). Ratio > 1: the path is worse than any correct implementation of its own model — B393's case C,
  flagged `DEFECT?`, inspected before anything is claimed. `rel_rms_err` is reported beside it, descriptive only.
  For cuBLAS paths (`p63_compare.CUBLAS_PATHS`: the int4 dequant + matmul, `Int4Linear`'s cached-bf16 route, dense
  bf16) torch's default `allow_bf16_reduced_precision_reduction = True` lets cuBLAS round split-K partials to bf16,
  which no fp32-accumulation bound covers. Those calls are rerun with the flag off; the DEFECT line reads that ratio,
  and the default-flag ratio and whether the flag changed any bit are reported beside it. The readings themselves run
  under the defaults, which are what is served.
- **Determinism gate (G0).** In each stack's first sub-arm the control is run twice and one verify window twice; both
  must be bit-identical, and every module replay at n = 1 must reproduce its recording. A stack failing G0 is an
  instrument fault: nothing from it is read (rc 14).

### What was found while building the instrument (disclosed: it changed the design)

- **ULPs are not a metric here either.** B393 found bf16 ULPs unbounded near cancellation. Per-layer hidden states
  have near-zero elements in every row, so a max-ULP per layer is dominated by them. The line for exactness is bit
  equality; magnitudes are relative (rel L2, max |d|); ULPs are reported only over elements ≥ 2^-4 of the row RMS
  (`tests/test_p63_compare.py` pins the near-zero case).
- **The accuracy bound cannot separate operand models at served shapes.** The first draft classified REORDER vs
  PRECISION by whether both paths sat inside the exact-operand bound. Its accumulation term grows as K·S, and at
  K = 2048–4096 it exceeds the whole error a bf16-rounded weight adds (~√K·2^-8·|t|): a bf16-weight path sits inside
  the exact-weight bound. So the classes come from the kernels' source, and the bound decides only DEFECT
  (`test_the_bound_cannot_see_a_bf16_weight_at_served_k_which_is_why_it_is_not_a_classifier` pins the blind spot).
- **End to end cannot attribute a MoE route by its first differing layer.** Every stack has a row-count-dependent
  route in layer 0's attention (int4 GEMV vs cuBLAS; bf16 cuBLAS by M; SDPA by mask; the norm fold at 64 rows) before
  the first MoE layer runs, so every sub-arm's first difference is predicted at layer 0. The module replay and the
  kernel census carry the per-route attribution; the end-to-end reading sizes the total.
- **Three more, found by the rehearsal below:** the router epilogue's dtype change (a route, now P7), the need for a
  cuBLAS accuracy model (torch's default bf16 reduced-precision reduction; cuBLAS paths are rerun with it off for the
  DEFECT line), and the split-K bit-invariance of the int4 GEMV when every split owns one KU iteration (P1's text).
  The probe also stopped letting one module's replay error discard a sub-arm.

### The rehearsal (NAS RTX A2000, free; NOT the lane's reading)

Receipts and the full account: [`rehearsal-a2000/`](rehearsal-a2000/README.md). OLMoE-1B-7B-0924 from the LAN model
store on the NAS RTX A2000 (sm_86, 26 SMs), all three stacks and their fifteen sub-arms, this branch's tree,
grouped-nf4-gemm `66d41c8`, 14.7 min of GPU under the shared lock. It is a plumbing rehearsal: a different family,
shapes and SM class, and B393 showed sm_86 and sm_120 differ in these very kernels' bits. What it read, stated so the
predictions below can be seen not to have been fitted to it:

- **G0** held on every stack.
- **Census:** int4 GEMV → itself EXACT (1,280 / 1,280 rows at T = 160, although its split count changed from 16 to
  1); → dequant + bf16 matmul PRECISION (0 / 1,280, rel L2 1.0e-2); → grouped int4 GEMM REORDER (1,035 / 1,280). NF4
  scalar decode → itself EXACT, → M-tile PRECISION. `combine_rows` EXACT across T and inside B393's bound (0.995).
- **Replay:** the experts module is exact at every n under singleton (int4 with the fix, NF4), at 16 / 17 and not 160
  under device grouping on the int4 store, and never on the default routes; attention projections, router and lm_head
  never above one row; the torch RMSNorm is row-invariant; the fold kernel is exact to 17 rows and not at 160.
- **End to end:** no mode exact anywhere; every first difference at layer 0 (`attn_core`, or `attn_in` in the prefill
  of the one stack with the norm fold on); KL mean 5.2e-4 to 4.3e-3 nats/token, top-1 ≥ 0.9688.
- **`E4B_FUSE_COMBINE=0`:** KL mean 4.6e-4 (NF4) and 1.2e-3 (int4), one flip in 160 each.
- **Defect scan:** 227 records, none outside its model's bound with fp32 split-K reduction; 35 cuBLAS records were
  outside it under torch's default flag, by up to 7.35×.
- Its three REFUTED rows are all `GNF4_GEMV_DOTPAD=0`, whose predictions are the 5090's (on 26 SMs the default decode
  already is the scalar path).

## Family and box

**Qwen/Qwen3-30B-A3B** @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` on one RTX 5090 (32 GB, 170 SMs, sm_120).

- It is the family every int4 serving claim was measured on (`e4b.serve.b16.qwen3-30b.int4.5090`, P54/P57/P58/P59),
  so "which routes are exact" is asked of the configuration that is actually served.
- It fits: NF4 experts ~16 GB; the int4 store replaces them (the NF4 stacks are freed); P59 served the same stack on
  this class.
- It exercises every route: 128 experts, top-8 (so a 17-token call routes 136 rows, and a 160-token prefill 1,280 —
  above `_int4_gemv_decode`'s 256); gate_up (1536, 2048) and down (2048, 768) are the shapes `_DOTPAD_CONFIGS` names,
  so on a ≥ 160-SM part its NF4 decode runs the dot-pad kernel and `GNF4_GEMV_DOTPAD=0` selects a different one; its
  per-head q/k norms take the rope-norm fold; its router takes the softmax-top-k epilogue; its attention fuses q/k/v.
- The int4 pack is RTN from the source checkpoint (`enable_serve_experts_int4`, P59's `int4` arm), not the licensed
  GPTQ artifact (P55x): the question is arithmetic, not quality, and RTN costs no calibration pass.

## Predictions (written before the lane's data; code-derived for the 5090)

Cross-architecture caveat, from B393: neither `combine_rows`' bits nor torch's own chain are the same on sm_86 and
sm_120. The A2000 rehearsal's pattern need not carry, and nothing below is read from it.

- **G0** — every determinism check bit-identical; every replay at n = 1 faithful.
- **P1 — kernel census, at T ∈ {16, 17, 160}:**
  - int4 GEMV at T = 1 → int4 GEMV at T: **EXACT** (> 64 SMs: the split count is the N-only plan at every row count;
    one program per row; partials reduced in split order). This is the `DEVICE_GROUPING` ≤ 256-row route and, with the
    fix, the singleton route. (On a ≤ 64-SM part the row term changes sk, but a change of sk is still bit-invariant
    when every split owns one KU iteration: the reduce adds the partials from zero in split order, which IS the
    one-program sequential sum. The A2000 rehearsal read EXACT for that reason at OLMoE's gate_up; it is a property of
    the plan at a shape, not of the part.)
  - int4 GEMV → dequant + bf16 matmul (the default T > 1 route): **PRECISION**.
  - int4 GEMV → grouped int4 GEMM (`DEVICE_GROUPING` > 256 rows): **REORDER** (the same int8 · int4 operands and exact
    int32 block dots; the fp32 scale sums group differently).
  - NF4 decode → NF4 decode at T·k rows (the singleton route): **EXACT** (the dot-pad kernel: one program per row, no
    split-K unless `GNF4_GEMV_SPLITK` is set).
  - NF4 decode → M-tile (the default and device routes): **PRECISION** (bf16 vs TF32 weights).
  - scalar NF4 GEMV (`GNF4_GEMV_DOTPAD=0`) at T = 1 → at T: **REORDER** (the K1 winner splits gate_up 16 ways at 8
    rows; at 128+ rows the universal plan does not split).
  - every path inside its own model's bound (no `DEFECT?`; for cuBLAS paths, with fp32 split-K reduction).
- **P2 — module replay:** experts module EXACT at n ∈ {16, 17} under `DEVICE_GROUPING` and NOT at 160; EXACT at every n
  under singleton (int4 with the fix; NF4); NOT at every n on the default and `combine0` routes, on the NF4 device
  route and on `singleton.dotpad0`; the int4 attention projections, the bf16 projections, the lm_head and the router
  NOT (cuBLAS picks its kernel by M); the int4 stack's input norm EXACT at 16 and 17 (the fold kernel) and NOT at 160
  (the torch chain). The HF torch RMSNorm carries no prediction (torch's reduction layout depends on the row count or
  not; either is read).
- **P3 — combine census:** `combine_rows` EXACT at every T (one program per token, slot order, a block size that does
  not depend on T). The torch chain carries no prediction. Both inside B393's bound.
- **P4 — T = 1 controls across sub-arms:** bit-identical to `hf.default`'s for `hf.singleton` and `hf.device` (the
  flags act only at T > 1); NOT for `hf.combine0` (B393: reorder-class), `hf.singleton.dotpad0` (a different T = 1
  kernel) and every `paged.*` (fp8 K/V).
- **P5 — B393's end-to-end size:** `hf.combine0` against `hf.default`, T = 1 controls: **held** if KL mean ≤ 0.01
  nats/token and argmax flips ≤ 3 % of positions; **refuted** if KL mean > 0.03 (a quantisation-tier size — NF4 vs
  bf16 experts measured 0.029); between, stated as such. Expected first non-equal layer: 0.
- **P7 — router output dtype by row count** (registered from the code after the rehearsal found it, before any 5090
  data): on the `int4` stack (`E4B_FUSE_ROUTER_EPI=1`) the router's replay output dtype differs at n = 160 and not at
  n ∈ {16, 17}; on `nf4` and `int4nf` it never differs.
- **D — the defect scan:** no record, in the census, the module replay or the combine census, is outside its model's
  bound (cuBLAS paths read with fp32 split-K reduction).
- **P6 — end to end:** no sub-arm is exact in any mode, and every first difference is at layer 0 (above). Size band:
  KL mean ≤ 0.02 nats and top-1 ≥ 0.95 (P59's reorder-class read was 0.0044 / 97.7 %); the shipped KL-from-checkpoint
  bar (≤ 0.10 nats, top-1 ≥ 0.93) is the outer limit a route's own T = 1 vs T > 1 difference may reach.

## Decision rule

- **¬G0** → the stack is an instrument fault; nothing from it is read; no second box on a disappointing result.
- **An EXACT route** (P2 and P1 agreeing) → registered as a measured claim at its shapes, box and cut: dispatch
  routes in this repository's `docs/claims.json` (`e4b.serve.p63.qwen3.<route>.row-exact.5090.<date>`), kernel
  properties in grouped-nf4-gemm's register (kernel first). Each gets a GPU test asserting `torch.equal` across the row
  counts it was read at (skipping on CPU), beside `tests/test_singleton_groups.py`, whose docstring is corrected to say
  it pins the dispatch algebra, not the arithmetic. `hot_residency.py:233-240` is corrected to the measured statement.
- **A REORDER route** → licensed the way P59 licensed fused q/k/v: a measured reorder-class claim with its row counts,
  fraction of rows differing, max relative L2 and bound ratio. Nothing may claim bit-identity through it (speculative
  identity, B=1 vs B=16 parity). Its quality licence stands.
- **A PRECISION route** → the route computes a different function at T = 1 than at T > 1. Recorded in
  `docs/STATUS.md` with the end-to-end size; each path keeps the quality licence it was measured on, and no licence
  transfers from one row count to the other. If its own T = 1 vs T > 1 difference exceeds the shipped KL bar (P6's
  outer limit) the T > 1 path is filed as a quality issue with its own lane. Whether an exact alternative (for the int4
  store: `DEVICE_GROUPING`'s GEMV up to 256 rows) should become a default is a separate lane with its own speed and KL
  gates; P63 moves no default.
- **P7 held** (the router epilogue's fp32-vs-bf16 routing weights) → the fold is PRECISION-class by row count;
  recorded in `docs/STATUS.md`, and a follow-up issue asks whether the fused router should round its weights to the
  module's dtype as the upstream router does (a one-line change that would make the two row counts the same function;
  it needs its own KL read, not this lane's).
- **`DEFECT?`** (a path outside its own model's bound) → filed against the kernel before anything is claimed (B393's C).
- **A refuted prediction** is reported as refuted, with the mechanism it disproves; no prediction is re-derived after
  the data.
- **P5** is B393's end-to-end size, recorded on grouped-nf4-gemm#393 / the B393 claim either way.

## What this lane does not say

No bf16 reference model (61 GB does not fit); no perplexity; one fixture, one family, one box. The paged attention's
row-count-planned `n_split` is capped by this fixture's short block table at both row counts, so a long-context
difference there is not covered. B=16 batched decode (sixteen sequences, one row each) is a different shape from a
16-row verify (one sequence): the census's T = 16 speaks for its kernels, the end-to-end reading does not.

## Box, cost, receipts, exit codes

- **Box:** one RTX 5090 on Vast verified/secure, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. The runner refuses
  any other class (rc 15) and a dud box (rc 10).
- **Download:** Qwen3-30B-A3B at the pinned revision (~61 GB bf16, the HF token staged from `~/.config/hf/token`,
  never on a command line), baked to an NF4 arena on the box (P39's `k8_bake.py`); nothing else.
- **Proving rental first (the standing rule for any guard > 1 h).** One RTX 5090 under the same provider class and
  image, ≤ $0.15 and ≤ 10 min, running the launcher's trivial command (`nvidia-smi`) at the SAME e4b commit on a clean
  tree: it proves attach, pre-flight, command handoff, receipt, ledger row, teardown proof and zero live afterwards.
  The registered run launches only after it returns OK; a failed proving run is a row and a defect, not a retry; and
  nothing merges to e4b `main` between the two (a proving run is evidence for one commit).
- **Cost:** the registered run under a **1.5 h guard** at ≤ $0.66/h, so ≤ $1.00; with the proving rental ≤ $1.15;
  lane ceiling $1.50, hard stop $2. Expected ~50 min: install ~8, fetch ~10, bake ~5, `int4` ~10, `nf4` ~7,
  `int4nf` ~7. A 1 h guard was considered and rejected: the runner refuses to start a stack within 10 minutes of the
  deadline, and ~50 minutes leaves no room for a slow fetch.
- **Runner:** [`p63_drive.sh`](p63_drive.sh) (controller: staged-file pin, nonce handshake, heartbeat/liveness, fetch)
  and [`p63_run.sh`](p63_run.sh) (box). e4b is installed at the driver's own HEAD (refused if dirty); grouped-nf4-gemm
  at `GNF4_SHA` (required; **`f88df1e`**, grouped-nf4-gemm#397's merge, or later — the first cut carrying the side
  diagnostic below; its probed kernels differ from the rehearsal's `66d41c8` only in two docstrings). The staged pieces
  are pinned by [`staged.sha256`](staged.sha256), checked in CI by `tests/test_p63_staged_pin.py`. Dry runs before
  registration ([`rehearsal-a2000/runner_dryrun.txt`](rehearsal-a2000/runner_dryrun.txt)): the driver's dry run and its
  refusals (no launcher environment, no `GNF4_SHA`, a drifted staged piece, a dirty tree: rc 78 each), and the box
  script staged flat in a GPU-less container (staging verified, then DUD BOX rc 10 with its nonce, exit-code and
  TP_DONE markers written; a drifted piece rc 9). The staged probe itself ran on real CUDA in the rehearsal.
- **Receipts:** `out/<stack>/p63_arm.json` + `p63_detail.pt` per stack, `RESULTS-p63-generated.md` + `p63_rep.json`
  (the reducer on the box), `summary.txt`, `versions.txt`, `forensics.txt`, `logs/`, the teardown proof, into
  `bench/p63/receipts/`.
- **Side diagnostic (disclosed, not part of the decision):** after the reducer the runner runs grouped-nf4-gemm #397's
  `kernel/receipts-b393/a2000-fma-attribution/fma_attribution.py` from the clone at `GNF4_SHA` (present at `f88df1e`),
  into `fma_attribution_5090.json`: what sm_120's `combine_rows` computes. Guarded by an existence check, non-fatal, it
  never changes the exit code.
- **Exit codes:** 78 refusal (config), 9 stage/install/tripwire, 10 dud box, 15 wrong class, 11 fetch, 12 bake,
  14 instrument fault (G0), 3 a stack errored, 30 no time for a stack, 42 a stack wrote no receipt. The verdict is
  read from the reducer's JSON, never from the exit code.

Amendments, dated, go below this line before any data is read.
