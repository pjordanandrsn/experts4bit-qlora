# DQ11 draft: dense adapter fusion headroom and shipped-path position

Issue #1083, DQ1 ranked item 2. This lane asks whether building a fused
dense LoRA path in e4b is worth further implementation work. The Unsloth
fusion-off ablation estimates a lever's headroom on its own backbone; the
Loggetta comparison tests our actual user path. These answer different
questions. e4b does not currently ship fused dense LoRA; its expert fusion
does not establish that capability.

**DRAFT_NOT_LAUNCHABLE.** This PR fixes the prospective question, recipe,
quality bar, decision and orchestration. Its executable box path is a
stubbed CPU dry run only. No scientific worker or GPU experiment is
authorized by merging this draft. A reviewed instrument amendment must
implement and seal the science closure below, run its CI dry run, and merge
before any box. No data have been collected for DQ11. DQ10 remains frozen:
its model/source/driver pins, reserve hypotheses and refusals do not move.

## Prior art searched before drafting

- [Han, Unsloth Mistral benchmark, 2023-12-14](https://unsloth.ai/blog/mistral-benchmark):
  dense NF4, seven projections, rank 16, dropout zero, sequence 2048,
  accumulation and checkpointing. Its open-source comparison includes
  changes to casting, attention, RoPE, normalization, loss and manual
  autograd. The published ablation motivates isolating adapter fusion;
  whole-framework ratios cannot identify its contribution. Training loss
  is not our heldout quality gate. Proprietary variants are excluded.
- [Han, Hugging Face guest article, 2024-01-10](https://huggingface.co/blog/unsloth-trl):
  59 runs on T4/A100 across four datasets, all attention/MLP adapters,
  rank 16 and checkpointing, with Transformers 4.36/Torch 2.1.1 SDPA.
  This is evidence that the dense lever is already shipped elsewhere,
  not a prediction for RTX 5090 or a quality receipt for our versions.
- Its [Mistral/Alpaca notebook](https://huggingface.co/datasets/unsloth/notebooks/blob/aa69f6646ed471e34759dde278dd5249d62df15f/Alpaca_%2B_Mistral_7b_full_example.ipynb)
  formats instruction/input/response plus EOS, uses all seven projections,
  dropout zero and a short fixed-step training recipe. It does not supply
  the two-text non-inferiority test required here.
- [Hugging Face PEFT's Unsloth example](https://github.com/huggingface/peft/blob/main/examples/sft/run_unsloth_peft.sh)
  specifies packed Mistral QLoRA, all seven targets, NF4 double quantization
  and BF16. Its dropout 0.1 is ineligible for the pinned fused adapter
  bindings, so it cannot serve as our fusion-on control. This moving example
  is method context, not an instrument pin.

The old Colab baseline/open links on the vendor article require sign-in
in this environment; their hidden contents are not asserted as inspected.
No published ratio is reused as a measured DQ11 result.

## Subjects, sources and arithmetic

One public `mistralai/Mistral-7B-v0.1` checkpoint, revision
`27d67f1b5f57dc0953326b2601d68371d40ea8da`, and its unchanged tokenizer.
No remote model code, credential, prequantized vendor replacement, RoPE
extension, context ladder, smaller fallback model or model search.

The baseline uses Loggetta `34ecb6cec6f43a6f8607ff9f192749fdc7b587e9`:
an actually admitted dense NF4/stream plan, `dense_train.prepare`, and the
shipped `experts4bit_train.train_loop`. Keep the development-executor opt-in,
mandatory streamed admission margin, non-reentrant checkpointing, pinned
homes, late-bound frozen-weight backward and `train_prefetch=True`.
Require one nonempty offload handle per decoder layer and no streamed
trainable adapter. Planner refusal ends the draw; never construct a fake
feasible plan or use DQ10's reserve policy. This Mistral/recipe is outside
DQ7's measured subject range: retain that warning and infer no capacity.

The instrument amendment's reviewed merged e4b commit is derived from the
clean launcher checkout and compared to installed VCS metadata, never a
future SHA typed into this draft. GNF4, if installed for that environment,
is the released CI pin `724ccc454f006c1a46836e434e997f31f293747f`.
Register Python 3.11, Torch 2.12.1+cu130, Transformers 5.5.0,
bitsandbytes 0.50.2, PEFT 0.21.2, Unsloth 2026.9.14 and Zoo 2026.9.9.
The two Unsloth wheel hashes in `dq11_protocol.json` were independently
verified from PyPI. No package fallback. The amendment must pin all
resolved transitive wheels, including attention dependencies, before draw;
it must not let one arm silently install a different Torch/TF/PEFT/bnb.

| Arm | Backbone/setup | Adapter path |
|---|---|---|
| L | Shipped Loggetta dense NF4, streamed | Actual PEFT-on-bnb forward, e4b streaming as above |
| U | Pinned Unsloth, resident NF4 | Eligible QKV/O/MLP fused bindings on every layer |
| U0 | Same Unsloth loader/backbone/settings as U | Restore only original QKV/O and MLP adapter bindings |

U0 must explicitly restore `original_apply_qkv`, `original_apply_o` and
the MLP class forward (or `_unsloth_forward` if tiled), following the
pinned source's fusion-decline logic. Do not enable FSDP to induce it, set
dropout, modify targets, or switch checkpointing. A tiled MLP or another
late patch that cannot be audited identically produces VOID, not fallback.
Other Unsloth patches remain common to U/U0. U/U0 measure this adapter
implementation bundle, including dequantization/autograd and cast/add
semantics, not a universal cost of two LoRA GEMMs. L/U also differ in
residency and broader framework operations; their ratio is whole-path only.

All arms store A/B in FP32, as the Loggetta setup selects. This does not
mean FP32 adapter compute in every arm. In pinned U, `matmul_lora` casts
A/B to the activation dtype and adds their product into the base result
with `addmm_`; its custom backward also casts factors. L uses the actual
PEFT bnb wrapper's compute/cast path and rounds the delta to base-result
dtype before adding. U0's executed PEFT path must be observed rather than
assumed. Different add/accumulation rounding is expected; cross-arm
bitwise equality is not the quality bar. No FP32-storage receipt may be
described as FP32 per-operation compute without observation.

After every final loader/patch step, call #1538's
`bench/dq1/adapter_path_audit.py::audit_adapter_paths` and record current
forward/apply_qkv/apply_o module, qualname, source SHA and runtime-code SHA,
Python version, A/B storage dtypes and declared compute. Add explicit MLP
`forward`/`_unsloth_forward` identities with `callable_identity`. Match
against the amendment's reviewed wheel/source manifest and required
bindings; refuse unknown/empty or partial engagement. Capture again after
proof and after training. Bindings alone do not prove execution. Separate
untimed proof processes must record all layer bindings firing forward AND
backward and per-operation operand/result dtypes for delta GEMMs/adds.
Then remove observers for timed processes and verify identical census.
Globals/closures/bound values outside the helper must be sealed separately.

## Matched initialization, data, work and scoring

Use the same source shard bytes. Quantize the same BF16 decoder tensors
with NF4/block64/double quantization, BF16 base compute, and retain a
canonical per-projection hash of packed bytes and all quantization state.
All three loaders must agree; identical model names or source revisions
alone do not prove identical quantization. Record norm/embedding/head
values and dtypes separately. Unsloth's legitimate operation differences
are declared; changed frozen values, target set or tokenizer produce VOID.

Rank 16, alpha 32, all seven Q/K/V/O/gate/up/down projections, zero dropout,
bias none, no DoRA/RSLoRA/LoftQ. Seed 3407. Generate one canonical FP32
adapter initialization file before reading any trained loss: A uses the
shipped PEFT initializer, B is zero. Copy exact keys, shape, dtype and bytes
to every arm; per-process seeding is insufficient. Optimizer state starts
empty in each process. Register the shipped torch AdamW defaults
(betas 0.9/0.999, epsilon 1e-8, weight decay 0.01), LR 2e-4 constant,
no LR warmup, gradient clip 1.0, no autocast, TF32 off, cache off.

Fixed sequence 2048, micro-batch 1, accumulation 1, forty optimizer
updates per arm. Forty distinct packed blocks; all 2047 shifted labels
per block are trained, no pad/mask/packing differences or repeated data.
Updates 1..5 are timing warmup but remain in the trained model. Updates
6..40 are timed by the same shipped synchronized loop in all arms. No
separate trainer, optimizer, loss-chunk patch, or loss-substitution wrapper.
Full-logit semantics are required; record Unsloth's internal loss path
separately. Compile/load/proof/quality time are outside steady-step
timings but inside wallclock, end-to-end cost and load peaks.

`yahma/alpaca-cleaned` revision `12567cabf869d7c92e573c7c783905fc160e9639`:
retain original row order, format instruction/input/response with the
notebook template and EOS, tokenize without extra special tokens. Hold out
the last 512 original rows before building the forty training blocks from
the preceding rows. Concatenate within each split, cut nonoverlapping
2048-token blocks and use the first 40 train / first 8 heldout blocks.
The second scoring text is the first eight concatenated 2048-token blocks
of `Salesforce/wikitext`, `wikitext-2-raw-v1`, test, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Keep empty rows in source
identity; join rows with newline. No truncation/repetition if insufficient.
Seal source-file, row-selection, tokenizer, train and both heldout token
hashes and label counts in the instrument amendment before launch.
The draft contains no invented token/checkpoint hashes.

Train and score six fresh isolated processes on one physical card, order
L/U/U0/U0/U/L. Every process starts from the same initial adapter and
empty optimizer; these are two repetitions of one seed, not six seeds.
Use a common heldout scorer: explicit shifted-token FP32 cross entropy on
returned logits with no labels passed to framework loss functions, then
PPL=exp(total NLL / exact shifted-target count). Score initial and final
checkpoints on both texts, retain NLL/count/PPL and every step time/loss.
Record final adapter hashes; repetitions may differ numerically and must
both be kept, with no best-run selection.

## Quality first, decision and predictions before data

Apply the existing **K8 uncalibrated |delta PPL| <= 0.05** budget, unchanged,
on each scoring text, initially and after 40 updates, for L/U and U0/U
within each reversed-order repetition. Finite values, exact scored token
hash/counts and matched step count are prerequisites. A finite quality
failure is **QUALITY_FAIL**, not VOID; retain timings but make no position
or fusion recommendation. Initial quality failure stops before training.
Training loss alone never licenses a position claim or instruction quality.

For a valid, quality-passing draw, report median updates 6..40 separately
in each process, raw step distributions, each paired ratio, allocated and
reserved training peaks, driver/load peaks and end-to-end time/cost.
Let fusion headroom H=(median(U0)-median(U))/median(U0).
**BUILD_CANDIDATE** only if H >= 0.05 in both repetitions and both pairs
pass quality. Otherwise **DEFER_FUSION**. Five percent is an ex ante
engineering priority cutoff, not significance, a noise floor, guaranteed
e4b gain or permission to ship a new path.

Report L/U whole-path ratio independently. Unsloth faster by at least 5%
in both quality-passing repetitions licenses only a bounded position on
this model/recipe/card. If order reverses the sign or either improvement is
below 5%, report **NO_STABLE_POSITION**. No population confidence interval
or extrapolation to large models, other links, seeds or cards. A future
e4b implementation requires its own matched arithmetic/correctness review.

Prospective predictions: U will beat streamed L by more than 5% in both
repetitions; U will beat U0 by 5..15%; eligible U will perform BF16 adapter
GEMMs despite FP32 storage. Quality passing is uncertain because arithmetic
differs. If U/U0 fail quality, precision isolation is the next question,
not a widened gate. If quality passes and H <5%, dense fusion is deferred.

## VOID, lifecycle and draft closure

VOID: mismatched source/runtime/quantization/init/data/work; missing or
changed loaded or executed bindings; partial fusion/stream engagement;
different U/U0 non-adapter setup; observer left in timed code; nonfinite
loss/gradient/scoring; missing/duplicate arm; OOM/refusal/deadline before
the full valid pair; undisclosed loss/attention/checkpoint fallback; or
unproved teardown. Keep every partial/refusal receipt, never resize,
retune/retry a failed draw without a reviewed amendment. A valid negative
result stays negative.

One normal 32 GB **RTX 5090**, secure provider under existing policy,
two-hour hard wallclock, 320 GB ordered storage, 100 GB transport allowance,
one draw `dq11-5090-1`. Total runtime/storage reservation <=$1.70 plus
transport <=$1.10 = **$2.80 maximum**, comfortably below the maintainer's
$15 no-ask ceiling. Actual quote/role/global policy admission still bind;
refuse a higher reservation, another card, or a draw without teardown
guard. Do not import DQ10's exact-driver restriction: record the admitted
driver, PCIe link and power limit; refuse non-normal VRAM/card or failed
standard VRAM/egress probes. No offer/rental query belongs to this draft.

The executable dry run exercises closure verification, nonce markers,
preflight -> prepare/seal -> binding/precision proof -> initial quality ->
the six train/final-quality stages -> quality-before-decision -> teardown
proof, including refusal/failure branches. All emitted values are stub
events; it cannot create scientific PASS, a position verdict, compute,
HTTP/SSH traffic or measurements. CI runs `tests/test_dq11_draft.py`
without framework/GPU imports or optional skips.

Launch remains blocked until a separate instrument amendment supplies
real workers, a complete resolved environment, canonical data/init hashes,
binding/precision observations in an untimed proof contract, and executable
quality/reducer checks. CI must then run those same box branches against
external stubs, not just this event-plan fixture. Review and merge precede
all box work. Record launch nonce, immutable source/instrument/manifest
hashes, actual invoice and guarded teardown plus verified instance absence.
Publish later scientific receipts in a separate results PR; exclude model
weights, caches, plural `adapters/`, credentials and private bus addresses.
