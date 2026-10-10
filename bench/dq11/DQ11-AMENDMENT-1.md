# DQ11 amendment 1: sealed science instrument and controlled comparison

This prospective amendment follows the merged [registration](DQ11-PREREG.md).
It supplies the real workers, resolved installation, and canonical inputs that
the draft required before a box. It changes no update count, quality gate,
decision cutoff, model, source pin, arm order, price ceiling, or hard deadline.
The original draft and its dry-run closure remain an immutable record.
This amendment itself must merge after review before any acquisition or run.

## Scope of the question and every eventual readout

The U and U0 arms set `UNSLOTH_COMPILE_DISABLE=1` and
`UNSLOTH_RETURN_LOGITS=1`. Their headroom comparison tests the eligible adapter
fusion bundle in that shared configuration. Every preregistration interpretation,
decision, and eventual whole-stack claim is **against Unsloth in this matched,
eager, full-logits configuration, not Unsloth's default**. There is no U-default
arm. L/U describes the complete shipped paths under these conditions; it cannot
establish position against default Unsloth. The U/U0 bundle includes its actual
BF16 adapter GEMMs and addition/rounding, autograd and dequantization behavior;
it is not an isolated GEMM speed limit. No default-speed, capacity, shipping,
or cross-arm bitwise-equivalence claim is licensed.

## Installation and inputs sealed before readings

[requirements.in](requirements.in) records the resolution request.
[requirements.lock](requirements.lock) and [wheels.json](wheels.json) select
101 individual Linux x86_64 / CPython 3.11 compatible wheels with exact URLs,
versions and SHA-256 hashes, including Torch 2.12.1+cu130, Transformers 5.5.0,
PEFT 0.21.2, bitsandbytes 0.50.2, Unsloth 2026.9.14 and Zoo 2026.9.9.
The resolution used uv 0.13.0, `--python-version 3.11`,
`--python-platform x86_64-manylinux_2_28`, `--only-binary :all:`,
the official cu130 PyTorch index, and `--index-strategy unsafe-best-match`.
The committed result wins over future resolution. The box verifies every wheel
before installing locally with `--no-index --no-deps --require-hashes`.
The reviewed launcher derives e4b from its clean, merged HEAD; Loggetta remains
the registered `34ecb6cec6f43a6f8607ff9f192749fdc7b587e9` source object.
No GNF kernels are required by this dense instrument.

[locked_inputs.json](locked_inputs.json) records actual CPU-built bytes:

| asset | bytes | SHA-256 |
|---|---:|---|
| `adapter_init.safetensors` | 167816528 | `d481abefae51719e0f45111b3a404e97549fbe9db5d16556aafa313e7730515c` |
| `tokens.json` | 554866 | `078bc61ccdbb15e8b679663f4714583c6996cd4c51f95e250c5372890bd0f396` |

[dq11_build_inputs.py](dq11_build_inputs.py) uses the recorded CPU builder
versions and PEFT's actual `reset_lora_parameters(..., True)` initializer:
seed 3407, FP32 Kaiming-uniform A and zero B, seven projections per layer,
448 tensors and 41943040 elements. All arms copy those exact tensors rather
than relying on different model-construction RNG consumption. The registered
Alpaca prompt, EOS, original-order split, and Wiki newline concatenation are
encoded by Tokenizers 0.22.2 with automatic special-token insertion disabled.
The manifest seals raw source bytes, the 40 training blocks and both eight-block
heldouts. Their 2047 shifted targets per block give 16376 targets per heldout.
The builder downloads public tokenizer/config/data files only, never model
weights. The 168 MB initialization is an external sealed input, not a repository
binary. The seal and independent rebuild used CPython 3.11.17 on macOS arm64.
Rebuild it in a fresh directory with the recorded environment; a changed
asset hash refuses staging and execution.

[model_files.json](model_files.json) records the pinned checkpoint's files.
Small config/tokenizer hashes came from actual CPU downloads. The two weight
hashes are explicitly Hub-reported LFS hashes, not a claim of downloaded weight
verification. The box checks every file's size and SHA-256 before loading it.
It first downloads config, applies actual shipped Loggetta admission and memory
policy, and refuses before full weights if that admission fails. The normal
single 32 GB RTX 5090, bnb mirror tripwire, inherited VRAM/egress probes and
120 GB free-disk check must pass. DQ10's exact driver restriction is not reused.

## Executed proof, then clean workers

The box runs fresh proof processes L/U/U0 with no optimizer update, followed by
the registered six fresh read processes L/U/U0/U0/U/L. L uses
`dense_train.prepare` and its admitted streaming plan, preserving non-reentrant
checkpointing, pinned homes and prefetch engagement. Both U arms use the pinned
FastLanguageModel NF4 double-quantized BF16 backbone and adapters. U0 restores
the pinned original QKV/O bindings and class MLP forward; unknown, tiled or
ineligible bindings refuse. All read arms use Loggetta's actual shared
`experts4bit_train.train_loop`, sealed packed tokens, plain Torch AdamW, forty
updates, clipping 1, and five trained timing warmups.

The pre-update proof records loaded callable identities plus actual forward,
backward, adapter GEMM operand/result dtype and all 448 FP32 gradient witnesses.
Source authority comes from verified wheel source members and pinned Git blobs.
Loaded Python code is compared with compilation of those authorized source bytes
without executing a reference file; unknown closure or partial bindings refuse.
The audit does not infer execution from a qualified name. U must execute every
layer's fused QKV, O and MLP forward/backward. L/U0 must execute all 224 adapter
projections. The operand witness must observe BF16 forward rank GEMMs for U and
FP32 for L/U0. All frozen parameters' represented values and packed NF4/state
bytes must agree across arms and remain unchanged within each arm; floating
storage dtypes are recorded separately from canonical FP32 value hashes.

Observers are removed in `finally`, before any timed worker starts. In the proof
process the same initial forward/backward is repeated cleanly; its loss and all
gradient bytes must match the observed call. This is an observer neutrality
guard within an arm, not cross-arm quality or bitwise parity. A failure voids
the instrument without assigning a cause to a framework. Fused backward routing
uses the pinned plain `custom_saved_tensors` packed-weight tuple: it never
unpacks checkpoint-managed `ctx.saved_tensors` before the real backward.

Clean workers recheck bindings, canonical inputs, runtime versions, source
authority and frozen bytes. They record load seconds, load allocator/driver/host
peaks, process seconds, shipped training memory/timing and raw loss curves.
The clean process's initial quality must pass before its first update. Common
initial/final scoring uses full returned logits, explicit FP32 shifted CE and
exact target counts on both texts. No autocast or TF32 is allowed. Inherited
Unsloth switches refuse; only the two registered switches are set by the worker.

## Reduction and transport

[science.sha256](science.sha256) is the flat code/config transport closure;
[inputs.sha256](inputs.sha256) verifies the external payload after placement.
[dq11_science_stage.py](dq11_science_stage.py) checks closure and asset bytes
before transport. [dq11_science_drive.sh](dq11_science_drive.sh) requires a clean
merged checkout before using the existing TC1 transport and nonce lifecycle.
It does not acquire compute. [dq11_science_run.sh](dq11_science_run.sh) enforces
the launcher's hard deadline with a 300-second teardown reserve and bounded
phases, checks proof/initial quality before starting the six reads, and retains
logs and nonce markers on failure. No fallback, retry or replacement draw is
authorized. Any refusal needs review before another draw.

[dq11_reduce.py](dq11_reduce.py) applies the unchanged uncalibrated K8 absolute
perplexity budget of 0.05 to L/U and U0/U, initially and after forty updates, on
both texts and both repetitions. Finite quality failure gives `QUALITY_FAIL`
with no recommendation; malformed, partial, reordered or nonfinite receipts
give `VOID`. Complete quality-passing readings stay `AWAITING_TEARDOWN` until
the registered guard's proof verifies successful destruction and instance
absence. A supplied invalid teardown gives `VOID`. Only that final reduction
can yield `VALID_CONTROLLED_READ`, then the original two-repetition headroom
cutoff yields `BUILD_CANDIDATE` or `DEFER_FUSION`. Its scope field always carries
the controlled-comparison sentence above. A box's TC1 success marker means
receipt collection can finish; it cannot replace guard proof or license a claim.

Mandatory CPU tests exercise the real reducer, closure/staging, shell lifecycle,
source-code mutation refusal and checkpoint-safe observer cleanup with synthetic
fixtures. Those fixtures are test data, not scientific evidence. Local CPU
validation does not prove that the registered GPU path ran. This amendment
publishes no timing, quality, capacity or competitive result.
