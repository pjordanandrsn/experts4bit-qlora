# DQ11 A2000 correctness rehearsal

This opt-in path uses the real DQ11 L/U/U0 loaders, callable census, observer,
bitwise loss/gradient removal proof, scoring and shipped forty-update loop.
It produces `dq11-rehearsal-*` receipts. The scientific reducer refuses them;
no rehearsal timing or quality value licenses a DQ11 result or replacement draw.

The separate 101-wheel cu128 lock retains 80 scientific package versions.
`wheels.json` lists every changed or removed CUDA package/version. Torch 2.12.1,
torchvision 0.27.1 and torchao 0.18.0 have no cu128 build in the official index;
the coherent set is Torch 2.11.0+cu128, torchvision 0.26.0+cu128, torchao
0.17.0+cu128 and Torch's required Triton 3.6.0. Scientific locks are unchanged.

During the real Loggetta preparation only, a scoped adapter delegates to the
real `enable_dense_offload` with `min_bytes=0`. Tiny projections otherwise fall
below the default 1 MiB threshold, leaving no streamed bytes and failing the
engagement assertion. The adapter is refused in science mode and the original
export is restored immediately, including on preparation errors. Science keeps
its default threshold. This makes the tiny L arm exercise the same streaming
branch as full Mistral without changing Loggetta's API or production code.

The local random Mistral has 32 layers, hidden size 128, intermediate size 256,
64 vocabulary entries and all 224 seven-projection modules / 448 FP32 rank-16
adapter slots. Original ordered blocks are mapped modulo 64 into synthetic
tokens, retaining sequence length 2048, forty training blocks and eight blocks
for each quality check. This is explicitly synthetic correctness work.

Run only from the reviewed source closure in an owned QNAP Pool 2 checkout,
with a new workspace and the original canonical tokens file:

```bash
export DQ11_REHEARSAL=1 DQ11_REHEARSAL_TINY_MODEL=1
export DQ11_W=/work/new-owned-rehearsal
export DQ11_REHEARSAL_TOKENS=/work/canonical/tokens.json
# E4B_SHA is the reviewed full source commit; TC1_RUN_NONCE is a unique local
# rehearsal nonce; TC1_DEADLINE_EPOCH is the bounded local deadline.
bash bench/dq11/dq11_science_run.sh
```

A real single RTX A2000 12GB with CUDA 12.8 is required. At least 6 GiB must be
free; this process is capped at at most 3 GiB. Do not run the scientific VRAM
fill or transport probe, change the QNAP driver, or disturb household GPU users.
Claim `gpu:a2000` on the bus before execution, then release it after completion.
The runner creates a fresh Python 3.11 venv, performs the real hash-locked
bootstrap and source-authority installation, builds/seals its local model and
inputs, applies actual Loggetta admission, then executes all three proofs and
all six reads in the original order. Every phase respects deadline minus the
300-second reserve. JSON refusal/completion and nonce markers are preserved.

Rehearsal workspace filenames `wheels.json`, `locked_inputs.json` and
`science.sha256` let the original arm consume its separate inputs/closure.
`REHEARSAL.json` marks their purpose and `science-reference.sha256` preserves
the reviewed scientific reference. These files never replace scientific locks
or canonical assets in the checkout. A tiny-model flag without explicit mode
is refused. With rehearsal unset, execution tests compare the entire command
sequence and subprocess environment to the sealed 858e7f69 runner on success
and six refusal paths.

The A2000 rehearsal cannot cover cu130 wheel loading or sm_120 kernels. Those
remain part of the actual 5090 instrument, with the previous bootstrap evidence
preserved. Any fourth draw still requires its sibling amendment, remaining
budget and explicit maintainer ACK.
