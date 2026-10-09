# CUDA paged-server smoke

From the checkout being tested, in a Linux CUDA environment with e4b's serving
dependencies and current grouped-nf4-gemm installed:

```bash
PYTHONPATH=. python bench/smoke/gpu_serve_smoke.py --output-dir /tmp/e4b-serve-smoke
```

Use a new output directory each time. The command reports PASS/FAIL for each of
four families (`qwen3_moe`, `granitemoe`, `mixtral`, and hybrid `qwen3_5_moe`),
first with the default NF4 stack, then with the applicable int4 stack.
Qwen3-MoE, GraniteMoE and Mixtral require expert and attention int4. Qwen3.5's
served text tower currently lacks an expert-int4 source convention
([#1485](https://github.com/pjordanandrsn/experts4bit-qlora/issues/1485)); its cell
requires attention int4, keeps experts NF4, and explicitly prints expert-int4
**UNSUPPORTED** with the convention's refusal text. Once that convention exists,
the same cell also requires expert int4. Execution failures never become skips.
Any failure, including a missing CUDA device, worker crash or the whole-suite
five-minute deadline, returns nonzero. `summary.json` records the exceptions,
source package digest, actual residency classes, lever engagement, graph
decisions and generated token IDs; each cell also keeps its worker log.

To exercise expert-int4 source reads from an actual local checkpoint directory
(requires the local-directory reader fix in
[#1490](https://github.com/pjordanandrsn/experts4bit-qlora/pull/1490)):

```bash
PYTHONPATH=. python bench/smoke/gpu_serve_smoke.py --int4-source local --output-dir /tmp/e4b-local-int4-smoke
```

The option applies to int4 cells and is forwarded to every isolated worker.
Default and fusion cells retain their existing source selection. Every cell
records the selected checkpoint source; int4 cells also record `int4_source`.
The default `offline-hub` mode retains the synthetic cache fixture for Hub-reader
coverage and testing historical source revisions. Both modes use the same
local random weights and stay offline.

Two additional `folds` cells enable all four fusion knobs at `auto` on
Qwen3-MoE and Mixtral, for ten cells in all. Both require residual licensing on
every served MoE layer, no partial licence or probe error, and nonzero glue/r2
engagement. Qwen3's shipped default and int4 cells also assert that licence;
its default allowlist already enables these folds.

Every cell creates a tiny seeded random checkpoint, tokenizer and NF4 arena.
The synthetic checkpoint lives in a private, content-addressed offline HF cache
fixture because the server's int4 reader uses `snapshot_download`. These fixture
IDs name no upstream model. Network/model downloads are disabled before worker
imports. Solver bandwidth inputs are synthetic and unused after the server's
default all-VRAM placement override. Each cell runs in a fresh process with
inherited `E4B_*` knobs removed, calls the real `serve_paged.build_engine`, checks
that both layers use `_HybridTier`, then finishes two requests, each with a
512-token prefill and four output tokens. The second request exercises slot
reset, including the hybrid's linear state. No server implementations are
mocked or replaced.
The composite Qwen3.5 checkpoint also includes visual and MTP distractor tensors,
including a full MTP decoder block with its own experts. Both the arena bake and
the served build must see exactly the text tower's two MoE layers by module name.

**Graph coverage depends on the GPU.** The shipped `auto` default resolves to
eager decode on the sm_86 A2000: its result explicitly says graphs were not
exercised. On sm_89+ the same command requires every default decode bucket to
capture and at least one graph replay during generation; eager fallback fails
the smoke. Prefill graph capture or its refusal is reported as the server
resolves it. This is a build/generation correctness check with random weights,
not a quality, capacity or speed result.

## Container recipe

Use a CUDA-capable host, Docker's NVIDIA runtime, and a C compiler for the
hybrid tier's compile-at-first-use native CPU kernels. This recipe installs
dependencies during setup; the smoke itself stays offline:

```bash
docker run --rm --gpus all --ipc=host -v "$PWD":/src -w /src \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel bash -lc '
    apt-get update && apt-get install -y build-essential git
    python -m pip install "bitsandbytes==0.50.2" "transformers==5.18.0" \
      "grouped-nf4-gemm==0.44.0" -e ".[serve]"
    PYTHONPATH=. python bench/smoke/gpu_serve_smoke.py --output-dir /tmp/e4b-smoke
  '
```

For a lane with an already pinned environment, run only the one-line command in
that environment; do not replace its dependencies. The source checkout selected
by `PYTHONPATH` is what is tested, and the output records its package digest
separately from installed distribution metadata. GNF4 provenance records the
actual imported kernel module paths and content hashes, plus native C sources;
its installed version is labelled as metadata. The kernel digest includes loaded
flat modules sharing the actual kernel source directory (which may include other
flat modules in a shared site-packages directory); their names make that scope
explicit. Run under the shared GPU's
resource claim. Keep correctness receipts separate from code changes.
