# Solutions — one page per problem

Each page answers one ordinary problem in the same order: the direct
answer, the symptoms people search for, why it happens, which of the two
projects solves it, the canonical install command, the smallest correct
example, the observable result, the supported scope, the limitations, and
the evidence (active claim IDs in [`claims.json`](claims.json), or the
statement that the page describes a capability without a measurement).

| the problem | page |
|---|---|
| `load_in_4bit=True` loads my MoE but it still OOMs; the experts stay bf16 | [bitsandbytes-moe-load-in-4bit-still-ooms.md](solutions/bitsandbytes-moe-load-in-4bit-still-ooms.md) |
| I need QLoRA / LoRA on the fused experts themselves | [qlora-fused-moe-experts.md](solutions/qlora-fused-moe-experts.md) |
| The quantised experts still do not fit VRAM | [run-moe-larger-than-vram.md](solutions/run-moe-larger-than-vram.md) |
| The experts do not fit host RAM either: serve or train from NVMe | [offload-moe-experts-to-cpu-or-nvme.md](solutions/offload-moe-experts-to-cpu-or-nvme.md) |
| Serve a large MoE on one consumer NVIDIA GPU | [serve-large-moe-on-a-consumer-gpu.md](solutions/serve-large-moe-on-a-consumer-gpu.md) |
| My experts are native MXFP4 (gpt-oss, DeepSeek-V4) | [mxfp4-moe-training-and-residency.md](solutions/mxfp4-moe-training-and-residency.md) |

## Which of the two packages

- **`experts4bit-qlora`** (this repository, `pip install experts4bit-qlora`)
  owns model loading, quantisation orchestration, adapters, training,
  residency integration and serving.
- **[`grouped-nf4-gemm`](https://github.com/pjordanandrsn/grouped-nf4-gemm)**
  (`pip install grouped-nf4-gemm`, or through `pip install
  "experts4bit-qlora[fast]"`) owns the packed-expert kernels: the grouped
  NF4 and native MXFP4 GEMMs, the int4 decode GEMV and its calibrated
  packer, the FP8 paged attention, the decode glue, and the host/NVMe arena
  primitives. Questions about a kernel's layout, its Triton requirements,
  a CUDA/Triton kernel change, or checkpoint-byte provenance go there; its
  own solution index is
  [docs/SOLUTIONS.md](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/docs/SOLUTIONS.md).

Lookup aliases (`e4b`, `e4b-qlora`, `experts4bit`, `expertsnbit`, `experts-mxfp4`) install this package;
always install and cite `experts4bit-qlora`.

## Environment, in one line

Linux, an NVIDIA CUDA GPU, torch ≥ 2.2, bitsandbytes ≥ 0.43; transformers
≥ 5.0 for the streaming loader (`[train]`); grouped-nf4-gemm ≥ 0.28.0 and
Triton on an sm_80+ GPU for the kernel path (`[fast]`). CI tests Python
3.11. No macOS, Windows, ROCm or XPU. The current position, including what
is measured-private and what is open, is [`STATUS.md`](STATUS.md).

## Limitations that apply to every page

- Linux + NVIDIA CUDA only; Python 3.11 is what CI tests. The kernel path
  needs Triton on an sm_80+ GPU.
- Nothing falls back silently: an unsupported family fails fast with a
  named error, and every `enable_*` returns a count or a non-empty handle
  list, or raises — the caller asserts it.
- A model that already fits in bf16 with headroom gains nothing here:
  4-bit is a memory trade, and on the measured comparator it cost energy
  (`e4b.train.energy-honest.scoped-a2000`).
- Numbers live in [`claims.json`](claims.json) with their status; a page
  quotes claim IDs, never figures, and a retired claim is never current.
