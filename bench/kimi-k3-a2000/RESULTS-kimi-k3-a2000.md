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
  is a reference for the other.
- **The prefill took 270.5 s against 178.3 s.** This was not investigated.
  Decode steps agree within 1.2 %, and each one reads the same 108.8 GB of dense
  weights, so the gap is not in the dense path's steady state.
- **VRAM peak rose 0.25 GB** on the generation pass (4.07 → 4.32 GB) and did
  not move on the perplexity or routing passes. This was not investigated.

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
  band and not a benchmark.
- It is not a speed claim beyond this host. Dense bytes from the SSD dominate
  each decode step, so the step time belongs to the pool and not to the kernels.
- It is not an e4b load path. The driver builds the model on `meta` and wires the
  engines itself. `load_moe_4bit_streaming` is not involved.
