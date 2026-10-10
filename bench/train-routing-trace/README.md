# Training routing trace: which experts a QLoRA step touches, and what a residency tier would hold

**Status:** the harness only. No trace has been captured yet and no number is claimed. The capture runs once this
lands, from the merged commit, on the project's RTX A2000 12 GB, with no rental and no timings.

## Why

Every Qwen3-30B-A3B expert-locality number so far comes from **decode** traces (#1469). The training-side residency
evidence is on OLMoE. This directory measures the training side for Qwen3-30B-A3B:
- how many distinct experts a training micro-batch touches per layer;
- how much of that set carries from step to step;
- at which VRAM expert budgets a cache would hold the working set.

Counts and bytes only.

## The plan (fixed before capture)

| | |
|---|---|
| model | `Qwen/Qwen3-30B-A3B @ ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| recipe | the TC3 A2000 secondary arm, `fused_attn4_m_offload_mb1`: `OFFLOAD_EXPERTS`, micro-batch 1 × accumulation 8, gradient checkpointing, TC1's matched adapters and tokens (register `e4b.train.frontier.qwen3.a2000-12gb.2026-10-02`) |
| size | 20 optimizer steps = 160 micro-batches (a 5-step fallback if the slot is short, stated in the record) |
| recorded | per (step, micro-batch, pass, layer): the router's top-8 expert ids for every token; pass = `fwd` or `recompute` |
| checks | **completeness guard** (`--expect-mb`, the accumulation): a trace is VOID unless every step holds exactly that many micro-batches, each with every layer in BOTH passes, nothing recorded after the last optimizer step, and records exist at all. `save()` writes the verdict, `run_capture.py` exits 3 on VOID, and `replay.py` refuses a VOID trace and re-checks the file itself. Forward ids == recompute ids for every micro-batch (`differ` is VOID unless the difference is the finding) |
| replay | rows = (layer, expert), 6,144 in total; budgets {128, 384, 768, 1024, 3216, 6144}; policies: profile top-N (fitted on steps 1–5, scored on 6–20), LFU with LRU tie-break, LRU, Belady |
| outputs | hit rate and staged bytes per step **per pass** (`fwd`, `recompute`, `dgrad`) and pooled. The dgrad touches each layer's set right after its recompute, so pooled alone is inflated. Also distinct experts per layer and step-to-step Jaccard |
| row bytes | one expert's packed NF4 (gate, up, down) plus fp32 absmax per 64 weights. Cited: `engines/offload.py:4-5,251` (the staged uint8 packed and float32 absmax), `engines/offload.py:737-743` (double-quantized absmax refused under offload), `loader.py:1177` (blocksize 64; the TC1 harness passes none and records `nf4/64`). `OFFLOAD_EXPERTS` itself stages whole layer stacks; the row figure prices a row-granular tier of the same layout |
| controls | a shuffled-routing null (same set sizes, random members, the forward set kept for recompute and dgrad); the #1469 Qwen3 decode traces through the same replay |
| not here | timings or any speed statement; index-vs-profile `TRAIN_VRAM_FRAC` (a separate comparison); any change to placement code |

## Files

- `routing_recorder.py`: forward hooks on every `layers.<i>.mlp.gate`. It records the top-k ids per pass, checks
  forward against recompute, and runs the completeness guard. `install_backward_labelling` covers both
  `torch.Tensor.backward` and `torch.autograd.backward`.
  - The dgrad runs on the recompute's routing, so the recompute set is also the dgrad set. This is by construction, not
    separately hooked.
  - A forward with gradients on and no backward after it is reported as `no_recompute`, and it makes the trace VOID.
- `run_capture.py`: runs `bench/tc1/tc1_arm.py` **unchanged**, with the recorder attached to the model `load_e4b`
  returns. Router calls inside any backward are labelled `recompute`, and the optimizer step closes a step. It writes
  `trace.npz` plus `trace.npz.meta.json`, which holds the environment, the model's config and index hashes, the
  harness arguments and the summary with its verdict. It exits 3 on VOID.
- `replay.py`: replays a training trace or a #1469 decode trace through the four policies at the six budgets. CPU only.
- `tests/test_train_routing_trace.py`: CPU tests.
  - **The forward/recompute check is armed:** a toy router whose recompute of one micro-batch routes differently is
    reported as exactly that micro-batch and layer.
  - **The completeness guard is armed by two cases:**
    - a model whose MoE block routes from `gate.weight` without calling the hooked module (zero records);
    - a training loop that reaches backward through an unlabelled `torch.autograd.backward`, so the recompute is
      mislabelled as new micro-batches.
    Both must come out VOID, and replay refuses them. With the labelling installed, the same `torch.autograd.backward`
    path is complete.
  - Per-pass reporting is tested: the dgrad pass hits fully at a budget that holds a layer's set.

## Run

```sh
# on the A2000 host, from this commit, with the TC1 harness's own arguments for the mb1 offload arm
python bench/train-routing-trace/run_capture.py --trace-out /path/trace.npz --expect-mb 8 -- <tc1_arm.py arguments>
python bench/train-routing-trace/replay.py --trace /path/trace.npz --expect-mb 8 --fit-steps 5 --out replay.json
python bench/train-routing-trace/replay.py --trace /path/trace.npz --expect-mb 8 --fit-steps 5 --null-seed 1 \
    --out null.json
python bench/train-routing-trace/replay.py --decode <gnf4>/bench/cold-engine/routing-trace/qwen3_prose.jsonl \
    --fit-steps 128 --out decode-prose.json
```
