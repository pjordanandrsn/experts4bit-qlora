# DQ3 — a streamed frozen-weight QLoRA prototype: do overlapped dense-layer prefetches give bitwise-identical training at resident speed and a layer-count-sized VRAM saving?

Registered 2026-10-05, before any code or box. The lane name is claimed by `prereg/dq3`. Work item: #1083.

The owner's instruction (Claude Code desktop chat, 2026-10-05; relayed on #1083) is "register the streaming prototype and
coordinate with the maintainer". The maintainer session confirmed no conflicting work on `engines/dense_offload.py`, and
its gates are adopted below (marked **[M]**).

This lane is licensed by DQ2's registered consequence (C_ALIVE, [RESULTS-dq2.md](../dq2/RESULTS-dq2.md)): a streamed
frozen-weight prototype on a branch, gated on gradient parity, step time and a write-after-read fence. **No new
repository** — the repository gate in [SUMMARY-dq1.md](../dq1/SUMMARY-dq1.md) stands until this lane and its matched-work
successor exist.

## Stages

1. **Registration (this document, [DESIGN-dq3.md](DESIGN-dq3.md)).** No code.
2. **Implementation PR.**
   - An **opt-in, off by default** training-prefetch path in `experts4bit_qlora/engines/dense_offload.py`.
   - Its CPU and GPU tests: the fence race test with a mutation arm, the off-path byte-identity test, and the phase
     schedule tests.
   - The lane harness (`bench/dq3/`) and reducer, with self-test and mutants.
   - A $0 RTX A2000 rehearsal through the local pool: correctness and engagement only, no timing.
3. **The read.** One RTX 5090 on PCIe gen 5 x16, through `pod-launch.sh` with the PCIe band and
   `preflight_bandwidth: none`, in the standing no-ask tier.

No default changes in any stage. Turning the path on by default would need its own registered licence.

## Subject (fixed)

| | |
|---|---|
| model | `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137` (64 layers; DQ2's subject at full depth), loaded with transformers `BitsAndBytesConfig` (nf4, double-quant, bf16 compute) |
| adapters | PEFT `get_peft_model`, `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0)` on q/k/v/o/gate/up/down (so PEFT's `lora.bnb.Linear4bit`, fp32 adapters — asserted per module in the receipt) |
| step | micro-batch 1 × 2048 tokens (fixed token ids from a seeded generator, identical across arms), causal-LM loss, HF non-reentrant gradient checkpointing, AdamW (lr 2e-4), `torch.manual_seed(0)`, SDPA |
| streamed bytes | each decoder layer's `Linear4bit` packed weights (Parameters); `quant_state` tensors and norms stay resident (~7.6 MB/layer of absmax state) |
| host | RTX 5090, PCIe gen max 5 / width max 16 (refused otherwise, rc 13) |
| software | torch 2.8 / CUDA 12.8 image, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2, experts4bit-qlora at the launch commit |

## Arms (each a fresh process; palindrome **R S S0 S0 S R**)

| arm | what |
|---|---|
| **R** | resident: no dense offload |
| **S** | streamed with overlap: `enable_dense_offload(model, train_prefetch=True)` — the prototype |
| **S0** | streamed synchronously: today's grad-mode `dense_offload` (copy, then compute) — the no-overlap control |

Each arm runs two passes in one process, both from the same seed and the same initial adapter weights.

1. **Parity pass, deterministic.** 2 steps under `torch.use_deterministic_algorithms(True)`,
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` and the SDPA math backend. It records the loss and every LoRA gradient's sha256
   after each step. A fused attention backward may be nondeterministic, so bitwise parity is judged only here, where an
   R-against-R control proves the baseline is deterministic.
2. **Timing pass, realistic.** The default SDPA backend, no deterministic mode, from a re-initialised optimizer and
   adapters: 2 warm steps, then 6 timed steps.

The timing pass records:
- per-step wall time with a device sync;
- the loss (finite);
- the allocator peak;
- the pinned-host allocator's reserved bytes;
- the prototype's counters:
- prefetches issued, overlapped (copy event already complete when the consumer waited) and blocking, per phase
  (forward, backward) **[M]**;
- fence waits;
- residency high-water.

## The rule (written now; the reducer that implements it is stage 2's, with a self-test and mutants)

The verdict is the first of these that applies:

1. **VOID** when any of the following holds:
   - the device is not exactly `NVIDIA GeForce RTX 5090`, or the link is not gen 5 x16;
   - an arm is missing or errored, or the model revision or software differs across arms;
   - any arm's engagement is wrong: PEFT's bnb LoRA wrappers ×448 (7 × 64), fp32 adapters; S and S0 report their dense
     offload active over 64 layers; R reports none.
   Also VOID when R1's and R2's parity-pass hashes differ: the deterministic baseline is not deterministic, so bitwise
   parity cannot be judged.
2. **FUNCTION_FAIL** when, in the parity pass, the S or S0 loss at either step, or any LoRA gradient sha256, is not
   **bitwise-identical (`torch.equal`)** to R's **[M]**. A tolerance would hide a stale-slot fence bug. This is a
   defect; the lane stops, and nothing is timed or claimed. The timing pass's losses must be finite (otherwise
   FUNCTION_FAIL as well).
3. **NOISY** when the R self-pair (R2 / R1 mean step time) falls outside [0.97, 1.03].
4. **READ.** Graded:

| axis | rule |
|---|---|
| **step** | **PASS** if T(S) / T(R) ≤ 1.10; **SLOW** otherwise. |
| **coverage** **[M]** | **PASS** if S's steady-state blocking fetches are ≤ 2 per step (one per phase's first layer at most) and its overlapped prefetches are ≥ 2 × (64 − 1) per step, forward and backward both; **SCHEDULE_FAIL** otherwise. A forward-only schedule fails here. |
| **capacity** **[M]** | **PASS** if peak_alloc(R) − peak_alloc(S) ≥ 0.9 × the slot-count prediction (64 − 2) × 243.8 MB = 15.12 GB, i.e. ≥ 13.6 GB; **SHORT** otherwise. |
| **lane** | **PROTO_PASS** if step, coverage and capacity all PASS. |

Also reported, not graded:
- T(S0) / T(R), the overlap's value;
- the pinned-host reserved bytes against the 64 × 243.8 MB requested **[M]**;
- blocking fetches per phase.

## Predictions (stamped before any code; bases from rented-5090 data only **[M]**)

The bases:
- **DQ2 run 2** (gen 5 x16, 48.6 GB/s). Per layer at 2048 tokens: T_fwd 15.02 ms, T_bwd 31.96 ms, copy 5.25 ms (the
  copy moved packed + state; packed alone is ~5.0 ms). Rmin = 2.86, and the forward slows 1.6% under DMA.
- **Byte arithmetic** from the model config.

| reading | band |
|---|---|
| parity (deterministic pass) | R1 = R2 bitwise; S and S0 bitwise equal to R on both steps' loss and every gradient |
| T(S) / T(R) | [1.00, 1.06] (copies hidden with Rmin ≈ 2.9; DMA slowdown ≈ 1.6%; scheduler overhead small) |
| T(S0) / T(R) | [1.12, 1.35] (2 × 64 synchronous ~5 ms copies ≈ 0.64 s added to a ≈ 3.1 s step) |
| blocking fetches, S, steady state | ≤ 2 per step |
| VRAM saving | [13.6, 15.2] GB (slot-count prediction 15.12 GB) |
| pinned reserved / requested | [1.10, 1.25] (per-tensor power-of-two rounding of the seven homes: 1.135 computed) |

## Consequence (registered)

| outcome | action |
|---|---|
| PROTO_PASS | **DQ4**, a matched-work lane on a model larger than VRAM on the 5090 (a dense ≥ 50B model), streamed QLoRA against FSDP-QLoRA with CPU offload. The opt-in stays off by default; turning it on is a separate registered question. The repository gate is re-examined only after DQ4. |
| step SLOW, coverage PASS | Report. One amendment may change the schedule (e.g. prefetch depth 2); a second SLOW closes the overlap line at this card's link. |
| coverage SCHEDULE_FAIL | The schedule is wrong (backward not covered). Fix in a new implementation PR and re-run; this is not a timing result. |
| capacity SHORT | Report what stayed resident (the allocator breakdown); the capacity claim is restated to what was measured. |
| FUNCTION_FAIL | Stop. Defect issue, fence or schedule fix, a new rehearsal and a new registration. Nothing ships. |
| VOID / NOISY | One re-run on another gen 5 host; a second stops the lane as an instrument finding. |

## Correctness tests (stage 2, all in CI or on the A2000)

- **Fence race test with a mutation arm [M].** It mirrors grouped-nf4-gemm #467's: a sleep kernel delays the consumer so
  that a copy into a still-read slot would land early. With the fence the gradients stay bitwise-equal; the mutant
  (fence wait removed) must fail.
- **Off-path byte identity [M].**
  - With `train_prefetch` off, every existing `tests/test_dense_offload.py` test passes unchanged.
  - A new test asserts the off path creates no side stream and no fence event.
  - `tests/test_dense_and_nvme_compose.py` and `tests/test_linear_state_dense_parity_gpu.py` also pass.
- **Schedule tests.**
  - Forward order prefetches i+1, and backward recompute order prefetches i−1.
  - At most two layers are resident at any point.
  - The counters match a 4-layer toy model's expected counts exactly.
- **Toy parity.** On a 4-layer Qwen3-shaped model with `Linear4bit` and PEFT, under the parity pass's deterministic
  settings: S, S0 and R give `torch.equal` loss and gradients over 3 steps, and R against R is bitwise too.

## Cost

Download about 65 GB (the bf16 checkpoint; quantized on load) and about 40 min of box time in total. Guard 1.5 h at the
policy rate ($0.85/h), so it is preceded by a proving rental (guard > 1 h), per policy. Estimated actual about $0.7–1.0.
This is the standing no-ask tier.
