# Is there a dense low-bit training primitive in this stack? Research note, 2026-10-04

Written before any DQ1 measurement. It covers what the code says, what earlier receipts already settle, and the cheapest
experiment that could falsify the idea. Lane DQ1 ([DQ1-PREREG.md](DQ1-PREREG.md)) is that experiment.

Sources read:
- experts4bit-qlora `origin/main` 7b0726ca;
- grouped-nf4-gemm `origin/main` 91ff8da (0.39.0 + #467);
- bitsandbytes `main` (0.50.0.dev0, `autograd/_functions.py`, `backends/cuda/ops.py`);
- Loggetta bc3e228;
- adertha-agents 876c7c1d.

File references are to those heads.

## 1. How much of the stack already serves dense QLoRA?

Less than the package names suggest, and nearly all of it is bitsandbytes underneath.

- **Works unchanged when called directly.** All of it is built on bnb `Linear4bit`; none of it uses gnf4's format.
  - `LoRALinear` (`lora.py:764`), which runs two matmuls and plain autograd.
  - `quantize_attention_projections_4bit` and `quantize_frozen_linears_4bit` (`lora.py:953, 1007`), which produce bnb
    NF4 with double-quant on.
  - Structural attention discovery (`lora.py:816`).
  - Fused RMSNorm training (`engines/rmsnorm_train.py:137`) and fused RoPE training (`engines/rope_train.py:173`).
  - `enable_dense_offload` (`engines/dense_offload.py:287`).
- **Generic, but switched on only through MoE.**
  - Fused RoPE and RMSNorm are enabled only when an `ExpertsLoRA` is patched (`engines/fast.py:879`).
  - `prepare_qlora_training` and `estimate_qlora_footprint` refuse a model with no fused expert stack
    (`recipe.py:103`).
- **Refuses dense outright.**
  - The loader (`loader.py:1103`: llama, qwen2 and qwen3 raise `NotImplementedError`).
  - `train.py:386`.
  - The HF arm of both training harnesses (`bench/tc1/tc1_arm.py:1847`, `bench/tp4/tp4_arm.py:1073`).
  - The README says to use bitsandbytes directly (`README.md:81`). gnf4's README says the same
    ("nothing here helps a single `nn.Linear`", `README.md:76`).
- **Missing.**
  - Dense-MLP LoRA.
  - An end-to-end dense training step in any test.
  - Overlapped weight streaming under autograd. `dense_offload` prefetches only for no-grad forwards (`:336`). Under
    grad it stages synchronously into one slot (`:296`).
  - Dense support in Loggetta: its model provider is `describe_moe` only (`model.py:16`).

## 2. Can the grouped primitive serve as a G=1 dense prototype?

It already runs at G=1, and today it is not a packed kernel at all.

On the A6000 on 2026-08-15, the packed G=1 kernel lost to bitsandbytes at every row count, forward and dgrad alike
(0.22–0.80×). That was a structural loss: the expert-major access pattern moved one big matrix at 8% of bandwidth. That
receipt lives in session memory only, not in either repo.

Since then, gnf4 0.39.0's `GNF4_TRAIN_GEMM=auto` sends any call with at most 16 groups, off sm_90, down the **dense
route**: gnf4's Triton dequant, then `torch.mm` (`nf4_route.py:85-86, 116, 233`). So G=1 is now structurally the same as
bitsandbytes' large-M path: decode the whole weight, then cuBLAS. The only difference is whose dequant kernel runs.
*(Corrected 2026-10-05.)* This note first quoted a code comment's A2000 timing of the two decoders here. An A2000 timing
is not speed evidence under the testbed policy, so it should not have informed the G1 prediction. DQ1's 5090 read
measures the decoders directly: gnf4's `dequant_groups` takes 0.94–1.01× bnb's time on the large shapes and 0.65× on
kv_proj ([RESULTS-dq1.md](RESULTS-dq1.md), run 2).

**The prediction follows.** G=1 auto should sit at parity with bnb, within the dequant's share. It cannot exceed the
dequant share by construction. DQ1 checks this, and re-reads the fused route on the 5090.

## 3. Which baselines are relevant?

- **bitsandbytes 0.50.x `matmul_4bit`.** This is the incumbent, and it already dispatches.
  - Forward: a fused 4-bit GEMM for small M, chosen per-arch by a CUDA heuristic that has an sm_120 block; otherwise
    dequant + `F.linear`. Read on bnb `main` @3343bac (June, 0.50.0.dev0; `backends/cuda/ops.py:586, 822`), which
    predates the 0.50.2 the census installs. The independent review read 0.50.1's sm_120 block as capping the fused
    path at M ≤ 256, which would make bnb dequant + cuBLAS at every census row. DQ1 records bnb's decision per cell
    rather than assuming either.
  - Backward: always `dequantize_4bit` + matmul (`_functions.py`, `MatMul4Bit.backward`).
  - **The shape- and arch-dependent dispatch this investigation might have proposed already ships upstream.**
- **HF + PEFT + bitsandbytes** and **Unsloth** at the training-step level. TC1's harness has both arms, but its HF arm
  refuses dense today.
- **bf16 cuBLAS on a resident decoded weight.** Not a QLoRA option, since it costs 4× the memory, but it is the floor:
  `(bnb − bf16) / bnb` bounds what any low-bit kernel can save on the base linears.
- **Marlin and Machete** (vLLM): dense W4A16 at small M. Inference-only and outside training scope; recorded so the
  decode question isn't reopened by accident.
- **For a larger-than-VRAM operating point:** FSDP-QLoRA with CPU offload, and HF `device_map` offload. These would be
  the comparison if streaming pans out.

## 4. First target model

**Qwen3-32B** (`Qwen3ForCausalLM`, config read from the LAN copy: 64 layers, hidden 5120, intermediate 25600, 64 q heads /
8 kv heads, head_dim 128, untied embeddings, no attention biases).

- **It is conventionally dense.** No hybrid linear attention, unlike the Qwen3.5/3.8 line.
- **Every framework supports it**: PEFT, Unsloth and bnb.
- **It is on the LAN store.**
- **Its NF4 weights fit an RTX 5090.** About 20 GB with bf16 embeddings, leaving about 11 GB for activations, so
  sequence length and micro-batch are real constraints there, and a 24 GB card is a fit boundary.

Qwen2.5 is out because e4b's 4-bit attention path refuses biased projections (`lora.py:979`). Llama-3.1-70B is the
natural second target only if streaming is alive, because its NF4 weights (~37 GB) do not fit 32 GB.

## 5. The cheapest falsifying experiment

**DQ1 is a shape-level census on one RTX 5090. It needs no checkpoint, no training and no download.**

Arms, on identical NF4 weights:

- **bf16** floor
- **bnb** `matmul_4bit`
- **dq**, explicit dequant + mm
- **gnf4a**: gnf4 G=1 auto
- **gnf4f**: gnf4 G=1 fused

Measured for each arm:

- forward and dgrad on Qwen3-32B's five linear shapes at M = 512 to 8192;
- each decoder alone;
- a PEFT-shaped LoRA delta;
- pinned H2D bandwidth for one layer's NF4 bytes, alone and under GEMM load.

These readings bound the two axes a dense primitive could win on:

- **Speed.** H(M), the most any low-bit kernel could save on the base linears. A training step cannot gain more from a
  better 4-bit GEMM than its linears' share times H.
- **Capacity.** Rmin(M): each phase's compute (the forward; the recompute + dgrad) over one layer's weight transfer. If
  Rmin stays well above 1 at
  training M, a dense model whose frozen weights live in pinned host memory could hide its weight
  traffic behind compute ("near-resident speed" is the hypothesis a later matched-work lane would test, not a DQ1 reading).
  That is a model-size Pareto point nobody in the baseline list ships for single-GPU QLoRA.

The streaming idea was refuted for **MoE** (e4b `bench/host-ram-ceiling/RESULTS-prefetch.md`; gnf4#60, closed
2026-10-04). That refutation does not carry over. It failed because the next layer's routing is unknowable, so a
prefetch had to fetch 1.9× the bytes. A dense layer's next weights are known exactly, and every fetched byte is used.

## 6. What would justify further implementation

Either of:

- **Speed.** H(2048) ≥ 0.15 (S_ALIVE). A perfect low-bit kernel could save at least 15% of base-linear time at
  QLoRA-typical rows. Next: a full-step profile of HF+PEFT+bnb on Qwen3-32B to get the linears' share, before any kernel.
- **Capacity.** Rmin(2048) ≥ 1.25, with concurrent DMA costing the GEMMs ≤ 5% and keeping ≥ 80% of its bandwidth
  (C_ALIVE). Next: a streamed frozen-weight prototype under autograd in a branch. Concretely: e4b's `dense_offload`
  extended to overlap under grad, or a tensor-level stream in gnf4. Then a matched-work full-step comparison, at a
  model that fits, against HF+PEFT+bnb and Unsloth, and at one that doesn't fit, against FSDP-QLoRA with offload.

## 7. What would justify a new repository

Not this lane. At most, DQ1 can show that a capacity primitive is worth prototyping. A repo is earned only after the
prototype and its matched-work result clear the eleven-point gate in the brief, all of these at once:

- a tensor-level API (pinned host weight + quant metadata + stream + layer schedule in; a device view out);
- no notion of experts or model families;
- no dependency on e4b, Loggetta or Adertha;
- more than one plausible caller;
- a measured win at matched work.

Today the strongest candidate for a home is grouped-nf4-gemm's host-transfer layer. `host_gather` is already
tensor-level and works for any pinned row stack (`host_gather.py:73`). The candidate for not splitting is
`dense_offload` inside e4b.

## 8. What would tell us to stop

- **Speed.** S_DEAD (H(2048) < 0.10 and H(4096) < 0.07) plus G1_PARITY closes the dense-speed-kernel line. G=1 stays an
  internal route, and nothing new gets built for speed. G1_LOSS would mean the dense route needs a shape guard. That
  goes back to the gnf4 maintainer; it is not a new primitive.
- **Capacity.** C_DEAD (Rmin(4096) < 1) closes the streaming line on this card's link.
- **Both dead** is outcome E (negative), and that is a fine result.

## 9. Delegation

Three archaeology agents, run in parallel and read-only, covered e4b, gnf4, and Loggetta + adertha-agents. Their
reports are the file references above. Two more:

- One reviews DQ1's harness and pre-registration before the box.
- One synthesises the phase summary after the read.

The peer maintainer session owns the gnf4/e4b release and CHANGELOG sequencing. It was told DQ1's scope over the session
bus and agreed (bench-only; separate from gnf4#60 and e4b#945).

## 10. Before the decisive run

Each of these must exist first:

- **Work item and authorization.** An experts4bit-qlora issue holding this note and the owner's directive. The
  authorization is the standing no-ask tier, issue #564's comment of 2026-09-26.
- **Lane name claimed** by pushing `prereg/dq1`.
- **Pre-registration and harness merged** to e4b main, CI green.
- **$0 rehearsal** of the census and reducer on the A2000 through the local pool (`tools/pool run --gpu cuda`, the
  scheduler's lane for that card), so a rented box never runs untested code.
- **Driver dry run** (`TC1_DRIVE_DRYRUN=1`).
- **Manifest and launch** through `tools/pod-launch.sh` on the mini, with receipt and ledger row committed to the store.

## 11. Resources

- **Rehearsal:** the local pool's `qnap-gpu` slot (exclusive `gpu:a2000`), about 20 min.
- **Decisive run:** one RTX 5090 on `vast:verified-secure` at the policy rate of $0.85/h, with a 0.75 h guard. The
  estimate is about 25 min of box time, under $0.50 actual. No proving rental is needed, since the guard is under 1 h.

## 12. Stop conditions by phase

| phase | stop when |
|---|---|
| 1 read | done (this note) |
| 2 census (DQ1) | the reducer's verdicts: S_DEAD + C_DEAD → stop (E); VOID/NOISY → one re-run, then stop and report the instrument |
| 3–4 baseline | only after S_ALIVE or C_ALIVE; stop if the matched baseline puts the linears below 50% of the step |
| 5 profile | stop if the measured bottleneck isn't one a low-bit primitive touches |
| 6 prototype | stop if parity or quality gates fail twice, or the matched-work gain is < 1.10× on speed and < 1.25× on capacity |
