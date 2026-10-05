### DQ3 registered: a streamed frozen-weight QLoRA prototype — bitwise-identical training, resident speed, a layer-count VRAM saving? (registration only; no code)

- **Licensed** by DQ2's C_ALIVE. Stage 1 of 3: the pre-registration and design note only (`bench/dq3/`). Stage 2 is an
  opt-in, off-by-default training prefetch in `engines/dense_offload.py`, with its tests and a $0 A2000 rehearsal.
  Stage 3 is one gen 5 x16 RTX 5090 read.
- **Subject and arms.** Qwen3-32B @ `9216db57`, HF + PEFT + bnb QLoRA, 2048-token micro-batch, non-reentrant
  checkpointing. Three arms: R (resident), S (streamed, overlapped) and S0 (today's synchronous grad-mode offload).
- **Gates**, with the maintainer session's additions:
  - bitwise parity of loss and every LoRA gradient in a deterministic pass, with an R-vs-R control;
  - step time T(S)/T(R) ≤ 1.10;
  - steady-state blocking fetches ≤ 2 per step, forward and backward both;
  - a VRAM saving ≥ 13.6 GB, against a 15.12 GB slot-count prediction;
  - pinned-host bytes read from the allocator;
  - a fence race test with a mutation arm;
  - an off path that stays byte-identical.

  Bands come from DQ2's 5090 readings only.
