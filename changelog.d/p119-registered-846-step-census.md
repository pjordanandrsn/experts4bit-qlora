### P119 registered (#846): a descriptive census of the 64-slot server's steps, per kernel, on one RTX 5090 (bench and tests only)

- **Why.** SC2e serves 12 req/s at 64 slots; the capacity model on its costs says prefill and the decode step set the
  next ceiling, and neither has been attributed by kernel on this stack (SC1b read B = 16 before the folds, lean glue,
  bulk KV bookkeeping and wide buckets).
- **The box** (`bench/p119/p119_box.py`): SC2e's int4 stack built eager with one slot; `torch.profiler` on the eager
  twins of the served graphs (P109: bit-identical to the replays) for decode at 16, 32 and 64 rows and 64 as four
  16-row pieces, the 512-token prefill with `E4B_PAGED_LAST_LOGITS` off and on, and the LM head alone. Shared with the
  TTFT lane, so one profiling run serves both.
- **The rule** (`bench/p119/p119_reduce.py`, 22 self-test cases): READ or VOID only; it checks every bracket ran its
  registered split and tabulates device ms by class, the marginal cost per decode row and the D2D copies per layer.
- **Proof** on Granite (guard 0.75 h); reading guard 1.5 h; lane ceiling $3.00.
