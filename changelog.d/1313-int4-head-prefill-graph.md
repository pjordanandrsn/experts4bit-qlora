### Fix: the int4 lm_head no longer crashes the default graph server at build

- **The crash.** With `E4B_SERVE_LMHEAD_INT4_CALIB=1`, or any int4 lm_head, `serve_paged.build_engine` failed while
  engaging the prefill graph, with `AttributeError: 'Int4Linear' object has no attribute 'weight'`.
  - `PagedModelRunner.enable_prefill_graph` read the head's `.weight.shape[0]` to draw its warm-up prompts.
  - The int4 head carries a packed grid and `N`, never a `weight`.
  - Lane P125's proof (`p125-prove-1`, #1313) found it.
- **The fix.** The runner now reads the head's width through `_output_width(model)`: `N` on an `Int4Linear`, the
  weight's rows on a dense head.
- **Nothing else needed changing.** This was the only serving-path read of the head's `.weight`. The other
  `get_output_embeddings()` callers take the module, not its weight.
