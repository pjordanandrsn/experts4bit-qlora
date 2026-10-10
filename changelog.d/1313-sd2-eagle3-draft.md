### SD2 build 3 (#1313): the EAGLE-3 draft and its auxiliary states in the package

- **`engines/eagle3_draft.py`** holds lane SD2's draft for one speculating sequence. It repeats SD1's arithmetic
  (`bench/sd1/sd1_eagle3.py`, vLLM's EAGLE-3 semantics) with a KV cache:
  - `Eagle3Drafter.prefill` writes the prompt's context.
  - `extend_and_draft` writes every verified row at a static shape (SGLang's draft-extend) and chains k drafts from
    the accepted row. Its inputs, the accept count and the base position included, are device tensors, so it can be
    captured.
  - `load_head` checks the pinned head's size (through a symlink), sha256 and config, and refuses an unsupported head.
- **`AuxStates`** installs forward pre-hooks that copy the residual stream entering layers `(2, L // 2, L − 3)` into
  static buffers, for prefill chunks and decode rows.
- **Not wired yet.** Nothing in the server uses it yet. `bench/sd2/PREREG-sd2.md` B3 and B4.
