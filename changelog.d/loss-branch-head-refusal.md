### The plan prices the stock loss when the chunked loss refuses the LM head

- `chunked_lm_loss_refusal` refuses an LM head whose width differs from the config's vocabulary (quietly under `auto`), and the run then
  takes the stock loss. The planner now prices the same. `MoETopology.config_vocab_size` records the config's vocabulary beside
  `vocab_size` (the head's width), and `recipe._head_refusal` names a mismatch. `_loss_chunk` then returns the stock loss, and the
  held-out loss line says the model is not patched.
- `tests/test_vocab_width_one_source.py`: with the chunked loss forced on, a head 128 wide on a 97-token vocabulary prices stock in both
  the training and the held-out loss, while the matching head prices chunked.
