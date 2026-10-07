### `E4B_CKPT_OFFLOAD=reentrant` (a diagnostic), and the offload's guards for a future default

- `E4B_CKPT_OFFLOAD=reentrant` routes the checkpointed decoder layers through PyTorch's reentrant checkpoint without the host-memory hook,
  so their inputs stay on the GPU. It separates the two halves of `E4B_CKPT_OFFLOAD=1` (the checkpoint flavour and the copies) for TC1
  amendment 62, which asks which of them made amendment 59's field-recipe step faster. Gradients equal Hugging Face's checkpointing
  exactly in the tests. TC1 receipts record the routed function (`ckpt_offload_funcs`).
- `E4B_CKPT_OFFLOAD` stays opt-in: TC1 amendment 59's rule asks for a torch 2.8 read on a host-bound box first (amendment 63). The values
  are now parsed strictly (`0` / `1` / `reentrant`, plus on / off spellings; anything else is an error).
- Ready for when it does become the default (`CKPT_OFFLOAD_DEFAULT`, still off): a default-path request leaves a dense-offloaded model alone
  and is silent without checkpointing, and `enable_dense_offload` warns when it finds offloaded checkpoints, because that pairing is
  untested.
