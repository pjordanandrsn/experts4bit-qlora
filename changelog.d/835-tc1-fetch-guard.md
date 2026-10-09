### TC1's model fetch: brotli>=1.2.0 in venv-e4b, a fetch probe, and rc 15 when nothing staged (#835)

- **What happened.** On `tc1-5090-143` (P129 Phase 2's first box) the model fetch failed, every arm was written as a `not_run` stub,
  and the box still ended OK. venv-e4b sees the base image's site-packages, whose brotli predates 1.2. huggingface_hub 2.x downloads
  through httpx2, which advertises brotli whenever it can import it, and passes `output_buffer_limit=` to `Decompressor.process`. Only
  brotli 1.2 or later accepts that, so a brotli-encoded response failed with `process() takes no keyword arguments`.
- **The fix.**
  - venv-e4b installs `brotli>=1.2.0`, which shadows the base image's copy.
  - The e4b tripwire refuses a visible brotli older than 1.2, and records the hub, httpx2 and brotli versions.
- **Two guards.**
  - A fetch probe downloads the registered Qwen3 pin's small files through the same client right after venv-e4b's tripwire, before any
    other venv is built. A failure refuses the box with rc 15 (`TC1_FETCH_PROBE=0` skips it).
  - A box on which no family's model staged ends rc 15, so its receipt reads HARNESS_ERROR, not OK.
