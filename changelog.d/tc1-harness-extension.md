### TC1 harness extension: an observe-only hook and three registered options, shared by P237 and P238

- **`TC1_ARM_HOOK=module:factory`.** A generic hook that only observes. The module must match its `TC1_ARM_HOOK_SHA256`, the lane's staged
  pin. Its phases:
  - `post_import(args)`;
  - `post_load(args, model)`, before the first forward;
  - `step_edge(i)`, before each optimizer step and once after the last;
  - `record()`, written into the receipt's `arm_hook`.
  A wrong sha, an import or a factory failure exits 10 and never falls back. The selftest pins observe-only: a hook returns the same
  losses as a baseline, and a mutant that scales the loss is caught.
- **`--vram-cap-gib X`.** Sets the per-process memory fraction before the load and reads the getter back. A torch without the getter
  refuses, because Unsloth's planner would silently read 1.0.
- **`--unsloth-offload-layers N|auto`** (Unsloth only, needs the cap). Passes `from_pretrained(offload_layers, max_memory)` and records the
  swapped layers. A loader that does not name `offload_layers` refuses.
- **`--layer-major 1`** (e4b only). The step body is `experts4bit_qlora`'s `layer_major_step`. Its drift guard runs once, before the timed
  window, with the guard's gradients discarded. The loop keeps the clock, the step edges and the profile window, and both paths build
  their micro-batches in one place.
- **`build_parser()`.** tc1_arm's command line as a function. A test pins every action against the parser at 1f43d364.
- Unset, nothing changes: the selftest asserts that a default arm's receipt carries none of the new keys.
- **`bench/tc1/tc1_drive.sh`** hands the box's exit code to the launcher through `bench/common/drive_tail.sh`. TC1's registered box codes
  are 9 12 15 19 78 130. Its allowlist row is dropped.
