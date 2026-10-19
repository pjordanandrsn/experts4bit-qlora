### DQ11 MLP-adapter BUILD Phase 1: the A2000 runner, and Amendment 3 (#1083)

`bench/dq11/dq11_mlp_phase1_run.py` runs Phase 1 over the real L fixture on the RTX A2000, at $0 and for correctness
only. Each part is its own process and writes one JSON record:
- **`stage-check`:** checks the model snapshot against `model_files.json` and the sealed inputs against
  `locked_inputs.json`.
- **`reads`:** the per-module read at Amendment 1's state, on layers 0, 15 and 31.
- **`reported`:** the same read after two AdamW updates (reported only).
- **The step witnesses:** `step-today`, `step-off` and `step-fused`.
- **K8's short run:** `k8-unfused` and `k8-fused`.
- **`reduce`:** writes the outcome.

Every part refuses without the A2000 bus claim, on any other device, or when the installed PEFT, transformers,
bitsandbytes, accelerate and torch are not the A2000 lock's (`rehearsal/wheels.json`). It records each package's
version, its locked wheel's sha256, and whether the installed files are that wheel's.

The per-module read gains three things:
- a second self-check, the reference's `_LoraLinear` path with every split off;
- a floor folded in one draw at a time (each draw holds about 210 MB at the fixture's size);
- element counts and ULP distances for the tensors held bit for bit.

`DQ11-MLP-BUILD-AMENDMENT-3.md` registers the runner's terms before data:
- PEFT 0.21.2 is the version of record. The DQ11 locks pin it, and the seat's 0.19.1 is not the fixture's.
- The canonical initializer is "PEFT's own initialization".
- It fixes the step witnesses, including the recompute count call for call.
- K8's short run is 8 updates.
