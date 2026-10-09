# DQ10 amendment 2: retained frozen-weight pricing and source pin

Before any DQ10 holdout reading, source inspection found that Loggetta omitted
frozen codes below the unchanged offload engine's 1 MiB streaming threshold.
SmolLM3's K/V projections stay on the GPU: 36 layers × 2 matrices × 524,288 bytes
= 37,748,736 bytes (36 MiB). This is a structural source correction, not a fit.

Re-pin Loggetta to its reviewed PR40 merge `8c9800f01ab378e7f0cf1ff3f82538e744750f7f` in runtime.json and
refresh only that file's checksum in instrument.sha256. The packaged policy
SHA remains `766020d5d9b848e533d87bb64f231976c74b7c55b0a85946b1e002db04fb5d59`.
All other eighteen stage-closure members and installed dense_offload.py remain
byte-identical to original DQ10 registration `5959328d21f6f606ecd67b4943219bab5fd99df8`.
This explicitly supersedes amendment 1's original Loggetta pin and complete
instrument byte-identity requirement for those two files only.

The correction is exactly zero on every DQ9 fitting arm. Config-only tests of
Qwen3-14B, Llama-3.1-8B and Qwen3-32B at both placements and all three rungs
preserve the prior allocator estimates, covering all sixteen fitting rows.
Keep frozen resident f=1/5, C=660602880 B and streamed f=2197697/5494591,
C=731906048 B without refitting. No activation coefficient, policy value,
subject/config, recipe, rung, five one-byte gate or licensing rule changes.

The first failed draw and its own ledger row remain preserved. Its rc14 egress
refusal occurred before install, proof, admission or reading. Amendment 1's
single replacement `dq10-5090-2`, committed failed-host exclusion, $0.85 total
hourly ceiling, two-hour guard and $2.80 reservation remain unchanged. Require
clean merged source/checksum closure, four fresh pinned CUDA correctness proofs
and all twelve policy admissions before checkpoint generation or reading.

Launch only after this amendment and the exact-driver offer guard are reviewed
and merged. Set vast_required_driver=595.91.07 in the reviewed manifest to avoid
buying out-of-scope advertised drivers, retaining the original actual box-side
rc19 check. No matching in-budget host means no purchase. The replacement keeps
all failure/no-redraw, fetch, invoice and verified-teardown requirements.
Observation import, opt-in removal and independent 24 GB transfer stay unlicensed.
