# DQ11 real initial quality on A2000

Prospective, initial-only supplement to experts4bit-qlora#1083, authorized
by the maintainer on 2026-10-10 (bus messages
`20261010T173132Z-86316-119774bdbc58` and
`20261010T180251Z-5403-755bf6bda96a`). The later clarification
`20261010T182644Z-27149-a83a7a8837ec` permits CPU review now, with the PR
unlabelled. Merge and execute only after the sixth tiny rehearsal at
`047efb36e81c56d0e176b7cc1e79c6ab34b8c323`.

Use the original Mistral revision, canonical 448 FP32 adapter tensors,
canonical token bytes, original scorer and unchanged absolute 0.05 PPL K8
comparison. The separate entrypoint `dq11_a2000_quality.py` selects the
original 101-package A2000/cu128 lock, identifies sm86, and marks every
receipt non-scientific. Preparation uses the existing L/U loaders and
shipped read policy with an explicit default-false `quality_read` opt-in.
The original science entrypoints still reject A2000. This replaces misuse
of the tiny-model override; it adds no trainer, optimizer or backward proof.

Require fresh exclusive bus ownership before framework imports, a unique
owned workspace and nonce, and exact merged source authority. Require at
least 9 GiB free; cap the owned process at the lesser of 9 GiB and 75% of
device memory. This cap accommodates the real resident model while leaving
space outside the process. Check frozen base values, bindings, policy and
absence of gradients before and after scoring. Keep raw scores, identities
and source/runtime provenance for independent CPU re-derivation.

The two-hour deadline includes a 300-second teardown reserve. The existing
durable phase helper runs preparation, L, U and the primary report. Optional
U0 and its separate report run last, only within the remaining budget;
their refusal preserves the completed primary report. No performance,
capacity, full-science, shipping or paid-run authority follows from this
read. An out-of-budget real L/U comparison remains `INITIAL_QUALITY_FAIL`
and requires prospective review before any spending.

The new helper and this note join the transport closure. Reseal the two
preparation modules; preserve the original scorer, reducer, inputs,
initializer, scientific runtime, runners and prior amendments byte for
byte. Since paid execution imports preparation, the actual merged candidate
head requires a new full short $0 path before a paid amendment.
