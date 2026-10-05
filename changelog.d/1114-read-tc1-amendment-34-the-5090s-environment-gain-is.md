### Read: TC1 amendment 34 — the 5090's environment gain is torch 2.12's (P63 HELD, 0.905), not transformers 5.5's (P62 FALSIFIED, 1.005)

- `tc1-5090-78` ($1.73, EPYC 7B13, 60-step load-gated draws, prebound launches off): e4b's matched arm in venv-e4b (torch 2.8,
  transformers 5.18), venv-e4b-tf55 (torch 2.8, transformers 5.5, built on the box) and venv-unsloth (torch 2.12.1, transformers 5.5).
  transformers alone 1.005 [0.995, 1.014]; torch 2.12 + triton 3.7 0.905 [0.892, 0.918]; the whole environment 0.909 (P64 HELD);
  held-out within 0.0013 (P65 HELD). With amendment 32's triton-alone 0.992, nearly all of the gain is torch 2.12's.
- By the registered rule (P63 HELD) the install section says that torch 2.12 runs e4b's host-bound training step faster on an
  RTX 5090. Row `e4b.train.env-split.qwen3.5090.2026-10-05`.
- The gate voided six draws, some under load from this campaign's own boxes on the same machine; the first attempts read 1.004 / 0.907 /
  0.911, so no verdict changes.
