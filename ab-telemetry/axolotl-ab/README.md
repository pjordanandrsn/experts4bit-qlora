# Axolotl-native expert_offload A/B (trainer-run, PR #3797)

Ran through `axolotl train` on the PR branch (fork base fa42e43 + local commits through
fd12f92: plain-DDP support and grouped-3D quantize_moe_experts offload), answering the
maintainer ask on #3797. One 2x-GPU pod, all four arms in a single session; the exact GPU
model and library versions are in run.log's environment-probe lines (CUDA devices / VERSIONS —
axolotl's pins upgrade torch past the image's 2.8.0). Deltas from the shipped
example config: seed 42, max_steps 150, eval_steps 50, no saves, wandb offline, flash_attention
false; DDP arms differ from single arms only by visible-GPU count. See reduction.txt for numbers,
configs/ for exact arms, HANDOFF.md in ~/code/axolotl-ab for the runbook.
