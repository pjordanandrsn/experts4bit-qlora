# tp4 — same-box training head-to-head: e4b (GitHub `main`) vs Unsloth vs plain HF+PEFT+bnb, every supported MoE family

Pre-registration: [`TP4-PREREG.md`](TP4-PREREG.md) (read it first; the harness refuses a run that does not name it).

| file | role |
|---|---|
| `tp4_arm.py` | per-arm driver, three frameworks through ONE `run_arm` (copy of `bench/tp3/tp3_arm.py` + T11–T16; `--selftest` on CPU) |
| `tp4_alpaca.py` | builds the fixed Alpaca text from the pinned `unsloth/alpaca-cleaned` revision (seed 3407, 1,200 / 48 rows; registered sha) |
| `tp4_run.sh` | BOX side: installs the three environments, fetches each family, tokenises once, runs the arms in the registered order, reduces |
| `tp4_drive.sh` | CONTROLLER side (the launcher's `--command`): stages, starts under a nonce, heartbeats, fetches receipts |
| `tp4_reduce.py` | the per-family support / validity / position / quality table, the anchor and the P1–P9 scoring (stdlib only) |
| `../../tests/test_tp4_arm.py` | CI: the selftest end to end, the `--prereg` refusal, the dataset builder's refusal path |

Boxes (one RTX 5090 each, launched in parallel through `adertha-agents/tools/pod-launch.sh`):
`TP4_BOX=A` granite · olmoe · gpt-oss + the Qwen3 anchor pair (tp2's fixture) + the two NOT_RUN rows;
`TP4_BOX=B` qwen3 · qwen3_5 (Qwen3.6-35B-A3B); `TP4_BOX=C` gemma4 · mixtral.

Receipts land privately first (the adertha receipt store), then the curated bundle is committed here as
`bench/h2h-<date>/tp4/` exactly as tp2's was.
