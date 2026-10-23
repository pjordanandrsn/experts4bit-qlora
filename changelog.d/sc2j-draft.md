### SC2j: the shared KV pool's capacity read at the default 4,096 tokens a slot (registration and box Q)

- `bench/sc2/SC2j-PREREG.md`: claim-all against shared, both at `E4B_PAGED_MAX_SEQS=auto`, on Qwen3-30B-A3B int4 on one
  RTX 5090. It carries:
  - a contract arm: shared at 32 slots token-identical to claim-all at 32 on one fixed schedule;
  - SC2e's ladder, plus a long-context ladder whose reservations must bind (a PRESSURE gate);
  - a re-run of P117's teacher-forced width check at the launch stack.
  A licence would cover one allowlisted pair, `qwen3_moe` int4 on sm_120.
- Box Q (`bench/sc2/sc2j_box_q.sh`, `sc2j_reduce.py`, `sc2j_contract.py`, `sc2j_prompts.py`) is wired into
  `bench/sc1/sc1_run.sh` and `sc1_drive.sh` beside boxes A-P, with the staged pin regenerated. `tests/test_sc2j_box.py`
  holds the lane's constants, Guard Fit's static form, the cap and the wiring; the other boxes' tests learn the new
  letter.
