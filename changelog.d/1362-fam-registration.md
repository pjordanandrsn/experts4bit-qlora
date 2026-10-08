### FAM registered (#1362): a quality instrument with each family's own neutral floor, and the fused B=1 stack at T == 1 on gpt-oss-20b, Granite and Qwen3.6

- **Why.** Phase C's SANE bar (argmax ≥ 0.95) was set on Qwen3 and sits on Qwen3's and Granite's own neutral floors
  (0.953–0.964). gpt-oss-20b failed it at 0.924 without its floor ever being drawn, and on SC2g's path, not the default
  server's. The allowlist now needs a T == 1 read at reading size.
- **What.** `bench/fam/{PREREG-fam.md,fam_box.py,fam_reduce.py,fam_run.sh,fam_drive.sh,staged.sha256}`, with
  `tests/test_fam_box.py`, `tests/test_fam_staged_pin.py` and `tests/test_fam_split1_gpu.py`.
  - **Instrument:** P115's teacher-forced quality instrument at its registered bytes, per (text, shape, set): two
    texts, T == 1 and T == 12, three disjoint window sets.
  - **Floor:** each family's own neutral draws: prefill chunking, half-batch grouping, and one KV split
    (`n_split=1`).
  - **Gate:** relative to the floor's worst draw, with a backstop (|bias| ≤ 0.02 nats, agreement ≥ 0.90).
  - **Mutants:** P108's scale mutant must fail. A graded × 0.90 mutant that passes makes the read UNRESOLVED; a PASS is
    a null read of that size (no effect as large as a 10 % softmax-temperature change).
- **Readings.** gpt-oss-20b one knob per arm (and all `auto`) on the default server, plus an anchor cell at Phase C's
  SC2g setting; Granite and Qwen3.6 all `auto`.
- **Rule.** VOID → UNRESOLVED → FAIL → **PASS**, which licenses a separate allowlist PR.
- **Budget.** Proof on Granite (guard 0.75 h), reading guard 4.5 h; lane ceiling $10.
- **Not measured yet.** No box runs before this page merges.
