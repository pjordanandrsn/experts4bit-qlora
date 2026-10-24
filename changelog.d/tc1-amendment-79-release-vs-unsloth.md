### TC1 amendment 79: the field recipe's same-stack position at the 0.58.0 release, against Unsloth 2026.10.8

- A new family, `qwen3pos79`, which is amendment 69's box unchanged: e4b's defaults (`fused_attn4_m_fd`) against Unsloth (`ckpt_unsloth_m_fv`,
  grouped_mm), two profiled draws each in ABBA order, both in venv-unsloth (torch 2.12.1), TC1's field recipe over 60 steps.
- The pins are constants of the reviewed kit, held equal across `tc1_run.sh`, `tc1_reduce.py`, the launch template and the registration.
  They are e4b 0.58.0 at its tag commit, grouped-nf4-gemm 0.51.0 at `30adb1bf`, and unsloth / unsloth_zoo 2026.10.8. The box refuses any
  other value before staging (rc 78).
- `bench/tc1/release_proof.py` (byte-identical to loggetta's, sha256 `aff3fc80…`) proves Unsloth's wheels against PyPI's published bytes
  after every install and before any arm (rc 9 on a refusal). Every arm then reads under the proof's fresh bytecode prefix.
- The box records the knob environment every arm inherits (`POS79_KNOB_ENV`), and the rule reads 0.58.0's defaults as unset knobs, not
  as 0.49.0's values, with what they resolved to reported beside the quote. The gate refuses `TC1_E4B_ENV` or `TC1_ARM_EXTRA_ENV`
  carrying anything, and a chunked-loss run-time refusal VOIDs an e4b row. `tc1_arm.py` is unchanged, since bench/q235's kit pins it.
- The rule, in order: NOT_READ on any box fact, or without two VALID profiled draws a side; UNSTABLE past TC1's
  stability bound; QUALITY_DIFFERS past 0.01 held-out; otherwise QUOTED, with the device ratio first and the wall ratio on the named
  host CPU. One box starts: $5.50 worst, $5.93 with a retry id, inside the row's $6.
