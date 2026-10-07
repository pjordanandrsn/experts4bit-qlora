### TC1 amendment 61 registered: the routed-expert combine over row chunks on packed rows (P166-P169), and a CUDA host floor

- Token `qwen3combck` runs e4b's matched arm on packed rows with `E4B_CKPT_OFFLOAD=1`, `E4B_COMBINE_CHUNK=0` against the default (row
  chunks), two draws each in A B B A order, with Unsloth's matched arm beside it and peaks split by phase.
- P166: the training-phase peak falls by 0.3 GB or more. P167: c1 / c0 at most 1.02. P168: step-0 held-out within 0.0001 per draw pair and
  held-out at N within 0.005. P169: c1's training peak at most 0.7 GB above Unsloth's.
- `tc1_arm.py` records `combine_chunk` (the variable, chunked forwards and backwards, the gate). The reducer adds the family, `combck_why`
  and `score_combck`. Self-test 126.
- `tc1_run.sh` probes the image's torch after the driver gate: torch that imports but cannot use the GPU refuses the box with code 18
  (`BOX_REFUSED cuda=unusable`), so the receipt names the machine for exclusion instead of a harness error.
