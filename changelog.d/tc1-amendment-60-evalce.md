### TC1 amendment 60 registered: the held-out loss from the logits in chunks on packed rows (P162-P165)

- Token `qwen3evalce` runs e4b's matched arm on packed rows with `E4B_CKPT_OFFLOAD=1`, `E4B_CHUNKED_EVAL_LOSS` 0 against 1, two draws
  each in A B B A order, with Unsloth's matched arm beside it and peaks split by phase.
- P162: the evaluation-phase peak falls by 3.5 GB or more. P163: with the switch on, every draw's evaluation peak sits below its training
  peak. P164: step-0 held-out within 0.0001 per draw pair. P165: held-out at N within 0.005.
- The reducer adds the family, `evalce_why` and `score_evalce`. Self-test 125.
