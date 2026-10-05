### TC1 amendment 44 registered: e4b's chunked LM loss as `auto` — its default decision (P99–P103)

- Token `qwen3chunkauto`: amendment 41's box with `E4B_CHUNKED_LM_LOSS=auto` (#1178) against the default at the field recipe, shipped and
  matched arms, two load-gated draws a side, venv-unsloth, avoiding amendment 41's busy host. P99: the 1 GiB gate never fires there
  (`chunked_calls` 0, `small_calls` 240 on every `auto` arm); P100 / P101: `auto` / off ≤ 1.02; P102: the matched peak not above +0.05 GB;
  P103: held-out within 0.005.
- With P99–P103 and amendment 43's P98 HELD, `auto` becomes e4b's default. Registered before amendment 43's box was read.
- `tc1_reduce.py`: the family, its engagement predicate (`chunk_auto_why`), P99's scorer, P100–P103 through amendment 41's scorer;
  self-test 106 cases.
