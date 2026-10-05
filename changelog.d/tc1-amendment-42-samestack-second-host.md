### TC1 amendment 42 registered: the same-stack position on a second host, on the current code (bench and tests only)

- **Why.** The Qwen3-30B-A3B position to quote (Unsloth/e4b 2.352, amendment 33) is one box on one host model. Amendment 38's box stepped
  e4b 37-44 % faster on an EPYC 9655 than on two other hosts, and TC2 amendment 9 traced most of Mixtral's 0.836 to the host.
- **The box** (token `qwen3samestackh2`): amendment 33's box on a machine other than 145701, on the current code. P94 Unsloth/e4b in
  [1.9, 2.9]; P95 the environment in [0.80, 0.95]. A reading outside makes STATUS quote the two hosts' readings as a range.
- The reducer reads it with amendment 25's scorer (one new self-test case).
