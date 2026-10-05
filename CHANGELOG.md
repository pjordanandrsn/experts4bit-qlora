# Changelog

## Unreleased

### TC1 amendment 39 registered: the packed 4,096-token regime with both frameworks on one stack (bench and tests only)

- **Why.** Every TC1 position reads the field recipe, whose Alpaca rows carry about 1,000–1,400 real tokens per step against a nominal
  16,384: a host-bound regime. Packed full-length rows put about 12× the tokens through each step, where the device does most of the work.
- **The box** (token `qwen3samestack4k`): amendment 25's same-stack family on rows of exactly 4,096 real tokens, micro-batch 1 × accum 4,
  30 steps, held-out at 0 and N, load-gated draws, every e4b arm resident at defaults. P84 Unsloth/e4b in [0.80, 1.60]; P85 the
  environment in [0.80, 1.00]; P86 every e4b arm completes resident. A stable reading is recorded whichever side it favours.
- **`TC1_FREE_OUTPUTS=1`** (`tc1_arm.py`): each micro-batch's output is released once its loss is read. Unset, the previous micro-batch's
  output, its logits included, stays live through the next forward and the optimizer step, inside every arm's peak. Off by default, so
  every earlier box keeps its instrument; the packed family requires it.
- **The builder.** `tc1_arm.py --prepare --pack 1` renders each example with the same template and tokenizer call, minus the
  per-example truncation. It concatenates the token lists with the tokenizer's EOS between examples and cuts rows of exactly `--seq`
  tokens. Labels are the input ids, nothing is padded, and attention is full causal across example boundaries (standard packing).
  - At 4,096 tokens the registered 1,200 + 48 rows fill only ~57 + 2 rows. So the pools extend the registered text in its own order:
    tp4_alpaca.py's pinned source and seed. `pack_pools` refuses unless the shuffled prefix reproduces the registered rows. The train
    pool is the 1,200 registered rows and then the next 4,800 examples. The held-out pool is the 48 registered rows and then the next
    352. The pools are disjoint.
  - On Qwen3-30B-A3B at the pin: 284 train rows and 8 held-out rows, every row 4,096 tokens, tokens sha `d2a501eba57d`. The first
    235,852 train tokens are the field recipe's 1,200 rows, unchanged.
- **Unchanged when off.** With `--pack 0` (the default) the tokens file is byte-identical to before: the field-recipe file for
  Qwen3-30B-A3B rebuilds to tokens sha `bfc742f67e37`, the sha TC3-PREREG cites. A test pins a small file's sha from the old code.
- **The box.** `TC1_PACK=1` (forwarded by `tc1_drive.sh`, recorded on the FIXTURE line) packs `tc1_prepare`'s tokens, at least
  steps x micro-batch x accum rows. The new token `qwen3samestack4k` is amendment 25's same-stack family in this regime. The box refuses
  the token without `TC1_PACK=1 TC1_SEQ=4096`, and refuses `TC1_PACK=1` beside a field-recipe token. The arm refuses a packed file whose
  seq is not its own `--seq`, and each receipt's `tokens` records `pack`.
- **The reducer.** The family is read with amendment 25's scorer and `score_packed4k`, against its own fixture. A row is VOID unless its
  tokens are packed at seq 4,096, micro-batch 1 x accum 4, with 16,384 real tokens and none padded on every step, so the field recipe's
  receipts never pass under this token. One new self-test case (101).