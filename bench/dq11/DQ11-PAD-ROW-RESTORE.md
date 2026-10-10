# DQ11 matched checkpoint padding row

The real Mistral initial-only A2000 read on merged e4b
`3daa0747578ada197e2fd8fbc2b3871e2788448d` completed L and U scoring on
2026-10-10, then refused at the report's strict frozen-value equality check.
The refused receipts remain immutable. There is no quality verdict from that
run, no SCI result, and no paid launch authorization.

The only differing frozen parameter was `embed_tokens.weight`, BF16 in both
arms, shape `[32000, 4096]`. CPU extraction from the pinned checkpoint reproduced
L's canonical FP32-value hash
`ad5b3153f038ee10f8bbb87f6cc48e86ac8e52bea547d43d239355c684b62280`.
Zeroing only row 0 reproduced U's hash
`fb7297148c129aeaae184e69e56b490557b65104e09cd681c66dddd0ab337fb1`;
zeroing row 1 or row 2 did not. The installed Unsloth tokenizer repair maps the
missing pad token to `<unk>` (id 0), and its loader zeroes the resulting padding
embedding row. The equality check correctly rejected those different values.

The matched configuration restores that row from the original local checkpoint
immediately after loading U or U0, before installing adapters. A safetensors
slice reads only the original row, from the indexed real-model shard or the
single-file Tiny checkpoint. The registered row is 0, independent of whether
the loaded embedding's `padding_idx` is set or absent; restoration never silently
skips because of that metadata. L does not call the restore. Full frozen-value
equality, the canonical initializer, original inputs and the 0.05 absolute PPL
budget remain unchanged.

Every U/U0 receipt names the deviation from default Unsloth:
"pad row restored (Unsloth's patch_tokenizer maps absent pad to unk 0 and zeroes
embedding row 0)". It records the restored row and its source-value hash, and
counts token 0 in every complete input block passed to the heldout scorer.
The eventual RESULTS must say in one plain sentence that the matched U/U0
configuration restores Unsloth's zeroed padding row from the checkpoint.

The refused run's canonical Alpaca heldout (8 blocks), WikiText test (8 blocks)
and train (40 blocks) each contain zero token-0 inputs. The checkpoint's
`lm_head` is untied, so that row did not affect the scores. These facts explain
the harmlessness of the loader mutation; they are not a substitute result.
The maintainer's 2026-10-10 ruling requires a fresh real-model read after this
fix is reviewed and merged. No derived identity or corrected copy of the refused
receipts will be used.

The loader is paid-imported. After merge, run the real initial-only read first,
then the seventh full short $0 rehearsal on the same actual merged head, in one
fresh sole A2000 claim if the queue and reserve permit. The sixth full-path
acceptance covers `3daa0747` only. Neither A2000 run licenses 5090/cu130 science,
speed, capacity or shipping claims, or a paid draw.
