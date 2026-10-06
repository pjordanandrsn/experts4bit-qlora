### `serve_paged` captures only the decode buckets its sequences can use

- `PagedServeConfig.validate` now keeps the buckets below `max_seqs`, then `max_seqs` itself, capped at the largest
  given (`serve_recipe.usable_buckets`), and logs the change.
- A decode step never carries more rows than sequences, and the runner pads a step to the next bucket, so a bucket
  above `max_seqs` never runs. It still cost a captured graph and, as the largest, the scratch slots: each a full slot
  of a hybrid model's linear-attention state, ~62 MiB on Qwen3.6-35B-A3B.
- Lane SV3 (#1232) found worse: on Qwen3.6 served for one sequence, buckets 2–16 failed to capture.
- The serve estimate sizes the scratch slots by the same rule.
