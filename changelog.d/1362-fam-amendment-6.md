### FAM Amendment 6 (#1362): Gemma-4-26B-A4B joins the T == 1 read, as its own registration

`bench/fam/` gains Gemma-4-26B-A4B (pinned `4d7ae498`) with ON_glue, ON_epi and ON_auto. The r2 fold refuses Gemma-4's
extra norms and engages nothing, so there is no r2 arm. Its census (`0 / 271 / [0, 0] / 30` at auto) and per-step glue
calls are pinned on CPU at the real depth and layer pattern. A Gemma-4 proof runs before its reading, and its box
refuses before the fetch below 120 GB free. The read cannot speak to the 1024-token sliding window, which never binds
at 640 tokens.
