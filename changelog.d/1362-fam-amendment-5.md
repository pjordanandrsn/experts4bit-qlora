### FAM Amendment 5 (#1362): Mixtral-8x7B joins the T == 1 read, as its own registration

`bench/fam/` gains Mixtral-8x7B-Instruct (pinned `eba92302`) with one knob per arm (ON_glue, ON_r2, ON_epi, ON_auto).
Its census and per-step glue calls are pinned at the real depth on CPU. The box records the build's `fusion_report`,
and the reducer VOIDs unless every fused router keeps fp32 weights (`fp32_upstream` 32). A family's own proof
(`--proof --families mixtral`) runs before its reading. Mixtral's box refuses on disk at the start (220 GB) and again
before its 93 GB fetch (165 GB). The lane ceiling rises to $26. The rule, gates and predictions for the other families
are unchanged.
