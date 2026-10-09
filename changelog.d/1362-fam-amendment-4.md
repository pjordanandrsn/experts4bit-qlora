### FAM Amendment 4 (#1362): a speed read of the router epilogue on Qwen3.6, before any default flip

On Qwen3.6, FAM's T == 1 quality read passed, but only the router epilogue engages there. Nothing has measured what it
buys, so `bench/fam/fam_speed.py` times it at one row with P124's interleaved method: two blocks, one runner per setting
in each, and strict lockstep alternation. Each runner binds its own linear-state slot, because the hybrid's one pool is
shared. `bench/fam/fam_speed_reduce.py` reads FASTER (both ratios ≤ 0.98), SLOWER, NO_GAIN or NOISY (blocks over 1.5 %
apart), self-tested on 30 cases. `serve_paged.FUSION_UNLICENSED` now cites FAM's reads for gpt-oss, Qwen3.5/3.6-MoE and
Granite-MoE. No default changes.
