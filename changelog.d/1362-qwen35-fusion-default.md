### Behaviour change (#1362): the B=1 fusion knobs default to `auto` on Qwen3.5/3.6-MoE

An unset `E4B_PAGED_FUSE_QKV`, `E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2` or `E4B_FUSE_ROUTER_EPI` now resolves to `auto`
on Qwen3.5/3.6-MoE (`qwen3_5_moe_text`, the text tower `serve_paged` builds, and `qwen3_5_moe`), as it already does on
Qwen3-MoE. Only the router epilogue engages there. Lane FAM licensed it on two reads: quality at T == 1 against the
family's own floor (`e4b.serve.fam.fused-stack-t1.qw36.5090.2026-10-09`), and one request decoding in 0.973 of the
step (`e4b.serve.fam.router-epilogue-speed.qw36.5090.2026-10-09`). The epilogue changes arithmetic, so greedy text
can change on these models. The way back is `E4B_FUSE_ROUTER_EPI=0`. `FUSION_UNLICENSED` no longer lists the family.
