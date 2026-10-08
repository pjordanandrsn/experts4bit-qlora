### P123 Amendment 1 (#1313): the proof checks that every router licenses on a 5090

- **What changes.** The proof gains one build of the default server on Granite with `E4B_FUSE_ROUTER_EPI=auto` alone
  (`p123_box.py --mode routers`). It VOIDs unless all 32 routers license.
- **Why.** That is lane FAM's `fam-prove-1` case, which licensed 27 of 32 before #1398 judged the probe's decisive rows
  on fp32 CPU logits. So a PROVED proof doubles as #1398's regression check on a 5090.
- **Unchanged.** The reading already VOIDs on any `qwen3_moe` router census short of 48. The reducer's self-test now
  has 23 cases and the box's 9; the three lane scripts are re-pinned before any box ran.
