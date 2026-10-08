### P121 registered (#846): K25 (`E4B_NF4_GROUPED_SMALLM=auto`, the default) against the NF4 M-tile on Qwen3-30B-A3B at the served W16 step, quality and speed on one box (bench and tests only)

- **Why.** K25 has no family gate and takes Qwen3's W16 step (128 routed rows). Its licence (P93's speed, P96's
  quality) read Granite and OLMoE only. FAM0's inventory (#1368) lists it as the one unread cell in Qwen3's serving row.
- **The box** (`bench/p121/p121_box.py`, from P115 Phase D's):
  - **speed:** ABBA arms K0 (`0`) and K1 (`auto`) on the default graph server at 16 slots, through W16 and W1. W1 is a
    token-identity null control, since T == 1 runs the same kernels in both arms.
  - **quality:** P115 Phase B's teacher-forced instrument at 16 windows a pass on wikitext and c4val1, against P110's
    floor, with a mutant that must fail.
  - **route counter:** wraps K25 and the M-tile, so every record shows which kernel ran.
  - **continuity row:** P96's arms at T == 1, reported with no bar.
- **The rule** (`bench/p121/p121_reduce.py`, 36 self-test cases), first match wins: VOID, NOISY, FUNCTION_FAIL,
  QUALITY_FAIL (P110's bar), SLOWER (g16 < 1.00), else LICENSED. The perplexity move and each arm's peak
  allocated and reserved memory are reported, not gated.
- **Premise on the card:** the decode-graph bucket tests and K25's row-exactness GPU tests, 11 passed.
- **Budget:** proof on Granite (guard 0.75 h), reading guard 2.0 h, lane ceiling $3.00.
