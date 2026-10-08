### TC1 amendment 71 read; amendment 72 registered

- **Amendment 71** (`tc1-5090-139`, RTX 5090, AMD EPYC 7763, $2.75, every arm profiled): grouped-nf4-gemm's
  `NF4_QLORA_SINGLE_LADDER=auto` laddered only the fp32-adapter arm. `aten::bmm`'s CPU time per call fell to 0.120 of
  the unladdered, device time rose 4.3 % (fp32) and 0.2 % (bf16), and held-out was unchanged (P217–P219 HELD). The step
  times went unread (P215, P216 UNTESTED): the host's load rose through the box and three arms' draws differed by more
  than 5 %. `auto` stays opt-in. Register row `e4b.train.single-ladder-auto.field.5090.2026-10-08`.
- **Amendment 72** registered: the same box (token `qwen3slauto`, no new code) on a third host, off amendments 70's and
  71's machines. If its P215 and P216 rows hold (P220, P221) with P219, `auto` becomes grouped-nf4-gemm's default. If
  either is UNTESTED again, no further host-load re-run is registered.
