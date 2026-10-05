### Read: DQ1 run 2 — no dense W4A16 speed primitive at QLoRA rows (S_DEAD, G1_PARITY, GF_LOSS); streaming marginal on PCIe 4.0 (bench only)

- `dq1-5090-2` ($0.10, RTX 5090, PCIe 4.0 x16, under Amendment 1). The lane READs; 6/250 self-pairs are out of band.
- **Speed.** A perfect bf16-math 4-bit kernel could save at most ~10% of base-linear time at 2048 tokens, ~5% at 4096
  (H 0.097 / 0.050; 0.097 is within draw noise of the line, and the consequence is the same either way). That share is
  bnb's dequant, within ±10%.
- **G=1 and fused.** grouped-nf4-gemm at G=1 (`auto` = dense route) is at parity (1.002–1.010). Its packed kernel is
  3.2–4.5× slower. bitsandbytes 0.50.2 takes dequant + cuBLAS at every census row on sm_120.
- **LoRA.** PEFT's unfused delta costs ~9–10% of base-linear time at ≥ 2048 tokens, as much as or more than the whole
  dequant headroom.
- **Streaming.** DMA costs the GEMMs ≤ 2.5%. The forward phase binds: Rmin 1.07 at 2048, 1.93 at 4096 on 27.9 GB/s.
  Model-size-free rule: break-even at M ≈ 0.26·F/B. C_MARGINAL licenses no prototype; a PCIe 5.0 lane is the
  registered next step.
- **Write-ups.** `bench/dq1/RESULTS-dq1.md` and `SUMMARY-dq1.md`: speed is a negative result, there is no new
  repository, and the next lanes are ranked.
