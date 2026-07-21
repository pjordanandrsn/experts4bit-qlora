# Benchmarks

*(Moved out of the top-level README for length; linked from it.)* [← back to README](../README.md)

```bash
# Runs on stock bitsandbytes:
python bench/bench_energy_excluded.py                    # memory wall + tokens-per-joule vs batch

# Require bitsandbytes >= 0.50 — measure the upstream matmul_4bit routing (#1965):
python bench/_upstream/bench_matmul4bit.py --mode both   # equivalence + latency/memory
python bench/_upstream/bench_energy.py                   # joules/op: bf16 vs dequant vs matmul_4bit
```

The LoRA-placement ablation (which of experts / attention / router to train) and full energy
analysis are written up in [`docs/METHODOLOGY.md`](https://github.com/pjordanandrsn/experts4bit-qlora/blob/v0.6.3/docs/METHODOLOGY.md). Short version: on Alpaca
the placements are largely **redundant**, attention-only is the efficiency pick, and training the
router **hurts**.
