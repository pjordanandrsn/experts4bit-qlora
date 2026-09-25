# Benchmarks

*(Moved out of the top-level README for length; linked from it.)* [← back to README](../README.md)

```bash
# Runs on stock bitsandbytes:
python bench/bench_energy_excluded.py                    # memory wall + tokens-per-joule vs batch

# Measured on the bitsandbytes#1965 fork build (0.50.0.dev0) — the `[packed, 1]` matmul_4bit routing; whether a stock bitsandbytes >= 0.50.0 reproduces it is open (#392, see BITSANDBYTES.md):
python bench/_upstream/bench_matmul4bit.py --mode both   # equivalence + latency/memory
python bench/_upstream/bench_energy.py                   # joules/op: bf16 vs dequant vs matmul_4bit
```

The LoRA-placement ablation (which of experts / attention / router to train) and full energy
analysis are written up in [`docs/METHODOLOGY.md`](METHODOLOGY.md). Short version: on Alpaca
the placements are largely **redundant**, attention-only is the efficiency pick, and training the
router **hurts**.
