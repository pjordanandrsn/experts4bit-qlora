### Training routing trace: Qwen3-30B-A3B result and the cross-step bar

`bench/train-routing-trace/RESULTS-qwen3-30b-a2000.md` reads the A2000 capture against the bar registered before it:
INCONCLUSIVE (LFU 1.36x the shuffled null at 768 of 6,144 rows; Belady 1.25x). The best possible frozen set reaches
only 1.09x the null, so training-side expert residency is not worth building now. A training micro-batch touches
about 83 of 128 experts per layer, which leaves routed-only staging (about 35% fewer bytes) as an untimed candidate. `cross_step_bar.py` computes
the bar (a policy frozen at each step boundary, scored on the next step's forward) and its reporting reads; its tests
check that its policies decide exactly as `replay.py`'s and that the verdict can come out both ways. Counts and bytes
only, no timings.
