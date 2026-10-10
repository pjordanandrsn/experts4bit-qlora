### Training routing trace harness (no capture yet)

`bench/train-routing-trace/` records which experts a QLoRA training step routes to, per micro-batch and per pass, and
replays the trace through residency policies to give hit rates and staged bytes. No timings.
- `routing_recorder.py` hooks every `layers.<i>.mlp.gate`. It records the forward and the checkpoint recompute, and
  checks that they route identically; a test arms that check with a deliberately divergent recompute.
- `run_capture.py` runs the TC1 harness unchanged, with the recorder attached.
- `replay.py` scores profile, LFU, LRU and Belady at six expert-row budgets, with a shuffled-routing null and the #1469
  decode traces as controls.

The Qwen3-30B-A3B capture on the RTX A2000 runs from the merged commit. No number is claimed until then.
