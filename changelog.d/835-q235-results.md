### Q235 results (#835): MILESTONE PASS, one full-depth QLoRA optimizer step on Qwen3-235B-A22B-Instruct-2507

`bench/q235/RESULTS-q235.md` records Q235's eight run ids and the milestone read. `q235-pro6000-8` ran under Amendment 16 on
one RTX PRO 6000 Blackwell Workstation Edition: all 94 decoder layers, every routed expert, two Shape B optimizer steps, a
fresh-process repeat. `q235_verdict.py` reads MILESTONE PASS, "rules 1-10 hold", against the full-depth layer-streamed bf16
reference, and the maintainer re-derived it on CPU. The receipts (`bench/q235/receipts/`) are the box's own outputs,
byte-identical to the private receipt store, with relocation manifests for the per-layer tensors. No default, gate or claim
beyond the record changes.
