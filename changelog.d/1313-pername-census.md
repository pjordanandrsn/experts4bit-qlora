### The per-name served census (#1313): where the B=1 step's moe_expert time goes, kernel by kernel

The B=1 step's moe_expert class runs 0.388 ms (9.7 % of the step) over K33's isolated time for the same two expert GEMVs,
and the class census cannot say where. This registers P136's box on Qwen3-30B-A3B at W1 with one added step:
`bench/pername/pername_table.py` splits every in-graph kernel of each kept replay into its idle gap, its own advance of
the critical path, and the time it ran under its predecessor (programmatic dependent launch). The split closes each
replay exactly. The rule is fixed before data. The swiglu-into-down fusion is read next iff `_swiglu_rows` removes at
least 3 % of the step; otherwise the top line by addressable time is proposed. An instrument: no default or speed claim
changes.
