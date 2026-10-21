### Bench: phase stamps for new rental kits

`bench/common/phases.sh` gives runners `phase_start NAME KIND` and `phase_end NAME`, which append one JSON line per phase change to `phases.jsonl`. The kind is setup, measure, reduce or other. `bench/common/phases.py` reads and checks the file, reporting seconds per kind, the first measurement's start, and any phase left open. New kits write it, so a later read of setup time against measurement time needs no log parsing. P149's earlier stamps read as kind `unknown`. Existing kits and receipts are unchanged.
