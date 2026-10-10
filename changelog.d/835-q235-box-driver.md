### Q235: the box driver and gate 2's time-left fit (bench/q235)

- **What it adds.** The lifecycle of the registered Q235 box around the already merged step (`q235_box.py`), and the fit that
  sizes its time-left guards. `q235_drive.sh` runs `bench/tc1/tc1_drive.sh` with a closure that `q235_stage.py` seals against
  the committed `q235.sha256`; on the box `q235_run.sh`:
  - re-checks that closure and provides git;
  - refuses a host below the registered floors with the launcher's host codes (18: card class, VRAM or host RAM; 13: disk);
  - installs the 56 hash-locked wheels of `wheels.json` (resolved natively for CPython 3.11 on manylinux 2.28 x86_64: torch
    2.11.0+cu128, transformers 5.19.0, bitsandbytes 0.50.2, gate 1's environment) and grouped-nf4-gemm and this repository
    at their commits;
  - fetches the 129 files of `model_files.json` at `ac9c66cc`, each checked by sha256 or git blob id;
  - renders the tokens and runs the two-step box and its fresh-process repeat.

  Every phase runs under an alarm capped by the deadline and is timed into `receipts/phase-timings.jsonl`.
- **Rehearsals.** `Q235_REHEARSAL=a2000` (a 4-layer slice of the checkpoint, fetched and verified the same way) and `cpu` (a
  tiny model with the registered tokenizer) run the same script; neither is ever science-eligible. Science-mode constants live in
  the sealed closure; overrides apply to rehearsals only.
- **Gate 2.** `q235_fit.py` scales each measured phase to the registered model (A2000 rates as an upper bound, x 94/4 layers;
  bytes over the host's listed bandwidth) with a 1.5x margin, and fails when a sealed cap, the wallclock or the cost at the
  host's rate does not fit.
