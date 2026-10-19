# `q235-pro6000-8`: the box under Amendment 16, MILESTONE PASS on rules 1-10

**The run:** one RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB, cc 12.0, driver 595.71.05) on machine 141671,
Vast instance 56688656. On the box from 07:11:20Z to 09:03:42Z on 2026-10-19. Cost $4.567: GPU $2.910, download $1.094 for
420 GB, storage $0.505, upload $0.058.

**The code:** e4b `27200917` (Amendment 16's merge) and grouped-nf4-gemm `724ccc45`. The host: an Intel Xeon Gold 6330
(56 physical cores, 112 logical) with 503.5 GiB of RAM.

**What ran:**
- install, the host-rate check and the fetch (Qwen/Qwen3-235B-A22B-Instruct-2507 at `ac9c66cc`, 61 min), then the tokens;
- the box: load, two Shape B steps, and the held-out and dX records;
- the fresh-process repeat of step 0, with the in-process plant passes (routing and epilogue).

The box phase took 2,047 s under its 8,653 s cap. The repeat took 843 s under its 1,439 s cap. Every phase exited 0.

**Verdict: MILESTONE PASS**, "rules 1-10 hold" (`verdict.27200917.json`, from `q235_verdict.py` at e4b `27200917`). The
reference was the full-depth layer-streamed bf16 reference on this box's tokens, run on the lab's RTX A2000 (`read/`). The
maintainer re-derived the verdict on CPU with identical rule booleans and instrument checks. See `../../RESULTS-q235.md`.

## Files

- **`tc1/receipts/`:** the box's records.
  - `box/box.json`, `box/nf4.json`, `box/load_layers.jsonl`;
  - the repeat's, with its plant passes under `repeat/`;
  - `box-cap.json`, `repeat-cap.json`, `cpu.json`, `host.json`, `hostrate.json`, `fetch.json`, `card.txt`;
  - `phase-timings.jsonl`, `run.json`, and the tokens' and held-out rows' sha256 records.
- **`tc1/summary.txt`, `tc1/outer.log`, `tc1/logs/`:** the phase log and each phase's log.
- **`tc1/install.json`, `tc1/ds_manifest.json`:** the install record and the dataset manifest.
- **`tc1/RELOCATED.sha256`:** the 380 per-layer tensors (8.2 GB) moved out of the receipt store, by sha256. Copies are on the
  lab NAS and the controller host.
- **`read/`:** the A2000 read's record.
  - the read log and the verdict it wrote;
  - `reference.json`, `ref-layers.sha256` and the per-layer times;
  - the stager's log;
  - `RELOCATED.sha256` for the 16.0 MB reference tensor.
- **`STORE-ONLY.sha256`:** the files kept only in the private receipt store, by sha256 and store path. These are the token
  and held-out tensors; the read's lab scripts, stage record and own `SHA256SUMS` (which name lab paths); and the fetch
  log, which carries carriage returns from the download's progress output.
- **`teardown-proof.json`, `q235_fetch_gate.json`:** the launcher's teardown proof and its pre-rental fetch-gate record.

The launcher's receipt and ledger row are in the private receipt store. `SHA256SUMS` covers every file here, and every
file but this README is byte-identical to the store's copy.
