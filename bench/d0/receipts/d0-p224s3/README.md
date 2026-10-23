# d0-p224s3: D0 on the p224-shake-3 fixtures (OLMoE-1B-7B, CPU)

The run's outputs, as written. The registration is [`../../PREREG-d0.md`](../../PREREG-d0.md) with its Amendment 1. The
result is [`../../RESULTS-d0.md`](../../RESULTS-d0.md), and the re-derivation is [`../../REDERIVE.md`](../../REDERIVE.md).

- **Fixtures:** `fixture-dq` (double-quantized expert absmax) and `fixture-nodq` (fp32 absmax): r = 8, alpha 16, trained 200
  steps on `tatsu-lab/alpaca` at `dce01c9b`, text column, seed 42, on OLMoE-1B-7B-0924-Instruct at `7f1c97f4`, by the P224
  shakedown's declared fixture step. `fixture-base` is the trained base's stored expert stacks. The tensors are not here.
- **Host:** a capped CPU container (2 CPUs, 16 GB, no GPU), fixtures mounted read-only. One run, 2026-10-22, 16:36:19Z to
  20:24:34Z ([`run.log`](run.log)). The environment is [`pip-freeze.txt`](pip-freeze.txt): torch 2.8.0+cpu,
  bitsandbytes 0.50.2, transformers 5.17.0, experts4bit-qlora `a40077ba`, Loggetta `90a7697d`. grouped-nf4-gemm is not
  installed. `tools/d0lib.py` imports its two reference packers from a checkout at `c6215964`, whose files equal that
  commit's blobs ([`tools/gnf4-packers.sha256`](tools/gnf4-packers.sha256)).
- **Tools:** [`tools/`](tools/) is a byte copy of the tools that ran. Their paths are the run host's.
  - The registration did not pin the tools' hashes, and no copy was stored before the run. That the computation tools were
    unchanged since 2026-10-21 rests on the author's statement. `d0_data.py` is the exception, per the deviation below.
  - `run_d0.sh` is orchestration only, written before the run.
- **Deviation D-1 (before any pass or read):** an earlier start of the same run refused at its data step. Loggetta's profile
  check reported "tokenizer differ". The only differing field was the tokenizer's recorded `name`: the plan recorded the hub
  id, and D0 loads the same files from the fixture folder.
  - `d0_data.py` now asserts that every other `tokenizer_identity` field equals the plan's (class, vocabulary size, EOS,
    chat-template sha256) and exits `D0_DATA VOID` otherwise. Only then does it align the name. Loggetta's own check still
    decides on the encoded stream, rows, tokens, format and options.
  - A negative test (an edited chat template) exits 1.
  - That earlier start was stopped before any pass. Its refused data step is kept in the private store.
- **Per fixture** (`dq/`, `nodq/`):
  - `data.log` and `heldout.json`: the unseen proof and the 64 held-out rows, 7,642 scored tokens.
  - `pass.log` and `pass/`: `cells_LNN.csv`, `checks_LNN.csv`, `routing_LNN.csv` per layer, and `arms.csv` (15 arms x 64
    rows).
  - `step1-complete.log`, `reduce.log` and `summary.json`.
- **Plumbing gate:** [`gate.log`](gate.log) (2 layers, max |dlogit| 5.72e-06 against transformers).
- **Sums:** [`SHA256SUMS`](SHA256SUMS) covers every file here. [`STORE-ONLY.sha256`](STORE-ONLY.sha256) lists what stays
  private.
