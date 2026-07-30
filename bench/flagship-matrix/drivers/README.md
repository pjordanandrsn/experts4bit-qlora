# The code that produced this matrix

These ran on rented pods and were never committed. The first ten cells' driver
survived only in a scratchpad; the pod it ran on is long gone. A result whose
driver cannot be read is not reproducible, and — as it turned out — not
auditable either: the vacuous B1 gate documented in
[`../RESULTS-flagship-matrix.md`](../RESULTS-flagship-matrix.md) was only found
by reading this code.

## First ten cells (Qwen3-30B-A3B) — **as they ran, unmodified**

| file | what it is |
|---|---|
| `n9_cell.py` | one cell: load → LoRA → arm the arm → 200 steps → receipt. **This is the driver that produced every committed `Qwen3-30B-A3B__*__*.json`.** |
| `n9_matrix.py` | the in-process loop over model × dataset × arm; superseded by a shell loop, kept because it is the same measurement code |
| `n9_datasets.py` | generates the five synthetic datasets; seeds 1000–1004 reproduce the registered sha256s |

**Deliberately not fixed.** `n9_cell.py:55` hashes the packed bytes with
`getattr(m, attr)`, which under `offload=True` reads a 0-element placeholder —
the defect that made B1 vacuous. Patching it here would misrepresent what ran.
The corrected form is `n17_cell.py`, next to it, so the two can be diffed.

## Second ten cells (the prereg's second model) — corrected

| file | what it is |
|---|---|
| `n17_setup.sh` | install + **gate**. Nine gates (sm_89, gemma4 present, fused kernel importable, the offload post-hook fix, a real device quantize) plus all five dataset sha256s, then writes a `SETUP_OK` sentinel. |
| `n17_matrix.sh` | five datasets × two arms; refuses to start without `SETUP_OK`; skips any cell whose receipt already exists, so a dead pod loses at most one cell |
| `n17_cell.py` | the corrected cell driver |

`n17_cell.py` adds exactly what
[`../../../docs/PREREG-flagship-matrix-model2.md`](../../../docs/PREREG-flagship-matrix-model2.md)
C1 registers and `n9_cell.py` lacked:

1. **Hashes come from `state_dict()`**, not module attributes — under offload the
   registered tensors are placeholders and the CPU home is where the bytes live.
2. **`bytes_hashed > 0` and `empties_skipped == 0` are asserted**, and a
   byte-flip positive control must fire before any training step. A byte-identity
   check with no demonstrated failure mode is a constant function.
3. **The arm must be the arm.** `enable_fast_train()` returns 0 silently when
   `grouped-nf4-gemm` is missing, which would run the reference path under a
   fused label; the fused arm now asserts a non-zero patch count.

The setup gate exists because the previous second-model attempt was armed with
**no install step** and all ten cells failed in seconds with
`ModuleNotFoundError` — a pod's provisioning paid for nothing. A gate failure
now costs one minute instead of ten cells.
