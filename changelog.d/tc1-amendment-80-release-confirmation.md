### TC1 amendment 80 registered: the confirmation read of amendment 79, with a version-aware engagement check and a card floor (e4b#2129)

- Amendment 80 reads amendment 79's box again at the 0.58.1 release. It closes the two defects e4b#2129 found:
  - Unsloth 2026.10.8 runs its expert GEMMs on its own Triton kernels, which TC1's engagement check (F15 v1, `torch._grouped_mm` only)
    cannot see;
  - amendment 79's card held a flat 2400 MHz with NVML's Reliability clock reason set.
- **Every distribution comes from PyPI.** The arms' experts4bit-qlora 0.58.1, grouped-nf4-gemm 0.51.0, unsloth and unsloth_zoo 2026.10.8 are
  PyPI's wheels, each proven by its bytes with `release_proof.py`: a fail-fast copy after install, and a binding copy in the arms' import
  environment before the first arm.
- **`bench/tc1/f15v2.py`, new:** Unsloth's expert-GEMM engagement counted on the dispatch the pinned unsloth_zoo actually takes, with exact
  bars tied to the arm's own expert-forward count:
  - 9E grouped GEMM and 2E dW launches on the Triton path, derived from the release-proven source;
  - 11E on the torch path, registered for zoo 2026.9.9 only.
  A mixed dispatch, a bypass or a double dispatch VOIDs the draw. It also gives a profile's kernels-only device time, without the
  `Optimizer.step` annotation span TC1's `device_ms` includes. F15 v1 is printed for this family, not binding.
- **`bench/common/card_clock.py`, new, shared:** a loaded SM-clock host floor. A 20 s bf16 burn must read a busy median >= 2700 MHz on the
  RTX 5090 with no slowdown or cap reason, else rc 18. The same check runs on each draw's clock log, which VOIDs a capped draw. The reason
  names are NVML's, from nvidia-ml-py 13.615.71. The 2700 MHz floor is calibrated on TC1's 76 committed RTX 5090 runs: 59 clear it.
- **The profile window moves to steps 20..22.** Amendment 79's two Unsloth draws differed 16 % in steps 3..5 because draw 1 compiled
  Unsloth's Mega-cache bundle and draw 2 loaded it. Each draw's cache state is reported beside its row.
- **Predictions:** TC1's P225 (F15 v2 reads ENGAGED on both Unsloth draws) and P226 (the device ratio Unsloth / e4b < 1.00).
- **Budget:** $5.94 worst. `tc1_arm.py` is unchanged. `tests/test_card_clock.py` and `tests/test_tc1_pos80.py` read real committed
  receipts: amendment 69's clean card and torch path, and amendment 79's capped card and Triton path.
