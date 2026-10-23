"""``E4B_TRAIN_DETERMINISTIC=1``: the fused training path's deterministic mode (opt-in; no default changes).

Eager training on the fused path is not run-to-run deterministic on CUDA. Two sources were found and measured:

* the MoE token gather's backward (``engines/fast.py``): ``index_select``'s ``index_add_`` adds each token's k gradient
  copies with float atomics. ``E4B_TRAIN_DET_GATHER=1`` takes the batched path's deterministic backward instead;
* the attention backward: SDPA's flash, memory-efficient and cuDNN backwards are nondeterministic by default, and
  become deterministic only under ``torch.use_deterministic_algorithms(True)`` with ``warn_only=False``.

This knob turns on, together:

1. ``E4B_TRAIN_DET_GATHER=1``;
2. ``torch.use_deterministic_algorithms(True)`` scoped to the BACKWARD only (:func:`deterministic_backward`): the forward
   and its loss run with it off, because the CUDA NLL loss raises under it; the checkpoint recompute runs inside the
   backward, so it is covered;
3. ``CUBLAS_WORKSPACE_CONFIG`` (``:4096:8`` or ``:16:8``), which cuBLAS reads when it creates its handle, so it must be in
   the environment before the process touches CUDA. :func:`enable_training_determinism` refuses by name when it is not;
4. ``torch.utils.deterministic.fill_uninitialized_memory = False``, so the mode does not pay a fill on every ``empty``.

THE DETERMINISTIC MODE IS THIS KNOB PLUS A SELF-IDENTITY CHECK IN THE GATE'S OWN RUN, NEVER THE FLAG ALONE. A gate that
states "bitwise to eager" on the training path is stated against eager under this mode, and the same run must show the
mode self-identical: two blocks of three steps, every trainable gradient bitwise equal to the first step's. Turning the
fill off (4) is safe only while no kernel on the path reads memory it did not write; a new or changed kernel could
break that silently. If the in-run check fails, the gate reads VOID, and the same run with the fill ON (the mode
without piece 4) tells an uninitialised read (the fill restores identity) from a nondeterministic op (it does not).
"""
from __future__ import annotations

import contextlib
import os

import torch

#: The CUBLAS_WORKSPACE_CONFIG values that make cuBLAS deterministic (torch's reproducibility notes).
CUBLAS_DETERMINISTIC = (":4096:8", ":16:8")
#: Set by :func:`enable_training_determinism`; :func:`deterministic_backward` refuses to run the mode without it.
STATE = {"enabled": False}


def deterministic_requested() -> bool:
    """``E4B_TRAIN_DETERMINISTIC=1``: anything else is off."""
    return os.environ.get("E4B_TRAIN_DETERMINISTIC", "0").strip() == "1"


def enable_training_determinism() -> None:
    """Turn on pieces 1, 3 (checked) and 4 for this process. Call once at trainer start, before any CUDA work.

    Refuses (``RuntimeError``) when CUDA is already initialized (checked, not a calling convention: a cuBLAS handle made
    before this call never read the setting); when ``CUBLAS_WORKSPACE_CONFIG`` is not one of
    :data:`CUBLAS_DETERMINISTIC`, since setting it here would be too late for a cuBLAS handle that may already exist; and
    when ``E4B_TRAIN_DET_GATHER`` is set to
    anything but ``1`` (the two knobs would disagree). It never runs silently non-deterministic."""
    if torch.cuda.is_initialized():
        raise RuntimeError("E4B_TRAIN_DETERMINISTIC=1 must be enabled before any CUDA work: CUDA is already initialized, so a "
                           "cuBLAS handle may exist that never read CUBLAS_WORKSPACE_CONFIG. Enable it at trainer start.")
    cw = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "").strip()
    if cw not in CUBLAS_DETERMINISTIC:
        raise RuntimeError(
            "E4B_TRAIN_DETERMINISTIC=1 needs CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) in the environment before the "
            f"process starts (cuBLAS reads it when it creates its handle); it is {cw or 'unset'!r}. Refusing rather than "
            "training non-deterministically.")
    gather = os.environ.get("E4B_TRAIN_DET_GATHER")
    if gather is not None and gather.strip() != "1":
        raise RuntimeError(f"E4B_TRAIN_DETERMINISTIC=1 with E4B_TRAIN_DET_GATHER={gather!r}: the deterministic mode needs "
                           "the deterministic token gather; unset E4B_TRAIN_DET_GATHER or set it to 1")
    os.environ["E4B_TRAIN_DET_GATHER"] = "1"
    from torch.utils import deterministic as _deterministic   # a bare `import torch.utils...` here would make torch local
    _deterministic.fill_uninitialized_memory = False
    STATE["enabled"] = True


@contextlib.contextmanager
def deterministic_backward(enabled: bool | None = None):
    """Piece 2: ``torch.use_deterministic_algorithms(True)`` for the body (the backward) and OFF after it, even when it
    raises. ``enabled`` defaults to :func:`deterministic_requested`; off, it does nothing. On, it refuses when the mode
    is already on at entry (a previous step left it on, which would change that step's forward) and when
    :func:`enable_training_determinism` was not called in this process."""
    on = deterministic_requested() if enabled is None else enabled
    if not on:
        yield
        return
    if not STATE["enabled"]:
        raise RuntimeError("deterministic_backward under E4B_TRAIN_DETERMINISTIC=1 before enable_training_determinism(): "
                           "call it once at trainer start, before any CUDA work")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("deterministic algorithms are already on before the backward: a previous step left them on")
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)
