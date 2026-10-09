# Retained dependency closure

Run `ra_closure.py` with the selected interpreter's `python -I -S -B`, passing
`--lock`, `--image`, `--wheels` and an absolute new `--out` receipt. It verifies
the complete archive inventory and image binding before and after reading
top-level METADATA. Only the verified packaging wheel is imported to parse
requirements, version constraints and markers.

The graph starts at the frozen lock roots, expands extras until stable and
requires every active dependency to be pinned at a compatible version. Python
bounds, undeclared extras, missing dependencies, surplus pins, direct URLs and
host-kernel-dependent markers refuse. The receipt records the explicit
Linux/CPython 3.11.13 marker environment, active edges and metadata hashes.

This establishes closure of the supplied archived metadata for that environment.
It does not prove a full-image installation, startup behavior, release/source
binding, runtime imports, GPU engagement or launch authority. CPU synthetic
graph controls exercise refusal cases; the real retained-wheel check is separate.
