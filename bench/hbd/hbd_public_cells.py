#!/usr/bin/env python3
"""HBD's public cell projections (bench/hbd/PREREG-hbd.md): from each cell receipt, the two fields hbd_reduce.cell_view
reads, ``{"status", "hp1"}``, written deterministically. A cell receipt also carries the rented host's identifiers, so it
stays in the private receipt store; its projection is public, and with a spec that points at the projections the
registered reducer rebuilds ``result.json`` from public files alone.

    python hbd_public_cells.py RECEIPT_DIR OUT_DIR

RECEIPT_DIR holds ``hbd_spec.json`` and the cell receipts it names; OUT_DIR gets ``cells/<cell>.json`` and
``hbd_spec.public.json``, which keeps every other path of the spec (the records and the probes) as it is.
"""
from __future__ import annotations

import argparse
import json
import os

FIELDS = ("status", "hp1")


def project(receipt):
    """The fields the reducer reads, and nothing else."""
    return {k: receipt[k] for k in FIELDS if k in receipt}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("receipt_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    spec = json.load(open(os.path.join(a.receipt_dir, "hbd_spec.json")))
    os.makedirs(os.path.join(a.out_dir, "cells"), exist_ok=True)
    public = {"cells": {}}
    for cell, files in sorted(spec["cells"].items()):
        receipt = json.load(open(os.path.join(a.receipt_dir, files["receipt"])))
        with open(os.path.join(a.out_dir, "cells", f"{cell}.json"), "w") as f:
            json.dump(project(receipt), f, indent=1, sort_keys=True)
            f.write("\n")
        public["cells"][cell] = dict(files, receipt=f"cells/{cell}.json")
    with open(os.path.join(a.out_dir, "hbd_spec.public.json"), "w") as f:
        json.dump(public, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"HBD public cells: {len(public['cells'])} projections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
