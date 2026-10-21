#!/usr/bin/env python3
"""HPN's public projections (bench/hpn/HPN-DESIGN.md): from each input hpn_reduce reads, only the fields it reads, written
deterministically under ``proj/``. The original files also carry the host's identifiers and the run's own paths, so they stay
in the private receipt store; with the projections and a spec that points at them, the registered reducer rebuilds
``hpn/result.json`` from public files alone.

    python hpn_public_inloop.py RUN_DIR OUT_DIR

The fields kept:
- an in-loop arm receipt: ``eval_curve`` (its step-0 per-row losses);
- an evaluator output: per base, ``digest``, ``n_slots``, ``digest_matches_training_run``, ``status`` and ``zero_adapter``
  (``row_losses``, ``row_mean``, ``token_weighted``);
- a reference (F) output: ``reader``, ``digest``, ``n_slots``, ``runs`` and ``row_mean``.

OUT_DIR gets ``proj/<path in the spec>`` for each input and ``hpn_spec.public.json``, which names the projections.
"""
from __future__ import annotations

import argparse
import copy
import json
import os

BASE_FIELDS = ("digest", "n_slots", "digest_matches_training_run", "status", "zero_adapter")
FREF_FIELDS = ("reader", "digest", "n_slots", "runs", "row_mean")


def project_inloop(r):
    return {k: r[k] for k in ("eval_curve",) if k in r}


def project_eval(r):
    return {"bases": [{k: b[k] for k in BASE_FIELDS if k in b} for b in r.get("bases", [])]}


def project_fref(r):
    return {k: r[k] for k in FREF_FIELDS if k in r}


def _write(out_dir, rel, obj):
    path = os.path.join(out_dir, "proj", rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    return "proj/" + rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    spec = json.load(open(os.path.join(a.run_dir, "hpn_spec.json")))
    load = lambda rel: json.load(open(os.path.join(a.run_dir, rel)))  # noqa: E731
    public, n = copy.deepcopy(spec), 0
    for d in public["depths"]:
        for grp in [d] + ([d["unsloth"]] if d.get("unsloth") else []):
            grp["inloop"] = [_write(a.out_dir, rel, project_inloop(load(rel))) for rel in grp["inloop"]]
            grp["eval"] = [_write(a.out_dir, rel, project_eval(load(rel))) for rel in grp["eval"]]
            grp["fref"] = _write(a.out_dir, grp["fref"], project_fref(load(grp["fref"])))
            n += len(grp["inloop"]) + len(grp["eval"]) + 1
    with open(os.path.join(a.out_dir, "hpn_spec.public.json"), "w") as f:
        json.dump(public, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"HPN public projections: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
