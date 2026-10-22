"""An HO2 box receipt's public copy: the box's files that carry no host identifier, plus derived copies of the four Loggetta
training receipts, from which HO2's reducer rebuilds the box's verdict.

Each row's Loggetta receipt (``run/receipts_<tag>/*.json``) records the GPU's UUID twice, at ``hardware.gpu.uuid`` and
``plan.hardware.gpu.uuid``, so the originals stay in the private receipt store, byte-exact. ``derive`` writes copies with exactly
those two values set to ``null``, serialized the way Loggetta writes a receipt (``json.dumps(..., indent=1, sort_keys=True)``, no
trailing newline), so each copy differs from its original in those two lines only. It refuses if a derived file still names the
box's instance or machine, or any GPU UUID. HO2's reducer never reads the UUID.

``rebuild`` lays the committed ``ho2/`` files and the derived receipts out as the box's run directory, then runs this checkout's
``ho2_reduce.py`` on it with the pins the box recorded (its ``KNOBS`` line). The result must equal the box's own
``ho2/run/verdict_ho2.json`` byte for byte.

    python bench/ho2/ho2_public.py derive STORE_RUN_DIR OUT_PUBLIC_DIR
    python bench/ho2/ho2_public.py rebuild RECEIPT_DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GPU_UUID = re.compile(r"\b(?:GPU|MIG)-[0-9a-f]{8}-[0-9a-f]{4}-")
UUID_PATHS = (("hardware", "gpu", "uuid"), ("plan", "hardware", "gpu", "uuid"))
KNOBS = re.compile(r"^KNOBS e4b=([0-9a-f]{40}) gnf4=([0-9a-f]{40}) loggetta=([0-9a-f]{40}) ", re.M)


def row_receipts(ho2_dir):
    """The rows' Loggetta receipts, relative to ``ho2_dir``: ``run/receipts_<tag>/*.json``."""
    return sorted(os.path.relpath(p, ho2_dir) for p in glob.glob(os.path.join(ho2_dir, "run", "receipts_*", "*.json")))


def identifiers(store_run_dir):
    """The strings that name this box's host: the instance and machine ids the launcher recorded in ``instance.json``."""
    inst = json.load(open(os.path.join(store_run_dir, "instance.json")))
    return [str(inst[k]) for k in ("instance_id", "machine_id") if inst.get(k)]


def leaks(text, ids):
    """The identifiers ``text`` still carries: any of ``ids`` and any GPU UUID."""
    return sorted({i for i in ids if i in text} | set(GPU_UUID.findall(text)))


def _null(js, path):
    node = js
    for k in path[:-1]:
        node = node.get(k) if isinstance(node, dict) else None
        if node is None:
            return
    if isinstance(node, dict) and path[-1] in node:
        node[path[-1]] = None


def derive(store_run_dir, out):
    ids = identifiers(store_run_dir)
    ho2 = os.path.join(store_run_dir, "ho2")
    written = {}
    for rel in row_receipts(ho2):
        js = json.load(open(os.path.join(ho2, rel)))
        for path in UUID_PATHS:
            _null(js, path)
        written[rel] = json.dumps(js, indent=1, sort_keys=True)
    if not written:
        raise SystemExit(f"derive refused: no row receipts under {ho2}/run/receipts_*/")
    bad = {f: leaks(t, ids) for f, t in written.items() if leaks(t, ids)}
    if bad:
        raise SystemExit(f"derive refused: a derived file still names the host: {bad}")
    for rel, t in written.items():
        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
        open(os.path.join(out, rel), "w", encoding="utf-8").write(t)
    return sorted(written)


def rebuild(receipt_dir, out_json):
    """Run this checkout's reducer on ``RECEIPT_DIR/ho2/run`` with ``RECEIPT_DIR/public``'s receipts laid in, at the box's pins."""
    ho2 = os.path.join(receipt_dir, "ho2")
    m = KNOBS.search(open(os.path.join(ho2, "summary.txt"), encoding="utf-8").read())
    if not m:
        raise SystemExit("rebuild refused: summary.txt carries no KNOBS line with the box's pins")
    run_id = json.load(open(os.path.join(ho2, "run", "verdict_ho2.json")))["run_id"]
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "run")
        os.makedirs(run)
        shutil.copy(os.path.join(ho2, "run", "phases.jsonl"), run)
        pub = os.path.join(receipt_dir, "public")
        for rel in row_receipts(pub):
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
            shutil.copy(os.path.join(pub, rel), os.path.join(tmp, rel))
        out = os.path.join(tmp, "verdict_ho2.json")
        subprocess.run([sys.executable, os.path.join(HERE, "ho2_reduce.py"), "--dir", run, "--run-id", run_id, "--e4b", m.group(1),
                        "--gnf4", m.group(2), "--loggetta", m.group(3), "--out", out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(out, out_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    d.add_argument("store_run_dir")
    d.add_argument("out")
    r = sub.add_parser("rebuild")
    r.add_argument("receipt_dir")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "derive":
        print("derived:", ", ".join(derive(a.store_run_dir, a.out)))
    else:
        rebuild(a.receipt_dir, a.out)
        print("rebuilt", a.out)


if __name__ == "__main__":
    main()
