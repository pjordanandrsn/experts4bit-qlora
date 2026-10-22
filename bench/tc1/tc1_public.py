"""A TC1 box receipt's public copy: the files that carry no host identifier, plus derived copies of the three that do, from
which the registered reducer rebuilds the box's RESULTS.

The box's own ``box.json``, cell receipts and ``summary.txt`` name the rented host (the instance id, the container's host name, the
GPU's UUID), so they stay in the private receipt store, byte-exact, with ``RESULTS-tc1.md`` (which embeds ``box.json``). ``derive``
writes copies of those three with exactly the identifying fields removed:
- ``box.json`` without ``instance_id`` and ``hostname``;
- each cell receipt without ``env.host``'s ``uuid``, ``vast_instance_id`` and ``container_host``;
- ``summary.txt`` without its lines naming the instance or the host (the BOX line). The reducer reads only its ``fam/fw/tag rc=``
  and ``LOADGATE`` lines.
Everything else is unchanged, and ``derive`` refuses if a derived file still names the instance, the host or a GPU UUID.

``rebuild`` puts the committed identifier-free files and the derived copies in one ``tc1/`` directory and runs this checkout's
``tc1_reduce.py`` on it. The result differs from the store's ``RESULTS-tc1.md`` only in the two ``box.json`` lines removed.

    python bench/tc1/tc1_public.py derive STORE_TC1_DIR OUT_PUBLIC_DIR
    python bench/tc1/tc1_public.py rebuild RECEIPT_DIR --md OUT.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BOX_DROP = ("instance_id", "hostname")
HOST_DROP = ("uuid", "vast_instance_id", "container_host")
GPU_UUID = re.compile(r"\b(?:GPU|MIG)-[0-9a-f]{8}-[0-9a-f]{4}-")
DERIVED = ("box.json", "summary.txt")


def _cells(d):
    """The cell receipts: top-level ``*.json`` files carrying ``env.host`` (the arm records ``tc1_arm.py`` writes)."""
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and f not in DERIVED:
            try:
                js = json.load(open(os.path.join(d, f)))
            except (OSError, ValueError):
                continue
            if isinstance(js, dict) and isinstance((js.get("env") or {}).get("host"), dict):
                out.append(f)
    return out


def identifiers(store_tc1):
    """The strings that name this box's host: the instance id and host name from its ``box.json``."""
    box = json.load(open(os.path.join(store_tc1, "box.json")))
    return [str(box[k]) for k in BOX_DROP if box.get(k)]


def leaks(text, ids):
    """The identifiers ``text`` still carries: any of ``ids`` and any GPU UUID."""
    return sorted({i for i in ids if i in text} | set(GPU_UUID.findall(text)))


def derive(store_tc1, out):
    ids = identifiers(store_tc1)
    os.makedirs(out, exist_ok=True)
    box = json.load(open(os.path.join(store_tc1, "box.json")))
    written = {"box.json": json.dumps({k: v for k, v in box.items() if k not in BOX_DROP}, indent=1)}
    for f in _cells(store_tc1):
        js = json.load(open(os.path.join(store_tc1, f)))
        for k in HOST_DROP:
            js["env"]["host"].pop(k, None)
        written[f] = json.dumps(js, indent=1)
    lines = open(os.path.join(store_tc1, "summary.txt"), encoding="utf-8").read().splitlines(keepends=True)
    written["summary.txt"] = "".join(ln for ln in lines if not leaks(ln, ids))
    bad = {f: leaks(t, ids) for f, t in written.items() if leaks(t, ids)}
    if bad:
        raise SystemExit(f"derive refused: a derived file still names the host: {bad}")
    for f, t in written.items():
        open(os.path.join(out, f), "w", encoding="utf-8").write(t)
    return sorted(written)


def rebuild(receipt_dir, md):
    """Run the reducer on ``RECEIPT_DIR/tc1`` (the committed files) overlaid with ``RECEIPT_DIR/public`` (the derived ones)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "tc1")                # the reducer titles its report with the directory's name
        shutil.copytree(os.path.join(receipt_dir, "tc1"), work)
        pub = os.path.join(receipt_dir, "public")
        for f in os.listdir(pub):
            if f in DERIVED or f.endswith(".json"):
                shutil.copy(os.path.join(pub, f), os.path.join(work, f))
        out = os.path.join(tmp, "RESULTS-tc1.md")
        subprocess.run([sys.executable, os.path.join(HERE, "tc1_reduce.py"), work, "--md", out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(out, md)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    d.add_argument("store_tc1")
    d.add_argument("out")
    r = sub.add_parser("rebuild")
    r.add_argument("receipt_dir")
    r.add_argument("--md", required=True)
    a = ap.parse_args()
    if a.cmd == "derive":
        print("derived:", ", ".join(derive(a.store_tc1, a.out)))
    else:
        rebuild(a.receipt_dir, a.md)
        print("rebuilt", a.md)


if __name__ == "__main__":
    main()
