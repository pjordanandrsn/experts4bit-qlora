#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc5_record.py -- lane SC5 (#1478 item 4): assemble the box's files into the record ``sc5_reduce.py`` reads.

The box (``sc5_box_m.sh``) writes, under ``--dir``:
- ``blocks/<NN>_d<D>_<memory>_<framework>_b<k>/``, one directory per block in run order (``NN`` = its position):
  - ``block.json``: ``framework``, ``block`` (k), ``draw``, ``memory``, ``ready``, ``versions_ok``, ``kv_tokens``,
    ``kv_rounding``, ``peak_mem_mib``;
  - ``c1.json``, ``c16.json``, ``c64.json``: ``sc5_driver.py run`` outputs, whose ``summary`` is the cell.
- ``quality/``:
  - ``ref_sha256.txt``: the sha256 the box computed over the reference's bytes;
  - ``windows.json``: the windows file, whose ``windows_sha256`` is recorded;
  - ``cmp_<row>.json``: ``sc5_quality.py compare`` outputs, one per row (``e4b``, ``vllm``, ``sglang``, ``e4b_decode``,
    ``floor``).

A block directory whose ``block.json`` is missing is recorded as never ready. A missing cell file is left out, so the
reducer VOIDs that block for the missing cell. Nothing is inferred or filled in.

    sc5_record.py --dir DIR --ref-sha256 HEX --out sc5.json
    sc5_record.py --self-test
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FRAMEWORKS = ("e4b", "vllm", "sglang")
CELLS = ("1", "16", "64")
BLOCK_DIR = re.compile(r"^(\d+)_d(\d+)_(default|matched)_(e4b|vllm|sglang)_b([12])$")


def load_block(d: Path, draw: str, memory: str, framework: str, k: int) -> dict:
    meta_path = d / "block.json"
    if meta_path.is_file():
        b = json.loads(meta_path.read_text())
    else:
        b = {"ready": False, "versions_ok": False, "error": "block.json missing"}
    b.update({"framework": framework, "block": k, "draw": draw, "memory": memory})
    cells = {}
    for c in CELLS:
        f = d / f"c{c}.json"
        if f.is_file():
            cells[c] = json.loads(f.read_text())["summary"]
    b["cells"] = cells
    return b


def assemble(root: Path, ref_sha256_expected: str) -> dict:
    draws: dict = {}
    seen = []
    for d in sorted((root / "blocks").iterdir()) if (root / "blocks").is_dir() else []:
        m = BLOCK_DIR.match(d.name)
        if not d.is_dir() or not m:
            continue
        nn, draw, memory, fw, k = m.groups()
        seen.append(int(nn))
        draws.setdefault(draw, {}).setdefault(memory, []).append(load_block(d, draw, memory, fw, int(k)))
    if seen != sorted(set(seen)):
        raise ValueError(f"block run positions repeat: {seen}")
    q = root / "quality"
    rows = {}
    for f in sorted(q.glob("cmp_*.json")) if q.is_dir() else []:
        rows[f.stem[len("cmp_"):]] = json.loads(f.read_text())
    ref_sha = (q / "ref_sha256.txt").read_text().split()[0] if (q / "ref_sha256.txt").is_file() else None
    wsha = json.loads((q / "windows.json").read_text()).get("windows_sha256") if (q / "windows.json").is_file() else None
    return {"frameworks": list(FRAMEWORKS), "draws": draws,
            "quality": {"reference_sha256": ref_sha, "reference_sha256_expected": ref_sha256_expected,
                        "windows_sha256": wsha, "rows": rows}}


def self_test() -> int:
    import tempfile

    import sc5_reduce as red
    ok = []
    want = red._record()
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        nn = 0
        for draw, mems in want["draws"].items():
            for memory in red.MEMORY:
                for b in mems[memory]:
                    d = root / "blocks" / f"{nn:02d}_d{draw}_{memory}_{b['framework']}_b{b['block']}"
                    d.mkdir(parents=True)
                    meta = {k: v for k, v in b.items() if k != "cells"}
                    (d / "block.json").write_text(json.dumps(meta))
                    for c, cell in b["cells"].items():
                        (d / f"c{c}.json").write_text(json.dumps({"summary": cell, "requests": []}))
                    nn += 1
        q = root / "quality"
        q.mkdir()
        (q / "ref_sha256.txt").write_text(want["quality"]["reference_sha256"] + "  ref.json\n")
        (q / "windows.json").write_text(json.dumps({"windows_sha256": want["quality"]["windows_sha256"]}))
        for name, row in want["quality"]["rows"].items():
            (q / f"cmp_{name}.json").write_text(json.dumps(row))
        rec = assemble(root, want["quality"]["reference_sha256_expected"])
        # the round trip reduces to the very verdict the reducer's own fixture gives
        ok.append(red.reduce_obj(rec) == red.reduce_obj(want))
        ok.append(red.reduce_obj(rec)["void"] == [] and red.reduce_obj(rec)["verdict"] == "READ")
        # a missing cell file is a VOID for its block, never a filled-in cell
        victim = sorted((root / "blocks").iterdir())[0]
        (victim / "c64.json").unlink()
        void = red.reduce_obj(assemble(root, want["quality"]["reference_sha256_expected"]))["void"]
        ok.append(any("cell C=64 missing" in v for v in void))
        # a block that never wrote block.json is recorded as never ready
        (victim / "block.json").unlink()
        void = red.reduce_obj(assemble(root, want["quality"]["reference_sha256_expected"]))["void"]
        ok.append(any("never ready" in v for v in void))
        # the wrong reference hash VOIDs every quality row
        rr = red.reduce_obj(assemble(root, "c" * 64))
        ok.append(rr["quality"] == {} and any("sha256 differs" in v for v in rr["quality_void"]))
    print(f"sc5_record self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir")
    ap.add_argument("--ref-sha256")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.dir and a.ref_sha256 and a.out):
        ap.error("--dir --ref-sha256 --out, or --self-test")
    rec = assemble(Path(a.dir), a.ref_sha256)
    json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
    n = sum(len(v) for m in rec["draws"].values() for v in m.values())
    print(f"SC5_RECORD draws={sorted(rec['draws'])} blocks={n} quality_rows={sorted(rec['quality']['rows'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
