# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's checkpoint on the box: the files ``model_files.json`` locks, at its revision, each verified by content.

- **full** (the box): all locked files are fetched with huggingface_hub's ``snapshot_download`` at the pinned revision into
  ``--dest``. Every file is then checked against the lock: its size, and its sha256 (LFS files: the shards and
  ``tokenizer.json``) or its git blob id (the rest). Any absent, extra-sized or mismatched file refuses.
- **slice N** (the A2000 rehearsal): the same, for the non-shard files and the shards that hold the first N decoder layers and
  every non-layer tensor (from the locked index), then ``mk_slice.py`` builds the depth-truncated view beside them.
- **only GLOB...** (the CPU rehearsal): the same, for the locked files the patterns name (the tokenizer's).

Records ``fetch.json``: files, bytes, the download and hashing seconds, and any refusal. Exits 0 when every file matches.

    python q235_fetch.py --lock model_files.json --dest DIR [--slice N --slice-dir DIR | --only GLOB...] [--workers 8]
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INDEX = "model.safetensors.index.json"


def content_id(path: str, lfs: bool) -> str:
    """sha256 of the bytes (LFS files), or the git blob id: sha1 over ``blob <size>\\0`` + the bytes (the rest)."""
    h = hashlib.sha256() if lfs else hashlib.sha1(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def slice_files(lock: dict, index: dict, n: int) -> list:
    """The locked names a depth-n slice needs: every non-shard file, and each shard holding a layer < n or a non-layer tensor."""
    need = set()
    for key, shard in index["weight_map"].items():
        parts = key.split(".")
        if not (len(parts) > 2 and parts[0] == "model" and parts[1] == "layers" and int(parts[2]) >= n):
            need.add(shard)
    return [f["file"] for f in lock["files"] if not f["file"].endswith(".safetensors") or f["file"] in need]


def verify(lock: dict, dest: Path, names: list, workers: int) -> list:
    """Refusals for ``names`` under ``dest`` against the lock (size first, then content; hashlib releases the GIL on large
    buffers, so the files hash in parallel on threads)."""
    rows = {f["file"]: f for f in lock["files"]}
    out, jobs = [], []
    for name in names:
        r, p = rows[name], dest / name
        if not p.is_file():
            out.append(f"{name}: absent")
        elif p.stat().st_size != r["size"]:
            out.append(f"{name}: {p.stat().st_size} bytes against the lock's {r['size']}")
        else:
            jobs.append((name, str(p), bool(r.get("lfs_sha256")), r.get("lfs_sha256") or r["blob_id"]))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        got = list(pool.map(content_id, [j[1] for j in jobs], [j[2] for j in jobs]))
    out += [f"{j[0]}: content {g} against the lock's {j[3]}" for j, g in zip(jobs, got) if g != j[3]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--slice", type=int, default=None, help="rehearsal: fetch only what the first N decoder layers need")
    ap.add_argument("--slice-dir", type=Path, default=None)
    ap.add_argument("--only", nargs="+", default=None, help="CPU rehearsal: only the locked files these patterns name")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("fetch.json"))
    a = ap.parse_args()
    lock = json.loads(a.lock.read_text())
    if lock.get("schema") != "q235-model-files/1" or not lock.get("files") or len(lock["files"]) != lock.get("n_files"):
        sys.exit("refusing: model_files.json is not a complete q235-model-files/1 lock")
    from huggingface_hub import hf_hub_download, snapshot_download
    if a.slice is not None and a.only:
        sys.exit("refusing: --slice and --only together")
    mode = f"slice {a.slice}" if a.slice is not None else (f"only {' '.join(a.only)}" if a.only else "full")
    rec = {"schema": "q235-fetch/1", "repo": lock["repo"], "revision": lock["revision"], "mode": mode}
    t0 = time.time()
    names = [f["file"] for f in lock["files"]]
    if a.slice is not None:
        hf_hub_download(lock["repo"], INDEX, revision=lock["revision"], local_dir=a.dest)
        idx_problems = verify(lock, a.dest, [INDEX], 1)
        if idx_problems:
            sys.exit("refusing: " + "; ".join(idx_problems))
        names = slice_files(lock, json.loads((a.dest / INDEX).read_text()), a.slice)
    elif a.only:
        names = [n for n in names if any(fnmatch.fnmatch(n, g) for g in a.only)]
        if not names:
            sys.exit("refusing: --only names no locked file")
    snapshot_download(lock["repo"], revision=lock["revision"], local_dir=a.dest, allow_patterns=names, max_workers=a.workers)
    rec["download_seconds"] = round(time.time() - t0, 1)
    t1 = time.time()
    problems = verify(lock, a.dest, names, a.workers)
    rec.update(files=len(names), bytes=sum(f["size"] for f in lock["files"] if f["file"] in set(names)),
               hash_seconds=round(time.time() - t1, 1), refusals=problems, ok=not problems)
    if not problems and a.slice is not None:
        if a.slice_dir is None:
            sys.exit("refusing: --slice needs --slice-dir")
        mk = Path(__file__).with_name("mk_slice.py")
        rec["slice"] = json.loads(subprocess.check_output([sys.executable, str(mk), str(a.dest), str(a.slice_dir), str(a.slice)], text=True))
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("mode", "files", "bytes", "download_seconds", "hash_seconds", "ok")}))
    sys.exit(0 if rec["ok"] else 1)


if __name__ == "__main__":
    main()
