# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's checkpoint on the box: the files ``model_files.json`` locks, at its revision, each verified by content.

- **full** (the box): all locked files are fetched with huggingface_hub's ``snapshot_download`` at the pinned revision into
  ``--dest``. Every file is then checked against the lock: its size, and its sha256 (LFS files: the shards and
  ``tokenizer.json``) or its git blob id (the rest). Any absent, extra-sized or mismatched file refuses.
- **slice N** (the A2000 rehearsal): the same, for the non-shard files and the shards that hold the first N decoder layers and
  every non-layer tensor (from the locked index), then ``mk_slice.py`` builds the depth-truncated view beside them.
- **only GLOB...** (the CPU rehearsal): the same, for the locked files the patterns name (the tokenizer's).

**The pull rate is measured on the fetch itself.** The fetch starts with a timed first slice -- the non-shard files and the first
``--probe-shards`` shards, through the same ``snapshot_download`` and workers, in a child process -- and records its bytes, seconds
and rate. The slice has a deadline, its bytes over ``--min-gb-s``: a host that has not finished it by then is refused at once, and so
is one that finished below the rate; either exits 14 (the launcher's host code for a pull rate under the floor) before paying for the
rest. Above it, the fetch goes on and skips what the slice already holds.

Records ``fetch.json``: files, bytes, the slice's rate, the download and hashing seconds, and any refusal. Exits 0 when every file
matches.

    python q235_fetch.py --lock model_files.json --dest DIR [--slice N --slice-dir DIR | --only GLOB...] [--workers 8]
                         [--min-gb-s X --probe-shards 8]
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


def probe_names(names: list, shards: int) -> list:
    """The fetch's timed first slice: every non-shard file, and the first ``shards`` shards in name order."""
    first = sorted(n for n in names if n.endswith(".safetensors"))[:max(0, shards)]
    return [n for n in names if not n.endswith(".safetensors")] + first


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
    ap.add_argument("--min-gb-s", type=float, default=0.0, help="refuse (exit 14) when the first slice pulls slower than this")
    ap.add_argument("--probe-shards", type=int, default=8)
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
    sizes = {f["file"]: f["size"] for f in lock["files"]}
    first = probe_names(names, a.probe_shards)
    slice_bytes = sum(sizes[n] for n in first)
    deadline = slice_bytes / (a.min_gb_s * 1e9) if a.min_gb_s else None
    child = ("import json, sys; from huggingface_hub import snapshot_download; a = json.loads(sys.argv[1]); "
             "snapshot_download(a['repo'], revision=a['rev'], local_dir=a['dest'], allow_patterns=a['names'], max_workers=a['workers'])")
    arg = json.dumps({"repo": lock["repo"], "rev": lock["revision"], "dest": str(a.dest), "names": first, "workers": a.workers})
    tp = time.time()
    try:
        r = subprocess.run([sys.executable, "-c", child, arg], timeout=deadline)
        finished, rc = True, r.returncode
    except subprocess.TimeoutExpired:
        finished, rc = False, None
    dt = max(time.time() - tp, 1e-3)
    done = sum(sizes[n] for n in first if (a.dest / n).is_file() and (a.dest / n).stat().st_size == sizes[n])
    rec["probe"] = {"files": len(first), "slice_bytes": slice_bytes, "completed_bytes": done, "seconds": round(dt, 1),
                    "gb_s": round(done / dt / 1e9, 4), "finished": finished, "deadline_s": round(deadline, 1) if deadline else None,
                    "floor_gb_s": a.min_gb_s}
    if finished and rc != 0:
        sys.exit(f"the first slice's download failed (rc {rc})")
    if a.min_gb_s and (not finished or rec["probe"]["gb_s"] < a.min_gb_s):
        why = (f"the first {len(first)} files ({slice_bytes / 1e9:.1f} GB) were not done by the {deadline:.0f} s deadline "
               f"({done / 1e9:.1f} GB complete)" if not finished else f"pull rate {rec['probe']['gb_s']} GB/s < {a.min_gb_s}")
        rec.update(ok=False, refusals=[why])
        a.out.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({"refused": why}))
        sys.exit(14)
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
