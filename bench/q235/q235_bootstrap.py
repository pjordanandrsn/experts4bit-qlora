# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's box install (bench/q235/PREREG-q235.md, "Before any rental", gates 3-4). Never creates compute.

1. Every wheel in the hash-locked ``wheels.json`` is fetched with DQ11's checked opener (allowlisted origins only), its
   sha256 and size checked, and the set installed by pip with ``--require-hashes --no-index --no-deps --only-binary=:all:``.
2. grouped-nf4-gemm and this repository are installed from git at their pinned commits (git is provided first by
   ``dq11_require_git.sh``), and pip's recorded commit for each must equal the pin.
3. ``install.json`` records the lock's sha256, the commits and the versions the interpreter then imports.

    python q235_bootstrap.py --cache DIR          (E4B_SHA and GNF4_SHA from the environment; 40-hex each)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dq11_bootstrap import fetch_wheel, read_wheel, validate_wheel_origins
from dq11_common import file_sha

GIT = {"grouped-nf4-gemm": ("https://github.com/pjordanandrsn/grouped-nf4-gemm.git", "GNF4_SHA", "grouped_nf4_gemm"),
       "experts4bit-qlora": ("https://github.com/pjordanandrsn/experts4bit-qlora.git", "E4B_SHA", "experts4bit_qlora")}
VERSIONS = ("torch", "transformers", "bitsandbytes", "accelerate", "huggingface_hub", "safetensors", "tokenizers", "numpy",
            "triton", "grouped-nf4-gemm", "experts4bit-qlora")


def check_lock(lock: dict) -> list:
    """The lock's rows, refused unless every one is complete, unique, https, safely named and from an allowlisted origin."""
    if lock.get("schema") != "q235-wheels/1":
        raise ValueError(f"wheels.json schema {lock.get('schema')!r} is not q235-wheels/1")
    rows = lock.get("packages") or []
    if not rows:
        raise ValueError("wheels.json locks no package")
    seen = set()
    for r in rows:
        name = r.get("name", "")
        if not name or name.lower() in seen:
            raise ValueError(f"wheels.json: missing or repeated name {name!r}")
        seen.add(name.lower())
        if Path(r.get("filename", "")).name != r.get("filename") or not r["filename"].endswith(".whl"):
            raise ValueError(f"{name}: unsafe or non-wheel filename {r.get('filename')!r}")
        if not str(r.get("url", "")).startswith("https://"):
            raise ValueError(f"{name}: not an https URL")
        if not re.fullmatch("[0-9a-f]{64}", str(r.get("sha256", ""))) or not isinstance(r.get("size"), int) or r["size"] <= 0:
            raise ValueError(f"{name}: no sha256 or size")
    validate_wheel_origins(rows)
    return rows


def fetch(rows: list, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for r in rows:
        path = cache / r["filename"]
        if not (path.is_file() and path.stat().st_size == r["size"] and file_sha(path) == r["sha256"]):
            part = path.with_suffix(".partial")
            with fetch_wheel(r["url"]) as source, part.open("wb") as target:
                while block := read_wheel(source, r["url"], 1 << 20):
                    target.write(block)
            part.replace(path)
        if path.stat().st_size != r["size"] or file_sha(path) != r["sha256"]:
            raise ValueError(f"{r['name']}: the fetched wheel does not match the lock")


def pins() -> dict:
    out = {}
    for repo, (url, env, _) in GIT.items():
        sha = os.environ.get(env, "")
        if not re.fullmatch("[0-9a-f]{40}", sha):
            raise ValueError(f"{env} is not a 40-hex commit")
        out[repo] = (url, sha)
    return out


def installed_commit(dist: str):
    from importlib.metadata import distribution
    raw = distribution(dist).read_text("direct_url.json")
    return (json.loads(raw).get("vcs_info") or {}).get("commit_id") if raw else None


def install(directory: Path, cache: Path) -> dict:
    lock_path = directory / "wheels.json"
    rows = check_lock(json.loads(lock_path.read_text()))
    commits = pins()
    fetch(rows, cache)
    req = cache / "requirements.local.txt"
    req.write_text("".join(f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}\n" for r in rows))
    pip = [sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
    subprocess.run(pip + ["--no-deps", "--no-index", "--only-binary=:all:", "--require-hashes", "--find-links", str(cache), "-r", str(req)],
                   check=True)
    for repo, (url, sha) in commits.items():
        subprocess.run(pip + ["--no-deps", "--no-build-isolation", f"git+{url}@{sha}"], check=True)
        got = installed_commit(repo)
        if got != sha:
            raise ValueError(f"{repo}: pip recorded commit {got}, not the pin {sha}")
    probe = ("import json, importlib.metadata as m; print(json.dumps({d: m.version(d) for d in %r}))" % (VERSIONS,))
    versions = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    rec = {"schema": "q235-install/1", "wheel_lock_sha256": file_sha(lock_path), "wheels": len(rows),
           "commits": {repo: sha for repo, (_, sha) in commits.items()}, "versions": versions, "python": sys.version.split()[0]}
    (directory / "install.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--directory", type=Path, default=Path.cwd())
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(install(a.directory, a.cache)))


if __name__ == "__main__":
    main()
