"""Hash-locked box installation and source authority; never creates compute."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse
import urllib.request
import zipfile

from dq11_common import file_sha


def install(directory, cache):
    wheels = json.loads((directory / "wheels.json").read_text())["packages"]
    cache.mkdir(parents=True, exist_ok=True)
    authority, local_requirements = set(), []
    for row in wheels:
        if urlparse(row["url"]).hostname not in {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}:
            raise ValueError("unregistered wheel origin")
        if Path(row["filename"]).name != row["filename"]:
            raise ValueError("unsafe wheel filename")
        path = cache / row["filename"]
        if not path.is_file():
            temporary = path.with_suffix(".partial")
            with urllib.request.urlopen(row["url"], timeout=60) as source, temporary.open("wb") as target:
                while block := source.read(1 << 20):
                    target.write(block)
            temporary.replace(path)
        if file_sha(path) != row["sha256"]:
            raise ValueError("resolved wheel hash mismatch")
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith(".py"):
                    authority.add(hashlib.sha256(archive.read(name)).hexdigest())
        local_requirements.append(f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}")
    requirements = cache / "requirements.local.txt"
    requirements.write_text("\n".join(local_requirements) + "\n")
    subprocess.run(["python", "-m", "pip", "install", "--no-deps", "--no-index", "--only-binary=:all:",
                    "--require-hashes", "--find-links", str(cache), "-r", str(requirements)], check=True)
    import os

    e4b_sha = os.environ["E4B_SHA"]
    if not re.fullmatch("[0-9a-f]{40}", e4b_sha):
        raise ValueError("invalid reviewed source object")
    for repository, sha, package in (("experts4bit-qlora", e4b_sha, "experts4bit_qlora"),
                                      ("loggetta", "34ecb6cec6f43a6f8607ff9f192749fdc7b587e9", "loggetta")):
        url = "https://github.com/pjordanandrsn/" + repository + ".git"
        clone = cache / (repository + "-" + sha)
        if not clone.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "fetch", "origin", sha], check=True)
        names = subprocess.check_output(["git", "-C", str(clone), "ls-tree", "-r", "--name-only", sha, package], text=True).splitlines()
        for name in names:
            if name.endswith(".py"):
                data = subprocess.check_output(["git", "-C", str(clone), "show", sha + ":" + name])
                authority.add(hashlib.sha256(data).hexdigest())
        subprocess.run(["python", "-m", "pip", "install", "--no-deps", "--no-build-isolation", "git+" + url + "@" + sha], check=True)
    result = {"schema": "dq11-source-authority/1", "sha256": sorted(authority),
              "wheel_lock_sha256": file_sha(directory / "wheels.json"), "e4b_sha": e4b_sha}
    (directory / "source_authority.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path, default=Path("/root/.cache/dq11-wheels"))
    args = parser.parse_args()
    install(args.directory, args.cache)
