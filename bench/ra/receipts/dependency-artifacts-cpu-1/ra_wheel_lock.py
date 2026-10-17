"""Bind retained archives to the reviewed proof image and installer inventory.

This is an artifact check. Installation, dependency closure, startup adapters,
release/source binding and on-card engagement are separate required gates.
"""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import urllib.parse
import zipfile
from pathlib import Path

REQUIRED = {"torch": "2.8.0+cu128", "triton": "3.4.0", "transformers": "5.18.0",
            "bitsandbytes": "0.50.2", "peft": "0.21.2"}
RELEASES = {"experts4bit-qlora", "grouped-nf4-gemm"}
HOSTS = {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org", "pypi.nvidia.com"}


def require(ok, message):
    if not ok:
        raise ValueError("wheel lock: " + message)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def digest(path):
    require(path.is_absolute() and path.is_file() and
            not any(p.is_symlink() for p in (path, *path.parents)), "nonregular/symlink input")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(lock, image_bytes):
    require(set(lock) == {"schema", "image_pin_sha256", "roots", "wheels"} and lock["schema"] == 1, "schema/fields")
    require(lock["image_pin_sha256"] == hashlib.sha256(image_bytes).hexdigest(), "proof image binding")
    image = json.loads(image_bytes)
    require(image["image"]["os"] == "linux" and image["image"]["architecture"] == "amd64" and
            image["python"]["version"] == "3.11.13" and image["python"]["soabi"] == "cpython-311-x86_64-linux-gnu",
            "selected interpreter/platform")
    require(isinstance(lock["roots"], list) and lock["roots"] and
            all(isinstance(r, str) and r and "\n" not in r for r in lock["roots"]), "resolver roots")
    require(isinstance(lock["wheels"], list) and lock["wheels"], "empty inventory")
    names, filenames = {}, set()
    for pin in lock["wheels"]:
        require(set(pin) == {"name", "version", "filename", "url", "sha256", "size"}, "wheel fields")
        name, filename = pin["name"], pin["filename"]
        require(isinstance(name, str) and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is not None and
                canonical(name) == name and name not in names, "duplicate/noncanonical distribution")
        require(isinstance(filename, str) and re.fullmatch(r"[A-Za-z0-9_.+\-]+\.whl", filename) is not None and
                filename not in filenames, "duplicate/unsafe filename")
        parts = filename[:-4].split("-")
        require(len(parts) in (5, 6) and canonical(parts[0]) == name and parts[1] == pin["version"], "filename identity")
        py, abi, platform = parts[-3:]
        tags = py.split(".")
        compatible = (abi == "none" and bool(set(tags) & {"py3", "py311", "cp311"}) or
                      abi == "cp311" and "cp311" in tags or
                      abi == "abi3" and any(re.fullmatch(r"cp3(\d+)", t) and
                          2 <= int(t[3:]) <= 11 for t in tags))
        require(compatible, "interpreter wheel tag")
        require(platform == "any" or all(re.fullmatch(r"manylinux(?:1|2010|2014|_2_\d+)_x86_64", p)
                for p in platform.split(".")), "platform wheel tag")
        require(all(int(n) <= 35 for n in re.findall(r"manylinux_2_(\d+)", platform)), "image glibc ceiling")
        require(re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) is not None and
                type(pin["size"]) is int and pin["size"] > 0, "archive hash/size")
        url = urllib.parse.urlsplit(pin["url"])
        require(url.scheme == "https" and url.hostname in HOSTS and url.port in (None, 443) and
                not url.username and not url.password and not url.query and not url.fragment and
                urllib.parse.unquote(url.path.rsplit("/", 1)[-1]) == filename, "artifact origin")
        names[name], _ = pin, filenames.add(filename)
    require(all(names.get(n, {}).get("version") == v for n, v in REQUIRED.items()), "registered versions")
    require({"pip", "uvicorn", *RELEASES} <= names.keys(), "installer/server/release inventory")
    return names


def verify_archives(lock, image_bytes, root):
    names = validate(lock, image_bytes)
    require(root.is_absolute() and root.is_dir() and
            not any(p.is_symlink() for p in (root, *root.parents)), "archive root")
    require({p.name for p in root.iterdir()} == {p["filename"] for p in names.values()}, "archive inventory")
    startup = {}
    for name, pin in names.items():
        path = root / pin["filename"]
        require(digest(path) == pin["sha256"] and path.stat().st_size == pin["size"], "retained archive bytes")
        with zipfile.ZipFile(path) as archive:
            entries = archive.namelist()
            require(len(entries) == len(set(entries)), "duplicate ZIP member")
            metadata = [p for p in entries if p.count("/") == 1 and p.endswith(".dist-info/METADATA")]
            require(len(metadata) == 1, "archive top-level METADATA: " + name)
            info = email.parser.BytesParser().parsebytes(archive.read(metadata[0]))
            require(bool(info["Name"]) and canonical(info["Name"]) == name and
                    info["Version"] == pin["version"], "archive identity")
            # Report a pending installation gate, never authorize startup hooks.
            hooks = [p for p in entries if p.endswith(".pth")]
            if hooks:
                startup[name] = hooks
    common = {n: {k: p[k] for k in ("version", "filename", "sha256", "size")}
              for n, p in names.items() if n not in RELEASES}
    return {"schema": 1, "verified_archives": len(names), "common_dependencies": len(common),
            "common_archive_lock_sha256": hashlib.sha256(json.dumps(common, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(), "startup_adapters_required": startup,
            "proves_installation": False, "proves_dependency_closure": False, "proves_gpu_engagement": False}


def provenance_manifest(lock, image_bytes, root, venv, imports):
    # Recheck bytes at the handoff; the isolated installed-payload probe then
    # owns RECORDs, generated scripts, import ownership and post-import bytes.
    verify_archives(lock, image_bytes, root)
    return {"schema": 1, "venv": str(venv), "wheels": [
        {"path": str(root / p["filename"]), "sha256": p["sha256"]} for p in lock["wheels"]], "imports": imports}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--wheels", required=True, type=Path)
    args = ap.parse_args()
    digest(args.lock)
    digest(args.image)
    print(json.dumps(verify_archives(json.loads(args.lock.read_bytes()), args.image.read_bytes(),
                                   args.wheels), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
