"""Check frozen dependency closure from retained wheels with python -I -S -B.

Only the hash-verified packaging wheel is imported. No installed dependency,
release module, startup hook or GPU code is imported or executed.
"""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import importlib.util
import json
import re
import sys
import sysconfig
import zipfile
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError("dependency closure: " + message)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


ENVIRONMENT = {
    "implementation_name": "cpython", "implementation_version": "3.11.13",
    "os_name": "posix", "platform_machine": "x86_64", "platform_python_implementation": "CPython",
    "platform_system": "Linux", "python_full_version": "3.11.13", "python_version": "3.11",
    "sys_platform": "linux", "platform_release": "", "platform_version": "",
}


def closure(packages, roots, environment, Requirement, Version, SpecifierSet):
    """Monotone extras expansion; refuse missing, incompatible or surplus pins."""
    require(set(environment) == set(ENVIRONMENT) and environment == ENVIRONMENT,
            "registered marker environment")
    require(packages and roots, "empty graph")
    extras, pending, edges = {}, [], set()

    def select(req, parent):
        require(req.url is None, "direct-URL requirement needs reviewed adapter")
        name = canonical(req.name)
        require(name in packages, "missing dependency: " + name)
        package = packages[name]
        require(req.specifier.contains(Version(package["version"]), prereleases=True),
                "dependency version: " + name)
        requested = {canonical(e) for e in req.extras}
        require(requested <= set(package["extras"]), "undeclared dependency extra: " + name)
        if name not in extras:
            extras[name] = set()
            pending.append(name)
        if requested - extras[name]:
            extras[name].update(requested)
            pending.append(name)
        edges.add((parent, name, str(req)))

    for raw in roots:
        req = Requirement(raw)
        require(req.marker is None, "conditional root needs reviewed adapter")
        select(req, "ROOT")
    while pending:
        name = pending.pop()
        package = packages[name]
        require(SpecifierSet(package["requires_python"]).contains(Version(environment["python_full_version"]),
                prereleases=True), "Requires-Python: " + name)
        for raw in package["requires_dist"]:
            req = Requirement(raw)
            if req.marker is not None:
                require(not re.search(r"\bplatform_(?:release|version)\b", str(req.marker)),
                        "host kernel marker needs reviewed adapter")
            if req.marker is None or any(req.marker.evaluate({**environment, "extra": extra})
                                         for extra in {"", *extras[name]}):
                select(req, name)
    require(set(extras) == set(packages), "unreachable pinned distributions: " +
            ",".join(sorted(set(packages) - set(extras))))
    return {"distributions": {n: {"version": packages[n]["version"], "extras": sorted(extras[n])}
                              for n in sorted(extras)},
            "edges": [list(edge) for edge in sorted(edges)], "marker_environment": environment}


def load_tool():
    path = Path(__file__).absolute().with_name("ra_wheel_lock.py")
    spec = importlib.util.spec_from_file_location("ra_closure_lock", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def verify(lock, image_bytes, root):
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,
            "python -I -S -B required")
    require(sys.version_info[:3] == (3, 11, 13) and sys.platform == "linux" and
            sysconfig.get_config_var("SOABI") == "cpython-311-x86_64-linux-gnu",
            "selected proof interpreter required")
    tool = load_tool()
    before = tool.verify_archives(lock, image_bytes, root)
    names = tool.validate(lock, image_bytes)
    require("packaging" in names and not any(n == "packaging" or n.startswith("packaging.") for n in sys.modules),
            "packaging must come from verified archive")
    parser = root / names["packaging"]["filename"]
    with zipfile.ZipFile(parser) as archive:
        require(not any(p.endswith((".pth", ".pyc")) for p in archive.namelist()), "parser startup/bytecode")
    sys.path.insert(0, str(parser))
    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        packages, metadata_hashes = {}, {}
        for name, pin in names.items():
            with zipfile.ZipFile(root / pin["filename"]) as archive:
                paths = [p for p in archive.namelist() if p.count("/") == 1 and p.endswith(".dist-info/METADATA")]
                require(len(paths) == 1, "top-level METADATA")
                data = archive.read(paths[0])
            msg = email.parser.BytesParser().parsebytes(data)
            require(len(msg.get_all("Name", [])) == len(msg.get_all("Version", [])) == 1 and
                    canonical(msg["Name"]) == name and msg["Version"] == pin["version"], "metadata identity")
            require(len(msg.get_all("Requires-Python", [])) <= 1, "duplicate Requires-Python")
            declared = [canonical(e) for e in msg.get_all("Provides-Extra", [])]
            require(len(declared) == len(set(declared)), "duplicate declared extra")
            packages[name] = {"version": pin["version"], "requires_python": msg.get("Requires-Python", ""),
                              "requires_dist": msg.get_all("Requires-Dist", []), "extras": declared}
            metadata_hashes[name] = hashlib.sha256(data).hexdigest()
        result = closure(packages, lock["roots"], ENVIRONMENT, Requirement, Version, SpecifierSet)
        for name, module in sys.modules.copy().items():
            if name == "packaging" or name.startswith("packaging."):
                require(getattr(module, "__file__", "").startswith(str(parser) + "/packaging/"),
                        "parser module owner")
        require(tool.verify_archives(lock, image_bytes, root) == before, "archive inputs changed during closure")
    finally:
        sys.path.remove(str(parser))
    return {"schema": 1, **result, "metadata_sha256": metadata_hashes,
            "packaging_wheel_sha256": names["packaging"]["sha256"],
            "common_archive_lock_sha256": before["common_archive_lock_sha256"],
            "proves_dependency_closure": True, "proves_installation": False,
            "proves_release_source_binding": False, "proves_gpu_engagement": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("lock", "image", "wheels", "out"):
        ap.add_argument("--" + flag, required=True, type=Path)
    args = ap.parse_args()
    tool = load_tool()
    lock_hash, image_hash = tool.digest(args.lock), tool.digest(args.image)
    result = verify(json.loads(args.lock.read_bytes()), args.image.read_bytes(), args.wheels)
    require(tool.digest(args.lock) == lock_hash and tool.digest(args.image) == image_hash, "manifest changed")
    require(args.out.is_absolute(), "absolute output required")
    result.update(lock_sha256=lock_hash, image_sha256=image_hash)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
