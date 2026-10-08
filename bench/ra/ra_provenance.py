"""Independent wheel/install/import probe; run with the selected venv's python -I -B.

Stdlib until wheel bytes are verified. A manifest is an input to review, not
spending authority, feature engagement, or proof clearance.
"""
from __future__ import annotations

import argparse
import base64
import configparser
import csv
import email.parser
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import re
import stat
import sys
import sysconfig
import zipfile
from pathlib import Path, PurePosixPath

RELEASES = {"e4b": "experts4bit-qlora", "gnf4": "grouped-nf4-gemm"}


def require(ok, message):
    if not ok:
        raise ValueError("wheel provenance: " + message)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def regular(path, boundary=None):
    require(path.is_absolute() and path.is_file() and
            not any(p.is_symlink() for p in (path, *path.parents)), "nonregular/symlink file")
    if boundary is not None:
        require(path.is_relative_to(boundary), "installed path escapes venv")
    return path


def digest(path):
    with regular(path).open("rb") as stream:
        return sha(stream)


def record_rows(data):
    rows = list(csv.reader(io.StringIO(data)))
    require(all(len(r) == 3 for r in rows) and len({r[0] for r in rows}) == len(rows), "duplicate/malformed RECORD")
    return {r[0]: (r[1], r[2]) for r in rows}


def wheel(pin):
    require(set(pin) == {"path", "sha256"}, "wheel pin fields")
    path = Path(pin["path"])
    require(path.suffix == ".whl" and digest(path) == pin["sha256"], "wheel archive digest")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len({e.filename for e in entries}) == len(entries), "duplicate ZIP entry")
        files = {}
        for entry in entries:
            parts = PurePosixPath(entry.filename).parts
            require(parts and not entry.filename.startswith("/") and ".." not in parts and
                    "\\" not in entry.filename and not stat.S_ISLNK(entry.external_attr >> 16), "unsafe ZIP entry")
            if entry.is_dir():
                continue
            require(not entry.filename.endswith((".pth", ".pyc")), "startup/bytecode wheel payload refused")
            with archive.open(entry) as stream:
                files[entry.filename] = (sha(stream), entry.file_size)
        metadata = [p for p in files if p.endswith(".dist-info/METADATA")]
        require(len(metadata) == 1, "one wheel METADATA required")
        info = metadata[0].rsplit("/", 1)[0]
        msg = email.parser.BytesParser().parsebytes(archive.read(metadata[0]))
        require(msg["Name"] and msg["Version"], "missing wheel identity")
        name, version = canonical(msg["Name"]), msg["Version"]
        require(canonical(path.name.split("-")[0]) == name and path.name.split("-")[1] == version,
                "wheel metadata/name/version")
        records = record_rows(archive.read(info + "/RECORD").decode())
        require(set(records) == set(files), "wheel RECORD coverage")
        for rel, (actual, size) in files.items():
            hashed, length = records[rel]
            if rel == info + "/RECORD":
                require(hashed == length == "", "wheel RECORD self hash")
            else:
                expected = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
                require(hashed == expected and length == str(size), "wheel RECORD bytes")
        entrypoints = configparser.ConfigParser(interpolation=None)
        entrypoints.optionxform = str
        if info + "/entry_points.txt" in files:
            entrypoints.read_string(archive.read(info + "/entry_points.txt").decode())
    return {"name": name, "version": version, "info": info, "files": files,
            "wheel_sha256": pin["sha256"], "entrypoints": entrypoints, "path": str(path)}


def destination(rel, site):
    parts = PurePosixPath(rel).parts
    if parts[0].endswith(".data"):
        require(len(parts) > 2 and parts[1] in ("purelib", "platlib"), "wheel relocation needs reviewed adapter")
        parts = parts[2:]
    return site.joinpath(*parts)


def generated_scripts(package, prefix):
    expected = {}
    for group in ("console_scripts", "gui_scripts"):
        entries = package["entrypoints"]
        if not entries.has_section(group):
            continue
        # Only called after every installed wheel payload, including pip's,
        # has been compared to its archive. Derive the installer's own template.
        from pip._internal.operations.install.wheel import PipScriptMaker, get_console_script_specs
        from pip._vendor.distlib.util import get_export_entry
        require("ENSUREPIP_OPTIONS" not in os.environ, "unexpected installer environment")
        maker = PipScriptMaker(None, str(prefix / "bin"))
        specifications = (get_console_script_specs(dict(entries.items(group))) if group == "console_scripts" else
                          [name + " = " + value for name, value in entries.items(group)])
        for specification in specifications:
            name = specification.split("=", 1)[0].strip()
            require(re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None, "entry-point name")
            entry = get_export_entry(specification)
            require(entry is not None and entry.suffix is not None, "entry-point target")
            text = maker._get_script_text(entry).encode("utf-8")
            shebang = maker._get_shebang("utf-8")
            expected[prefix / "bin" / name] = shebang + text
    return expected


def probe(manifest):
    require(set(manifest) == {"schema", "venv", "wheels", "imports"} and manifest["schema"] == 1,
            "manifest fields/schema")
    prefix = Path(manifest["venv"])
    require(prefix.is_absolute() and not any(p.is_symlink() for p in (prefix, *prefix.parents)) and
            Path(sys.prefix) == prefix and sys.prefix != sys.base_prefix and sys.flags.isolated and
            sys.dont_write_bytecode, "selected isolated venv/python required")
    sites = {Path(sysconfig.get_path(k)) for k in ("purelib", "platlib")}
    require(all(p.is_relative_to(prefix) for p in sites), "site directory outside venv")
    packages = {}
    for pin in manifest["wheels"]:
        package = wheel(pin)
        require(package["name"] not in packages, "duplicate wheel distribution")
        packages[package["name"]] = package
    installed = {}
    for dist in importlib.metadata.distributions(path=[str(p) for p in sorted(sites)]):
        name = canonical(dist.metadata["Name"])
        require(name not in installed, "duplicate installed distribution")
        installed[name] = dist
    require(set(installed) == set(packages), "installed inventory differs from wheel manifest")
    owned, extras, trees, payload_hashes = {}, {}, {}, {}
    for name, package in packages.items():
        dist = installed[name]
        require(dist.version == package["version"], "installed version differs from wheel")
        site = Path(dist.locate_file(""))
        require(site in sites, "distribution not in selected site")
        payload = {}
        for rel, (expected, size) in package["files"].items():
            if rel == package["info"] + "/RECORD":
                continue  # Installer rewrites this, checked separately below.
            path = regular(destination(rel, site), prefix)
            require(digest(path) == expected and path.stat().st_size == size, "installed bytes differ from wheel")
            require(path not in owned, "two wheels own one installed file")
            owned[path] = name
            payload_hashes[path] = expected
            payload[str(path.relative_to(prefix))] = expected
        trees[name] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        text = dist.read_text("RECORD")
        require(text is not None, "installed RECORD missing")
        rows = record_rows(text)
        recorded = {}
        for rel, (hashed, size) in rows.items():
            path = regular(Path(dist.locate_file(rel)).absolute().resolve(), prefix)
            # Reject symlinks on the unresolved path too (resolve alone hides them).
            regular(Path(dist.locate_file(rel)).absolute(), prefix)
            require(path not in recorded, "duplicate installed target")
            require(path not in owned or owned[path] == name, "installed RECORD claims another wheel's file")
            recorded[path] = name
            if path == site / package["info"] / "RECORD":
                require(hashed == size == "", "installed RECORD self hash")
            else:
                actual = digest(path)
                expected = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
                require(hashed == expected and size == str(path.stat().st_size), "installed RECORD bytes")
        require(set(payload) <= {str(p.relative_to(prefix)) for p in recorded}, "installed RECORD omits wheel file")
        extras[name] = {p for p in recorded if p not in owned}
    for name, paths in extras.items():
        package = packages[name]
        if any(p.parent == prefix / "bin" for p in paths):
            require("pip" in packages, "script verification requires pinned pip wheel")
        scripts = generated_scripts(package, prefix)
        require(set(scripts) <= paths, "installed entry-point missing from RECORD")
        for path in paths:
            if path in scripts:
                require(path.read_bytes() == scripts[path], "generated entry-point bytes")
            else:
                require(path.parent.name == package["info"], "unrecognized installed file")
                if path.name == "INSTALLER":
                    require(path.read_bytes() == b"pip\n", "installer identity")
                elif path.name == "REQUESTED":
                    require(path.read_bytes() == b"", "REQUESTED contents")
                elif path.name == "direct_url.json":
                    direct = json.loads(path.read_bytes())
                    require(direct == {"url": Path(package["path"]).as_uri(),
                            "archive_info": {"hash": "sha256=" + package["wheel_sha256"],
                                             "hashes": {"sha256": package["wheel_sha256"]}}}, "direct/editable origin")
                else:
                    require(path.name == "RECORD", "unrecognized install metadata")
            require(path not in owned, "duplicate installed owner")
            owned[path] = name
    for site in sites:
        for path in site.rglob("*"):
            require(not path.is_symlink(), "symlink in site tree")
            if path.is_file():
                require(path in owned, "unlisted site file/bytecode")
    installed_hashes = {p: digest(p) for p in owned}
    require(set(manifest["imports"]) == set(RELEASES), "release import set")
    imports = {}
    for alias, name in RELEASES.items():
        modules = manifest["imports"][alias]
        require(name in packages and isinstance(modules, list) and modules and
                len(set(modules)) == len(modules), "release import modules")
        paths = {}
        for module in modules:
            imported = importlib.import_module(module)
            path = Path(imported.__file__).absolute()
            require(owned.get(path) == name and path in payload_hashes and digest(path) == payload_hashes[path],
                    "import resolves outside selected wheel")
            paths[module] = str(path)
        imports[alias] = {"venv": str(prefix), "wheel_verified": True, "tree_sha256": trees[name], "modules": paths}
    # An import may initialize or mutate its own package. Recheck source bytes
    # and refuse module origins outside the verified wheel payload or stdlib.
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    stdlib_zip = stdlib.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    for entry in sys.path:
        path = Path(entry).absolute().resolve()
        require(bool(entry) and (path == stdlib_zip or path.is_relative_to(stdlib) and
                "site-packages" not in path.parts or any(path.is_relative_to(p) for p in sites)),
                "import added foreign search path")
    for module in list(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if not origin or origin.startswith("<"):
            continue
        path = Path(origin).absolute()
        if path == Path(__file__).absolute():
            continue
        if path in payload_hashes:
            require(digest(path) == payload_hashes[path], "import mutated wheel payload")
        else:
            require(path.resolve().is_relative_to(stdlib) and "site-packages" not in path.parts,
                    "loaded module outside verified wheels/stdlib: " + str(path))
    for path, expected in installed_hashes.items():
        require(digest(path) == expected, "import mutated verified installation")
    for site in sites:
        require(all(not p.is_symlink() and (not p.is_file() or p in owned) for p in site.rglob("*")),
                "import created unlisted site file")
    distributions = {n: {"version": p["version"], "wheel_sha256": p["wheel_sha256"],
                         "tree_sha256": trees[n]} for n, p in packages.items()}
    common = {n: v for n, v in distributions.items() if n not in RELEASES.values()}
    common_digest = hashlib.sha256(json.dumps(common, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": 1, "venv": str(prefix), "python": sys.version,
            "platform": sysconfig.get_platform(), "soabi": sysconfig.get_config_var("SOABI"), "imports": imports,
            "distributions": distributions, "wheel_lock_sha256": common_digest,
            "proves_gpu_engagement": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    regular(args.manifest)
    require(args.out.is_absolute(), "absolute receipt output required")
    original = digest(args.manifest)
    result = probe(json.loads(args.manifest.read_bytes()))
    require(digest(args.manifest) == original, "manifest changed during verification")
    result["manifest_sha256"] = original
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
