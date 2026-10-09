"""Verify installed wheel payload with python -I -S -B; never activate site.

The proposed startup registry permits byte inspection only. The existing
release-import probe still refuses startup hooks. This is no launch authority.
"""
from __future__ import annotations

import argparse
import base64
import importlib.metadata
import importlib.util
import json
import platform
import sys
import sysconfig
from pathlib import Path


def load_provenance():
    path = Path(__file__).absolute().with_name("ra_provenance.py")
    spec = importlib.util.spec_from_file_location("ra_prestartup_provenance", path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return helper


def layout(manifest, helper):
    require = helper.require
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, "python -I -S -B required")
    require(set(manifest) == {"schema", "venv", "image", "image_sha256", "startup_registry_sha256", "wheels"}
            and manifest["schema"] == 1 and manifest["wheels"], "prestartup manifest fields/schema")
    prefix = Path(manifest["venv"])
    require(prefix.is_absolute() and prefix.is_dir() and
            not any(p.is_symlink() for p in (prefix, *prefix.parents)), "nonregular venv prefix")
    executable = Path(sys.executable).absolute()
    require(executable == prefix / "bin/python", "selected venv executable required")
    config = helper.regular(prefix / "pyvenv.cfg")
    rows = [line.split("=", 1) for line in config.read_text().splitlines() if line.strip()]
    require(all(len(r) == 2 for r in rows), "venv config syntax")
    cfg = {k.strip(): v.strip() for k, v in rows}
    require(len(cfg) == len(rows) and
            {"home", "include-system-site-packages", "version"} <= cfg.keys() and
            cfg.keys() <= {"home", "include-system-site-packages", "version", "executable", "command", "prompt"},
            "venv config fields")
    real = executable.resolve(strict=True)
    base = Path(sys._base_executable).resolve(strict=True)
    require(cfg["include-system-site-packages"] == "false" and cfg["version"] == platform.python_version() and
            Path(cfg["home"]).resolve() == base.parent and
            ("executable" not in cfg or Path(cfg["executable"]).resolve() == base), "venv base/system-site identity")
    # <=3.13 -S leaves sys.prefix at the base. Derive sites from the verified
    # venv location, rather than rewriting interpreter globals or invoking site.
    sites = {Path(sysconfig.get_path(k, vars={"base": str(prefix), "platbase": str(prefix)}))
             for k in ("purelib", "platlib")}
    require(all(p.is_relative_to(prefix) and p.is_dir() for p in sites), "venv site layout")
    image_path = Path(manifest["image"])
    require(helper.digest(image_path) == manifest["image_sha256"], "reviewed image manifest bytes")
    image = json.loads(image_path.read_bytes())
    architecture = {"x86_64": "amd64", "AMD64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine())
    require(image["python"]["version"] == platform.python_version() and
            image["python"]["soabi"] == sysconfig.get_config_var("SOABI") and
            image["python"]["executable_sha256"] == helper.digest(real) and
            image["python"]["executable_sha256"] == helper.digest(base) and
            image["image"]["os"] == sys.platform and image["image"]["architecture"] == architecture,
            "image interpreter binary/ABI/platform")
    return prefix, sites, config, real, helper.digest(real)


def verify(manifest):
    helper = load_provenance()
    require = helper.require
    prefix, sites, config, real_executable, executable_hash = layout(manifest, helper)
    base_executable = Path(sys._base_executable).resolve(strict=True)
    base_hash = helper.digest(base_executable)
    config_hash = helper.digest(config)
    registry_path = Path(__file__).absolute().with_name("startup-audit-adapters.json")
    require(helper.digest(registry_path) == manifest["startup_registry_sha256"], "reviewed startup registry bytes")
    registry = json.loads(registry_path.read_bytes())
    require(set(registry) == {"schema", "status", "adapters"} and registry["schema"] == 1 and
            registry["status"] == "PROPOSED_AUDIT_ONLY" and isinstance(registry["adapters"], list),
            "proposed startup byte registry required")
    packages = {}
    for pin in manifest["wheels"]:
        package = helper.wheel(pin, audit_startup=registry["adapters"])
        require(package["name"] not in packages, "duplicate retained distribution")
        packages[package["name"]] = package
    distributions = {}
    for dist in importlib.metadata.distributions(path=[str(p) for p in sorted(sites)]):
        name = helper.canonical(dist.metadata["Name"])
        require(name not in distributions, "duplicate installed distribution")
        distributions[name] = dist
    require(set(distributions) == set(packages), "installed inventory differs from retained archives")
    owned, hashes, extras, trees = {}, {}, {}, {}
    for name, package in packages.items():
        dist = distributions[name]
        require(dist.version == package["version"], "installed version differs")
        site = Path(dist.locate_file(""))
        require(site in sites, "installed distribution site")
        tree = {}
        for rel, (expected, size) in package["files"].items():
            if rel == package["info"] + "/RECORD":
                continue
            path = helper.regular(helper.destination(rel, site), prefix)
            require(helper.digest(path) == expected and path.stat().st_size == size, "installed payload bytes")
            require(path not in owned, "two wheels own installed file")
            owned[path], hashes[path] = name, expected
            tree[str(path.relative_to(prefix))] = expected
        trees[name] = helper.hashlib.sha256(json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        text = dist.read_text("RECORD")
        require(text is not None, "installed RECORD missing")
        recorded = set()
        for rel, (hashed, size) in helper.record_rows(text).items():
            unresolved = helper.regular(Path(dist.locate_file(rel)).absolute(), prefix)
            path = helper.regular(unresolved.resolve(), prefix)
            require(path not in recorded and (path not in owned or owned[path] == name), "installed RECORD ownership")
            recorded.add(path)
            if path == site / package["info"] / "RECORD":
                require(hashed == size == "", "installed RECORD self hash")
            else:
                actual = helper.digest(path)
                expected = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
                require(hashed == expected and size == str(path.stat().st_size), "installed RECORD bytes")
        require({p for p, owner in owned.items() if owner == name} <= recorded, "installed RECORD coverage")
        extras[name] = recorded - owned.keys()
    # Reject site extras before allowing installer imports: a resealed RECORD
    # cannot turn an unknown .py into an importable installer dependency.
    for name, paths in extras.items():
        package = packages[name]
        for path in paths:
            require(path not in owned, "installed RECORD claims another wheel payload")
            if path.parent == prefix / "bin":
                require("pip" in packages, "generated script needs pinned pip")
                continue
            require(path.parent == Path(distributions[name].locate_file("")) / package["info"],
                    "unknown installed extra")
            if path.name == "INSTALLER":
                require(path.read_bytes() == b"pip\n", "installer identity")
            elif path.name == "REQUESTED":
                require(path.read_bytes() == b"", "REQUESTED contents")
            elif path.name == "direct_url.json":
                require(json.loads(path.read_bytes()) == {"url": Path(package["path"]).as_uri(),
                        "archive_info": {"hash": "sha256=" + package["wheel_sha256"],
                                         "hashes": {"sha256": package["wheel_sha256"]}}}, "direct/editable origin")
            else:
                require(path.name == "RECORD", "unknown installer metadata")
        if any(package["entrypoints"].has_section(group) for group in ("console_scripts", "gui_scripts")):
            require("pip" in packages, "entry-point verification needs pinned pip")
    for package in packages.values():
        require(helper.digest(Path(package["path"])) == package["wheel_sha256"], "archive changed before imports")
    extra_hashes = {p: helper.digest(p) for paths in extras.values() for p in paths}
    # All installed metadata/RECORD bytes and payload are verified before any
    # installer code is imported. Adding paths does not process .pth files.
    for site in sites:
        for path in site.rglob("*"):
            require(not path.is_symlink() and (not path.is_file() or path in owned or
                    any(path in extra for extra in extras.values())), "unlisted site file/bytecode")
    sys.path.extend(str(p) for p in sorted(sites))
    try:
        for name, paths in extras.items():
            package = packages[name]
            scripts = helper.generated_scripts(package, prefix)
            require(set(scripts) <= paths, "generated script missing from RECORD")
            for path in paths:
                if path in scripts:
                    require(path.read_bytes() == scripts[path], "generated script bytes")
                else:
                    require(path.parent == Path(distributions[name].locate_file("")) / package["info"],
                            "unknown installed extra")
                require(path not in owned and helper.digest(path) == extra_hashes[path], "duplicate/changed installed extra")
                owned[path], hashes[path] = name, extra_hashes[path]
        stdlib = Path(sysconfig.get_path("stdlib")).resolve()
        stdlib_zip = stdlib.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
        for entry in sys.path:
            path = Path(entry).absolute().resolve()
            require(bool(entry) and (path == stdlib_zip or
                    path.is_relative_to(stdlib) and "site-packages" not in path.parts or
                    any(path.is_relative_to(p) for p in sites)), "installer added foreign search path")
        for module in list(sys.modules.values()):
            origin = getattr(module, "__file__", None)
            if not origin or origin.startswith("<"):
                continue
            path = Path(origin).absolute()
            # One adjacent tool implements the same-process continuation;
            # no manifest may add arbitrary caller modules to this exception.
            if path in (Path(__file__).absolute(), Path(helper.__file__).absolute(),
                        Path(__file__).absolute().with_name("ra_handoff.py")):
                continue
            require((path in hashes and owned[path] not in helper.RELEASES.values()) or
                    (path.resolve().is_relative_to(stdlib) and "site-packages" not in path.parts),
                    "loaded code outside verified installer dependencies/stdlib")
        for path, expected in hashes.items():
            require(helper.digest(path) == expected, "installer import changed installed bytes")
        for site in sites:
            require(all(not p.is_symlink() and (not p.is_file() or p in owned) for p in site.rglob("*")),
                    "installer import created site file")
    finally:
        for site in sorted(sites):
            sys.path.remove(str(site))
    require(Path(sys.executable).resolve() == real_executable and helper.digest(real_executable) == executable_hash,
            "interpreter changed")
    require(Path(sys._base_executable).resolve() == base_executable and
            helper.digest(base_executable) == base_hash, "base interpreter changed")
    require(helper.digest(config) == config_hash and helper.digest(Path(manifest["image"])) == manifest["image_sha256"] and
            helper.digest(registry_path) == manifest["startup_registry_sha256"], "bootstrap inputs changed")
    common = {n: {"version": p["version"], "wheel_sha256": p["wheel_sha256"], "tree_sha256": trees[n]}
              for n, p in packages.items() if n not in helper.RELEASES.values()}
    common_hash = helper.hashlib.sha256(json.dumps(common, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": 1, "venv": str(prefix), "python": platform.python_version(),
            "image_sha256": manifest["image_sha256"], "startup_registry_sha256": manifest["startup_registry_sha256"],
            "pyvenv_cfg_sha256": config_hash, "executable_sha256": executable_hash, "installed_files": len(hashes),
            "base_executable_sha256": base_hash, "copied_executable": real_executable != base_executable,
            "distributions": {n: {"version": p["version"], "wheel_sha256": p["wheel_sha256"],
                                   "tree_sha256": trees[n], "startup": p["startup"]} for n, p in packages.items()},
            "wheel_lock_sha256": common_hash, "proves_installed_payload": True, "startup_activated": False,
            "proves_full_image_installation": False, "proves_release_imports": False,
            "proves_release_source_binding": False, "proves_gpu_engagement": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    helper = load_provenance()
    before = helper.digest(args.manifest)
    result = verify(json.loads(args.manifest.read_bytes()))
    helper.require(helper.digest(args.manifest) == before and args.out.is_absolute(), "manifest changed/output path")
    result["manifest_sha256"] = before
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
