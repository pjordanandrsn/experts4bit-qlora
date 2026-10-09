"""Proposed same-process payload/startup/import handoff, python -I -S -B.

This draft execution adapter is not review or launch authority. The existing
PROPOSED_AUDIT_ONLY registry and provenance probe retain their refusals.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HOOK = (b"import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'local') == 'local'; "
        b"enabled and __import__('_distutils_hack').add_shim(); \n")


def load_prestartup():
    path = Path(__file__).absolute().with_name("ra_prestartup.py")
    spec = importlib.util.spec_from_file_location("ra_handoff_prestartup", path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return helper


def tree(sites, helper):
    files = {}
    for site in sites:
        for path in site.rglob("*"):
            helper.require(not path.is_symlink(), "handoff site symlink")
            if path.is_file():
                files[path] = helper.digest(path)
    return files


def origins(helper, hashes, owners, stdlib, tool_paths):
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not origin or origin.startswith("<"):
            continue
        path = Path(origin).absolute()
        helper.require(path in tool_paths or path in hashes and path in owners or
                       path.resolve().is_relative_to(stdlib) and "site-packages" not in path.parts,
                       "handoff foreign loaded module: " + name)
        if path in hashes:
            helper.require(helper.digest(path) == hashes[path], "handoff imported payload changed")


def activate(sites, expected, helper):
    """Replace native .pth execution with one literal, byte-bound adapter.

    site.main still derives the native isolated venv search path and prefix.
    Unknown files, calls, customisation modules and env overrides refuse.
    """
    import site
    helper.require("SETUPTOOLS_USE_DISTUTILS" not in os.environ and
                   not any(n == "distutils" or n.startswith("distutils.") or n == "_distutils_hack"
                           for n in sys.modules), "startup override/preloaded distutils")
    helper.require(not any((p / n).exists() for p in sites for n in
                           ("sitecustomize", "sitecustomize.py", "sitecustomize.pyc", "usercustomize",
                            "usercustomize.py", "usercustomize.pyc")), "startup customisation refused")
    seen = {}
    native = site.addpackage

    def guarded(sitedir, name, known_paths):
        path = Path(sitedir) / name
        helper.require(path in expected and seen.get(path, 0) < 2 and helper.digest(path) == expected[path] and
                       path.read_bytes() == HOOK, "unknown/changed/duplicate startup hook")
        seen[path] = seen.get(path, 0) + 1
        selected = importlib.machinery.PathFinder.find_spec("_distutils_hack", [str(p) for p in sorted(sites)])
        helper.require(selected is not None and Path(selected.origin).absolute() in
                       {p / "_distutils_hack/__init__.py" for p in sites}, "startup helper selection")
        module = importlib.import_module("_distutils_hack")
        helper.require(Path(module.__file__).absolute() in {p / "_distutils_hack/__init__.py" for p in sites},
                       "startup helper origin")
        module.add_shim()
        helper.require(sum(f is module.DISTUTILS_FINDER for f in sys.meta_path) == 1,
                       "startup shim not installed exactly once")
        return known_paths

    site.addpackage = guarded
    try:
        site.main()
    finally:
        site.addpackage = native
    # Native site.main visits the isolated venv site in venv() and again in
    # addsitepackages(). add_shim is idempotent; each visit must keep one finder.
    helper.require(set(seen) == set(expected) and all(n == 2 for n in seen.values()), "startup hook coverage")
    return {str(p): n for p, n in sorted(seen.items())}


def run(manifest, progress):
    pre = load_prestartup()
    helper = pre.load_provenance()
    require = helper.require
    require(set(manifest) == {"schema", "payload", "imports", "execution_policy_sha256"} and
            manifest["schema"] == 1, "handoff manifest fields/schema")
    payload = manifest["payload"]
    prefix, sites, config, real, executable_hash = pre.layout(payload, helper)
    base = Path(sys._base_executable).resolve(strict=True)
    # No wheel code may already be loaded in this fresh no-site interpreter.
    require(not any(getattr(m, "__file__", None) and any(Path(m.__file__).absolute().is_relative_to(p)
                    for p in sites) for m in sys.modules.values()), "handoff preloaded site code")
    policy_path = Path(__file__).absolute().with_name("startup-execution-proposal.json")
    require(helper.digest(policy_path) == manifest["execution_policy_sha256"], "execution proposal bytes")
    policy = json.loads(policy_path.read_bytes())
    require(set(policy) == {"schema", "status", "adapters"} and policy["schema"] == 1 and
            policy["status"] == "PROPOSED_EXECUTION_NOT_CLEARANCE", "execution proposal schema/status")
    require(set(manifest["imports"]) == set(helper.RELEASES) and
            all(isinstance(v, list) and v and len(set(v)) == len(v) and
                all(isinstance(n, str) and n and all(p.isidentifier() for p in n.split(".")) for n in v)
                for v in manifest["imports"].values()), "handoff release imports")
    environment = dict(os.environ)
    progress["phase"] = "PAYLOAD"
    verified = pre.verify(payload)
    progress["payload"] = verified
    registry = json.loads(Path(pre.__file__).with_name("startup-audit-adapters.json").read_bytes())
    packages = [helper.wheel(pin, audit_startup=registry["adapters"]) for pin in payload["wheels"]]
    hashes = tree(sites, helper)
    owners, hooks = {}, {}
    for package in packages:
        for rel in package["files"]:
            paths = [helper.destination(rel, p) for p in sites if helper.destination(rel, p) in hashes]
            require(len(paths) == 1, "handoff payload destination")
            require(rel == package["info"] + "/RECORD" or hashes[paths[0]] == package["files"][rel][0],
                    "handoff payload drift before startup")
            owners[paths[0]] = package["name"]
        if package["startup"]:
            matches = [a for a in policy["adapters"] if a == {"name": package["name"],
                "version": package["version"], "wheel_sha256": package["wheel_sha256"],
                "startup": package["startup"], "adapter": "setuptools_add_shim_default_local_v1"}]
            require(package["name"] == "setuptools" and len(matches) == 1 and
                    package["startup"] == {"distutils-precedence.pth": hashlib.sha256(HOOK).hexdigest()},
                    "unreviewed startup execution shape")
            for path in owners:
                if owners[path] == package["name"] and path.name == "distutils-precedence.pth":
                    hooks[path] = hashes[path]
    require({p for p in hashes if p.suffix == ".pth"} == set(hooks), "startup execution inventory")
    stdlib = Path(helper.sysconfig.get_path("stdlib")).resolve()
    original_paths = set(sys.path)
    original_meta = list(sys.meta_path)
    tool_paths = {Path(__file__).absolute(), Path(pre.__file__).absolute(), Path(helper.__file__).absolute()}
    origins(helper, hashes, owners, stdlib, tool_paths)
    input_hashes = {p: helper.digest(p) for p in tool_paths}
    input_hashes.update({config: helper.digest(config), policy_path: manifest["execution_policy_sha256"],
                    Path(payload["image"]): payload["image_sha256"],
                    Path(pre.__file__).with_name("startup-audit-adapters.json"): payload["startup_registry_sha256"]})
    progress["phase"] = "STARTUP"
    progress["startup_hooks"] = activate(sites, hooks, helper)
    require(Path(sys.prefix) == prefix and sys.flags.no_user_site and
            set(sys.path) == original_paths | {str(p) for p in sites}, "startup native venv/search path")
    extra_meta = [f for f in sys.meta_path if not any(f is old for old in original_meta)]
    expected_finder = getattr(sys.modules.get("_distutils_hack"), "DISTUTILS_FINDER", None)
    expected_meta = ([expected_finder] if hooks else []) + original_meta
    require(len(extra_meta) == bool(hooks) and len(sys.meta_path) == len(expected_meta) and
            all(a is b for a, b in zip(sys.meta_path, expected_meta)), "startup import machinery")
    origins(helper, hashes, owners, stdlib, tool_paths)
    require(tree(sites, helper) == hashes and dict(os.environ) == environment, "startup changed installation/environment")
    startup_paths = list(sys.path)
    progress["phase"] = "IMPORTS"
    imports = {}
    for alias, distribution in helper.RELEASES.items():
        imports[alias] = {}
        for name in manifest["imports"][alias]:
            search = [str(p) for p in sorted(sites)]
            for index, component in enumerate(name.split(".")):
                qualified = ".".join(name.split(".")[:index + 1])
                selected = importlib.machinery.PathFinder.find_spec(qualified, search)
                require(selected is not None and selected.origin is not None and
                        owners.get(Path(selected.origin).absolute()) == distribution,
                        "release import selection owner")
                search = selected.submodule_search_locations
                require(index == len(name.split(".")) - 1 or search is not None,
                        "release import parent is not a package")
            module = importlib.import_module(name)
            path = Path(module.__file__).absolute()
            require(owners.get(path) == distribution and helper.digest(path) == hashes.get(path),
                    "release import owner/bytes")
            imports[alias][name] = str(path)
    require(sys.path == startup_paths and
            len(sys.meta_path) == len(expected_meta) and
            all(a is b for a, b in zip(sys.meta_path, expected_meta)), "imports changed search machinery")
    origins(helper, hashes, owners, stdlib, tool_paths)
    require(tree(sites, helper) == hashes and dict(os.environ) == environment, "imports changed installation/environment")
    for path, value in input_hashes.items():
        require(helper.digest(path) == value, "handoff inputs changed")
    require(Path(sys.executable).resolve() == real and helper.digest(real) == executable_hash, "handoff interpreter changed")
    require(Path(sys._base_executable).resolve() == base and
            helper.digest(base) == verified["base_executable_sha256"], "handoff base interpreter changed")
    for package in packages:
        require(helper.digest(Path(package["path"])) == package["wheel_sha256"], "handoff archive changed")
    progress.update(status="PASSED", phase="COMPLETE", imports=imports, startup_activated=True,
                    installed_inventory_sha256=hashlib.sha256(json.dumps(
                        {str(p.relative_to(prefix)): h for p, h in hashes.items()},
                        sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    proves_requested_release_imports=True, proves_full_image_installation=False,
                    proves_gpu_engagement=False, proves_launch_authority=False)
    return progress


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--worker-spec", type=Path)
    ap.add_argument("--worker-sha256")
    args = ap.parse_args()
    helper = load_prestartup().load_provenance()
    helper.require((args.worker_spec is None) == (args.worker_sha256 is None), "worker pin pair")
    worker = None
    if args.worker_spec is not None:
        helper.require(args.worker_spec.is_absolute() and
                       helper.digest(args.worker_spec) == args.worker_sha256, "worker spec pin")
        worker = json.loads(args.worker_spec.read_bytes())
        if worker.get('phase') in ('tc1_training', 'tc1_training_profile'):
            pin = worker['handoff_manifest']
            helper.require(set(pin) == {'path', 'sha256'} and Path(pin['path']) == args.manifest and
                           helper.digest(args.manifest) == pin['sha256'], 'nested manifest pin before startup')
    helper.require(args.out.is_absolute() and not args.out.exists(), "fresh absolute handoff receipt")
    before = helper.digest(args.manifest)
    progress = {"schema": 1, "status": "FAILED", "phase": "PRECONDITIONS", "manifest_sha256": before}
    try:
        manifest = json.loads(args.manifest.read_bytes())
        result = run(manifest, progress)
        helper.require(helper.digest(args.manifest) == before, "handoff manifest changed")
    except Exception as error:
        progress.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        args.out.write_text(json.dumps(progress, sort_keys=True, indent=2) + "\n")
        raise
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if worker is not None:
        helper.require(helper.digest(args.worker_spec) == args.worker_sha256, "worker spec changed during handoff")
        path = Path(__file__).absolute().with_name("ra_verified_worker.py")
        helper.require(helper.digest(path) == worker["tools"][path.name], "worker helper pin")
        spec = importlib.util.spec_from_file_location("ra_verified_worker", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.execute(worker, args.worker_spec, args.worker_sha256, result, args.out, manifest)


if __name__ == "__main__":
    main()
