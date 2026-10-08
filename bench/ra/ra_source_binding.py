"""Bind complete release runtime wheel payload to committed Git objects.

Stdlib and the RA archive verifier only; never import release/build code.
Only the two recorded static setuptools layouts are supported.
"""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


def require(ok, message):
    if not ok:
        raise ValueError("source binding: " + message)


def load_tool():
    path = Path(__file__).absolute().with_name("ra_provenance.py")
    spec = importlib.util.spec_from_file_location("ra_source_archive", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def git(repo, *args, data=None):
    require(repo.is_absolute() and repo.is_dir(), "absolute source repository required")
    return subprocess.run(["git", "--no-replace-objects", "-C", str(repo), *args], input=data,
                          check=True, capture_output=True, timeout=60).stdout


def source_tree(repo, commit):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "full commit pin required")
    require(git(repo, "rev-parse", commit + "^{commit}").decode().strip() == commit, "commit identity")
    tree = {}
    for row in git(repo, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not row:
            continue
        header, path = row.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        rel = path.decode("utf-8")
        require(rel not in tree and PurePosixPath(rel).as_posix() == rel and
                not rel.startswith("/") and ".." not in PurePosixPath(rel).parts and "\\" not in rel,
                "unsafe/duplicate source path")
        tree[rel] = (mode, kind, oid)
    return tree


def blobs(repo, tree, paths):
    paths = sorted(set(paths))
    for path in paths:
        require(path in tree and tree[path][0] in ("100644", "100755") and tree[path][1] == "blob",
                "missing/nonregular Git source: " + path)
    data = git(repo, "cat-file", "--batch", data="".join(tree[p][2] + "\n" for p in paths).encode())
    result, offset = {}, 0
    for path in paths:
        end = data.index(b"\n", offset)
        oid, kind, size = data[offset:end].decode().split()
        size = int(size)
        require(oid == tree[path][2] and kind == "blob", "Git object response")
        start = end + 1
        result[path] = data[start:start + size]
        require(len(result[path]) == size and data[start + size:start + size + 1] == b"\n",
                "Git object length")
        offset = start + size + 1
    require(offset == len(data), "surplus Git object response")
    return result


def runtime_mapping(project, tree):
    """Derive payload paths, never accept a caller-supplied file allowlist."""
    require(project.get("build-system") == {"requires": ["setuptools>=77"], "build-backend": "setuptools.build_meta"},
            "build convention needs reviewed adapter")
    info = project["project"]
    require(not any(k in info for k in ("dynamic", "scripts", "gui-scripts", "entry-points")),
            "dynamic metadata/entry points need reviewed adapter")
    config = project["tool"]["setuptools"]
    name, mapping = info["name"], {}
    if name == "experts4bit-qlora":
        require(config == {"packages": {"find": {"include": ["experts4bit_qlora*"]}},
                           "package-data": {"experts4bit_qlora": ["py.typed", "README-LAYOUT.md"]}},
                "e4b packaging convention needs reviewed adapter")
        for path in tree:
            if path.startswith("experts4bit_qlora/") and path.endswith(".py"):
                mapping[path] = path
        for rel in config["package-data"]["experts4bit_qlora"]:
            path = "experts4bit_qlora/" + rel
            mapping[path] = path
    elif name == "grouped-nf4-gemm":
        require(set(config) == {"package-dir", "packages", "py-modules", "package-data"} and
                config["package-dir"] == {"": "kernel", "gnf4_native": "gnf4_native"} and
                config["packages"] == ["gnf4_native"] and config["package-data"] == {"gnf4_native": ["*.c"]},
                "gnf4 packaging convention needs reviewed adapter")
        modules = config["py-modules"]
        require(isinstance(modules, list) and modules and len(set(modules)) == len(modules) and
                all(isinstance(m, str) and re.fullmatch(r"[A-Za-z_]\w*", m) for m in modules),
                "flat module declaration")
        mapping.update({m + ".py": "kernel/" + m + ".py" for m in modules})
        for path in tree:
            if path.startswith("gnf4_native/") and path.count("/") == 1 and path.endswith((".py", ".c")):
                mapping[path] = path
    else:
        raise ValueError("source binding: unsupported release")
    require(mapping and any(p.endswith("__init__.py") for p in mapping), "empty/missing package initializer")
    return mapping


def verify_release(row, tool):
    require(set(row) == {"repo", "commit", "version", "wheel"}, "release manifest fields")
    repo, commit = Path(row["repo"]), row["commit"]
    tree = source_tree(repo, commit)
    config = blobs(repo, tree, ["pyproject.toml"])["pyproject.toml"]
    project = tomllib.loads(config.decode("utf-8"))
    mapping = runtime_mapping(project, tree)
    info = project["project"]
    require(info["version"] == row["version"], "source version differs from pin")
    package = tool.wheel(row["wheel"])
    require(package["name"] == info["name"] and package["version"] == row["version"], "source/wheel identity")
    payload = {p for p in package["files"] if not p.startswith(package["info"] + "/")}
    require(payload == set(mapping), "runtime payload coverage differs from committed packaging")
    readme, licenses = info["readme"], info["license-files"]
    require(isinstance(readme, str) and isinstance(licenses, list) and licenses and
            len(set(licenses)) == len(licenses) and all(isinstance(p, str) and "/" not in p for p in licenses),
            "readme/license convention needs reviewed adapter")
    sources = blobs(repo, tree, [*mapping.values(), readme, *licenses])
    observed = {}
    for rel, source in mapping.items():
        data = sources[source]
        value = hashlib.sha256(data).hexdigest()
        require(package["files"][rel] == (value, len(data)), "runtime source bytes: " + rel)
        observed[rel] = {"source": source, "git_blob": tree[source][2], "sha256": value, "size": len(data)}
    expected_info = {package["info"] + "/" + p for p in ("METADATA", "WHEEL", "top_level.txt", "RECORD")}
    expected_info.update(package["info"] + "/licenses/" + p for p in licenses)
    require(set(package["files"]) == payload | expected_info, "unreviewed generated wheel metadata")
    with zipfile.ZipFile(package["path"]) as archive:
        metadata_bytes = archive.read(package["info"] + "/METADATA")
        require(b"\r" not in metadata_bytes and b"\n\n" in metadata_bytes, "metadata newline convention")
        metadata = email.parser.BytesParser().parsebytes(metadata_bytes)
        for field, expected in [("Requires-Python", info["requires-python"]), ("License-Expression", info["license"])]:
            require(metadata.get_all(field) == [expected], "source metadata identity: " + field)
        require(metadata_bytes.split(b"\n\n", 1)[1] == sources[readme], "source readme differs from metadata")
        require(set(metadata.get_all("License-File", [])) == set(licenses), "declared license files")
        for path in licenses:
            require(archive.read(package["info"] + "/licenses/" + path) == sources[path], "source license bytes")
        top = {p.split("/", 1)[0].removesuffix(".py") for p in mapping}
        listed = archive.read(package["info"] + "/top_level.txt").decode().splitlines()
        require(len(listed) == len(set(listed)) and set(listed) == top, "top-level module census")
        wheel = email.parser.BytesParser().parsebytes(archive.read(package["info"] + "/WHEEL"))
        require(set(wheel.keys()) == {"Wheel-Version", "Generator", "Root-Is-Purelib", "Tag"} and
                wheel.get_all("Wheel-Version") == ["1.0"] and wheel.get_all("Root-Is-Purelib") == ["true"] and
                wheel.get_all("Tag") == ["py3-none-any"] and len(wheel.get_all("Generator", [])) == 1,
                "wheel layout convention needs reviewed adapter")
    require(tool.digest(Path(package["path"])) == package["wheel_sha256"] and source_tree(repo, commit) == tree,
            "archive/source inputs changed")
    return {"name": package["name"], "version": package["version"], "commit": commit,
            "wheel_sha256": package["wheel_sha256"], "pyproject_sha256": hashlib.sha256(config).hexdigest(),
            "runtime_files": observed, "runtime_tree_sha256": hashlib.sha256(
                json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def verify(manifest):
    require(set(manifest) == {"schema", "releases"} and manifest["schema"] == 1 and
            isinstance(manifest["releases"], list) and len(manifest["releases"]) == 2, "pair manifest schema")
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, "python -I -S -B required")
    tool = load_tool()
    results = [verify_release(row, tool) for row in manifest["releases"]]
    require({r["name"] for r in results} == set(tool.RELEASES.values()), "one release of each project required")
    return {"schema": 1, "releases": results, "proves_runtime_source_binding": True,
            "proves_reproducible_build": False, "proves_dependency_metadata_source_binding": False,
            "proves_installation": False, "proves_imports": False, "proves_gpu_engagement": False,
            "proves_release_authorization": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    tool = load_tool()
    before = tool.digest(args.manifest)
    result = verify(json.loads(args.manifest.read_bytes()))
    require(tool.digest(args.manifest) == before and args.out.is_absolute(), "manifest changed/output not absolute")
    result["manifest_sha256"] = before
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
