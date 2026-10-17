"""Compare release dependency metadata to pinned Git declarations, without site startup.

Only the retained, RECORD-verified packaging parser is imported. Runtime source
binding runs first; build backends and release modules are never imported.
"""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import importlib.util
import json
import sys
import tomllib
import zipfile
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError("source metadata: " + message)


def load_source():
    path = Path(__file__).absolute().with_name("ra_source_binding.py")
    spec = importlib.util.spec_from_file_location("ra_metadata_source", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


def marker_tree(values):
    """Canonical Boolean structure, preserving variable/value and comparison types."""
    def child(value):
        if isinstance(value, list):
            return marker_tree(value)
        require(isinstance(value, tuple) and len(value) == 3, "unsupported parser marker atom")
        left, op, right = value
        require(type(left).__name__ in ("Variable", "Value") and type(right).__name__ in ("Variable", "Value") and
                type(op).__name__ == "Op", "unsupported parser marker operand")
        return ["atom", type(left).__name__, left.value, op.value, type(right).__name__, right.value]

    def combine(op, nodes):
        flat = [part for node in nodes for part in (node[1:] if node[0] == op else [node])]
        flat = sorted(flat, key=lambda x: json.dumps(x, separators=(",", ":")))
        return flat[0] if len(flat) == 1 else [op, *flat]

    require(isinstance(values, list) and values and len(values) % 2 == 1, "unsupported parser marker structure")
    groups, atoms = [], []
    for index, value in enumerate(values):
        if index % 2 == 0:
            atoms.append(child(value))
        else:
            require(value in ("and", "or"), "unsupported marker Boolean operator")
            if value == "or":
                groups.append(combine("and", atoms))
                atoms = []
    groups.append(combine("and", atoms))
    return combine("or", groups)


def signature(raw, Requirement, Marker, canonical, extra=None):
    req = Requirement(raw)
    require(req.url is None, "direct-URL dependency needs reviewed adapter")
    marker = req.marker
    if extra is not None:
        marker = Marker(("(" + str(marker) + ") and " if marker is not None else "") +
                        "extra == " + json.dumps(extra))
    return {"name": canonical(req.name), "extras": sorted(canonical(e) for e in req.extras),
            "specifiers": sorted([part.operator, part.version] for part in req.specifier),
            "marker": marker_tree(marker._markers) if marker is not None else None}


def compare(project, metadata, Requirement, Marker, canonical):
    deps, optional = project.get("dependencies", []), project.get("optional-dependencies", {})
    require(isinstance(deps, list) and isinstance(optional, dict) and
            all(isinstance(e, str) and isinstance(rows, list) for e, rows in optional.items()), "static dependency fields")
    extras = [canonical(e) for e in optional]
    require(len(extras) == len(set(extras)), "ambiguous source extras")
    expected = [signature(raw, Requirement, Marker, canonical) for raw in deps]
    for extra, rows in optional.items():
        expected.extend(signature(raw, Requirement, Marker, canonical, canonical(extra)) for raw in rows)
    observed_extras = [canonical(e) for e in metadata.get_all("Provides-Extra", [])]
    require(len(observed_extras) == len(set(observed_extras)) and sorted(observed_extras) == sorted(extras),
            "Provides-Extra differs from source")
    observed = [signature(raw, Requirement, Marker, canonical) for raw in metadata.get_all("Requires-Dist", [])]
    def key(row):
        return json.dumps(row, sort_keys=True, separators=(",", ":"))
    require(len({key(row) for row in expected}) == len(expected) and
            len({key(row) for row in observed}) == len(observed), "duplicate dependency declaration")
    require(sorted(expected, key=key) == sorted(observed, key=key), "Requires-Dist differs from source")
    return {"extras": sorted(extras), "requires_dist": sorted(observed, key=key),
            "declarations_sha256": hashlib.sha256(json.dumps(sorted(observed, key=key),
                                                             sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def verify(manifest):
    require(set(manifest) == {"schema", "releases", "parser", "parser_lock_sha256"} and manifest["schema"] == 1,
            "manifest fields/schema")
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, "python -I -S -B required")
    source = load_source()
    tool = source.load_tool()
    binding = source.verify({"schema": 1, "releases": manifest["releases"]})
    lock_path = Path(__file__).absolute().with_name("proof-wheel-lock.json")
    require(tool.digest(lock_path) == manifest["parser_lock_sha256"], "reviewed parser lock digest")
    lock = json.loads(lock_path.read_bytes())
    pins = [p for p in lock["wheels"] if p["name"] == "packaging"]
    require(len(pins) == 1, "one retained parser pin required")
    parser = tool.wheel(manifest["parser"])
    require(parser["name"] == "packaging" and parser["version"] == pins[0]["version"] and
            parser["wheel_sha256"] == pins[0]["sha256"] and Path(parser["path"]).name == pins[0]["filename"],
            "parser archive differs from reviewed lock")
    require(not any(n == "packaging" or n.startswith("packaging.") for n in sys.modules),
            "parser must come from verified archive")
    path = parser["path"]
    sys.path.insert(0, path)
    try:
        from packaging.markers import Marker
        from packaging.requirements import Requirement
        results = {}
        for row in manifest["releases"]:
            repo = Path(row["repo"])
            tree = source.source_tree(repo, row["commit"])
            raw = source.blobs(repo, tree, ["pyproject.toml"])["pyproject.toml"]
            project = tomllib.loads(raw.decode())["project"]
            wheel = tool.wheel(row["wheel"])
            with zipfile.ZipFile(wheel["path"]) as archive:
                data = archive.read(wheel["info"] + "/METADATA")
            metadata = email.parser.BytesParser().parsebytes(data)
            results[wheel["name"]] = {"commit": row["commit"], "wheel_sha256": wheel["wheel_sha256"],
                                      "metadata_sha256": hashlib.sha256(data).hexdigest(),
                                      **compare(project, metadata, Requirement, Marker, tool.canonical)}
        for name, module in sys.modules.copy().items():
            if name == "packaging" or name.startswith("packaging."):
                require(getattr(module, "__file__", "").startswith(path + "/packaging/"), "parser module owner")
        require(tool.wheel(manifest["parser"])["wheel_sha256"] == parser["wheel_sha256"] and
                tool.digest(lock_path) == manifest["parser_lock_sha256"], "parser/lock changed")
        require(source.verify({"schema": 1, "releases": manifest["releases"]}) == binding, "release inputs changed")
    finally:
        sys.path.remove(path)
    return {"schema": 1, "releases": results, "runtime_source_binding": binding,
            "parser_wheel_sha256": parser["wheel_sha256"], "parser_lock_sha256": manifest["parser_lock_sha256"],
            "proves_dependency_metadata_source_binding": True, "proves_reproducible_build": False,
            "proves_release_authorization": False, "proves_installation": False, "proves_imports": False,
            "startup_activated": False, "proves_gpu_engagement": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    tool = load_source().load_tool()
    before = tool.digest(args.manifest)
    result = verify(json.loads(args.manifest.read_bytes()))
    require(tool.digest(args.manifest) == before and args.out.is_absolute(), "manifest changed/output not absolute")
    result["manifest_sha256"] = before
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
