"""Bind a materialized checkpoint snapshot to the registered Hugging Face revision.

Only repository-index and local file equality is established. No dataset,
calibration, tokenizer execution, model consumption, GPU or launch proof.
Run with selected python -I -S -B; it imports no release package.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import sys
import urllib.request
from pathlib import Path, PurePosixPath

LIMIT = 4 * 1024 * 1024
PEERS = ("ra_checkpoint", "ra_inputs", "ra_stage")


def require(ok, message):
    if not ok:
        raise ValueError("checkpoint binding: " + message)


def load_inputs():
    base = Path(__file__).absolute().parent
    for name in ("ra_stage", "ra_inputs"):
        require(name not in sys.modules or Path(sys.modules[name].__file__).absolute() == base / (name + ".py"),
                "foreign input helper preloaded")
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, base / (name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
    return sys.modules["ra_inputs"]


def clock():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(value, width):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(width) + "}", value) is not None


def safe_name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", value) and
            all(p not in (".", "..") for p in value.split("/")), "repository file path")
    return value


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def urls(model, revision):
    return ("https://huggingface.co/api/models/" + model + "/revision/" + revision + "?blobs=true",
            "https://huggingface.co/api/models/" + model + "/tree/" + revision + "?recursive=true&expand=false")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("checkpoint binding: redirects refused")


class Collector:
    def __init__(self, out, allowed):
        self.out, self.allowed, self.records = out, set(allowed), []

    def __call__(self, url):
        require(url in self.allowed, "fixed authority URL required")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        request = urllib.request.Request(url, headers={"User-Agent": "ra-checkpoint-binding", "Accept": "application/json"})
        started = clock()
        with opener.open(request, timeout=20) as response:
            require(response.status == 200 and response.geturl() == url, "authority response")
            require(not response.headers.get("Link") and not response.headers.get("Content-Encoding") and
                    response.headers.get_content_type() == "application/json", "unpaged plain JSON required")
            data = response.read(LIMIT + 1)
        record = {"url": url, "started_at": started, "finished_at": clock(),
                  "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                  "file": f"response-{len(self.records):02d}.json"}
        with (self.out / record["file"]).open("xb") as dest:
            dest.write(data)
        self.records.append(record)
        require(0 < len(data) <= LIMIT, "bounded nonempty authority JSON")
        return json.loads(data, object_pairs_hook=load_inputs().unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite authority JSON")))


def observe(model, revision, fetch):
    info_url, tree_url = urls(model, revision)
    info, tree = fetch(info_url), fetch(tree_url)
    require(isinstance(info, dict) and info.get("id") == model and info.get("modelId") == model and
            info.get("sha") == revision and info.get("private") is False and info.get("gated") is False and
            info.get("disabled") is False, "registered public revision identity")
    siblings = info.get("siblings")
    require(isinstance(siblings, list) and 0 < len(siblings) <= 1000 and isinstance(tree, list) and
            0 < len(tree) <= 2000, "bounded complete repository lists required")
    files, dirs = {}, set()
    for row in tree:
        require(isinstance(row, dict), "tree row")
        name = safe_name(row.get("path"))
        require(name not in files and name not in dirs and sha(row.get("oid"), 40), "unique addressed tree row")
        if row.get("type") == "directory":
            dirs.add(name)
            continue
        require(row.get("type") == "file" and type(row.get("size")) is int and row["size"] >= 0, "file tree row")
        lfs = row.get("lfs")
        binding = {"size": row["size"], "git_blob_oid": row["oid"], "lfs_sha256": None}
        if lfs is not None:
            require(isinstance(lfs, dict) and set(lfs) == {"oid", "size", "pointerSize"} and
                    sha(lfs["oid"], 64) and type(lfs["size"]) is int and lfs["size"] == row["size"], "LFS tree identity")
            pointer = ("version https://git-lfs.github.com/spec/v1\noid sha256:" + lfs["oid"] +
                       "\nsize " + str(lfs["size"]) + "\n").encode()
            require(type(lfs["pointerSize"]) is int and lfs["pointerSize"] == len(pointer) and
                    git_blob(pointer) == row["oid"], "canonical LFS pointer Git address")
            binding["lfs_sha256"] = lfs["oid"]
        files[name] = binding
    expected_dirs = {str(p) for name in files for p in PurePosixPath(name).parents if str(p) != "."}
    require(dirs == expected_dirs, "complete directory projection")
    seen = set()
    for row in siblings:
        require(isinstance(row, dict), "sibling row")
        name = safe_name(row.get("rfilename"))
        require(name in files and name not in seen and row.get("blobId") == files[name]["git_blob_oid"] and
                type(row.get("size")) is int and row["size"] == files[name]["size"], "info/tree file identity")
        seen.add(name)
        lfs = row.get("lfs")
        if files[name]["lfs_sha256"] is None:
            require(lfs is None, "surplus sibling LFS identity")
        else:
            require(isinstance(lfs, dict) and set(lfs) == {"sha256", "size", "pointerSize"} and
                    lfs["sha256"] == files[name]["lfs_sha256"] and type(lfs["size"]) is int and
                    lfs["size"] == files[name]["size"] and type(lfs["pointerSize"]) is int and
                    lfs["pointerSize"] == len(("version https://git-lfs.github.com/spec/v1\noid sha256:" +
                        lfs["sha256"] + "\nsize " + str(lfs["size"]) + "\n").encode()), "info/tree LFS identity")
    require(seen == set(files), "complete info/tree file set")
    return {"model": model, "revision": revision, "files": dict(sorted(files.items()))}


def bind(root, authority, expected):
    inputs = load_inputs()
    observed = inputs.inventory(root)
    require(observed == expected and set(observed) == set(authority["files"]), "complete reviewed checkpoint inventory")
    for name, row in authority["files"].items():
        path = root / name
        require(observed[name]["size"] == row["size"], "checkpoint size differs")
        if row["lfs_sha256"] is not None:
            require(observed[name]["sha256"] == row["lfs_sha256"], "materialized LFS bytes differ")
        else:
            require(row["size"] <= LIMIT, "bounded Git checkpoint file")
            before = inputs.file_record(path)
            require(git_blob(path.read_bytes()) == row["git_blob_oid"] and inputs.file_record(path) == before,
                    "checkpoint Git blob bytes differ")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
    require(required <= set(observed), "registered checkpoint/config/tokenizer/index set")
    for name in ("config.json", "tokenizer_config.json"):
        require(isinstance(inputs.read_json(root / name), dict), "checkpoint configuration object")
    index = inputs.read_json(root / "model.safetensors.index.json")
    weights = index.get("weight_map")
    require(isinstance(weights, dict) and weights and all(isinstance(k, str) and k for k in weights), "nonempty weight map")
    shards = {n for n in observed if n.endswith(".safetensors")}
    require(set(weights.values()) == shards and shards and
            all(authority["files"][n]["lfs_sha256"] is not None for n in shards), "complete LFS shard map")
    require(inputs.inventory(root) == observed, "checkpoint changed during binding")
    return {"files": observed, "file_count": len(observed), "bytes": sum(v["size"] for v in observed.values()),
            "shard_count": len(shards), "weight_names": len(weights)}


def check(manifest):
    inputs = load_inputs()
    require(set(manifest) == {"schema", "input_spec", "stage", "input_lock", "input_lock_sha256"} and
            type(manifest["schema"]) is int and manifest["schema"] == 1, "manifest schema")
    require(sha(manifest["input_lock_sha256"], 64), "reviewed input lock digest")
    spec = inputs.read_json(manifest["input_spec"])
    verified = inputs.verify(spec, manifest["stage"], manifest["input_lock"], manifest["input_lock_sha256"])
    lock = inputs.read_json(manifest["input_lock"])
    pins = inputs.read_json(Path(__file__).absolute().with_name("source-pins.json"))
    require(lock["model"] in pins["models"] and lock["revision"] == pins["models"][lock["model"]], "fixed registered model pin")
    return spec, lock, verified


def verify(manifest, fetch):
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, "python -I -S -B required")
    inputs = load_inputs()
    base = Path(__file__).absolute().parent
    paths = [base / (n + ".py") for n in PEERS] + [base / "source-pins.json"]
    paths += [inputs.absolute(manifest[k]) for k in ("input_spec", "input_lock")]
    before = {str(p): inputs.file_record(p) for p in paths}
    spec, lock, verified = check(manifest)
    authority = observe(lock["model"], lock["revision"], fetch)
    materialized = bind(inputs.absolute(spec["trees"]["checkpoint"]), authority, lock["trees"]["checkpoint"])
    require(observe(lock["model"], lock["revision"], fetch) == authority, "authority changed during binding")
    require(check(manifest) == (spec, lock, verified) and {str(p): inputs.file_record(p) for p in paths} == before,
            "input/helper binding changed")
    return {"schema": 1, "authority": authority, "checkpoint": materialized,
            "common_inputs": verified, "proves_hub_index_checkpoint_equality": True,
            "proves_dataset_authority": False, "proves_calibration_authority": False,
            "proves_tokenizer_execution": False, "proves_runtime_consumption": False,
            "proves_publisher_signature": False, "proves_gpu_engagement": False, "launch_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    inputs = load_inputs()
    out = inputs.absolute(args.out)
    manifest = inputs.read_json(args.manifest)
    spec, lock, _ = check(manifest)
    protected = [inputs.absolute(v) for v in spec["trees"].values()] + [inputs.absolute(v) for v in spec["files"].values()]
    protected += [inputs.absolute(manifest[k]) for k in ("input_spec", "input_lock", "stage")]
    protected += [inputs.absolute(args.manifest), Path(__file__).absolute().parent]
    require(not out.exists() and all(not (out == p or out in p.parents or p in out.parents) for p in protected),
            "fresh separate receipt directory")
    out.mkdir()
    manifest_before = inputs.file_record(args.manifest)
    fetch = Collector(out, urls(lock["model"], lock["revision"]))
    try:
        result = verify(manifest, fetch)
        require(inputs.file_record(args.manifest) == manifest_before, "manifest changed")
        result.update(status="PASS", pid=os.getpid(), manifest_sha256=manifest_before["sha256"])
    except Exception as error:
        result = {"schema": 1, "status": "FAILED", "error_type": type(error).__name__,
                  "manifest_sha256": manifest_before["sha256"]}
        raise
    finally:
        result["observations"] = fetch.records
        with (out / "result.json").open("x") as dest:
            dest.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
