"""Bind retained source/wheels to live GitHub release/tag and PyPI publication records.

Run with python -I -S -B. GitHub-reported signature verification is evidence from
that service, not local cryptographic verification or launch authorization.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


LIMIT = 4 * 1024 * 1024
PROJECTS = {"experts4bit-qlora", "grouped-nf4-gemm"}


def require(ok, message):
    if not ok:
        raise ValueError("publication binding: " + message)


def clock():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def timestamp(value):
    require(isinstance(value, str), "publication time required")
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.utcoffset() == datetime.timedelta(0), "UTC publication time required")
    return value


def load_metadata():
    path = Path(__file__).absolute().with_name("ra_source_metadata.py")
    spec = importlib.util.spec_from_file_location("ra_publication_metadata", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("publication binding: redirects refused")


class Collector:
    def __init__(self, out):
        self.out, self.records = out, []

    def __call__(self, url):
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme == "https" and parsed.netloc in ("api.github.com", "pypi.org") and
                not parsed.query and not parsed.fragment and parsed.username is None, "authority URL")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        request = urllib.request.Request(url, headers={"User-Agent": "ra-publication-binding", "Accept": "application/json"})
        started = clock()
        with opener.open(request, timeout=20) as response:
            require(response.status == 200 and response.geturl() == url, "authority response")
            data = response.read(LIMIT + 1)
        require(len(data) <= LIMIT, "authority response too large")
        def unique(rows):
            values = dict(rows)
            require(len(values) == len(rows), "duplicate JSON fields")
            return values
        record = {"url": url, "started_at": started, "finished_at": clock(),
                  "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                  "file": f"response-{len(self.records):02d}.json"}
        (self.out / record["file"]).write_bytes(data)
        self.records.append(record)
        return json.loads(data, object_pairs_hook=unique)


def observe(binding, row, source, fetch):
    name, version = binding["name"], binding["version"]
    require(name in PROJECTS and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version), "supported final release identity")
    tag = "v" + version
    api = "https://api.github.com/repos/pjordanandrsn/" + name
    release = fetch(api + "/releases/tags/" + tag)
    require(type(release["id"]) is int and release["id"] > 0 and release["tag_name"] == tag and
            release["draft"] is False and release["prerelease"] is False and
            release["html_url"] == "https://github.com/pjordanandrsn/" + name + "/releases/tag/" + tag,
            "published GitHub release identity")
    published = timestamp(release["published_at"])
    ref = fetch(api + "/git/ref/tags/" + tag)
    obj = ref["object"]
    require(ref["ref"] == "refs/tags/" + tag and obj["type"] == "tag" and
            re.fullmatch(r"[0-9a-f]{40}", obj["sha"]) and obj["url"] == api + "/git/tags/" + obj["sha"],
            "annotated tag reference")
    tagged = fetch(obj["url"])
    raw = source.object_bytes(Path(row["repo"]), {"tag": (obj["sha"], "tag")})["tag"]
    prefix = ("object " + binding["commit"] + "\ntype commit\ntag " + tag + "\n").encode()
    require(raw.startswith(prefix) and b"\n-----BEGIN PGP SIGNATURE-----\n" in raw,
            "local signed tag target/name")
    unsigned, signature = raw.split(b"-----BEGIN PGP SIGNATURE-----\n", 1)
    verification = tagged["verification"]
    require(tagged["sha"] == obj["sha"] and tagged["tag"] == tag and
            tagged["object"] == {"sha": binding["commit"], "type": "commit", "url": api + "/git/commits/" + binding["commit"]}
            and verification["verified"] is True and verification["reason"] == "valid" and
            verification["payload"].encode() == unsigned and
            verification["signature"].strip() == (b"-----BEGIN PGP SIGNATURE-----\n" + signature).decode().strip(),
            "GitHub verified tag differs from local object")
    verified_at = timestamp(verification["verified_at"])
    pypi = fetch("https://pypi.org/pypi/" + name + "/" + version + "/json")
    tool = source.load_tool()
    require(tool.canonical(pypi["info"]["name"]) == name and pypi["info"]["version"] == version,
            "published PyPI project/version")
    path = Path(row["wheel"]["path"])
    files = [p for p in pypi["urls"] if p["filename"] == path.name]
    require(len(files) == 1, "one published wheel file required")
    file = files[0]
    url = urllib.parse.urlsplit(file["url"])
    require(file["packagetype"] == "bdist_wheel" and file["yanked"] is False and
            file["digests"]["sha256"] == binding["wheel_sha256"] and type(file["size"]) is int and
            file["size"] == path.stat().st_size and url.scheme == "https" and url.netloc == "files.pythonhosted.org" and
            not url.query and not url.fragment and url.path.endswith("/" + path.name), "published wheel bytes/origin")
    return {"project": name, "version": version, "commit": binding["commit"], "tag_oid": obj["sha"],
            "github_release_id": release["id"], "published_at": published, "github_verified_at": verified_at,
            "github_reports_verified_tag": True, "wheel_filename": path.name, "wheel_sha256": binding["wheel_sha256"],
            "wheel_bytes": file["size"], "wheel_uploaded_at": timestamp(file["upload_time_iso_8601"]), "wheel_url": file["url"]}


def verify(manifest, fetch):
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, "python -I -S -B required")
    metadata = load_metadata()
    source = metadata.load_source()
    runtime = source.verify({"schema": 1, "releases": manifest["releases"]})
    rows = {source.load_tool().wheel(r["wheel"])["name"]: r for r in manifest["releases"]}
    bindings = {b["name"]: b for b in runtime["releases"]}
    before = {name: observe(bindings[name], rows[name], source, fetch) for name in sorted(bindings)}
    bound = metadata.verify(manifest)
    require(bound["runtime_source_binding"] == runtime, "source binding changed")
    after = {name: observe(bindings[name], rows[name], source, fetch) for name in sorted(bindings)}
    require(after == before, "publication identity changed during binding")
    require(source.verify({"schema": 1, "releases": manifest["releases"]}) == runtime, "source/archive changed")
    tool = source.load_tool()
    require(tool.digest(Path(manifest["parser"]["path"])) == bound["parser_wheel_sha256"] and
            tool.digest(Path(__file__).absolute().with_name("proof-wheel-lock.json")) == bound["parser_lock_sha256"],
            "parser/lock changed")
    return {"schema": 1, "publications": before, "source_binding": bound,
            "proves_published_release_index_binding": True, "proves_local_signature_verification": False,
            "proves_reproducible_build": False, "proves_release_authorization": False,
            "proves_installation": False, "startup_activated": False, "proves_release_imports": False,
            "proves_gpu_engagement": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    require(args.out.is_absolute() and not args.out.exists() and
            not any(p.is_symlink() for p in (args.out.parent, *args.out.parents)), "fresh absolute receipt directory")
    args.out.mkdir()
    fetch = Collector(args.out)
    helper = load_metadata().load_source().load_tool()
    before = helper.digest(args.manifest)
    try:
        result = verify(json.loads(args.manifest.read_bytes()), fetch)
        require(helper.digest(args.manifest) == before, "manifest changed")
        result.update(status="PASS", manifest_sha256=before)
    except Exception as error:
        result = {"schema": 1, "status": "FAILED", "error_type": type(error).__name__,
                  "manifest_sha256": before, "observations": fetch.records}
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    result["observations"] = fetch.records
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
