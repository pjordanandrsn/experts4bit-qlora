"""Hash-locked box installation and source authority; never creates compute."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
import urllib.request
import zipfile

from dq11_common import file_sha

ALLOWED_WHEEL_ORIGINS = frozenset({"files.pythonhosted.org", "download.pytorch.org",
                                 "download-r2.pytorch.org", "pypi.nvidia.com"})
FETCH_USER_AGENT = "dq11-bootstrap/1"


class WheelFetchError(RuntimeError):
    def __init__(self, url, status, detail):
        self.url, self.status = url, status
        super().__init__(f"wheel fetch refused: URL={url} status={status}: {detail}")


def fetch_wheel(url, *, probe=False):
    """One opener for pre-rental probes and full hash-checked downloads."""
    headers = {"User-Agent": FETCH_USER_AGENT}
    if probe:
        headers["Range"] = "bytes=0-0"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except HTTPError as error:
        raise WheelFetchError(url, error.code, str(error)) from error
    except (URLError, OSError) as error:
        raise WheelFetchError(url, "UNKNOWN", str(error)) from error
    if not 200 <= response.status < 300:
        status = response.status
        response.close()
        raise WheelFetchError(url, status, "non-2xx response")
    return response


def read_wheel(response, url, size):
    try:
        return response.read(size)
    except (URLError, OSError) as error:
        raise WheelFetchError(url, response.status, str(error)) from error


def preflight_urls(directory):
    """Read one byte from ALL locked URLs, without installation or compute."""
    packages = json.loads((directory / "wheels.json").read_text())["packages"]
    validate_wheel_origins(packages)
    urls = [line.split(" @ ", 1)[1].split()[0]
            for line in (directory / "requirements.lock").read_text().splitlines() if " @ " in line]
    if (len(urls) != 101 or len(packages) != 101 or len(set(urls)) != 101
            or set(urls) != {row["url"] for row in packages}
            or any(urlparse(url).scheme != "https" for url in urls)):
        raise ValueError("pre-rental gate requires all 101 matching HTTPS locked wheel URLs")
    results = []
    for row in packages:
        result = {"name": row["name"], "url": row["url"]}
        try:
            with fetch_wheel(row["url"], probe=True) as response:
                result.update(status=response.status, bytes_read=len(read_wheel(response, row["url"], 1)))
            if result["bytes_read"] != 1:
                result["error"] = "empty wheel response"
        except WheelFetchError as error:
            result.update(status=error.status, error=str(error))
        results.append(result)
    return {"schema": "dq11-wheel-fetch-gate/1", "user_agent": FETCH_USER_AGENT,
            "method": "GET bytes=0-0; read one byte then close",
            "wheel_lock_sha256": file_sha(directory / "wheels.json"), "count": len(results),
            "passed": all("error" not in row for row in results), "results": results}


def validate_wheel_origins(wheels, allowed=None):
    """Validate the entire committed lock before any download or installation."""
    allowed = ALLOWED_WHEEL_ORIGINS if allowed is None else allowed
    for row in wheels:
        origin = urlparse(row["url"]).hostname
        if origin not in allowed:
            raise ValueError("unregistered wheel origin: " + str(origin))


def install(directory, cache):
    wheels = json.loads((directory / "wheels.json").read_text())["packages"]
    validate_wheel_origins(wheels)
    cache.mkdir(parents=True, exist_ok=True)
    authority, local_requirements = set(), []
    for row in wheels:
        if Path(row["filename"]).name != row["filename"]:
            raise ValueError("unsafe wheel filename")
        path = cache / row["filename"]
        if not path.is_file():
            temporary = path.with_suffix(".partial")
            with fetch_wheel(row["url"]) as source, temporary.open("wb") as target:
                while block := read_wheel(source, row["url"], 1 << 20):
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
    parser.add_argument("--preflight", action="store_true", help="probe all locked URLs; never install or rent")
    args = parser.parse_args()
    if args.preflight:
        report = preflight_urls(args.directory)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["passed"] else 78)
    install(args.directory, args.cache)
