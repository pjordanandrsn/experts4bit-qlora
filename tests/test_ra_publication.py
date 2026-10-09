"""Synthetic HTTPS authority mutations; no local signature or release clearance claim."""

import importlib.util
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fixture = load("ra_publication_fixture", ROOT / "tests/test_ra_source_metadata.py")
publication = load("ra_publication_test", ROOT / "bench/ra/ra_publication.py")


def setup(tmp_path):
    manifest, tools = fixture.setup(tmp_path)
    (tools / "ra_publication.py").write_bytes((ROOT / "bench/ra/ra_publication.py").read_bytes())
    responses = {}
    for row in manifest["releases"]:
        name = Path(row["repo"]).name
        version = row["version"]
        tag = "v" + version
        api = "https://api.github.com/repos/pjordanandrsn/" + name
        commit = row["commit"]
        unsigned = (
            f"object {commit}\ntype commit\ntag {tag}\ntagger CPU fixture <fixture@example.invalid> "
            f"{int(time.time())} +0000\n\nSynthetic CPU tag, no cryptographic validity claim.\n"
        ).encode()
        signature = b"-----BEGIN PGP SIGNATURE-----\nSYNTHETIC CPU CONTROL ONLY\n-----END PGP SIGNATURE-----\n"
        oid = (
            subprocess.run(
                ["git", "-C", row["repo"], "hash-object", "-t", "tag", "-w", "--stdin"],
                input=unsigned + signature,
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        now = publication.clock()
        responses[api + "/releases/tags/" + tag] = {
            "id": 1,
            "tag_name": tag,
            "draft": False,
            "prerelease": False,
            "published_at": now,
            "html_url": "https://github.com/pjordanandrsn/" + name + "/releases/tag/" + tag,
        }
        responses[api + "/git/ref/tags/" + tag] = {
            "ref": "refs/tags/" + tag,
            "object": {"sha": oid, "type": "tag", "url": api + "/git/tags/" + oid},
        }
        responses[api + "/git/tags/" + oid] = {
            "sha": oid,
            "tag": tag,
            "object": {"sha": commit, "type": "commit", "url": api + "/git/commits/" + commit},
            "verification": {
                "verified": True,
                "reason": "valid",
                "verified_at": now,
                "payload": unsigned.decode(),
                "signature": signature.decode(),
            },
        }
        p = Path(row["wheel"]["path"])
        responses["https://pypi.org/pypi/" + name + "/" + version + "/json"] = {
            "info": {"name": name, "version": version},
            "urls": [
                {
                    "filename": p.name,
                    "packagetype": "bdist_wheel",
                    "yanked": False,
                    "digests": {"sha256": row["wheel"]["sha256"]},
                    "size": p.stat().st_size,
                    "url": "https://files.pythonhosted.org/synthetic/" + p.name,
                    "upload_time_iso_8601": now,
                }
            ],
        }
    return manifest, tools, responses


def invoke(tmp_path, manifest, tools, responses, drift=False, flags=("-I", "-S", "-B")):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "responses.json").write_text(json.dumps(responses))
    wrapper = tools / "fixture.py"
    wrapper.write_text(
        """import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('publication',Path(__file__).with_name('ra_publication.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
data=json.loads(Path(sys.argv[1]).read_text());counts={}
class Fake:
 def __init__(self,out):self.records=[]
 def __call__(self,url):
  counts[url]=counts.get(url,0)+1
  result=json.loads(json.dumps(data[url]))
  if DRIFT and '/releases/tags/' in url and counts[url]==2:result['id']+=1
  return result
m.Collector=Fake
sys.argv=[sys.argv[0],*sys.argv[2:]]
m.main()
""".replace("DRIFT", repr(drift))
    )
    return subprocess.run(
        [
            sys.executable,
            *flags,
            str(wrapper),
            str(tmp_path / "responses.json"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--out",
            str(tmp_path / "receipt"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_complete_binding_retains_bounded_scope(tmp_path):
    manifest, tools, responses = setup(tmp_path)
    p = invoke(tmp_path, manifest, tools, responses)
    assert p.returncode == 0, p.stderr
    result = json.loads((tmp_path / "receipt/result.json").read_text())
    assert result["status"] == "PASS" and result["proves_published_release_index_binding"] is True
    assert len(result["publications"]) == 2
    assert all(r["github_reports_verified_tag"] is True for r in result["publications"].values())
    assert all(
        v is False
        for k, v in result.items()
        if k.startswith("proves_") and k != "proves_published_release_index_binding"
    )
    assert result["startup_activated"] is False


@pytest.mark.parametrize(
    "mutant",
    [
        "release_id",
        "release_tag",
        "draft",
        "prerelease",
        "release_url",
        "release_time",
        "ref_name",
        "ref_kind",
        "ref_oid",
        "ref_url",
        "tag_name",
        "tag_oid",
        "target",
        "verified",
        "reason",
        "payload",
        "signature",
        "verified_time",
        "project",
        "version",
        "missing",
        "duplicate",
        "sha",
        "size",
        "yanked",
        "file_url",
        "file_kind",
        "http",
        "publication_drift",
    ],
)
def test_authority_and_artifact_mutations_refuse_with_failure_receipt(tmp_path, mutant):
    manifest, tools, responses = setup(tmp_path)
    release, ref, tag, pypi = [responses[k] for k in list(responses)[:4]]
    file = pypi["urls"][0]
    if mutant == "release_id":
        release["id"] = True
    elif mutant == "release_tag":
        release["tag_name"] = "foreign"
    elif mutant == "draft":
        release["draft"] = True
    elif mutant == "prerelease":
        release["prerelease"] = True
    elif mutant == "release_url":
        release["html_url"] = release["html_url"].replace("pjordanandrsn", "foreign")
    elif mutant == "release_time":
        release["published_at"] = None
    elif mutant == "ref_name":
        ref["ref"] = "refs/heads/main"
    elif mutant == "ref_kind":
        ref["object"]["type"] = "commit"
    elif mutant == "ref_oid":
        ref["object"]["sha"] = "short"
    elif mutant == "ref_url":
        ref["object"]["url"] = ref["object"]["url"].replace("pjordanandrsn", "foreign")
    elif mutant == "tag_name":
        tag["tag"] = "foreign"
    elif mutant == "tag_oid":
        tag["sha"] = "0" * 40
    elif mutant == "target":
        tag["object"]["sha"] = "0" * 40
    elif mutant == "verified":
        tag["verification"]["verified"] = False
    elif mutant == "reason":
        tag["verification"]["reason"] = "unsigned"
    elif mutant == "payload":
        tag["verification"]["payload"] += "modified"
    elif mutant == "signature":
        tag["verification"]["signature"] += "modified"
    elif mutant == "verified_time":
        tag["verification"]["verified_at"] = None
    elif mutant == "project":
        pypi["info"]["name"] = "foreign"
    elif mutant == "version":
        pypi["info"]["version"] = "1.0.0"
    elif mutant == "missing":
        pypi["urls"] = []
    elif mutant == "duplicate":
        pypi["urls"].append(file.copy())
    elif mutant == "sha":
        file["digests"]["sha256"] = "0" * 64
    elif mutant == "size":
        file["size"] += 1
    elif mutant == "yanked":
        file["yanked"] = True
    elif mutant == "file_url":
        file["url"] = file["url"].replace("files.pythonhosted.org", "foreign.invalid")
    elif mutant == "file_kind":
        file["packagetype"] = "sdist"
    elif mutant == "http":
        file["url"] = file["url"].replace("https:", "http:")
    p = invoke(tmp_path, manifest, tools, responses, drift=mutant == "publication_drift")
    assert p.returncode != 0, mutant
    result = json.loads((tmp_path / "receipt/result.json").read_text())
    assert result["status"] == "FAILED" and "proves_published_release_index_binding" not in result


@pytest.mark.parametrize("flags", [("-S", "-B"), ("-I", "-B"), ("-I", "-S")])
def test_isolated_no_site_controls(tmp_path, flags):
    manifest, tools, responses = setup(tmp_path)
    p = invoke(tmp_path, manifest, tools, responses, flags=flags)
    assert p.returncode != 0 and "python -I -S -B required" in p.stderr


@pytest.mark.parametrize("mutant", ["foreign", "http", "query", "redirect", "oversize", "duplicate_json", "status"])
def test_transport_refuses_foreign_or_ambiguous_response(tmp_path, monkeypatch, mutant):
    url = "https://pypi.org/pypi/experts4bit-qlora/0.0.0/json"
    if mutant == "foreign":
        url = url.replace("pypi.org", "foreign.invalid")
    elif mutant == "http":
        url = url.replace("https:", "http:")
    elif mutant == "query":
        url += "?token=synthetic"
    data = (
        b"x" * (publication.LIMIT + 1)
        if mutant == "oversize"
        else b'{"same":1,"same":2}'
        if mutant == "duplicate_json"
        else b"{}"
    )

    class Response:
        status = 503 if mutant == "status" else 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def geturl(self):
            return url + "/changed" if mutant == "redirect" else url

        def read(self, n):
            return data[:n]

    def opener(*handlers):
        assert any(isinstance(h, publication.NoRedirect) for h in handlers)
        assert next(h for h in handlers if isinstance(h, publication.urllib.request.ProxyHandler)).proxies == {}
        return types.SimpleNamespace(open=lambda *_args, **_kwargs: Response())

    monkeypatch.setattr(publication.urllib.request, "build_opener", opener)
    with pytest.raises(ValueError):
        publication.Collector(tmp_path)(url)


def test_collector_retains_raw_bytes_and_clock_reads(tmp_path, monkeypatch):
    data = b'{"public":true}\n'
    url = "https://pypi.org/pypi/experts4bit-qlora/0.0.0/json"
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def geturl(self):
            return url
        def read(self, _):
            return data
    monkeypatch.setattr(publication.urllib.request, "build_opener",
                        lambda *_: types.SimpleNamespace(open=lambda *_args, **_kwargs: Response()))
    fetch = publication.Collector(tmp_path)
    assert fetch(url) == {"public": True}
    record, = fetch.records
    assert (tmp_path / record["file"]).read_bytes() == data
    assert record["sha256"] == publication.hashlib.sha256(data).hexdigest()
    assert record["bytes"] == len(data)
    assert publication.timestamp(record["started_at"]) <= publication.timestamp(record["finished_at"])


def test_redirect_adapter_never_follows_redirect():
    with pytest.raises(ValueError, match="redirects refused"):
        publication.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://foreign.invalid")
