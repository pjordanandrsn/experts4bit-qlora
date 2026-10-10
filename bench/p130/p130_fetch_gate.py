#!/usr/bin/env python3
"""Lane P130's pre-rental fetch gate (bench/p130/PREREG-p130.md, "Budget, sequencing and STOP rules"; DQ11's launch-gate
pattern, bench/dq11/DQ11-AMENDMENT-3.md). Before the rental controller quotes, every locked thing the box will fetch is
resolved from a CPU host with the box's own clients, and any missing, unauthorized or mismatched entry refuses. It never
installs, never downloads a whole file, never quotes and never rents.

What the box fetches, and how each is resolved here:
- **the checkpoint** (``p130_run.sh``: ``hf_fetch_watchdog.py``, i.e. huggingface_hub's ``snapshot_download`` at the
  pinned revision with ``ALLOW``). ``model_info(files_metadata=True)`` at the revision lists every file with its size,
  blob id and LFS sha256. The files ``ALLOW`` selects (``filter_repo_objects``, as the download selects them) must include
  the safetensors index and every shard it names. For each, ``get_hf_file_metadata`` on its resolve URL (the HEAD the
  download makes) must answer with the revision's commit, the listed size, and as its etag the listed LFS sha256 (LFS
  files) or blob id (the rest);
- **the windows' corpus** (``p117_box.windows``: ``datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")``).
  The dataset's README and every file under ``wikitext-2-raw-v1/`` at ``main`` resolve the same way. P117's box does not
  pin the corpus by revision, so the commit ``main`` names is recorded and the reducer's windows digest carries the rest;
- **e4b at the launch commit and grouped-nf4-gemm at its pin** (pip's ``git+`` URLs): ``git fetch --depth 1`` of each
  commit by SHA must fetch exactly that commit;
- **the PyPI requirements** ``p130_run.sh`` installs: pip's own resolver (``install --dry-run --report``, no dependencies,
  wheels only) for the box's platform (``BOX_PYTHON`` on manylinux x86_64) picks each one's wheel, which must satisfy its
  pin, and every picked URL must answer a one-byte range GET. Their dependencies resolve on the box, as every lane's do.
There is no other asset: the staged files travel from the controller and are checked against ``staged.sha256``.

    python p130_fetch_gate.py --e4b-sha SHA --out GATE.json     (exit 0 passed, 1 refused, 2 usage)
    python p130_fetch_gate.py --self-test

The report (``schema`` p130-fetch-gate/1) is what ``p130_drive.sh`` requires (``P130_FETCH_GATE``): passed, for the
launch commit, and under ``MAX_AGE_S`` old.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from urllib.error import HTTPError, URLError

MODEL, REV = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
ALLOW = ("*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json")   # p130_run.sh's --allow
CORPUS, CORPUS_CONFIG = "Salesforce/wikitext", "wikitext-2-raw-v1"
GNF4_SHA = "724ccc454f006c1a46836e434e997f31f293747f"
REPOS = {"e4b": "https://github.com/pjordanandrsn/experts4bit-qlora.git",
         "gnf4": "https://github.com/pjordanandrsn/grouped-nf4-gemm.git"}
PYPI = ("transformers==5.17.0", "bitsandbytes==0.50.2", "datasets", "accelerate", "sentencepiece", "safetensors",
        "huggingface_hub>=1.31,<2", "pytest")                                                     # p130_run.sh's pip line
BOX_PYTHON = "3.11"                    # pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel's CPython
BOX_PLATFORMS = ("manylinux_2_28_x86_64", "manylinux_2_24_x86_64", "manylinux2014_x86_64", "manylinux_2_17_x86_64")
HUB_RANGE = ((1, 31), (2, 0))           # the box's huggingface_hub, the client this gate must be
USER_AGENT = "p130-fetch-gate/1"
MAX_AGE_S = 24 * 3600


# ---- the pure checks (the self-test drives these with fixtures) -------------------------------------------------------

def check_listing(listing: list, allow_selected: set, heads: dict, commit: str, *, need_index=True) -> list:
    """Refusals for one repository: ``listing`` rows {file, size, blob_id, lfs_sha256}; ``allow_selected`` the names the
    download would take; ``heads`` file -> {commit, size, etag} or {error, status}; ``commit`` the revision's."""
    out = []
    rows = {r["file"]: r for r in listing}
    if not allow_selected:
        out.append("no file selected")
    if need_index:
        idx = rows.get("model.safetensors.index.json")
        if idx is None or "model.safetensors.index.json" not in allow_selected:
            out.append("model.safetensors.index.json missing")
        shards = [f for f in allow_selected if f.endswith(".safetensors")]
        if not shards:
            out.append("no safetensors shard selected")
    for name in sorted(allow_selected):
        r, h = rows.get(name), heads.get(name)
        if r is None:
            out.append(f"{name}: selected but not listed")
            continue
        if h is None:
            out.append(f"{name}: not resolved")
            continue
        if "error" in h:
            out.append(f"{name}: {h.get('status')} {h['error']}")
            continue
        want_etag = r.get("lfs_sha256") or r.get("blob_id")
        if h.get("commit") != commit:
            out.append(f"{name}: resolved at commit {h.get('commit')}, not {commit}")
        if not r.get("size") or h.get("size") != r.get("size"):
            out.append(f"{name}: size {h.get('size')} against the listing's {r.get('size')}")
        if not want_etag or (h.get("etag") or "").strip('"') != want_etag:
            out.append(f"{name}: etag {h.get('etag')} against the listing's {want_etag}")
    return out


def check_shards(index: dict, allow_selected: set) -> list:
    """Every shard the safetensors index names is selected."""
    named = set((index or {}).get("weight_map", {}).values())
    if not named:
        return ["the safetensors index names no shard"]
    return [f"{s}: named by the index, not selected" for s in sorted(named - allow_selected)]


def check_git(results: dict, want: dict) -> list:
    return [f"{k}: {results.get(k)} is not {sha}" for k, sha in want.items() if results.get(k) != sha]


def check_pypi(specs: tuple, picked: list, probes: dict) -> list:
    """``picked`` pip's report rows {name, version, url, sha256}; ``probes`` url -> {status, bytes} or {error, status}."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    out = []
    by = {canonicalize_name(p["name"]): p for p in picked}
    for spec in specs:
        req = Requirement(spec)
        p = by.get(canonicalize_name(req.name))
        if p is None:
            out.append(f"{spec}: pip picked nothing")
            continue
        if not req.specifier.contains(p["version"], prereleases=True):
            out.append(f"{spec}: pip picked {p['version']}")
        if not p.get("url", "").startswith("https://") or not p["url"].endswith(".whl") or not p.get("sha256"):
            out.append(f"{spec}: not a hashed HTTPS wheel ({p.get('url')})")
            continue
        pr = probes.get(p["url"])
        if pr is None or "error" in pr or not 200 <= int(pr.get("status") or 0) < 300 or pr.get("bytes") != 1:
            out.append(f"{spec}: {p['url']} answered {pr}")
    return out


def hub_in_range(version: str) -> bool:
    v = tuple(int(x) for x in version.split(".")[:2])
    return HUB_RANGE[0] <= v < HUB_RANGE[1]


# ---- the network (resolved with the box's clients) ---------------------------------------------------------------

def _hub_repo(api, repo, revision, repo_type, allow=None, prefixes=None):
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import filter_repo_objects
    info = (api.model_info if repo_type == "model" else api.dataset_info)(repo, revision=revision, files_metadata=True)
    listing = [{"file": s.rfilename, "size": s.size, "blob_id": s.blob_id,
                "lfs_sha256": (s.lfs.sha256 if getattr(s, "lfs", None) else None)} for s in info.siblings]
    names = [r["file"] for r in listing]
    if allow is not None:
        selected = set(filter_repo_objects(names, allow_patterns=list(allow)))
    else:
        selected = {n for n in names if any(n == p or n.startswith(p) for p in prefixes)}
    heads = {}
    for name in sorted(selected):
        try:
            m = get_hf_file_metadata(hf_hub_url(repo, name, revision=revision, repo_type=repo_type),
                                     user_agent=USER_AGENT)
            heads[name] = {"commit": m.commit_hash, "size": m.size, "etag": m.etag}
        except Exception as e:  # noqa: BLE001 -- every failure is a refusal, with its status if it had one
            status = getattr(getattr(e, "response", None), "status_code", None)
            heads[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}", "status": status}
    return info.sha, listing, selected, heads


def _git(url, sha):
    with tempfile.TemporaryDirectory() as t:
        run = lambda *a: subprocess.run(["git", "-C", t, *a], capture_output=True, text=True, timeout=600)  # noqa: E731
        run("init", "-q")
        f = run("fetch", "-q", "--depth", "1", "--filter=blob:none", url, sha)
        if f.returncode != 0:
            return f"fetch failed: {f.stderr.strip()[:200]}"
        return run("rev-parse", "FETCH_HEAD").stdout.strip()


def _pip_pick(specs):
    with tempfile.TemporaryDirectory() as t:
        report = os.path.join(t, "report.json")
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--no-deps", "--only-binary=:all:",
               "--python-version", BOX_PYTHON, "--implementation", "cp", "--target", os.path.join(t, "target"),
               "--report", report, "--quiet", "--disable-pip-version-check"]
        for p in BOX_PLATFORMS:
            cmd += ["--platform", p]
        r = subprocess.run(cmd + list(specs), capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return None, f"pip could not resolve: {r.stderr.strip()[-400:]}"
        rows = []
        for item in json.load(open(report))["install"]:
            di = item.get("download_info") or {}
            rows.append({"name": item["metadata"]["name"], "version": item["metadata"]["version"], "url": di.get("url"),
                         "sha256": ((di.get("archive_info") or {}).get("hashes") or {}).get("sha256")})
        return rows, None


def _probe(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return {"status": resp.status, "bytes": len(resp.read(1))}
    except HTTPError as e:
        return {"status": e.code, "error": str(e)}
    except (URLError, OSError) as e:
        return {"status": None, "error": str(e)}


def gate(e4b_sha: str) -> dict:
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download
    rep = {"schema": "p130-fetch-gate/1", "generated_at": int(time.time()), "e4b_sha": e4b_sha, "user_agent": USER_AGENT,
           "clients": {"python": sys.version.split()[0], "huggingface_hub": huggingface_hub.__version__,
                       "git": subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip(),
                       "pip": subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True,
                                             text=True).stdout.split(" from ")[0]},
           "refusals": []}
    ref = rep["refusals"]
    if not hub_in_range(huggingface_hub.__version__):
        ref.append(f"huggingface_hub {huggingface_hub.__version__} is not the box's client (>=1.31,<2)")
    api = HfApi(user_agent=USER_AGENT)
    # the checkpoint
    try:
        sha, listing, selected, heads = _hub_repo(api, MODEL, REV, "model", allow=ALLOW)
        out = check_listing(listing, selected, heads, REV)
        if sha != REV:
            out.append(f"{MODEL}@{REV} resolves to {sha}")
        if "model.safetensors.index.json" in selected:
            with tempfile.TemporaryDirectory() as t:
                idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json", revision=REV, cache_dir=t)))
            out += check_shards(idx, selected)
        rep["model"] = {"repo": MODEL, "revision": REV, "files": len(selected),
                        "bytes": sum(r["size"] or 0 for r in listing if r["file"] in selected), "listing": listing,
                        "heads": heads, "refusals": out}
        ref += [f"model {x}" for x in out]
    except Exception as e:  # noqa: BLE001
        ref.append(f"model: {type(e).__name__}: {str(e)[:300]}")
    # the corpus
    try:
        sha, listing, selected, heads = _hub_repo(api, CORPUS, "main", "dataset", prefixes=(f"{CORPUS_CONFIG}/", "README.md"))
        out = check_listing(listing, selected, heads, sha, need_index=False)
        if not any(f.startswith(f"{CORPUS_CONFIG}/test") for f in selected):
            out.append(f"no {CORPUS_CONFIG}/test file")
        rep["corpus"] = {"repo": CORPUS, "config": CORPUS_CONFIG, "main": sha, "files": sorted(selected), "heads": heads,
                         "refusals": out}
        ref += [f"corpus {x}" for x in out]
    except Exception as e:  # noqa: BLE001
        ref.append(f"corpus: {type(e).__name__}: {str(e)[:300]}")
    # the two git commits
    want = {"e4b": e4b_sha, "gnf4": GNF4_SHA}
    got = {k: _git(REPOS[k], sha) for k, sha in want.items()}
    rep["git"] = {"want": want, "got": got}
    ref += [f"git {x}" for x in check_git(got, want)]
    # PyPI, by pip's resolver for the box's platform
    picked, err = _pip_pick(PYPI)
    if err:
        ref.append(f"pypi {err}")
        picked = []
    probes = {p["url"]: _probe(p["url"]) for p in picked if p.get("url")}
    rep["pypi"] = {"specs": list(PYPI), "box_python": BOX_PYTHON, "platforms": list(BOX_PLATFORMS), "picked": picked,
                   "probes": probes}
    if not err:
        ref += [f"pypi {x}" for x in check_pypi(PYPI, picked, probes)]
    rep["passed"] = not ref
    return rep


def report_ok(rep: dict, e4b_sha: str, now: float | None = None) -> str | None:
    """None if ``rep`` lets the driver stage for ``e4b_sha``; else why not (the driver's check, also tested here)."""
    now = time.time() if now is None else now
    if rep.get("schema") != "p130-fetch-gate/1":
        return f"schema {rep.get('schema')}"
    if rep.get("passed") is not True or rep.get("refusals"):
        return f"the gate refused: {rep.get('refusals')}"
    if rep.get("e4b_sha") != e4b_sha:
        return f"the gate resolved e4b {rep.get('e4b_sha')}, the launch commit is {e4b_sha}"
    if not 0 <= now - float(rep.get("generated_at") or 0) <= MAX_AGE_S:
        return f"the gate report is {int(now - float(rep.get('generated_at') or 0))} s old (at most {MAX_AGE_S})"
    return None


# ---- self-test ------------------------------------------------------------------------------------------------------

def self_test() -> int:
    cases = []
    C = "c" * 40
    listing = [{"file": "model.safetensors.index.json", "size": 10, "blob_id": "b1", "lfs_sha256": None},
               {"file": "model-00001.safetensors", "size": 99, "blob_id": "b2", "lfs_sha256": "s2"},
               {"file": "README.md", "size": 5, "blob_id": "b3", "lfs_sha256": None}]
    sel = {"model.safetensors.index.json", "model-00001.safetensors"}
    heads = {"model.safetensors.index.json": {"commit": C, "size": 10, "etag": '"b1"'},
             "model-00001.safetensors": {"commit": C, "size": 99, "etag": "s2"}}
    cases.append(("listing clean", check_listing(listing, sel, heads, C) == []))
    h = json.loads(json.dumps(heads))
    h["model-00001.safetensors"]["size"] = 98
    cases.append(("size mismatch", any("size 98" in x for x in check_listing(listing, sel, h, C))))
    h = json.loads(json.dumps(heads))
    h["model-00001.safetensors"]["etag"] = "other"
    cases.append(("hash mismatch", any("etag" in x for x in check_listing(listing, sel, h, C))))
    h = json.loads(json.dumps(heads))
    h["model.safetensors.index.json"]["commit"] = "d" * 40
    cases.append(("another commit", any("resolved at commit" in x for x in check_listing(listing, sel, h, C))))
    h = json.loads(json.dumps(heads))
    h["model-00001.safetensors"] = {"error": "HTTPError: 403 Forbidden", "status": 403}
    cases.append(("unauthorized", any("403" in x for x in check_listing(listing, sel, h, C))))
    cases.append(("selected but missing", any("not listed" in x for x in
                                              check_listing(listing, sel | {"tokenizer.json"}, heads, C))))
    cases.append(("no shard", any("no safetensors shard" in x for x in
                                  check_listing(listing, {"model.safetensors.index.json"}, heads, C))))
    cases.append(("shards all selected", check_shards({"weight_map": {"a": "model-00001.safetensors"}}, sel) == []))
    cases.append(("a shard not selected", check_shards({"weight_map": {"a": "model-00002.safetensors"}}, sel) != []))
    cases.append(("git ok", check_git({"e4b": "a" * 40}, {"e4b": "a" * 40}) == []))
    cases.append(("git missing", check_git({"e4b": "fetch failed: no such commit"}, {"e4b": "a" * 40}) != []))
    picked = [{"name": "transformers", "version": "5.17.0", "url": "https://files.pythonhosted.org/t.whl", "sha256": "x"},
              {"name": "huggingface-hub", "version": "1.33.0", "url": "https://files.pythonhosted.org/h.whl", "sha256": "y"}]
    probes = {"https://files.pythonhosted.org/t.whl": {"status": 206, "bytes": 1},
              "https://files.pythonhosted.org/h.whl": {"status": 206, "bytes": 1}}
    specs = ("transformers==5.17.0", "huggingface_hub>=1.31,<2")
    try:
        import packaging  # noqa: F401
        cases.append(("pypi clean", check_pypi(specs, picked, probes) == []))
        bad = [dict(picked[0], version="5.16.1"), picked[1]]
        cases.append(("pypi off its pin", any("picked 5.16.1" in x for x in check_pypi(specs, bad, probes))))
        cases.append(("pypi 403", any("answered" in x for x in check_pypi(
            specs, picked, {**probes, "https://files.pythonhosted.org/t.whl": {"status": 403, "error": "Forbidden"}}))))
        cases.append(("pypi nothing picked", any("picked nothing" in x for x in check_pypi(specs, picked[:1], probes))))
    except ImportError:
        cases.append(("packaging importable", False))
    cases.append(("hub client range", hub_in_range("1.33.0") and not hub_in_range("1.27.0") and not hub_in_range("2.2.0")))
    now = 1_000_000.0
    good = {"schema": "p130-fetch-gate/1", "passed": True, "refusals": [], "e4b_sha": "a" * 40, "generated_at": now - 60}
    cases.append(("report ok", report_ok(good, "a" * 40, now) is None))
    cases.append(("report for another commit", report_ok(good, "b" * 40, now) is not None))
    cases.append(("report stale", report_ok(dict(good, generated_at=now - MAX_AGE_S - 1), "a" * 40, now) is not None))
    cases.append(("report refused", report_ok(dict(good, passed=False, refusals=["x"]), "a" * 40, now) is not None))
    bad_names = [n for n, ok in cases if not ok]
    print(f"p130_fetch_gate self-test {'OK' if not bad_names else 'FAILED ' + str(bad_names)} ({len(cases)} cases)")
    return 0 if not bad_names else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--e4b-sha")
    ap.add_argument("--out")
    ap.add_argument("--check-report", help="the driver's check: exit 0 iff this report lets --e4b-sha stage")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.e4b_sha or len(a.e4b_sha) != 40 or any(c not in "0123456789abcdef" for c in a.e4b_sha):
        print("usage: --e4b-sha must be the 40-character launch commit")
        return 2
    if a.check_report:
        try:
            why = report_ok(json.load(open(a.check_report)), a.e4b_sha)
        except (OSError, ValueError) as e:
            why = f"unreadable: {e}"
        print("P130_FETCH_GATE " + ("ok" if why is None else f"REFUSED: {why}"))
        return 0 if why is None else 1
    if not a.out:
        print("usage: --out GATE.json")
        return 2
    rep = gate(a.e4b_sha)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    m = rep.get("model") or {}
    print(f"P130_FETCH_GATE {'PASSED' if rep['passed'] else 'REFUSED'} | model {m.get('files')} files "
          f"{(m.get('bytes') or 0) / 1e9:.1f} GB | corpus main {(rep.get('corpus') or {}).get('main')} "
          f"| git {rep['git']['got']} | pypi {[(p['name'], p['version']) for p in rep['pypi']['picked']]}")
    for r in rep["refusals"]:
        print(f"  REFUSED {r}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
