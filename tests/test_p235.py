# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P235 (bench/p235/PREREG-p235.md): P58's same-box vLLM head-to-head re-read at the release pins (e4b 0.58.0,
grouped-nf4-gemm 0.51.0, vLLM 0.32.0 and 0.30.0), each installed from PyPI with its provenance checked. The staged pins,
resolved the way test_p58_staged_pin.py resolves P58's and held to P58's digests; provenance.py's rules on fake
distributions (PyPI OK; editable, file://, another host, any git install, a wrong version or a missing
distribution refused) and its constants against the PREREG and the runner; the runner as P58's by named edits (every
changed line one of them), calling the check before the first e4b arm and in each vLLM venv before its arms; the
reducer as P58's with its labels renamed (its output on P58's receipts unchanged but for the title); and the drive's
pod-launch contract, refusals and dry run (P233's form)."""
import ast
import difflib
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import types

import pytest
from guard_outcomes import expect_outcome
from test_lane_guard_fit import fit_problems

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p235"
P58 = B / "p58"
PIN = LANE / "staged.sha256"
#: staged path (as the box sees it) -> the file in this checkout
SOURCES = {"p235_run.sh": LANE / "p235_run.sh", "p235_reduce.py": LANE / "p235_reduce.py",
           "provenance.py": LANE / "provenance.py", "release_proof.py": LANE / "release_proof.py",
           "p37_vllm.py": B / "h2h-20260905" / "p37" / "p37_vllm.py", "step_decomp.py": B / "p39" / "step_decomp.py",
           "k8_bake.py": B / "p39" / "k8_bake.py", "calib.json": B / "p39" / "calib.json",
           "hook/usercustomize.py": B / "p42" / "hook" / "usercustomize.py"}
RUN = (LANE / "p235_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p235_drive.sh").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p235.md").read_text(encoding="utf-8")
FLAT = " ".join(PREREG.split())     # the PREREG's wrapped lines joined: a phrase reads across a line break
NEW_FILES = [LANE / f for f in ("p235_run.sh", "p235_reduce.py", "provenance.py", "release_proof.py", "p235_drive.sh",
                                "staged.sha256", "PREREG-p235.md")] + [REPO / "tests" / "test_p235.py"]
# The registration's changelog fragment (changelog.d/p235-registered.md) was folded into CHANGELOG.md by the 0.58.1
# release, so it is no longer a file; its text is checked there with the rest of the release section.
#: loggetta's bench/release_proof.py, staged byte-identical: its source and sha256 (the ONE constant a new helper
#: version edits, beside its ONE staged.sha256 line)
RELEASE_PROOF = {"source": "loggetta bench/release_proof.py at f09e5676 (#126's merge)",
                 "sha256": "aff3fc80db8d35bbf94a9aaeed125c9d424cfe93bf16eb0e6868b6f9d03f13e0"}
#: the top-level import names of the dists the arms import, read from the published wheels (experts4bit-qlora 0.57.0,
#: grouped-nf4-gemm 0.50.0 and the released 0.51.0, vllm 0.32.0 and 0.30.0 x86_64; zip central directories) and from
#: grouped-nf4-gemm's pyproject py-modules + packages at v0.50.0 and main (the same 43 names): a staged file must not
#: shadow one
TOP_LEVEL = {
    "experts4bit-qlora": {"experts4bit_qlora"},
    "grouped-nf4-gemm": {"_triton_shim", "arena_experts", "arena_moe_patch", "cold_cpu_view", "cold_deadline",
                         "cpu_grouped", "dense_gemv", "dense_i8", "dev_row_cache", "fp8_kv", "fp8_paged_attn",
                         "gate_native_load_20b", "gguf_reader", "gnf4_native", "gptq_pack", "host_gather", "int4_b32",
                         "int4_pack_ref", "int4_smallm", "kquant_ref", "moonshot_gather", "mxfp4_grouped", "mxfp4_loader",
                         "mxfp4_native_load", "mxfp4_pack_ref", "mxfp4_pipelined", "mxfp4_qlora", "mxfp4_residency",
                         "nf4_grouped", "nf4_pack_ref", "nf4_qlora", "nf4_route", "nf4_smallm", "nvme_arena",
                         "nvme_bake_nf4", "nvme_reader", "nvme_residency", "reuse_profile", "row_pool",
                         "run_mxfp4_20b_qlora", "segmented_pool", "verify_provenance", "vram_slots"},
    "vllm": {"vllm"},
}
#: P58's names for what P235 renamed, so P58's text and this lane's compare line for line
TO_P58 = (("/root/p235", "/root/p58"), ("P235_", "P58_"), ("p235_reduce.py", "p58_reduce.py"),
          ("RESULTS-p235-generated.md", "RESULTS-p58-generated.md"), ("] p235: ", "] p58: "), ('"P235 arm=', '"P58 arm='),
          ("bench/p235/staged.sha256", "bench/p58/staged.sha256"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROV = _load("p235_provenance_under_test", LANE / "provenance.py")


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _resolve(name):
    """test_p58_staged_pin.py's map, for this lane's names: the runner, the reducer, the provenance check and the
    release proof from bench/p235; p37_vllm.py from bench/h2h-20260905/p37 (byte-identical to P37's); the hook from bench/p42; everything
    else (step_decomp.py, k8_bake.py, calib.json) from bench/p39."""
    if name in ("p235_run.sh", "p235_reduce.py", "provenance.py", "release_proof.py"):
        return LANE / name
    if name == "p37_vllm.py":
        return B / "h2h-20260905" / "p37" / name
    if name.startswith("hook/"):
        return B / "p42" / name
    return B / "p39" / name


@pytest.mark.parametrize("name,want", list(_pins(PIN).items()))
def test_staged_file_matches_its_pin(name, want):
    src = _resolve(name)
    assert src == SOURCES[name] and src.is_file(), (name, src)
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    assert got == want, f"{name} has changed without re-pinning: repo {got[:12]}, staged.sha256 {want[:12]}"


def test_every_staged_piece_is_pinned_and_the_harness_is_p58s():
    """The pins name exactly the staged pieces, and every piece but this lane's three is at P58's digest."""
    mine, p58 = _pins(PIN), _pins(P58 / "staged.sha256")
    assert list(mine) == list(SOURCES)
    shared = ("p37_vllm.py", "step_decomp.py", "k8_bake.py", "calib.json", "hook/usercustomize.py")
    assert all(mine[n] == p58[n] for n in shared), {n: (mine[n][:8], p58[n][:8]) for n in shared}
    assert set(p58) - set(shared) == {"p58_run.sh", "p58_reduce.py"}


def test_the_release_proof_is_loggettas_bytes():
    """bench/p235/release_proof.py is loggetta's helper byte for byte: the file, its staged.sha256 line and the one
    constant here agree, and nothing else in the lane carries the digest (a new helper version is two edits)."""
    data = (LANE / "release_proof.py").read_bytes()
    assert hashlib.sha256(data).hexdigest() == RELEASE_PROOF["sha256"] == _pins(PIN)["release_proof.py"]
    assert b"\r" not in data
    for f in ("p235_run.sh", "p235_drive.sh", "provenance.py"):
        assert RELEASE_PROOF["sha256"][:8] not in (LANE / f).read_text(encoding="utf-8"), f
    commit, merge = RELEASE_PROOF["source"].split(" at ")[1].split(" ", 1)
    assert f"is loggetta `bench/release_proof.py` at `{commit}` {merge}, sha256 `{RELEASE_PROOF['sha256']}`" in FLAT
    rp = _load("p235_release_proof_under_test", LANE / "release_proof.py")      # stdlib only; no network here
    assert rp.main([]) == 2 and rp.main(["experts4bit-qlora"]) == 2
    code = "\n".join(ln for ln in RUN.splitlines() if not ln.lstrip().startswith("#"))
    pins = re.findall(r"release_proof\.py ([^)>]+)\)", code)
    assert len(pins) == 4, pins                                  # the binding proofs and the fast-checks
    for pin in pins:
        for arg in pin.split():
            dist, sep, version = arg.strip('"').partition("==")
            assert sep and (PROV.E4B.get(dist) == version or (dist, version) in (("vllm", "${VVER[$VT]}"), ("vllm", "$VER"))), arg


def test_no_staged_file_shadows_an_imported_top_level_module():
    """The proofs and the arms run with sys.path[0] = $W and $W/hook on PYTHONPATH, so a staged module (or a directory
    the runner makes there) named like a top-level module of e4b, gnf4 or vLLM would be imported in its place."""
    names = set().union(*TOP_LEVEL.values())
    staged = {pathlib.PurePosixPath(n).stem for n in SOURCES} | {pathlib.PurePosixPath(n).parts[0] for n in SOURCES}
    made = set(re.findall(r"mkdir -p \$W/(\w+)", RUN)) | set(re.findall(r"\$W/(\w+)/", RUN))
    assert {"hook", "logs", "work_qwen3", "usercustomize", "step_decomp", "provenance", "release_proof"} <= staged | made
    assert not (staged | made) & names, sorted((staged | made) & names)
    assert "int4_b32" in names and "nf4_smallm" in names and len(TOP_LEVEL["grouped-nf4-gemm"]) == 43


# ------------------------------------------------------------------------------------------------ provenance --

def _fake(site, name, version, direct_url=None):
    """a dist-info for NAME-VERSION under `site`, with direct_url.json when given"""
    d = site / f"{name.replace('-', '_')}-{version}.dist-info"
    d.mkdir(parents=True)
    (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    if direct_url is not None:
        (d / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")
    return site


SHA = "1" * 40
WHEEL = "https://files.pythonhosted.org/packages/aa/bb/experts4bit_qlora-0.58.0-py3-none-any.whl"


def _one(tmp_path, direct_url=None, version="0.58.0", pin="0.58.0", name="experts4bit-qlora", **kw):
    site = _fake(tmp_path / "site", name, version, direct_url)
    return PROV.check_one(name, pin, path=[str(site)], **kw)


def test_a_pypi_wheel_without_direct_url_passes(tmp_path):
    rec, errs = _one(tmp_path)
    assert not errs and rec == {"version": "0.58.0", "provenance": "pypi", "direct_url": None}


@pytest.mark.parametrize("url", [WHEEL, "https://pypi.org/packages/x/experts4bit_qlora-0.58.0-py3-none-any.whl"])
def test_a_direct_url_on_pypi_passes(tmp_path, url):
    du = {"url": url, "archive_info": {"hashes": {"sha256": "0" * 64}}}
    rec, errs = _one(tmp_path, du)
    assert not errs and rec["provenance"] == "pypi" and rec["direct_url"] == du


def test_an_editable_install_is_refused(tmp_path):
    rec, errs = _one(tmp_path, {"url": "file:///src/experts4bit-qlora", "dir_info": {"editable": True}})
    assert rec is None and len(errs) == 1 and "an editable install" in errs[0], errs


@pytest.mark.parametrize("du", [{"url": "file:///tmp/experts4bit_qlora-0.58.0-py3-none-any.whl", "archive_info": {}},
                                {"url": "FILE:///src", "dir_info": {}}])
def test_a_file_url_is_refused(tmp_path, du):
    rec, errs = _one(tmp_path, du)
    assert rec is None and len(errs) == 1 and "a file:// install" in errs[0], errs


@pytest.mark.parametrize("url", ["https://example.com/experts4bit_qlora-0.58.0-py3-none-any.whl",
                                 "https://files.pythonhosted.org.example.com/x.whl", "not a url",
                                 "http://files.pythonhosted.org/packages/aa/bb/experts4bit_qlora-0.58.0-py3-none-any.whl"])
def test_another_host_is_refused(tmp_path, url):
    rec, errs = _one(tmp_path, {"url": url, "archive_info": {}})
    assert rec is None and len(errs) == 1 and "which is not PyPI" in errs[0], errs


def _vcs(url, commit=SHA):
    return {"url": url, "vcs_info": {"vcs": "git", "requested_revision": "v0.58.0", "commit_id": commit}}


@pytest.mark.parametrize("url", ["https://github.com/pjordanandrsn/experts4bit-qlora.git",    # canonical, any commit
                                 "https://github.com/someone-else/experts4bit-qlora.git",      # a fork with the tag
                                 "https://mirror.example.com/experts4bit-qlora.git"])
def test_every_git_install_is_refused_whatever_its_url_or_commit(tmp_path, url):
    """the amended rule's path (1): index wheels only, so no tag is resolved through the URL the install recorded"""
    rec, errs = _one(tmp_path, _vcs(url))
    assert rec is None and errs == [f"experts4bit-qlora: a git install ({url} @ {SHA}) is not an index release"]
    assert not hasattr(PROV, "TAG_COMMITS")


def test_a_git_file_url_is_refused(tmp_path):
    rec, errs = _one(tmp_path, _vcs("file:///tmp/fork/experts4bit-qlora"))
    assert rec is None and len(errs) == 1 and "a file:// install" in errs[0], errs


def test_a_wrong_version_is_refused(tmp_path):
    rec, errs = _one(tmp_path, version="0.57.0")
    assert rec is None and errs == ["experts4bit-qlora: installed version 0.57.0 is not the registered 0.58.0"]


def test_a_missing_distribution_is_refused(tmp_path):
    (tmp_path / "site").mkdir()
    assert PROV.check_one("grouped-nf4-gemm", "0.51.0", path=[str(tmp_path / "site")]) == (
        None, ["grouped-nf4-gemm: not installed"])


@pytest.mark.parametrize("name,version", [("experts4bit-qlora", "0.57.0"), ("grouped-nf4-gemm", "0.50.0"),
                                          ("vllm", "0.31.0"), ("vllm", "0.33.0"), ("torch", "2.8.0")])
def test_an_unregistered_pin_is_refused_before_the_lookup(tmp_path, name, version):
    site = _fake(tmp_path / "site", name, version)
    assert PROV.check_one(name, version, path=[str(site)]) == (None, [f"{name}=={version} is not a registered pin"])


def test_check_names_every_failure_and_an_empty_list_fails(tmp_path):
    site = _fake(tmp_path / "site", "experts4bit-qlora", "0.58.0", {"url": "file:///src", "dir_info": {"editable": True}})
    _fake(site, "grouped-nf4-gemm", "0.50.0")
    recs, errs = PROV.check([("experts4bit-qlora", "0.58.0"), ("grouped-nf4-gemm", "0.51.0")], path=[str(site)])
    assert recs == {} and len(errs) == 2
    assert PROV.check([], path=[str(site)]) == ({}, ["no distributions named"])


def _cli(site, out, *pins):
    """provenance.py as the box runs it, but with site-packages off (-S) and the fake site first on sys.path"""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(site)}
    return subprocess.run([sys.executable, "-S", str(LANE / "provenance.py"), "check", *pins, "--out", str(out)],
                          capture_output=True, text=True, env=env, timeout=60)


def test_the_cli_writes_provenance_only_when_every_dist_passes(tmp_path):
    site = _fake(tmp_path / "site", "experts4bit-qlora", "0.58.0")
    _fake(site, "grouped-nf4-gemm", "0.51.0", {"url": WHEEL.replace("experts4bit_qlora-0.58.0", "grouped_nf4_gemm-0.51.0"),
                                               "archive_info": {}})
    out = tmp_path / "provenance.json"
    r = _cli(site, out, "experts4bit-qlora=0.58.0", "grouped-nf4-gemm=0.51.0")
    assert r.returncode == 0, r.stdout + r.stderr
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert {k: (v["version"], v["provenance"]) for k, v in rec.items()} == {
        "experts4bit-qlora": ("0.58.0", "pypi"), "grouped-nf4-gemm": ("0.51.0", "pypi")}
    bad = tmp_path / "bad"
    _fake(bad, "vllm", "0.32.0", {"url": "file:///src/vllm", "dir_info": {"editable": True}})
    r = _cli(bad, tmp_path / "v.json", "vllm=0.32.0")
    assert r.returncode == 1 and "PROVENANCE FAIL: vllm: an editable install" in r.stdout and not (tmp_path / "v.json").exists()
    r = _cli(site, tmp_path / "w.json", "experts4bit-qlora")
    assert r.returncode == 1 and "is not NAME=VERSION" in r.stdout


def _vllm_constants(run):
    m = re.search(r"^VLLM_PRIMARY=(\S+); VLLM_SECONDARY=(\S+)   #", run, re.M)
    return m and {"primary": m.group(1), "secondary": m.group(2)}


def test_the_constants_are_the_preregs_and_the_runners():
    assert PROV.E4B == {"experts4bit-qlora": "0.58.0", "grouped-nf4-gemm": "0.51.0"}
    assert PROV.VLLM == {"primary": "0.32.0", "secondary": "0.30.0"}
    assert _vllm_constants(RUN) == PROV.VLLM
    assert 'vllm_install primary "$VLLM_PRIMARY"' in RUN and 'vllm_install secondary "$VLLM_SECONDARY"' in RUN
    for name, version in PROV.E4B.items():
        assert f'"{name}=={version}"' in RUN, name                      # the pip pin
        assert f" {name}={version}" in RUN, name                        # the provenance call
    assert f'experts4bit_qlora.__version__ == "{PROV.E4B["experts4bit-qlora"]}"' in RUN
    assert f'md.version("grouped-nf4-gemm") == "{PROV.E4B["grouped-nf4-gemm"]}"' in RUN
    assert "`experts4bit-qlora==0.58.0`" in FLAT and "`grouped-nf4-gemm==0.51.0`" in FLAT
    assert "| vLLM PRIMARY | latest PyPI at launch (0.30.0) | **0.32.0**, a constant" in FLAT
    assert "| vLLM SECONDARY | 0.29.0 | **0.30.0** (P58's primary), a constant" in FLAT


def test_no_git_install_and_no_latest_vllm_lookup_remain():
    assert "git+" not in RUN and "pip install" in RUN
    assert "pypi.org/pypi/vllm" not in RUN and "P235_VLLM_PRIMARY" not in RUN and "P235_VLLM_SECONDARY" not in RUN
    for line in RUN.splitlines():
        if re.search(r"pip install .*(experts4bit-qlora|grouped-nf4-gemm|vllm==)", line):
            assert "--index-url https://pypi.org/simple" in line and "--only-binary" in line, line


E4B_CHECK = ("python -I $W/provenance.py check experts4bit-qlora=0.58.0 grouped-nf4-gemm=0.51.0 --out $W/provenance.json "
             '> logs/provenance_e4b.log 2>&1 || { cat logs/provenance_e4b.log; say "PROVENANCE FAIL (e4b/gnf4)"; finish 9; }')
VLLM_CHECK = '  $W/venv-$TAG/bin/python -I $W/provenance.py check "vllm=$VER" --out $W/provenance_vllm_$TAG.json'


def test_the_e4b_check_runs_after_the_installs_and_before_the_tripwire_and_every_arm():
    lines = RUN.splitlines()
    k = lines.index(E4B_CHECK)
    gnf4 = next(i for i, ln in enumerate(lines) if '"grouped-nf4-gemm==0.51.0" > logs/pip_gnf4.log' in ln)
    trip = next(i for i, ln in enumerate(lines) if ln.startswith("PYTHONPATH=$W/hook python - <<'PYT'"))
    first_arm = next(i for i, ln in enumerate(lines) if re.match(r"can_run \d+ \S+ +&& \{ (e4b|vllm)_arm", ln))
    assert gnf4 + 1 == k < trip < first_arm
    assert not any(ln.startswith(("if ", "[ ")) and "provenance" in ln for ln in lines)      # never behind a condition


def test_each_vllm_venv_is_checked_before_its_arms_and_a_failure_refuses_that_build():
    lines = RUN.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("vllm_install(){"))
    end = next(i for i in range(start, len(lines)) if lines[i].endswith("VOK[$TAG]=1; }"))
    body = lines[start:end + 1]
    k = next(i for i, ln in enumerate(body) if ln.startswith(VLLM_CHECK))
    pip = next(i for i, ln in enumerate(body) if '"vllm==$VER"' in ln)
    imp = next(i for i, ln in enumerate(body) if "<<'PYV'" in ln)
    assert pip < k < imp
    fail = body[k + 1]
    assert fail.lstrip().startswith("|| {") and "VOK[$TAG]=0; return; }" in fail and "finish" not in fail
    calls = [i for i, ln in enumerate(lines) if re.match(r'(\[.*\] \|\| )?vllm_install (primary|secondary) ', ln)]
    first_vllm_arm = next(i for i, ln in enumerate(lines) if "vllm_arm $VT" in ln)
    assert len(calls) == 2 and all(end < c < first_vllm_arm for c in calls)
    assert 'vllm_arm(){ local VT=$1 ARM=$2 B=$3; [ "${VOK[$VT]:-0}" = 1 ] || { echo "vllm-$VT/$ARM B=$B SKIPPED' in RUN


E4B_PROOF = ("(cd $W && PYTHONPATH=$W/hook python $W/release_proof.py experts4bit-qlora==0.58.0 grouped-nf4-gemm==0.51.0) "
             "> $W/release_proof.jsonl 2> logs/release_proof_e4b.log || { cat logs/release_proof_e4b.log; "
             'say "RELEASE PROOF FAIL (e4b/gnf4)"; finish 9; }')
VLLM_PROOF = ('  (cd $W && PYTHONPATH=$W/hook $W/venv-$VT/bin/python $W/release_proof.py "vllm==${VVER[$VT]}") '
              "> $W/release_proof_vllm_$VT.jsonl 2> logs/release_proof_vllm_$VT.log \\")
FAST_E4B = ("(cd $W && PYTHONPATH=$W/hook python $W/release_proof.py experts4bit-qlora==0.58.0 grouped-nf4-gemm==0.51.0) "
            "> $W/release_proof_fastcheck.jsonl 2> logs/release_proof_fastcheck_e4b.log || { cat logs/release_proof_fastcheck_e4b.log; "
            'say "RELEASE PROOF FAIL (fast-check, e4b/gnf4)"; finish 9; }')
FAST_E4B_FRESH = ("proof_fresh $W/release_proof_fastcheck.jsonl 2 || { cat $W/release_proof_fastcheck.jsonl; "
                  'say "RELEASE PROOF FAIL (fast-check, e4b/gnf4): pycache_fresh is not true"; finish 9; }')
FAST_VLLM = ('  (cd $W && PYTHONPATH=$W/hook $W/venv-$TAG/bin/python $W/release_proof.py "vllm==$VER") '
             "> $W/release_proof_vllm_${TAG}_fastcheck.jsonl 2> logs/release_proof_fastcheck_vllm_$TAG.log \\")
FAST_VLLM_FAIL = ('    || { cat logs/release_proof_fastcheck_vllm_$TAG.log; say "RELEASE PROOF FAIL (fast-check, vllm $TAG) -- e4b arms '
                  'still run; NO $TAG COMPARATOR"; echo "RELEASE PROOF FAIL (fast-check, vllm $TAG): vllm==$VER, its rows are refused" '
                  '>> summary.txt; VOK[$TAG]=0; return; }')
FAST_VLLM_FRESH = ('  proof_fresh $W/release_proof_vllm_${TAG}_fastcheck.jsonl 1 || { say "RELEASE PROOF FAIL (fast-check, vllm $TAG): '
                   'pycache_fresh is not true -- NO $TAG COMPARATOR"; echo "RELEASE PROOF FAIL (fast-check, vllm $TAG): vllm==$VER, '
                   'pycache_fresh not true, its rows are refused" >> summary.txt; VOK[$TAG]=0; return; }')
E4B_FRESH = ("proof_fresh $W/release_proof.jsonl 2 || { cat $W/release_proof.jsonl; "
             'say "RELEASE PROOF FAIL (e4b/gnf4): pycache_fresh is not true"; finish 9; }')
VLLM_FRESH = ('  [ "${VOK[$VT]}" = 0 ] || proof_fresh $W/release_proof_vllm_$VT.jsonl 1 || { say "RELEASE PROOF FAIL (vllm $VT): '
              'pycache_fresh is not true -- NO $VT COMPARATOR"; echo "RELEASE PROOF FAIL (vllm $VT ${VVER[$VT]}): pycache_fresh '
              'not true, its rows are refused" >> summary.txt; VOK[$VT]=0; }')
INSTALLS = re.compile(r"pip install|python -m venv|\bvllm_install (primary|secondary)\b")


def _last_install(lines):
    return max(i for i, ln in enumerate(lines) if INSTALLS.search(ln) and not ln.lstrip().startswith("#"))


def test_the_e4b_proof_is_the_last_step_before_the_first_e4b_arm():
    """After every install (the vLLM venvs included), unconditional, in the arms' import environment, and nothing but
    the rc bookkeeping between it and the first e4b arm; a refusal is rc 9."""
    lines = RUN.splitlines()
    k = lines.index(E4B_PROOF)
    first_e4b = next(i for i, ln in enumerate(lines) if re.match(r"can_run \d+ e4b_\S+ +&& \{ e4b_arm", ln))
    assert _last_install(lines) < k < first_e4b == k + 2 and lines[k + 1] == E4B_FRESH
    assert not any(INSTALLS.search(ln) for ln in lines[k:first_e4b])


def test_the_e4b_fast_check_runs_after_provenance_and_before_the_tripwire_and_the_fetch():
    """Not ruled, but it stops a wrong wheel in minutes: the same proof in the arms' env right after the installs and
    provenance.py, unconditional, its pycache_fresh read next, both refusing rc 9, before the tripwire and the fetch."""
    lines = RUN.splitlines()
    k = lines.index(FAST_E4B)
    prov = lines.index(E4B_CHECK)
    trip = next(i for i, ln in enumerate(lines) if ln.startswith("PYTHONPATH=$W/hook python - <<'PYT'"))
    fetch = lines.index('say "fetch $MID @ $REV"')
    assert prov < k < trip < fetch and lines[k + 1] == FAST_E4B_FRESH
    assert all(ln.startswith("#") or ln.startswith(("cat logs/provenance_e4b.log", "proof_fresh(){")) for ln in lines[prov + 1:k])
    assert k < lines.index(E4B_PROOF)                     # the binding proof is a separate, later call
    assert "only the binding proof, the last step before the first e4b arm, is" in RUN


def test_each_vllm_fast_check_runs_in_its_install_and_refuses_that_build():
    """Inside vllm_install, right after that build's provenance check and before its import tripwire: a refusal sets
    VOK=0 and returns (that build's rows refused, its binding proof skipped), never ends the run."""
    lines = RUN.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("vllm_install(){"))
    end = next(i for i in range(start, len(lines)) if lines[i].endswith("VOK[$TAG]=1; }"))
    body = lines[start:end + 1]
    prov = next(i for i, ln in enumerate(body) if ln.startswith(VLLM_CHECK))
    k = body.index(FAST_VLLM)
    assert prov + 2 < k and body[k + 1] == FAST_VLLM_FAIL and body[k + 2] == FAST_VLLM_FRESH
    assert k < next(i for i, ln in enumerate(body) if "<<'PYV'" in ln)
    for fail in (FAST_VLLM_FAIL, FAST_VLLM_FRESH):
        assert fail.endswith("VOK[$TAG]=0; return; }") and "finish" not in fail
        assert 'echo "RELEASE PROOF FAIL (fast-check, vllm $TAG):' in fail
    loop = 'for VT in primary secondary; do [ "${VOK[$VT]:-0}" = 1 ] || continue'
    assert any(ln.startswith(loop) for ln in lines)       # a refused build's binding proof then does not run


def test_each_vllm_proof_is_the_last_step_before_that_builds_arms():
    """In that build's venv, as p37_vllm.py runs (cwd $W, PYTHONPATH=$W/hook), after every install, right before the
    vLLM arms with only e4b arms (no install) since the e4b proof; a refusal refuses that build's rows, never the run."""
    lines = RUN.splitlines()
    k = lines.index(VLLM_PROOF)
    loop = lines[k - 1]
    assert loop == 'for VT in primary secondary; do [ "${VOK[$VT]:-0}" = 1 ] || continue   # no venv: its rows already read SKIPPED'
    fail = lines[k + 1]
    assert fail.lstrip().startswith("|| {") and fail.endswith("VOK[$VT]=0; }") and "finish" not in fail and "return" not in fail
    assert lines[k + 2] == VLLM_FRESH and lines[k + 3] == "done"
    first_vllm = next(i for i, ln in enumerate(lines) if "vllm_arm $VT graph_r1" in ln)
    assert _last_install(lines) < lines.index(E4B_PROOF) < k < first_vllm == k + 4
    assert 'vllm_arm(){ local VT=$1 ARM=$2 B=$3; [ "${VOK[$VT]:-0}" = 1 ]' in RUN       # a refused build's arms skip


def test_no_release_proof_runs_isolated_or_behind_a_switch():
    """-I would drop PYTHONPATH and the script directory the arms import through, and the pycache prefix: never on a
    release_proof.py line. Neither proof sits behind an environment switch."""
    proof = [ln for ln in RUN.splitlines() if "release_proof.py" in ln and not ln.lstrip().startswith("#")]
    assert len(proof) == 5                         # the stage check, the two binding proofs and the two fast-checks
    assert not any(re.search(r"(^|\s)-I(\s|$)", ln) for ln in proof), proof
    assert not re.search(r"\$\{?P235_[A-Z_]*PROOF", RUN)


def _proof_fresh_code():
    m = re.search(r"""^proof_fresh\(\)\{ python -c '([^']+)' "\$1" "\$2"; \}$""", RUN, re.M)
    assert m, "proof_fresh is not defined as registered"
    return m.group(1)


@pytest.mark.parametrize("rows,n,ok", [
    ([{"dist": "a", "pycache_fresh": True}, {"dist": "b", "pycache_fresh": True}], 2, True),
    ([{"dist": "a", "pycache_fresh": True}, {"dist": "b", "pycache_fresh": False}], 2, False),
    ([{"dist": "a", "pycache_fresh": True}, {"dist": "b"}], 2, False),
    ([{"dist": "a", "pycache_fresh": "true"}], 1, False),
    ([{"dist": "a", "pycache_fresh": True}], 2, False),
    ([], 1, False),
    (None, 1, False),
])
def test_proof_fresh_reads_every_line_and_the_count(tmp_path, rows, n, ok):
    """The runner's own proof_fresh snippet, run: every line pycache_fresh true and one line per pin, else non-zero
    (a missing or empty proof file, a missing key, a false or a string fails)."""
    f = tmp_path / "proof.jsonl"
    if rows is not None:
        f.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    r = subprocess.run([sys.executable, "-c", _proof_fresh_code(), str(f), str(n)], capture_output=True, text=True)
    assert (r.returncode == 0) == ok, (r.returncode, r.stderr[-300:])


def test_the_pycache_prefix_is_fresh_exported_before_any_python_and_never_unset():
    lines = RUN.splitlines()
    k = lines.index('mkdir "$W/pycache-$NONCE" && export PYTHONPYCACHEPREFIX=$W/pycache-$NONCE '
                    'RELEASE_PROOF_FRESH_PYCACHE=$W/pycache-$NONCE || { say "refusing: no pycache prefix"; finish 78; }')
    m = re.fullmatch(r'mkdir "(\S+)" && export PYTHONPYCACHEPREFIX=(\S+) RELEASE_PROOF_FRESH_PYCACHE=(\S+) \|\| .*', lines[k])
    assert m and m.group(1) == m.group(2) == m.group(3)              # one value: the proof's pycache_fresh can be true
    assert lines[k - 1] == '[ -e "$W/pycache-$NONCE" ] && { say "refusing: $W/pycache-$NONCE exists"; finish 78; }'
    assert lines[k + 1] == 'echo "PYTHONPYCACHEPREFIX $PYTHONPYCACHEPREFIX" >> $W/versions.txt'
    first_python = next(i for i, ln in enumerate(lines) if re.search(r"\bpython3?\b|/pip\b", ln)
                        and not ln.lstrip().startswith("#"))
    assert k < first_python
    assert sum("PYTHONPYCACHEPREFIX" in ln for ln in lines) == 2
    assert not re.search(r"unset[^\n]*(PYTHONPYCACHEPREFIX|RELEASE_PROOF_FRESH_PYCACHE)|env -i|env -u|-u PYTHONPYCACHEPREFIX", RUN)
    assert sum("RELEASE_PROOF_FRESH_PYCACHE" in ln for ln in lines) == 1
    for ln in lines:                                       # every install leaves its .pyc out of the wheel's RECORD
        if re.search(r"pip install", ln):
            assert "--no-compile" in ln, ln
    assert "--exclude 'pycache-*'" in DRIVE


# ------------------------------------------------------------------------------------------------- runner --

def _as_p58(text):
    for new, old in TO_P58:
        text = text.replace(new, old)
    return text


def _body(text):
    """the script below its header comment (from `set -uo pipefail` on)"""
    return text[text.index("set -uo pipefail\n"):]


#: the line prefix of each line P235 adds to P58's runner body (P58's names): each named edit, and nothing else
RUN_EDITS = [
    "for v in P58_RUN_ID P58_DEADLINE_EPOCH P58_INSTANCE_ID E4B_SHA; do",
    "VLLM_PRIMARY=0.32.0; VLLM_SECONDARY=0.30.0   #",
    "# every python this runner starts writes and reads bytecode under a fresh, empty prefix",
    '[ -e "$W/pycache-$NONCE" ] && { say "refusing: $W/pycache-$NONCE exists"; finish 78; }',
    'mkdir "$W/pycache-$NONCE" && export PYTHONPYCACHEPREFIX=$W/pycache-$NONCE RELEASE_PROOF_FRESH_PYCACHE=$W/pycache-$NONCE ',
    'echo "PYTHONPYCACHEPREFIX $PYTHONPYCACHEPREFIX" >> $W/versions.txt',
    "for f in step_decomp.py k8_bake.py calib.json hook/usercustomize.py p37_vllm.py p58_reduce.py provenance.py "
    "release_proof.py staged.sha256; do",
    "# ---- install (image python): experts4bit-qlora 0.58.0 and grouped-nf4-gemm 0.51.0 as PyPI wheels",
    "# pins, --no-compile (install-time .pyc would land under the prefix, outside the wheel's RECORD)",
    "# unconditionally, before anything is timed (a version string is not a release)",
    'say "install experts4bit-qlora==0.58.0 + grouped-nf4-gemm==0.51.0 from PyPI',
    "perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --no-compile --index-url https://pypi.org/simple "
    '--prefer-binary --only-binary experts4bit-qlora "experts4bit-qlora==0.58.0" \\',
    '"transformers==5.17.0" "bitsandbytes==0.50.1"',
    "perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --no-compile --index-url https://pypi.org/simple "
    '--only-binary grouped-nf4-gemm --force-reinstall --no-deps "grouped-nf4-gemm==0.51.0"',
    E4B_CHECK,
    "cat logs/provenance_e4b.log",
    "import experts4bit_qlora",
    'assert experts4bit_qlora.__version__ == "0.58.0"',
    'assert md.version("grouped-nf4-gemm") == "0.51.0"',
    'assert hasattr(int4_b32, "gemv_counter_len")',
    'open("/root/p58/versions.txt", "a").write(f"e4b {e.__version__} (PyPI; provenance.json) harness@',
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee forensics.txt",
    "# ---- vLLM venvs: PRIMARY 0.32.0, SECONDARY 0.30.0 (the constants above)",
    "# here and its release proof right before its arms.",
    "# Each install or provenance failure is recorded",
    "# e4b arms); that build's rows read SKIPPED",
    "python -m venv $W/venv-$TAG && perl -e 'alarm 2700; exec @ARGV' $W/venv-$TAG/bin/pip install -q --no-input "
    '--no-compile --no-cache-dir --index-url https://pypi.org/simple --only-binary vllm "vllm==$VER"',
    VLLM_CHECK.strip(),
    '|| { cat logs/provenance_vllm_$TAG.log; say "PROVENANCE FAIL (vllm $TAG)',
    '{ echo "P58 arm=e4b_b${B}_$TAG gnf4=0.51.0 at=',
    "# ---- release proofs: the installed bytes ARE the published wheels', read as the arms import them",
    "# PYTHONPATH=$W/hook, the arms' python, no -I), after every install and right before that build's first arm;",
    E4B_PROOF,
    'for VT in primary secondary; do [ "${VOK[$VT]:-0}" = 1 ] || continue',
    VLLM_PROOF.strip(),
    '|| { cat logs/release_proof_vllm_$VT.log; say "RELEASE PROOF FAIL (vllm $VT)',
    "done",
    '# ---- the pre-install fit check (bench/common/README.md, "Guard fit"): before any install and any staged script.',
    "# term is conservative against P58's record: its whole run, boot and 18 arms included, took 42 min from the rental, and it",
    "# kept no per-phase times, so each term is P58's PREREG estimate with margin, or stated arithmetic",
    'INSTALL_TO_GATE_S=600 ',
    'NEED_VLLM_VENVS=1500 ',
    'NEED_FETCH=1960 ',
    'NEED_BAKE=500 ',
    'NEED_X=900 ',
    'RESERVE=600 ',
    'FIT_HEADROOM_PCT=125 ',
    'FIT=$((INSTALL_TO_GATE_S + NEED_VLLM_VENVS',
    '# rc 41: a refusal before the install, not a strike',
    '[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s',
    'echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s"',
    "# file whose lines are not all pycache_fresh true (or not one per pin) refuses like the proof itself. These are the",
    "# BINDING proofs, the only ones ruled; the fast-checks above only stop a wrong wheel early.",
    "# ---- release-proof FAST-CHECK (not ruled: only the binding proof, the last step before the first e4b arm, is): the",
    "# same staged release_proof.py in the arms' env, right after the installs, so a wrong wheel refuses (rc 9) before the",
    "# tripwire and the ~80 GB fetch, not after them. Its file travels with the receipts.",
    FAST_E4B,
    FAST_E4B_FRESH,
    "# release-proof FAST-CHECK in this venv (not ruled; the binding proof runs right before this build's arms)",
    FAST_VLLM.strip(),
    FAST_VLLM_FAIL.strip(),
    FAST_VLLM_FRESH.strip(),
    "proof_fresh(){ python -c 'import json, sys;",
    E4B_FRESH,
    VLLM_FRESH.strip(),
]


def test_the_runner_is_p58s_by_named_edits():
    """With the lane's names mapped back, every line P235 adds to P58's runner is one of the named edits and every edit
    is there: the arms, their order, the alarms, the STOP rules, the engagement greps and the reducer call are P58's."""
    diff = list(difflib.unified_diff(_body((P58 / "p58_run.sh").read_text(encoding="utf-8")).splitlines(),
                                     _body(_as_p58(RUN)).splitlines(), lineterm="", n=0))
    added = [ln[1:].strip() for ln in diff if ln.startswith("+") and not ln.startswith("+++")]
    removed = [ln[1:] for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    hits = {ln: [n for n in RUN_EDITS if ln.startswith(n)] for ln in added}
    assert all(len(h) == 1 for h in hits.values()), {ln: h for ln, h in hits.items() if len(h) != 1}
    assert sorted(n for h in hits.values() for n in h) == sorted(RUN_EDITS)
    assert len(removed) == 17, removed
    for gone in ("driver_version,uuid", "transformers==5.16.1", "git+https://github.com/pjordanandrsn/experts4bit-qlora.git",
                 "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git", "pypi.org/pypi/vllm/json", ":-0.29.0}"):
        assert any(gone in ln for ln in removed), gone


GUARD_S = 3 * 3600              # the registered guard, 3.0 h
BOOT_S = 300                    # launch to the check (bench/common/README.md, "Guard fit"; tests/test_p155.py)
SSH_READY_S, STAGING_MARGIN_S = 420, 120   # Vast's ssh wait and the staging margin (README rule 5)


def _term(name):
    m = re.search(rf"^{name}=(\d+) ", RUN, re.M)
    assert m, name
    return int(m.group(1))


def test_the_runner_checks_its_guard_fit_before_the_first_install():
    """The cross-lane rule's own check finds nothing wrong, and the refusal is rc 41 with a REFUSAL file and the
    GUARD_FIT line, unused by any other refusal."""
    assert fit_problems(RUN) == []
    lines = RUN.splitlines()
    k = lines.index('[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), '
                    '${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }')
    assert lines[k + 1] == 'echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt'
    assert RUN.count("finish 41") == 1
    first_install = next(i for i, ln in enumerate(lines) if re.search(r"pip install|python -m venv", ln)
                         and not ln.lstrip().startswith("#"))
    assert k < first_install and k > lines.index(": > summary.txt; echo \"$P235_INSTANCE_ID\" > INSTANCE_ID")


def test_the_registered_guard_holds_the_boot_and_the_fit_with_headroom():
    """The static side (tests/test_p155.py's form): (boot 300 + FIT) x 1.25 <= the 3.0 h guard, and the guard keeps
    the boot slack (ssh wait + staging) over the check's need. The terms are read from the runner."""
    assert "FIT=$((INSTALL_TO_GATE_S + NEED_VLLM_VENVS + NEED_FETCH + NEED_BAKE + NEED_X + RESERVE))" in RUN
    terms = {n: _term(n) for n in ("INSTALL_TO_GATE_S", "NEED_VLLM_VENVS", "NEED_FETCH", "NEED_BAKE", "NEED_X", "RESERVE")}
    assert terms == {"INSTALL_TO_GATE_S": 600, "NEED_VLLM_VENVS": 1500, "NEED_FETCH": 1960, "NEED_BAKE": 500, "NEED_X": 900,
                     "RESERVE": 600}
    assert terms["NEED_X"] == max(int(x) for x in re.findall(r"^can_run (\d+) ", RUN, re.M))   # the largest gate
    assert "$((now + need + 600))" in RUN and terms["RESERVE"] == 600               # can_run keeps 600 s after a step
    fit = sum(terms.values())
    pct = _term("FIT_HEADROOM_PCT")
    assert pct >= 125 and fit == 6060 and fit * pct // 100 == 7575
    assert (BOOT_S + fit) * 1.25 == 7950 <= GUARD_S
    assert GUARD_S - fit * pct // 100 >= SSH_READY_S + STAGING_MARGIN_S, "a slow-ssh host clears the fit check"
    assert 78.03e9 / 40e6 <= terms["NEED_FETCH"]                                 # 78.03 GB at the stated 40 MB/s
    for s in ("**FIT = 6,060 s, x 1.25 = 7,575 s.**", "(300 + 6,060) x 1.25 = 7,950 s <= the 3.0 h guard (10,800 s)",
              "the two checkpoints, 78.03 GB, at 40 MB/s: **1,960 s**", "A fit refusal (rc 41) comes before the install and costs cents",
              "two consecutive fit refusals count as one strike", "took 42 min (2,520 s) from the rental", "guard 3.0 h"):
        assert s in FLAT, s


def test_the_pins_the_prereg_states_are_the_runners():
    assert 'MID=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;' in RUN
    assert "GPTQ_MID=Qwen/Qwen3-30B-A3B-GPTQ-Int4; GPTQ_REV=9b534e4318b7ebc3c961a839f13eb18b1833f441" in RUN
    assert '"transformers==5.17.0" "bitsandbytes==0.50.1"' in RUN
    assert "GNF4_SHA" not in RUN and "E4B_SHA" in RUN.split("PYT", 1)[0]   # the launch commit is recorded, not installed


def _tripwire():
    head = "PYTHONPATH=$W/hook python - <<'PYT' || { say \"TRIPWIRE FAIL (e4b)\"; finish 9; }\n"
    assert RUN.count(head) == 1
    body = RUN[RUN.index(head) + len(head):]
    return body[:body.index("\nPYT\n")]


def test_the_tripwire_asserts_the_release_versions_first():
    """The version asserts sit in the e4b tripwire, whose failure finishes 9, ahead of every API assert."""
    tw = _tripwire()
    ast.parse(tw)
    lines = tw.splitlines()
    i = lines.index("import experts4bit_qlora")
    assert lines[i + 1].startswith('assert experts4bit_qlora.__version__ == "0.58.0", ')
    assert lines[i + 2].startswith('assert md.version("grouped-nf4-gemm") == "0.51.0", ')
    assert i < next(k for k, ln in enumerate(lines) if "_smallm_kernels" in ln)
    assert lines[0].startswith("import site, os, inspect, importlib.metadata as md")


@pytest.mark.parametrize("e4b,gnf4,ok", [("0.58.0", "0.51.0", True), ("0.57.0", "0.51.0", False),
                                         ("0.58.1", "0.51.0", False), ("0.58.0", "0.50.0", False),
                                         ("0.58.0", "0.51.1", False)])
def test_the_tripwire_refuses_any_other_release(monkeypatch, e4b, gnf4, ok):
    lines = _tripwire().splitlines()
    i = lines.index("import experts4bit_qlora")
    pin = "\n".join(lines[i:i + 3])
    monkeypatch.setitem(sys.modules, "experts4bit_qlora", types.SimpleNamespace(__version__=e4b))
    md = types.SimpleNamespace(version=lambda name: {"grouped-nf4-gemm": gnf4}[name])
    if ok:
        exec(pin, {"md": md})
    else:
        with pytest.raises(AssertionError, match="is not the 0.5"):
            exec(pin, {"md": md})


def test_the_runner_never_collects_the_gpu_uuid():
    """The runner never asks for the card's GPU UUID: no nvidia-smi query names uuid, and no nvidia-smi -L, -q or
    --list-gpus (which print it) runs. P58's query line is refused."""
    def bad(src):
        return [ln for ln in src.splitlines() if not ln.lstrip().startswith("#") and "nvidia-smi" in ln
                and ("uuid" in ln.lower() or re.search(r"nvidia-smi\s+(-L|-q|--list-gpus)(\s|$)", ln))]
    assert sum("nvidia-smi --query-gpu=" in ln for ln in RUN.splitlines()) >= 2   # the check has its input
    assert not bad(RUN), bad(RUN)
    assert bad(RUN.replace("driver_version --format", "driver_version,uuid --format"))
    assert bad((P58 / "p58_run.sh").read_text(encoding="utf-8"))
    assert bad("nvidia-smi -L") and bad("nvidia-smi -q | tee f") and bad("nvidia-smi --list-gpus")


def _docstring_split(text):
    a = text.index('"""') + 3
    b = text.index('"""', a)
    return text[a:b], text[b:]


def test_the_reducer_is_p58s_with_its_labels_renamed():
    """Below the docstring the reducer is P58's byte for byte but the title, and on P58's committed receipts it prints
    P58's table but for the title line."""
    mine_doc, mine = _docstring_split((LANE / "p235_reduce.py").read_text(encoding="utf-8"))
    p58_doc, p58 = _docstring_split((P58 / "p58_reduce.py").read_text(encoding="utf-8"))
    assert mine.count('"# P235 -- ') == 1 and mine.replace('"# P235 -- ', '"# P58 -- ') == p58
    changed = [k for k, (x, y) in enumerate(zip(mine_doc.splitlines(), p58_doc.splitlines())) if x != y]
    assert len(mine_doc.splitlines()) == len(p58_doc.splitlines()) and changed == [0, 1, 3, 7], changed
    rec = P58 / "receipts"
    outs = [subprocess.run([sys.executable, str(s), str(rec)], capture_output=True, text=True, timeout=120)
            for s in (LANE / "p235_reduce.py", P58 / "p58_reduce.py")]
    assert all(o.returncode == 0 for o in outs), [o.stderr for o in outs]
    (m, *m_rest), (p, *p_rest) = (o.stdout.splitlines() for o in outs)
    assert m.startswith("# P235 -- ") and p.startswith("# P58 -- ") and m_rest == p_rest and len(m_rest) > 40


# ----------------------------------------------------------------------------------------------------- drive --

def test_the_drive_follows_pod_launchs_contract():
    """pod-launch runs a reading's drive without E4B_SHA: the drive derives the launch commit (the harness's) from the
    clean HEAD, refuses an override, takes no grouped-nf4-gemm commit (the box installs PyPI's release), and fetches the
    box tree into $RUN_DIR/p235/, refusing a non-empty target."""
    assert 'if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi' in DRIVE
    assert 'E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)' in DRIVE and "is dirty" in DRIVE
    assert 'OUT="$RUN_DIR/p235"' in DRIVE and '"$RUN_DIR/"' not in DRIVE
    assert 'mkdir -p "$OUT" && [ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: $OUT is not an empty directory"; exit 22; }' in DRIVE
    assert ('case "$RUN_ID" in\n  p235-5090-1|p235-5090-2|p235-5090-3|p235-5090-r1|p235-5090-r2|p235-5090-r3|p235-5090-r4|'
            'p235-5090-r5|p235-5090-r6) :;;\n  *) say "refusing unregistered run ID') in DRIVE
    assert "GNF4" not in DRIVE and 'E4B_SHA=$E4B_SHA"' in DRIVE
    for name, path in SOURCES.items():
        assert path.relative_to(REPO).as_posix() in DRIVE, name
    assert "bench/p235/staged.sha256" in DRIVE and 'cp "$REPO/$f" "$STAGE/hook/"' in DRIVE
    assert "--include 'work_qwen3/' --include 'work_qwen3/bake.json' --exclude 'work_qwen3/*'" in DRIVE
    for word in ("vast.py", "destroy", "/api/", "rent-watch"):
        assert word not in DRIVE.split("set -uo pipefail", 1)[1], word


def _fetches_without_no_links(drive):
    """(rsync lines, the ones without --no-links): the drive's live rsync invocations, comments skipped."""
    lines = [ln for ln in drive.splitlines() if re.search(r"\brsync\b", ln) and not ln.lstrip().startswith("#")]
    return lines, [ln for ln in lines if not re.search(r"\brsync\s(?:[^|;&]*\s)?--no-links\b", ln)]


def test_the_fetch_never_imports_a_link():
    lines, bad = _fetches_without_no_links(DRIVE)
    assert len(lines) == 1 and not bad, (lines, bad)
    assert _fetches_without_no_links(DRIVE.replace(" --no-links", ""))[1]          # the old line is refused
    assert not _fetches_without_no_links("echo fetch\n")[0]                          # no rsync -> the count above fails


def _dry_env(tmp_path, run_id, **extra):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P235_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0", **extra}


def _drive(tmp_path, run_id, **extra):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    return subprocess.run(["bash", str(LANE / "p235_drive.sh")], capture_output=True, text=True,
                          env=_dry_env(tmp_path, run_id, **extra), timeout=120)


@pytest.mark.parametrize("run_id", ["p235-5090-4", "p58-5090-1", "p235-shake-1", "p235-5090-r7", "p235-5090-r0"])
def test_the_driver_refuses_another_run_id(tmp_path, run_id):
    out = _drive(tmp_path, run_id)
    assert out.returncode == 78 and "is not p235-5090-N or p235-5090-rN" in out.stdout + out.stderr


def test_the_driver_refuses_an_e4b_sha_override(tmp_path):
    out = _drive(tmp_path, "p235-5090-1", E4B_SHA="a" * 40)
    assert out.returncode == 78 and "E4B_SHA override" in out.stdout + out.stderr


def _clean_or_skip(dirty):
    """The driver refuses a dirty tree by design, so a dirty local checkout skips the dry run, named. Under CI the
    checkout is fresh: a dirty one is broken and fails, never a silent skip that drops the dry-run asserts."""
    if dirty.strip():
        why = f"the checkout is dirty ({dirty.splitlines()[0][:80]}): the driver refuses a dirty tree by design"
        if os.environ.get("CI"):
            pytest.fail(why + " (CI must run on a clean checkout)")
        pytest.skip(why)


@pytest.mark.parametrize("ci,outcome", [("1", pytest.fail.Exception), ("", pytest.skip.Exception)],
                         ids=["ci-fails", "local-skips"])
def test_a_dirty_checkout_fails_under_ci_and_skips_locally(monkeypatch, ci, outcome):
    if ci:
        monkeypatch.setenv("CI", ci)
    else:
        monkeypatch.delenv("CI", raising=False)
    with expect_outcome(outcome):
        _clean_or_skip(" M tests/test_p235.py\n")
    _clean_or_skip("")                                                   # a clean tree neither fails nor skips


@pytest.mark.parametrize("run_id", ["p235-5090-1", "p235-5090-3", "p235-5090-r6"])
def test_the_drivers_dry_run_names_the_launch(tmp_path, run_id):
    """The driver dry-runs on a clean checkout: it stages the eight pieces and the pin file, checks them there, and
    starts the runner with the harness's commit and no grouped-nf4-gemm commit (a GNF4_SHA in the environment is not
    passed on)."""
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    _clean_or_skip(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                  text=True).stdout)
    out = _drive(tmp_path, run_id, GNF4_SHA="b" * 40)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"DRYRUN stage {len(SOURCES) + 1} files -> root@h:/root/p235" in out.stdout and "bash p235_run.sh" in out.stdout
    assert f"P235_RUN_ID={run_id} " in out.stdout and "GNF4" not in out.stdout
    assert re.search(r" E4B_SHA=[0-9a-f]{40} ", out.stdout)
    assert f"fetch -> {tmp_path}/p235" in out.stdout


# ------------------------------------------------------------------------------------------ prereg, files --

def test_the_prereg_states_the_pins_bars_sequencing_and_limits():
    for s in ('`experts4bit_qlora.__version__ == "0.58.0"`', '`md.version("grouped-nf4-gemm") == "0.51.0"`',
              "A version string is not a release.", "Every git install is refused",
              "| transformers | 5.16.1 | **5.17.0** |", "| bitsandbytes | 0.50.1 | 0.50.1 (e4b's floor is `>=0.43`) |",
              "A newer vLLM is a new reading", "the run ends with rc 9 before the tripwire and before any arm",
              "that build's rows are refused", "`p37_vllm.py` has run against vLLM 0.28 to 0.30, not 0.32.0",
              "`Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`",
              "`Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e4318b7ebc3c961a839f13eb18b1833f441`", "`a8e6ea1d…` (B=1) and `f67e7e4d…` (B=16)",
              "must be inside **1.03×**", "only with an equal `prompts_sha` on both sides", "**VOID arms never enter a ratio**",
              "**1.087** at B=1 and **1.396** at B=16", "## What this lane cannot say", "TTFT and prefill",
              "vLLM reserves 0.90 of VRAM by policy", "`GNF4_GEMV_FUSED_REDUCE=1`", "an fp8-KV vLLM arm",
              "is not the server's scheduler", "**The engine advantage is understated, not inflated.**", "## Sequencing",
              "both releases (e4b 0.58.0, grouped-nf4-gemm 0.51.0) are published on PyPI", "The maintainer gives an Approved.",
              "The maintainer re-derives the reading from the receipts.", "A results PR follows.",
              "one RTX 5090 at ≤ $1.05/h, guard 3.0 h, download ≤ $0.02/GB × 90 GB",
              "$3.15 + $1.80 = **$4.95**, inside a **$6** lane ceiling", "`p235-5090-1..-3`", "`p235-5090-r1..-r6`",
              "**No shakedown is registered.**", "**not collected**",
              "**The release proof: the installed bytes.**", "is loggetta `bench/release_proof.py` at `f09e5676` (#126's merge)", "staged byte-identical",
              "staged byte-identical and pinned in `staged.sha256`", "`release_proof.jsonl` and `release_proof_vllm_<build>.jsonl`",
              '"RECORD equal"', "cwd `$W`, `PYTHONPATH=$W/hook`, the arms' python, no `-I`",
              "`PYTHONPYCACHEPREFIX=$W/pycache-<nonce>`", "Every install uses `--no-compile`.",
              "so the proof picks by arch (`wheel_pick` \"arch\")", "Neither wheel has `.data/` entries.",
              "If the helper refuses a build anyway, that build's rows are refused",
              "vLLM 0.32.0 and 0.30.0 installed from PyPI `--no-deps --no-compile` and proved as the runner runs it",
              "every top-level name the wheel installs imports from the proven install", "`pycache_fresh`",
              'not all `"pycache_fresh": true`, or not one per pin, refuses like the proof itself',
              "`RELEASE_PROOF_FRESH_PYCACHE` is exported on the same line with the same value",
              "**Release-proof fast-check** (each build; not ruled)", "before the tripwire and the model fetch",
              "its binding proof then does not run", "A wrong wheel at install now costs minutes, not the ~86 GB setup.",
              "only the binding proof, the last step before each build's first arm, is ruled"):
        assert s in FLAT, s
    assert "GNF4_SHA" not in PREREG and "placeholder" not in PREREG.lower()


def test_the_new_files_are_lf_only_and_the_shell_scripts_parse():
    for f in NEW_FILES:
        assert b"\r" not in f.read_bytes(), f
    if shutil.which("bash") is None:
        return
    for f in ("p235_run.sh", "p235_drive.sh"):
        out = subprocess.run(["bash", "-n", str(LANE / f)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
