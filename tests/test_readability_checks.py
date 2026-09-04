"""The agent-readability checks -- scripts/check_system_manifest.py,
scripts/check_dependency_floor.py and scripts/check_change_impact.py -- on
synthetic trees: what each must catch and what it must excuse.

Standard library + pytest + git; no torch, no network. The change-impact
check is exercised end to end on a throwaway git repository, because its
whole contract is "these files move together in one diff".
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_change_impact as cci  # noqa: E402
import check_dependency_floor as cdf  # noqa: E402
import check_system_manifest as csm  # noqa: E402

OTHER = frozenset({"torch", "bitsandbytes", "transformers"})


# ------------------------------------------------------------ pure functions --

def test_version_ranges():
    assert csm.version_in_range("0.35.0", ">=0.35.0")
    assert csm.version_in_range("0.35.2", ">=0.35.0")
    assert not csm.version_in_range("0.34.9", ">=0.35.0")
    assert csm.version_in_range("0.34.3", "0.34.x")
    assert csm.version_in_range("0.34.0", "0.34.*")
    assert not csm.version_in_range("0.35.0", "0.34.x")
    assert csm.version_in_range("0.30.0", ">=0.28.0,<0.31")
    assert not csm.version_in_range("0.31.0", ">=0.28.0,<0.31")
    assert csm.version_in_range("0.34.5", "~=0.34.1")
    assert not csm.version_in_range("0.35.0", "~=0.34.1")
    assert csm.version_in_range("1.0.0", "1.0.0")
    assert not csm.version_in_range("1.0.1", "1.0.0")
    assert csm.version_in_range("1.0", "1.0.0")
    with pytest.raises(ValueError):                 # unreadable ranges never pass
        csm.version_in_range("1.0.0", ">=1.0.0rc1")
    assert csm.floor_of(">=0.30.0") == "0.30.0"
    assert csm.floor_of(">=0.30.0,<1") == "0.30.0"
    assert csm.floor_of("0.34.x") is None
    assert csm.floor_of(">=1,>=2") is None


def test_fast_requirement_reads_the_extra():
    py = {"optional-dependencies": {"fast": ["torch>=2", "Grouped_NF4.gemm[x]>=0.30.0; python_version>'3'"]}}
    assert csm.fast_requirement(py) == ("Grouped_NF4.gemm[x]>=0.30.0; python_version>'3'", ">=0.30.0")
    assert csm.fast_requirement({"optional-dependencies": {"fast": ["torch>=2"]}}) is None
    assert csm.fast_requirement({}) is None


@pytest.mark.parametrize("line, versions", [
    ("grouped-nf4-gemm>=0.30.0", ["0.30.0"]),
    ("grouped-nf4-gemm >= 0.30.0 and Triton", ["0.30.0"]),
    ("grouped-nf4-gemm ≥ 0.28.0 and", ["0.28.0"]),
    ("requires grouped-nf4-gemm 0.28.0.", ["0.28.0"]),
    ("the `[fast]` extra pins >=0.28.0.", ["0.28.0"]),
    ("`grouped-nf4-gemm`>=0.28.0", ["0.28.0"]),
    ("--requires 'grouped-nf4-gemm>=0.30.0; extra == \"fast\"'", ["0.30.0"]),
    ("torch>=2.2.0, bitsandbytes >= 0.50.0 (no kernel package here)", []),
    ("bitsandbytes ≥ 0.50.0 can run packed cells; [fast] is the seam", []),
    ("python>=3.9.0 is what gnf4 needs", []),
    ("with grouped-nf4-gemm 0.26.0 the scales", []),
])
def test_statements(line, versions):
    assert cdf.statements(line, OTHER) == versions


def test_historical_markers_are_whole_words():
    assert cdf.historical_marker("the floor was grouped-nf4-gemm>=0.28.0") == "was"
    assert cdf.historical_marker("raised from >=0.28.0") == "raised from"
    assert cdf.historical_marker("pinned at v0.28.0") == "vX.Y.Z tag pin"
    assert cdf.historical_marker("sm_80 or newer") is None
    assert cdf.historical_marker("wasabi >= 0.28.0") is None


def test_all_symbols():
    assert cci.all_symbols('__all__ = ["a", "b"]\n__all__ += ["c"]') == {"a", "b", "c"}
    assert cci.all_symbols("x = 1") is None
    assert cci.all_symbols("def (") is None
    assert cci.all_symbols(None) is None


# ------------------------------------------------------------- the real tree --

def test_real_manifest_agrees_with_this_tree():
    r = subprocess.run([sys.executable, str(SCRIPTS / "check_system_manifest.py"), "--root", str(ROOT)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------- a synthetic tree --

def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _append(root: Path, rel: str, text: str = "\nchanged\n") -> None:
    p = root / rel
    p.write_text(p.read_text(encoding="utf-8") + text, encoding="utf-8")


def _manifest(floor: str) -> dict:
    return {"packages": {"kernels": {"import_names": ["nf4_grouped", "gptq_pack"]}},
            "compatibility": [{"consumer": "experts4bit-qlora", "consumer_versions": ">=0.35.0",
                               "kernel": "grouped-nf4-gemm", "floor": f">={floor}", "extra": "fast"}]}


PYPROJECT = '[project]\nname = "experts4bit-qlora"\nversion = "0.35.0"\n' \
            '[project.optional-dependencies]\nfast = ["grouped-nf4-gemm>={floor}"]\n'
CLAIMS = {"status_vocabulary": {"measured": "run, receipt public", "retired": "withdrawn"},
          "claims": [{"id": "x.one", "status": "measured", "value": 1.0}]}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "r"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _write(root, "pyproject.toml", PYPROJECT.format(floor="0.28.0"))
    _write(root, "docs/claims.json", json.dumps(CLAIMS))
    _write(root, "docs/STATUS.md", "# status\n")
    _write(root, "docs/capabilities.json", "{}\n")
    _write(root, "docs/SOLUTIONS.md", "grouped-nf4-gemm ≥ 0.28.0 and Triton\n")
    _write(root, "docs/solutions/a.md", "- Environment: engines need grouped-nf4-gemm>=0.28.0 (Triton).\n")
    _write(root, "docs/solutions/b.md", "- The floor was raised to grouped-nf4-gemm>=0.28.0 in 0.34.0; see pyproject.\n")
    _write(root, "docs/system-manifest.json", json.dumps(_manifest("0.28.0")))
    _write(root, ".github/workflows/ci.yml", "run: check --requires 'grouped-nf4-gemm>=0.28.0'\n")
    _write(root, "CHANGELOG.md", "# changelog\n")
    _write(root, "experts4bit_qlora/__init__.py", '__all__ = ["a", "b"]\n')
    _write(root, "experts4bit_qlora/fast.py", "import nf4_grouped\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _impact(root: Path, *extra: str, base: str = "HEAD") -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS / "check_change_impact.py"), "--root", str(root),
                           "--base", base, *extra], capture_output=True, text=True)


def test_clean_diff_and_no_base(repo: Path):
    r = _impact(repo)
    assert r.returncode == 0 and "no change-impact class" in r.stdout, r.stdout
    r = _impact(repo, base="")
    assert r.returncode == 0 and r.stdout.startswith("SKIP"), r.stdout
    r = _impact(repo, base="no-such-ref")
    assert r.returncode == 2, r.stdout


def test_floor_change_needs_its_companions(repo: Path):
    _write(repo, "pyproject.toml", PYPROJECT.format(floor="0.30.0"))
    r = _impact(repo)
    assert r.returncode == 1, r.stdout
    assert "CLASS dependency-floor" in r.stdout
    for companion in ("docs/system-manifest.json", "docs/capabilities.json", ".github/workflows/ci.yml",
                      "docs/SOLUTIONS.md", "docs/solutions/a.md"):
        assert f"MISSING: {companion}" in r.stdout, r.stdout
    assert "docs/solutions/b.md" not in r.stdout          # historical line: not a companion
    _write(repo, "docs/system-manifest.json", json.dumps(_manifest("0.30.0")))
    for companion in ("docs/capabilities.json", ".github/workflows/ci.yml", "docs/SOLUTIONS.md", "docs/solutions/a.md"):
        _append(repo, companion)
    r = _impact(repo)
    assert r.returncode == 0 and "OK: dependency-floor" in r.stdout, r.stdout


def test_dependency_floor_names_the_drifting_line(repo: Path):
    def run() -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SCRIPTS / "check_dependency_floor.py"), "--root", str(repo)],
                              capture_output=True, text=True)
    r = run()
    assert r.returncode == 0, r.stdout                       # documents agree with pyproject
    assert "0 historical line(s)" in r.stdout
    _write(repo, "pyproject.toml", PYPROJECT.format(floor="0.30.0"))
    r = run()
    assert r.returncode == 1, r.stdout
    assert "historical ('raised to'): docs/solutions/b.md:1" in r.stdout   # excused and listed, never failed
    assert "FAIL: docs/SOLUTIONS.md:1: states grouped-nf4-gemm >= 0.28.0" in r.stdout
    assert "FAIL: docs/solutions/a.md:1:" in r.stdout
    assert "FAIL: .github/workflows/ci.yml:1:" in r.stdout
    assert "FAIL: docs/system-manifest.json: record '>=0.35.0' floor '>=0.28.0' != 0.30.0" in r.stdout
    _write(repo, "docs/system-manifest.json", json.dumps(_manifest("0.30.0")))
    r = run()
    assert "OK: docs/system-manifest.json: record '>=0.35.0' floor >=0.30.0" in r.stdout


def test_claim_change_needs_status(repo: Path):
    claims = json.loads(json.dumps(CLAIMS))
    claims["claims"][0]["status"] = "retired"
    _write(repo, "docs/claims.json", json.dumps(claims))
    r = _impact(repo)
    assert r.returncode == 1 and "CLASS measured-result" in r.stdout and "MISSING: docs/STATUS.md" in r.stdout
    assert "status 'measured' -> 'retired'" in r.stdout
    r = _impact(repo, "--allow-claims-only")
    assert r.returncode == 0 and "WARN: measured-result" in r.stdout
    _append(repo, "docs/STATUS.md")
    r = _impact(repo)
    assert r.returncode == 0 and "OK: measured-result" in r.stdout


def test_public_api_change_warns_then_strict_fails(repo: Path):
    _write(repo, "experts4bit_qlora/__init__.py", '__all__ = ["a", "b", "c"]\n')
    r = _impact(repo)
    assert r.returncode == 0 and "WARN: public-api-change" in r.stdout and "+['c']" in r.stdout, r.stdout
    r = _impact(repo, "--strict")
    assert r.returncode == 1 and "FAIL: public-api-change" in r.stdout
    _append(repo, "docs/capabilities.json")
    _append(repo, "CHANGELOG.md")
    r = _impact(repo, "--strict")
    assert r.returncode == 0 and "OK: public-api-change" in r.stdout


def test_new_kernel_import_warns(repo: Path):
    _write(repo, "experts4bit_qlora/new_engine.py", "import nf4_grouped\nfrom gptq_pack import pack\n")
    r = _impact(repo)
    assert r.returncode == 0, r.stdout
    assert "CLASS new-kernel-capability" in r.stdout and "['gptq_pack']" in r.stdout, r.stdout
