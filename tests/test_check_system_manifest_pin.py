"""scripts/check_system_manifest.py: the CI kernel pin and the kernel-pinning extras.

The audit found CI installing the kernel from the v0.30.0 commit while the
kernel was at 0.30.1, and the ``test`` extra's floor two releases behind
``fast``'s -- neither compared to anything. These pin the parsing that the
check keys on; the network lookup itself is exercised by the check in CI.
"""
import importlib.util
import pathlib

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_system_manifest.py"
_spec = importlib.util.spec_from_file_location("check_system_manifest", _SCRIPT)
csm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csm)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_ci_kernel_pin_is_parsed_from_the_workflow():
    text = ('          pip install torch --index-url https://download.pytorch.org/whl/cpu\n'
            '          pip install "grouped-nf4-gemm @ git+https://github.com/o/grouped-nf4-gemm.git@d58b39fbb51f15721a32953a4dbd3808a5d1f3a6"\n'
            '          pip install -e ".[test]"\n')
    assert csm.ci_kernel_pin(text) == ("o/grouped-nf4-gemm", "d58b39fbb51f15721a32953a4dbd3808a5d1f3a6")
    assert csm.ci_kernel_pin("pip install -e .[test]\n") is None
    # a branch name is not a pin
    assert csm.ci_kernel_pin('pip install "grouped-nf4-gemm @ git+https://github.com/o/grouped-nf4-gemm.git@main"') is None


def test_the_repository_workflow_pins_a_full_sha():
    slug, sha = csm.ci_kernel_pin((ROOT / csm.CI_WORKFLOW).read_text(encoding="utf-8"))
    assert slug == "pjordanandrsn/grouped-nf4-gemm"
    assert len(sha) == 40


def test_release_tag_lines_are_parsed_and_annotated_tags_peeled(monkeypatch):
    out = ("94368fc7711f6eb40b24fbb1abd9b9d425efe8a5\trefs/tags/v0.30.0\n"
           "0de0195ed7557459050f3ca35a12051220ae3020\trefs/tags/v0.30.0^{}\n"
           "d58b39fbb51f15721a32953a4dbd3808a5d1f3a6\trefs/tags/v0.30.1\n"
           "abcdef0123456789abcdef0123456789abcdef01\trefs/heads/main\n")

    class P:
        returncode = 0
        stdout = out

    monkeypatch.setattr(csm.subprocess, "run", lambda *a, **k: P())
    tags = csm.release_tags("https://example.invalid/r.git")
    assert tags == {"0.30.0": "0de0195ed7557459050f3ca35a12051220ae3020", "0.30.1": "d58b39fbb51f15721a32953a4dbd3808a5d1f3a6"}


def test_release_tags_returns_none_without_network(monkeypatch):
    def boom(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(csm.subprocess, "run", boom)
    assert csm.release_tags("https://example.invalid/r.git") is None


def test_kernel_extras_lists_every_extra_that_pins_the_kernel():
    py = {"optional-dependencies": {"fast": ["grouped-nf4-gemm>=0.30.0"], "test": ["pytest>=7", "grouped-nf4-gemm>=0.30.0"],
                                    "serve": ["fastapi>=0.110"]}}
    assert csm.kernel_extras(py) == {"fast": "grouped-nf4-gemm>=0.30.0", "test": "grouped-nf4-gemm>=0.30.0"}


def test_the_repository_extras_are_at_or_above_the_fast_floor():
    py = csm.load_pyproject(ROOT)
    fast = csm.floor_of(csm.fast_requirement(py)[1])
    for extra, req in csm.kernel_extras(py).items():
        fl = csm.floor_of(csm.fast_requirement(py, csm.KERNEL_PACKAGE, extra)[1])
        assert fl is not None, (extra, req)
        assert csm.parse_version(fl) >= csm.parse_version(fast), f"extra {extra!r} floors {fl} below fast's {fast}"
