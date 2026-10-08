"""Hermetic Git and resealed-wheel controls; no release imports or GPU claims."""
import copy
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "bench/ra/ra_source_binding.py"
spec = importlib.util.spec_from_file_location("ra_source_binding_test", TOOL)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
spec = importlib.util.spec_from_file_location("ra_source_fixture", ROOT / "tests/test_ra_provenance.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def commit(repo):
    paths = git(repo, "ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    git(repo, "add", "--", *paths)
    git(repo, "commit", "--only", "-m", "Synthetic CPU source fixture\n\n"
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", *paths)
    return git(repo, "rev-parse", "HEAD")


def setup(tmp_path):
    releases = []
    for name in ["experts4bit-qlora", "grouped-nf4-gemm"]:
        repo = tmp_path / name
        repo.mkdir()
        git(repo, "init", "--quiet")
        git(repo, "config", "user.name", "RA CPU fixture")
        git(repo, "config", "user.email", "fixture@example.invalid")
        project = ('[build-system]\nrequires=["setuptools>=77"]\nbuild-backend="setuptools.build_meta"\n'
                   f'[project]\nname="{name}"\nversion="0.0.0"\nreadme="README.md"\n'
                   'license="MIT"\nlicense-files=["LICENSE"]\nrequires-python=">=3.10"\n')
        if name == "experts4bit-qlora":
            project += ('[tool.setuptools.packages.find]\ninclude=["experts4bit_qlora*"]\n'
                        '[tool.setuptools.package-data]\nexperts4bit_qlora=["py.typed","README-LAYOUT.md"]\n')
            mapping = {"experts4bit_qlora/__init__.py": "experts4bit_qlora/__init__.py",
                       "experts4bit_qlora/nested/engine.py": "experts4bit_qlora/nested/engine.py",
                       "experts4bit_qlora/py.typed": "experts4bit_qlora/py.typed",
                       "experts4bit_qlora/README-LAYOUT.md": "experts4bit_qlora/README-LAYOUT.md"}
            module = "experts4bit_qlora/__init__"
        else:
            project += ('[tool.setuptools]\npackage-dir={""="kernel",gnf4_native="gnf4_native"}\n'
                        'packages=["gnf4_native"]\npy-modules=["nf4_grouped"]\n'
                        '[tool.setuptools.package-data]\ngnf4_native=["*.c"]\n')
            mapping = {"nf4_grouped.py": "kernel/nf4_grouped.py", "gnf4_native/__init__.py": "gnf4_native/__init__.py",
                       "gnf4_native/native.c": "gnf4_native/native.c"}
            module = "nf4_grouped"
        files = {rel: b"# synthetic source; must never be imported\n" for rel in mapping}
        for rel, path in mapping.items():
            dest = repo / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(files[rel])
        (repo / "pyproject.toml").write_text(project)
        readme = "Synthetic source · UTF-8 body\n".encode()
        (repo / "README.md").write_bytes(readme)
        (repo / "LICENSE").write_bytes(b"Synthetic license\n")
        head = commit(repo)
        info = name.replace("-", "_") + "-0.0.0.dist-info"
        files.update({info + "/METADATA": (f"Metadata-Version: 2.4\nName: {name}\nVersion: 0.0.0\n"
                      "Requires-Python: >=3.10\nLicense-Expression: MIT\nLicense-File: LICENSE\n\n").encode() + readme,
                      info + "/licenses/LICENSE": b"Synthetic license\n",
                      info + "/WHEEL": b"Wheel-Version: 1.0\nGenerator: setuptools (synthetic)\n"
                      b"Root-Is-Purelib: true\nTag: py3-none-any\n",
                      info + "/top_level.txt": "\n".join(sorted({p.split('/')[0].removesuffix('.py')
                                                                for p in mapping})).encode() + b"\n"})
        pin = fixture.make_wheel(tmp_path, name, module, extra=files)
        releases.append({"repo": str(repo), "commit": head, "version": "0.0.0", "wheel": pin})
    return {"schema": 1, "releases": releases}


def invoke(tmp_path, manifest, *, flags=("-I", "-S", "-B")):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return subprocess.run([sys.executable, *flags, str(TOOL), "--manifest", str(path),
                           "--out", str(tmp_path / "result.json")], capture_output=True, text=True, timeout=30)


def reseal(tmp_path, row, mutate):
    with zipfile.ZipFile(row["wheel"]["path"]) as archive:
        files = {p: archive.read(p) for p in archive.namelist() if not p.endswith("/RECORD")}
    mutate(files)
    name = "experts4bit-qlora"
    row["wheel"] = fixture.make_wheel(tmp_path, name, "experts4bit_qlora/__init__", extra=files)


def test_complete_pair_binds_without_import_or_build(tmp_path):
    manifest = setup(tmp_path)
    result = invoke(tmp_path, manifest)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "result.json").read_text())
    assert receipt["proves_runtime_source_binding"] is True
    assert [len(p["runtime_files"]) for p in receipt["releases"]] == [4, 3]
    assert all(v is False for k, v in receipt.items() if k.startswith("proves_") and k != "proves_runtime_source_binding")


@pytest.mark.parametrize("mutation", ["payload", "extra", "missing", "typed", "readme", "license", "python",
                                     "python_duplicate", "top", "info_extra", "wheel_tag", "newline", "metadata_version"])
def test_resealed_archive_cannot_override_git_source(tmp_path, mutation):
    manifest = setup(tmp_path)
    row = manifest["releases"][0]
    info = "experts4bit_qlora-0.0.0.dist-info"

    def mutate(files):
        if mutation == "payload":
            files["experts4bit_qlora/nested/engine.py"] += b"# replaced\n"
        elif mutation == "extra":
            files["experts4bit_qlora/foreign.py"] = b"# foreign\n"
        elif mutation in ["missing", "typed"]:
            files.pop("experts4bit_qlora/" + ("nested/engine.py" if mutation == "missing" else "py.typed"))
        elif mutation == "readme":
            files[info + "/METADATA"] += b"changed\n"
        elif mutation == "license":
            files[info + "/licenses/LICENSE"] += b"changed\n"
        elif mutation == "python":
            files[info + "/METADATA"] = files[info + "/METADATA"].replace(b">=3.10", b">=3.11")
        elif mutation == "python_duplicate":
            files[info + "/METADATA"] = b"Requires-Python: >=3.10\n" + files[info + "/METADATA"]
        elif mutation == "top":
            files[info + "/top_level.txt"] += b"foreign\n"
        elif mutation == "info_extra":
            files[info + "/unknown"] = b"unreviewed\n"
        elif mutation == "wheel_tag":
            files[info + "/WHEEL"] = files[info + "/WHEEL"].replace(b"py3-none-any", b"cp311-none-any")
        elif mutation == "newline":
            files[info + "/METADATA"] = files[info + "/METADATA"].replace(b"\n", b"\r\n")
        elif mutation == "metadata_version":
            files[info + "/METADATA"] = files[info + "/METADATA"].replace(b"Version: 0.0.0", b"Version: 0.0.1")

    reseal(tmp_path, row, mutate)
    result = invoke(tmp_path, manifest)
    assert result.returncode != 0, mutation
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("mutation", ["body", "new_module", "symlink", "layout", "dynamic", "source_version",
                                     "short_pin", "unknown_pin", "archive_hash", "pair_duplicate", "schema"])
def test_source_and_manifest_mutations_refuse(tmp_path, mutation):
    manifest = setup(tmp_path)
    row = manifest["releases"][0]
    repo = Path(row["repo"])
    target = repo / "experts4bit_qlora/nested/engine.py"
    if mutation == "body":
        target.write_text("# changed committed source\n")
    elif mutation == "new_module":
        (target.parent / "new.py").write_text("# omitted module\n")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to("../__init__.py")
    elif mutation in ["layout", "dynamic", "source_version"]:
        p = repo / "pyproject.toml"
        text = p.read_text()
        if mutation == "layout":
            text = text.replace('"experts4bit_qlora*"', '"other*"')
        elif mutation == "dynamic":
            text = text.replace('[project]\n', '[project]\ndynamic=["dependencies"]\n')
        else:
            text = text.replace('version="0.0.0"', 'version="0.0.1"')
        p.write_text(text)
    elif mutation == "short_pin":
        row["commit"] = row["commit"][:8]
    elif mutation == "unknown_pin":
        row["commit"] = "0" * 40
    elif mutation == "archive_hash":
        row["wheel"]["sha256"] = "0" * 64
    elif mutation == "pair_duplicate":
        manifest["releases"][1] = copy.deepcopy(row)
    else:
        manifest["schema"] = 2
    if mutation in ["body", "new_module", "symlink", "layout", "dynamic", "source_version"]:
        row["commit"] = commit(repo)
    result = invoke(tmp_path, manifest)
    assert result.returncode != 0, mutation
    assert not (tmp_path / "result.json").exists()


def test_worktree_and_git_replace_do_not_change_pinned_objects(tmp_path):
    manifest = setup(tmp_path)
    row = manifest["releases"][0]
    repo = Path(row["repo"])
    target = repo / "experts4bit_qlora/nested/engine.py"
    target.write_text("# changed\n")
    other = commit(repo)
    git(repo, "replace", row["commit"], other)
    target.write_text("# dirty worktree differs again\n")
    result = invoke(tmp_path, manifest)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mutation", ["native_c", "missing_c", "missing_flat", "duplicate_flat", "package_dir",
                                     "package_list"])
def test_flat_native_packaging_and_c_sources_are_bound(tmp_path, mutation):
    manifest = setup(tmp_path)
    row = manifest["releases"][1]
    repo = Path(row["repo"])
    if mutation == "native_c":
        (repo / "gnf4_native/native.c").write_text("/* changed native code */\n")
    elif mutation == "missing_c":
        (repo / "gnf4_native/native.c").unlink()
    elif mutation == "missing_flat":
        (repo / "kernel/nf4_grouped.py").unlink()
    else:
        p = repo / "pyproject.toml"
        text = p.read_text()
        if mutation == "duplicate_flat":
            text = text.replace('py-modules=["nf4_grouped"]', 'py-modules=["nf4_grouped","nf4_grouped"]')
        elif mutation == "package_dir":
            text = text.replace('""="kernel"', '""="other"')
        else:
            text = text.replace('packages=["gnf4_native"]', 'packages=["other"]')
        p.write_text(text)
    row["commit"] = commit(repo)
    result = invoke(tmp_path, manifest)
    assert result.returncode != 0, mutation
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("flags", [("-S", "-B"), ("-I", "-B"), ("-I", "-S")])
def test_isolation_flags_are_required(tmp_path, flags):
    result = invoke(tmp_path, setup(tmp_path), flags=flags)
    assert result.returncode != 0
    assert "python -I -S -B required" in result.stderr
