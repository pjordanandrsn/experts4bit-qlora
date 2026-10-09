"""Source/metadata mutations using hermetic Git, copied tools and a test-host parser."""
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ra_metadata_fixture", ROOT / "tests/test_ra_source_binding.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def setup(tmp_path):
    manifest = fixture.setup(tmp_path)
    row = manifest["releases"][0]
    repo = Path(row["repo"])
    config = repo / "pyproject.toml"
    config.write_text(config.read_text().replace('[tool.setuptools.packages.find]', '''dependencies=[
  'Foo_Bar[Kind_One]>=1.0; sys_platform == "linux"', 'base>=2,<3'
]
[project.optional-dependencies]
fast=['speed>=1; python_version < "4"', 'other>=1; os_name == "posix" or platform_system == "Linux"']
train=['base>=2,<3']
[tool.setuptools.packages.find]'''))
    row["commit"] = fixture.commit(repo)
    headers = (b"Requires-Dist: foo-bar[kind-one]>=1.0; sys_platform == 'linux'\n"
               b"Requires-Dist: base<3,>=2\n"
               b"Requires-Dist: speed>=1; python_version < '4' and extra == 'fast'\n"
               b"Requires-Dist: other>=1; (os_name == 'posix' or platform_system == 'Linux') and extra == 'fast'\n"
               b"Requires-Dist: base<3,>=2; extra == 'train'\n"
               b"Provides-Extra: fast\nProvides-Extra: train")

    def add_headers(files):
        path = "experts4bit_qlora-0.0.0.dist-info/METADATA"
        head, body = files[path].split(b"\n\n", 1)
        files[path] = head + b"\n" + headers + b"\n\n" + body

    fixture.reseal(tmp_path, row, add_headers)
    # Re-archive the test host's observed parser, not the real Linux proof lock.
    dist = importlib.metadata.distribution("packaging")
    payload = {str(p): Path(dist.locate_file(p)).read_bytes() for p in dist.files
               if str(p).startswith("packaging/") and not str(p).endswith(".pyc")}
    parser = fixture.fixture.make_wheel(tmp_path, "packaging", "packaging/__init__",
                                        version=dist.version, extra=payload)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ["ra_source_metadata.py", "ra_source_binding.py", "ra_provenance.py"]:
        (tools / name).write_bytes((ROOT / "bench/ra" / name).read_bytes())
    lock = {"schema": 1, "wheels": [{"name": "packaging", "version": dist.version,
                                    "filename": Path(parser["path"]).name, "sha256": parser["sha256"]}]}
    (tools / "proof-wheel-lock.json").write_text(json.dumps(lock))
    manifest.update(parser=parser, parser_lock_sha256=hashlib.sha256((tools / "proof-wheel-lock.json").read_bytes()).hexdigest())
    return manifest, tools


def invoke(tmp_path, manifest, tools, flags=("-I", "-S", "-B")):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return subprocess.run([sys.executable, *flags, str(tools / "ra_source_metadata.py"),
                           "--manifest", str(path), "--out", str(tmp_path / "result.json")],
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("spelling", ["ordinary", "commuted", "parenthesized"])
def test_semantic_markers_bind_without_startup_or_release_import(tmp_path, spelling):
    manifest, tools = setup(tmp_path)
    if spelling != "ordinary":
        def mutate(files):
            path = "experts4bit_qlora-0.0.0.dist-info/METADATA"
            before = b"python_version < '4' and extra == 'fast'"
            after = (b"extra == 'fast' and python_version < '4'" if spelling == "commuted" else
                     b"(python_version < '4') and (extra == 'fast')")
            files[path] = files[path].replace(before, after)
        fixture.reseal(tmp_path, manifest["releases"][0], mutate)
    result = invoke(tmp_path, manifest, tools)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "result.json").read_text())
    assert receipt["proves_dependency_metadata_source_binding"] is True
    assert receipt["releases"]["experts4bit-qlora"]["extras"] == ["fast", "train"]
    assert len(receipt["releases"]["experts4bit-qlora"]["requires_dist"]) == 5
    assert receipt["startup_activated"] is False
    assert all(v is False for k, v in receipt.items() if k.startswith("proves_") and
               k != "proves_dependency_metadata_source_binding")


@pytest.mark.parametrize("mutation", ["version", "platform", "python", "extra_gate", "extra_or", "missing_gate",
                                     "missing", "surplus", "duplicate", "requested_extra", "url", "provides_missing",
                                     "provides_surplus", "provides_duplicate"])
def test_resealed_dependency_metadata_cannot_override_source(tmp_path, mutation):
    manifest, tools = setup(tmp_path)
    row = manifest["releases"][0]

    def mutate(files):
        path = "experts4bit_qlora-0.0.0.dist-info/METADATA"
        head, body = files[path].split(b"\n\n", 1)
        if mutation == "version":
            head = head.replace(b"base<3,>=2", b"base<4,>=2")
        elif mutation == "platform":
            head = head.replace(b"sys_platform == 'linux'", b"sys_platform == 'darwin'")
        elif mutation == "python":
            head = head.replace(b"python_version < '4'", b"python_version < '3.11'")
        elif mutation == "extra_gate":
            head = head.replace(b"extra == 'fast'", b"extra == 'train'")
        elif mutation == "extra_or":
            head = head.replace(b"and extra == 'fast'", b"or extra == 'fast'")
        elif mutation == "missing_gate":
            head = head.replace(b" and extra == 'fast'", b"")
        elif mutation == "missing":
            head = head.replace(b"Requires-Dist: base<3,>=2\n", b"")
        elif mutation == "surplus":
            head += b"\nRequires-Dist: foreign>=1"
        elif mutation == "duplicate":
            head += b"\nRequires-Dist: base<3,>=2"
        elif mutation == "requested_extra":
            head = head.replace(b"[kind-one]", b"[kind-two]")
        elif mutation == "url":
            head += b"\nRequires-Dist: foreign @ https://example.invalid/foreign.whl"
        elif mutation == "provides_missing":
            head = head.replace(b"Provides-Extra: train", b"")
        elif mutation == "provides_surplus":
            head += b"\nProvides-Extra: foreign"
        elif mutation == "provides_duplicate":
            head += b"\nProvides-Extra: fast"
        files[path] = head + b"\n\n" + body

    fixture.reseal(tmp_path, row, mutate)
    result = invoke(tmp_path, manifest, tools)
    assert result.returncode != 0, mutation
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("mutation", ["archive", "lock", "resealed_parser", "parser_name", "source_dependency",
                                     "source_extra", "source_duplicate", "source_extra_collision", "source_url"])
def test_parser_lock_and_source_mutations_refuse(tmp_path, mutation):
    manifest, tools = setup(tmp_path)
    if mutation == "archive":
        manifest["parser"]["sha256"] = "0" * 64
    elif mutation == "lock":
        manifest["parser_lock_sha256"] = "0" * 64
    elif mutation == "resealed_parser":
        path = Path(manifest["parser"]["path"])
        with path.open("ab") as stream:
            stream.write(b"changed bytes")
        manifest["parser"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == "parser_name":
        manifest["parser"] = manifest["releases"][0]["wheel"]
    else:
        row = manifest["releases"][0]
        repo = Path(row["repo"])
        p = repo / "pyproject.toml"
        text = p.read_text()
        if mutation == "source_dependency":
            text = text.replace('base>=2,<3', 'base>=2,<4')
        elif mutation == "source_extra":
            text = text.replace('train=', 'other=')
        elif mutation == "source_duplicate":
            text = text.replace("'base>=2,<3'\n", "'base>=2,<3', 'base>=2,<3'\n")
        elif mutation == "source_extra_collision":
            text = text.replace('train=', 'FAST=[]\ntrain=')
        else:
            text = text.replace('base>=2,<3', 'base @ https://example.invalid/base.whl')
        p.write_text(text)
        row["commit"] = fixture.commit(repo)
    result = invoke(tmp_path, manifest, tools)
    assert result.returncode != 0, mutation
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("flags", [("-S", "-B"), ("-I", "-B"), ("-I", "-S")])
def test_isolation_flags_required(tmp_path, flags):
    manifest, tools = setup(tmp_path)
    result = invoke(tmp_path, manifest, tools, flags)
    assert result.returncode != 0
    assert "python -I -S -B required" in result.stderr
