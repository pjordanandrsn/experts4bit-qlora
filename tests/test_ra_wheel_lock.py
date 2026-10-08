"""CPU archive binding mutations; fixtures are not proof installations."""
import copy
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "bench/ra/ra_wheel_lock.py"
spec = importlib.util.spec_from_file_location("ra_wheel_lock", TOOL)
locktool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(locktool)


def fixture(tmp_path):
    image = json.dumps({"image": {"os": "linux", "architecture": "amd64"},
                        "python": {"version": "3.11.13", "soabi": "cpython-311-x86_64-linux-gnu"}}).encode()
    root = tmp_path / "wheels"
    root.mkdir()
    rows = []
    versions = {**locktool.REQUIRED, "pip": "26.2.1", "uvicorn": "0.54.0",
                "experts4bit-qlora": "0.50.0", "grouped-nf4-gemm": "0.43.0"}
    for name, version in versions.items():
        filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        p = root / filename
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(f"{name.replace('-', '_')}-{version}.dist-info/METADATA",
                       f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        rows.append({"name": name, "version": version, "filename": filename,
                     "url": "https://files.pythonhosted.org/packages/" + filename.replace('+', '%2B'),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": p.stat().st_size})
    return {"schema": 1, "image_pin_sha256": hashlib.sha256(image).hexdigest(),
            "roots": [f"{n}=={v}" for n, v in versions.items()], "wheels": rows}, image, root


def test_archive_handoff_keeps_scope_explicit(tmp_path):
    lock, image, root = fixture(tmp_path)
    result = locktool.verify_archives(lock, image, root)
    assert result["verified_archives"] == 9
    assert result["common_dependencies"] == 7
    assert result["startup_adapters_required"] == {}
    assert not any(result[k] for k in ("proves_installation", "proves_dependency_closure", "proves_gpu_engagement"))
    imports = {"e4b": ["experts4bit_qlora"], "gnf4": ["nf4_grouped"]}
    manifest = locktool.provenance_manifest(lock, image, root, tmp_path / "venv", imports)
    assert manifest["imports"] == imports
    assert all(set(p) == {"path", "sha256"} for p in manifest["wheels"])


@pytest.mark.parametrize("mutation", ["image", "duplicate", "version", "filename", "origin", "credentials",
                                     "query", "digest", "size", "glibc", "architecture", "python_tag",
                                     "missing_pip", "schema", "extra_field", "roots"])
def test_lock_mutations_refuse(tmp_path, mutation):
    lock, image, root = fixture(tmp_path)
    row = lock["wheels"][0]
    if mutation == "image":
        image += b" "
    elif mutation == "duplicate":
        lock["wheels"].append(copy.deepcopy(row))
    elif mutation == "version":
        row["version"] = "9.9.9"
    elif mutation == "filename":
        row["filename"] = "../" + row["filename"]
    elif mutation == "origin":
        row["url"] = row["url"].replace("files.pythonhosted.org", "unknown.invalid")
    elif mutation == "credentials":
        row["url"] = row["url"].replace("https://", "https://user:secret@")
    elif mutation == "query":
        row["url"] += "?override=1"
    elif mutation == "digest":
        row["sha256"] = "f" * 64
    elif mutation == "size":
        row["size"] += 1
    elif mutation in {"glibc", "architecture", "python_tag"}:
        suffix = {"glibc": "py3-none-manylinux_2_39_x86_64.whl",
                  "architecture": "py3-none-manylinux_2_28_aarch64.whl",
                  "python_tag": "cp314-cp314-manylinux_2_28_x86_64.whl"}[mutation]
        row["filename"] = row["filename"].replace("py3-none-any.whl", suffix)
        row["url"] = "https://files.pythonhosted.org/packages/" + row["filename"]
    elif mutation == "missing_pip":
        lock["wheels"] = [p for p in lock["wheels"] if p["name"] != "pip"]
    elif mutation == "schema":
        lock["schema"] = 2
    elif mutation == "extra_field":
        lock["override"] = True
    else:
        lock["roots"] = []
    with pytest.raises(ValueError, match="wheel lock"):
        locktool.verify_archives(lock, image, root)


@pytest.mark.parametrize("mutation", ["extra", "symlink", "payload", "metadata"])
def test_retained_bytes_and_inventory_mutations_refuse(tmp_path, mutation):
    lock, image, root = fixture(tmp_path)
    row = lock["wheels"][0]
    p = root / row["filename"]
    if mutation == "extra":
        (root / "unregistered.whl").write_bytes(b"unexpected")
    elif mutation == "symlink":
        target = tmp_path / "outside.whl"
        p.rename(target)
        p.symlink_to(target)
    elif mutation == "payload":
        p.write_bytes(p.read_bytes() + b"changed")
    else:
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("torch-2.8.0.dist-info/METADATA", "Name: foreign\nVersion: 2.8.0+cu128\n")
        row.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size=p.stat().st_size)
    with pytest.raises(ValueError, match="wheel lock"):
        locktool.verify_archives(lock, image, root)


def test_startup_hook_requires_separate_reviewed_install_adapter(tmp_path):
    lock, image, root = fixture(tmp_path)
    row = lock["wheels"][0]
    p = root / row["filename"]
    with zipfile.ZipFile(p, "a") as z:
        z.writestr("unreviewed.pth", "import arbitrary\n")
    row.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size=p.stat().st_size)
    result = locktool.verify_archives(lock, image, root)
    assert result["startup_adapters_required"] == {"torch": ["unreviewed.pth"]}
    assert result["proves_installation"] is False


@pytest.mark.parametrize("tag", ["py311-none-any", "cp36-abi3-manylinux_2_28_x86_64",
                                "cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64"])
def test_python_specific_and_older_stable_abi_tags_are_compatible(tmp_path, tag):
    lock, image, root = fixture(tmp_path)
    row = lock["wheels"][0]
    before = root / row["filename"]
    row["filename"] = row["filename"].replace("py3-none-any", tag)
    before.rename(root / row["filename"])
    row["url"] = "https://files.pythonhosted.org/packages/" + row["filename"].replace('+', '%2B')
    assert locktool.verify_archives(lock, image, root)["verified_archives"] == 9


@pytest.mark.parametrize("vendored", [True, False])
def test_vendored_metadata_is_payload_and_duplicate_top_level_refuses(tmp_path, vendored):
    lock, image, root = fixture(tmp_path)
    row = lock["wheels"][0]
    p = root / row["filename"]
    rel = "vendor/package-1.dist-info/METADATA" if vendored else "foreign-1.dist-info/METADATA"
    with zipfile.ZipFile(p, "a") as z:
        z.writestr(rel, "Name: vendored-package\nVersion: 1\n")
    row.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size=p.stat().st_size)
    if vendored:
        assert locktool.verify_archives(lock, image, root)["verified_archives"] == 9
    else:
        with pytest.raises(ValueError, match="top-level METADATA"):
            locktool.verify_archives(lock, image, root)
