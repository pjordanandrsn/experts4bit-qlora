"""Real isolated CPU wheel installs plus provenance mutations; no GPU proof."""
import base64
import csv
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "bench/ra/ra_provenance.py"
spec = importlib.util.spec_from_file_location("ra_provenance", TOOL)
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


def make_wheel(root, name, module, *, extra=None, version="0.0.0"):
    info = name.replace("-", "_") + "-" + version + ".dist-info"
    files = {module + ".py": b"VALUE = 'synthetic CPU wheel'\n",
             info + "/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
             info + "/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    files.update(extra or {})
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    for rel, data in files.items():
        hashed = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        writer.writerow([rel, "sha256=" + hashed, len(data)])
    writer.writerow([info + "/RECORD", "", ""])
    files[info + "/RECORD"] = stream.getvalue().encode()
    path = root / (name.replace("-", "_") + "-" + version + "-py3-none-any.whl")
    with zipfile.ZipFile(path, "w") as archive:
        for rel, data in files.items():
            archive.writestr(rel, data)
    return {"path": str(path), "sha256": provenance.digest(path)}


def setup(tmp_path):
    prefix = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
    wheels = [make_wheel(tmp_path, "experts4bit-qlora", "ra_probe_e4b"),
              make_wheel(tmp_path, "grouped-nf4-gemm", "ra_probe_gnf4")]
    subprocess.run([sys.executable, "-m", "pip", "--python", str(prefix), "install", "--no-index", "--no-deps",
                    "--no-compile", *(w["path"] for w in wheels)], check=True, capture_output=True, text=True)
    python = prefix / "bin/python"
    site = Path(subprocess.check_output([str(python), "-I", "-B", "-c",
                "import sysconfig;print(sysconfig.get_path('purelib'))"], text=True).strip())
    manifest = {"schema": 1, "venv": str(prefix), "wheels": wheels,
                "imports": {"e4b": ["ra_probe_e4b"], "gnf4": ["ra_probe_gnf4"]}}
    return manifest, python, site


def probe(tmp_path, manifest, python, *, isolated=True):
    pin = tmp_path / "manifest.json"
    pin.write_text(json.dumps(manifest))
    argv = [str(python), *(["-I"] if isolated else []), "-B", str(TOOL), "--manifest", str(pin),
            "--out", str(tmp_path / "provenance.json")]
    return subprocess.run(argv, capture_output=True, text=True, timeout=30)


def reseal_record(site):
    record = next(site.glob("experts4bit_qlora-*.dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(record.read_text())))
    for row in rows:
        path = site / row[0]
        if path == record:
            continue
        data = path.read_bytes()
        row[1] = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        row[2] = str(len(data))
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(rows)
    record.write_text(text.getvalue())


def test_real_isolated_wheel_install_matches_archive_and_imports(tmp_path):
    manifest, python, site = setup(tmp_path)
    result = probe(tmp_path, manifest, python)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "provenance.json").read_bytes())
    assert receipt["proves_gpu_engagement"] is False
    for alias, module in (("e4b", "ra_probe_e4b"), ("gnf4", "ra_probe_gnf4")):
        observed = receipt["imports"][alias]
        assert observed["wheel_verified"] is True and len(observed["tree_sha256"]) == 64
        assert observed["venv"] == manifest["venv"]
        assert observed["modules"][module] == str(site / (module + ".py"))
    assert set(receipt["distributions"]) == {"experts4bit-qlora", "grouped-nf4-gemm"}


@pytest.mark.parametrize("mutation", [None, "resealed_script", "missing_script"])
def test_generated_scripts_derive_from_verified_installer(tmp_path, mutation):
    manifest, python, site = setup(tmp_path)
    # Re-archive the test host's real pip payload to exercise its installed
    # template without a network fixture. This is not a release wheel claim.
    dist = importlib.metadata.distribution("pip")
    extra = {}
    for rel in dist.files:
        if rel.parts[0] == "pip" and not str(rel).endswith(".pyc"):
            extra[str(rel)] = Path(dist.locate_file(rel)).read_bytes()
    extra[f"pip-{dist.version}.dist-info/entry_points.txt"] = dist.read_text("entry_points.txt").encode()
    installer = make_wheel(tmp_path, "pip", "ra_probe_installer", version=dist.version, extra=extra)
    info = "experts4bit_qlora-0.0.0.dist-info"
    entry = make_wheel(tmp_path, "experts4bit-qlora", "ra_probe_e4b", extra={
        "ra_probe_e4b.py": b"def main():\n    return 0\n",
        info + "/entry_points.txt": b"[console_scripts]\nra-probe = ra_probe_e4b:main\n"})
    manifest["wheels"] = [entry, manifest["wheels"][1], installer]
    subprocess.run([sys.executable, "-m", "pip", "--python", str(python.parent.parent), "install",
                    "--force-reinstall", "--no-index", "--no-deps", "--no-compile", entry["path"], installer["path"]],
                   check=True, capture_output=True, text=True)
    script = python.parent / "ra-probe"
    if mutation == "resealed_script":
        script.write_text(script.read_text() + "\n# altered wrapper\n")
        reseal_record(site)
    elif mutation == "missing_script":
        script.unlink()
    result = probe(tmp_path, manifest, python)
    if mutation:
        assert result.returncode != 0 and not (tmp_path / "provenance.json").exists()
    else:
        assert result.returncode == 0, result.stderr
        receipt = json.loads((tmp_path / "provenance.json").read_bytes())
        assert set(receipt["distributions"]) == {"experts4bit-qlora", "grouped-nf4-gemm", "pip"}


@pytest.mark.parametrize("mutation", ["wheel_hash", "wheel_bytes", "source", "resealed_source", "metadata",
    "missing_file", "missing_record", "record_coverage", "record_hash", "duplicate_record", "unlisted",
    "bytecode", "symlink", "editable", "inventory", "version", "wrong_venv", "nonisolated", "wrong_import",
    "duplicate_wheel", "import_mutation", "foreign_import", "foreign_search_path", "foreign_record"])
def test_install_and_import_mutations_refuse(tmp_path, mutation):
    manifest, python, site = setup(tmp_path)
    source = site / "ra_probe_e4b.py"
    info = next(site.glob("experts4bit_qlora-*.dist-info"))
    if mutation == "wheel_hash":
        manifest["wheels"][0]["sha256"] = "0" * 64
    elif mutation == "wheel_bytes":
        with Path(manifest["wheels"][0]["path"]).open("ab") as stream:
            stream.write(b"changed archive")
    elif mutation in ("source", "resealed_source"):
        source.write_text("VALUE = 'changed'\n")
        if mutation == "resealed_source":
            reseal_record(site)
    elif mutation == "metadata":
        (info / "METADATA").write_text("Name: experts4bit-qlora\nVersion: 0.0.0\nextra: changed\n")
    elif mutation == "missing_file":
        source.unlink()
    elif mutation == "missing_record":
        (info / "RECORD").unlink()
    elif mutation == "record_coverage":
        p = info / "RECORD"
        p.write_text("\n".join(line for line in p.read_text().splitlines() if not line.startswith("ra_probe_e4b.py,")) + "\n")
    elif mutation == "record_hash":
        p = info / "RECORD"
        p.write_text(p.read_text().replace("sha256=", "sha512=", 1))
    elif mutation == "duplicate_record":
        p = info / "RECORD"
        p.write_text(p.read_text() + p.read_text().splitlines()[0] + "\n")
    elif mutation == "foreign_record":
        p = next(site.glob("grouped_nf4_gemm-*.dist-info/RECORD"))
        data = source.read_bytes()
        hashed = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        p.write_text(p.read_text() + f"ra_probe_e4b.py,sha256={hashed},{len(data)}\n")
    elif mutation == "unlisted":
        (site / "foreign.py").write_text("VALUE=1\n")
    elif mutation == "bytecode":
        (site / "foreign.pyc").write_bytes(b"unverified bytecode")
    elif mutation == "symlink":
        foreign = tmp_path / "foreign.py"
        foreign.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(foreign)
    elif mutation == "editable":
        p = info / "direct_url.json"
        p.write_text(json.dumps({"url": tmp_path.as_uri(), "dir_info": {"editable": True}}))
        reseal_record(site)
    elif mutation == "inventory":
        extra = site / "foreign-0.0.0.dist-info"
        extra.mkdir()
        (extra / "METADATA").write_text("Name: foreign\nVersion: 0.0.0\n")
    elif mutation == "version":
        p = info / "METADATA"
        p.write_text(p.read_text().replace("Version: 0.0.0", "Version: 1.0.0"))
    elif mutation == "wrong_venv":
        manifest["venv"] = str(tmp_path / "other-venv")
    elif mutation == "wrong_import":
        manifest["imports"]["e4b"] = ["ra_probe_gnf4"]
    elif mutation == "duplicate_wheel":
        manifest["wheels"].append(manifest["wheels"][0])
    elif mutation in ("import_mutation", "foreign_import", "foreign_search_path"):
        # These are deliberately reviewed synthetic wheel payloads: changing
        # an archive after install is not used to manufacture a positive case.
        if mutation == "import_mutation":
            text = b"from pathlib import Path\nPath(__file__).write_text('changed')\n"
        else:
            foreign = tmp_path / "foreign.py"
            foreign.write_text("VALUE=1\n")
            text = ("import sys\nsys.path.insert(0," + repr(str(tmp_path)) + ")\n" +
                    ("import foreign\n" if mutation == "foreign_import" else "")).encode()
        pin = make_wheel(tmp_path, "experts4bit-qlora", "ra_probe_e4b", extra={"ra_probe_e4b.py": text})
        manifest["wheels"][0] = pin
        subprocess.run([sys.executable, "-m", "pip", "--python", str(python.parent.parent), "install",
                        "--force-reinstall", "--no-index", "--no-deps", "--no-compile", pin["path"]],
                       check=True, capture_output=True, text=True)
    result = probe(tmp_path, manifest, python, isolated=mutation != "nonisolated")
    assert result.returncode != 0, mutation
    assert not (tmp_path / "provenance.json").exists()


@pytest.mark.parametrize("mutation", ["traversal", "pth", "data", "zip_duplicate", "missing_record_entry"])
def test_wheel_archive_mutations_refuse(tmp_path, mutation):
    extras = {}
    if mutation == "traversal":
        extras["../outside.py"] = b"VALUE=1\n"
    elif mutation == "pth":
        extras["startup.pth"] = b"import foreign\n"
    elif mutation == "data":
        extras["experts4bit_qlora-0.0.0.data/scripts/tool"] = b"unreviewed script\n"
    pin = make_wheel(tmp_path, "experts4bit-qlora", "ra_probe_e4b", extra=extras)
    path = Path(pin["path"])
    if mutation == "zip_duplicate":
        with pytest.warns(UserWarning, match="Duplicate"):
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("ra_probe_e4b.py", b"duplicate")
    elif mutation == "missing_record_entry":
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("missing.py", b"VALUE=1\n")
    pin["sha256"] = provenance.digest(path)
    with pytest.raises(ValueError):
        package = provenance.wheel(pin)
        for rel in package["files"]:
            provenance.destination(rel, tmp_path)
