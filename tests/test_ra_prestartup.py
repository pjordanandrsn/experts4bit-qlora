"""No-site installed-payload controls with tiny CPU venvs, never GPU proof."""
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ra_prestartup_fixtures", ROOT / "tests/test_ra_provenance.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def setup(tmp_path, *, startup=False, pip=False):
    prefix = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
    python = prefix / "bin/python"
    code = ("import sys,sysconfig,json,platform,hashlib; from pathlib import Path; "
            "print(json.dumps({'image':{'os':sys.platform,'architecture':"
            "{'x86_64':'amd64','AMD64':'amd64','arm64':'arm64','aarch64':'arm64'}[platform.machine()]},"
            "'python':{'version':platform.python_version(),'soabi':sysconfig.get_config_var('SOABI'),"
            "'executable_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()}}))")
    image = tmp_path / "synthetic-cpu-image.json"
    image.write_bytes(subprocess.check_output([str(python), "-I", "-S", "-B", "-c", code]))
    site = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    hook_marker, import_marker = tmp_path / "hook-ran", tmp_path / "release-imported"
    extras = {"ra_probe_e4b.py": f"from pathlib import Path\nPath({str(import_marker)!r}).touch()\n".encode()}
    body = f"import pathlib; pathlib.Path({str(hook_marker)!r}).touch()\n".encode()
    if startup:
        extras["synthetic.pth"] = body
    wheels = [fixtures.make_wheel(tmp_path, "experts4bit-qlora", "ra_probe_e4b", extra=extras),
              fixtures.make_wheel(tmp_path, "grouped-nf4-gemm", "ra_probe_gnf4")]
    tool = tmp_path / "tool"
    tool.mkdir()
    for name in ("ra_prestartup.py", "ra_provenance.py", "startup-audit-adapters.json"):
        shutil.copyfile(ROOT / "bench/ra" / name, tool / name)
    if startup:
        registry = {"schema": 1, "status": "PROPOSED_AUDIT_ONLY", "adapters": [{
            "name": "experts4bit-qlora", "version": "0.0.0", "wheel_sha256": wheels[0]["sha256"],
            "startup": {"synthetic.pth": hashlib.sha256(body).hexdigest()}, "semantics": "synthetic no-execution control"}]}
        (tool / "startup-audit-adapters.json").write_text(json.dumps(registry))
    if pip:
        dist = importlib.metadata.distribution("pip")
        payload = {str(p): Path(dist.locate_file(p)).read_bytes() for p in dist.files
                   if p.parts[0] == "pip" and not str(p).endswith(".pyc")}
        payload[f"pip-{dist.version}.dist-info/entry_points.txt"] = dist.read_text("entry_points.txt").encode()
        wheels.append(fixtures.make_wheel(tmp_path, "pip", "ra_test_installer", version=dist.version, extra=payload))
    subprocess.run([sys.executable, "-m", "pip", "--python", str(prefix), "install", "--no-index", "--no-deps",
                    "--no-compile", *(p["path"] for p in wheels)], check=True, capture_output=True, text=True)
    manifest = {"schema": 1, "venv": str(prefix), "image": str(image),
                "image_sha256": fixtures.provenance.digest(image),
                "startup_registry_sha256": fixtures.provenance.digest(tool / "startup-audit-adapters.json"),
                "wheels": wheels}
    return manifest, python, site, tool, hook_marker, import_marker


def run(tmp_path, manifest, python, tool, *, flags=("-I", "-S", "-B")):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return subprocess.run([str(python), *flags, str(tool / "ra_prestartup.py"), "--manifest", str(path),
                           "--out", str(tmp_path / "result.json")], text=True, capture_output=True, timeout=45)


@pytest.mark.parametrize("startup,pip", [(False, False), (True, False), (False, True)])
def test_installed_payload_without_startup_or_release_import(tmp_path, startup, pip):
    manifest, python, site, tool, hook_marker, import_marker = setup(tmp_path, startup=startup, pip=pip)
    result = run(tmp_path, manifest, python, tool)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "result.json").read_bytes())
    assert receipt["proves_installed_payload"] is True and receipt["startup_activated"] is False
    assert receipt["proves_release_imports"] is receipt["proves_gpu_engagement"] is False
    assert not hook_marker.exists() and not import_marker.exists()


@pytest.mark.parametrize("mutation", ["archive_hash", "source", "resealed_source", "metadata", "record",
    "record_missing", "unlisted", "resealed_extra", "bytecode", "symlink", "registry_hash", "image_hash",
    "image_binary", "system_site", "cfg_duplicate", "wrong_executable", "nonisolated", "site_enabled",
    "script_missing", "script_resealed", "cross_record", "image_version", "image_abi", "image_platform",
    "system_home", "cfg_missing", "no_bytecode_flag", "hook_changed"])
def test_prestartup_mutations_refuse_before_receipt(tmp_path, mutation):
    manifest, python, site, tool, hook_marker, import_marker = setup(
        tmp_path, pip=mutation.startswith("script"), startup=mutation == "hook_changed")
    source = site / "ra_probe_e4b.py"
    flags = ("-I", "-S", "-B")
    if mutation == "archive_hash":
        manifest["wheels"][0]["sha256"] = "0" * 64
    elif mutation in ("source", "resealed_source"):
        source.write_text("CHANGED=1\n")
        if mutation == "resealed_source":
            fixtures.reseal_record(site)
    elif mutation == "metadata":
        next(site.glob("experts4bit_qlora-*.dist-info/METADATA")).write_text("Name: foreign\nVersion: 0.0.0\n")
    elif mutation in ("record", "record_missing"):
        p = next(site.glob("experts4bit_qlora-*.dist-info/RECORD"))
        p.unlink() if mutation == "record_missing" else p.write_text("changed\n")
    elif mutation in ("unlisted", "resealed_extra", "bytecode"):
        p = site / ("foreign.pyc" if mutation == "bytecode" else "foreign.py")
        p.write_bytes(b"FOREIGN=1\n")
        if mutation == "resealed_extra":
            record = next(site.glob("experts4bit_qlora-*.dist-info/RECORD"))
            hashed = base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip("=")
            record.write_text(record.read_text() + f"{p.name},sha256={hashed},{p.stat().st_size}\n")
    elif mutation == "symlink":
        target = tmp_path / "foreign.py"
        target.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(target)
    elif mutation in ("registry_hash", "image_hash"):
        manifest["startup_registry_sha256" if mutation == "registry_hash" else "image_sha256"] = "0" * 64
    elif mutation in ("image_binary", "image_version", "image_abi", "image_platform"):
        image = Path(manifest["image"])
        value = json.loads(image.read_bytes())
        if mutation == "image_binary":
            value["python"]["executable_sha256"] = "0" * 64
        elif mutation == "image_platform":
            value["image"]["os"] = "unregistered"
        else:
            value["python"]["version" if mutation == "image_version" else "soabi"] = "unregistered"
        image.write_text(json.dumps(value))
        manifest["image_sha256"] = fixtures.provenance.digest(image)
    elif mutation in ("system_site", "cfg_duplicate"):
        p = python.parent.parent / "pyvenv.cfg"
        p.write_text(p.read_text().replace("include-system-site-packages = false", "include-system-site-packages = true")
                     if mutation == "system_site" else p.read_text() + "home = duplicate\n")
    elif mutation == "wrong_executable":
        python = Path(sys.executable)
    elif mutation == "nonisolated":
        flags = ("-S", "-B")
    elif mutation == "site_enabled":
        flags = ("-I", "-B")  # No hook in this negative fixture.
    elif mutation == "no_bytecode_flag":
        flags = ("-I", "-S")
    elif mutation == "system_home":
        p = python.parent.parent / "pyvenv.cfg"
        p.write_text("\n".join("home = " + str(tmp_path) if line.startswith("home =") else line
                               for line in p.read_text().splitlines()) + "\n")
    elif mutation == "cfg_missing":
        (python.parent.parent / "pyvenv.cfg").unlink()
    elif mutation == "hook_changed":
        (site / "synthetic.pth").write_bytes(b"import unverified_hook\n")
        fixtures.reseal_record(site)
    elif mutation in ("script_missing", "script_resealed"):
        script = python.parent / "pip"
        if mutation == "script_missing":
            script.unlink()
        else:
            script.write_text(script.read_text() + "\n# changed wrapper\n")
            record = next(site.glob("pip-*.dist-info/RECORD"))
            rows = list(csv.reader(io.StringIO(record.read_text())))
            for row in rows:
                if row[0] == "../../../bin/pip":
                    row[1] = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(script.read_bytes()).digest()).decode().rstrip("=")
                    row[2] = str(script.stat().st_size)
            text = io.StringIO()
            csv.writer(text, lineterminator="\n").writerows(rows)
            record.write_text(text.getvalue())
    elif mutation == "cross_record":
        p = next(site.glob("grouped_nf4_gemm-*.dist-info/RECORD"))
        hashed = base64.urlsafe_b64encode(hashlib.sha256(source.read_bytes()).digest()).decode().rstrip("=")
        p.write_text(p.read_text() + f"ra_probe_e4b.py,sha256={hashed},{source.stat().st_size}\n")
    result = run(tmp_path, manifest, python, tool, flags=flags)
    assert result.returncode != 0 and not (tmp_path / "result.json").exists(), result.stdout
    assert not hook_marker.exists() and not import_marker.exists()
