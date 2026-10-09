"""Synthetic isolated startup/import handoff controls, no real proof install."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ra_handoff_fixtures", ROOT / "tests/test_ra_prestartup.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
spec = importlib.util.spec_from_file_location("ra_handoff_control", ROOT / "bench/ra/ra_handoff.py")
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def setup(tmp_path, *, hook=False, body=None, shim_body=None, copies=False):
    payload, python, site, tool, hook_marker, import_marker = fixtures.setup(tmp_path, copies=copies)
    for name in ("ra_handoff.py", "startup-execution-proposal.json"):
        shutil.copyfile(ROOT / "bench/ra" / name, tool / name)
    if body is not None:
        # Reseal a synthetic *retained* release archive too: the import body is
        # permitted payload, so only a post-import check can catch its effect.
        pin = fixtures.fixtures.make_wheel(tmp_path, "experts4bit-qlora", "ra_probe_e4b",
                                           extra={"ra_probe_e4b.py": body.encode()})
        for extra in next(site.glob("experts4bit_qlora-*.dist-info")).iterdir():
            if extra.name in ("INSTALLER", "REQUESTED", "direct_url.json"):
                extra.unlink()
        with fixtures.fixtures.provenance.zipfile.ZipFile(pin["path"]) as archive:
            for rel in archive.namelist():
                (site / rel).write_bytes(archive.read(rel))
        payload["wheels"][0] = pin
    if hook:
        module = b"import sys\nclass Finder:\n    def find_spec(self, fullname, path=None, target=None):\n        return None\nDISTUTILS_FINDER = Finder()\ndef add_shim():\n    if DISTUTILS_FINDER not in sys.meta_path:\n        sys.meta_path.insert(0, DISTUTILS_FINDER)\n"
        if shim_body is not None:
            module = shim_body.encode()
        pin = fixtures.fixtures.make_wheel(tmp_path, "setuptools", "ra_synthetic_setuptools", version="0.0.0",
            extra={"distutils-precedence.pth": handoff.HOOK, "_distutils_hack/__init__.py": module})
        with fixtures.fixtures.provenance.zipfile.ZipFile(pin["path"]) as archive:
            for rel in archive.namelist():
                target = site / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(rel))
        payload["wheels"].append(pin)
        adapter = {"name": "setuptools", "version": "0.0.0", "wheel_sha256": pin["sha256"],
                   "startup": {"distutils-precedence.pth": hashlib.sha256(handoff.HOOK).hexdigest()}}
        audit = {**adapter, "semantics": "SYNTHETIC shim only, not real setuptools"}
        execution = {**adapter, "adapter": "setuptools_add_shim_default_local_v1"}
        (tool / "startup-audit-adapters.json").write_text(json.dumps(
            {"schema": 1, "status": "PROPOSED_AUDIT_ONLY", "adapters": [audit]}))
        (tool / "startup-execution-proposal.json").write_text(json.dumps(
            {"schema": 1, "status": "PROPOSED_EXECUTION_NOT_CLEARANCE", "adapters": [execution]}))
        payload["startup_registry_sha256"] = fixtures.fixtures.provenance.digest(tool / "startup-audit-adapters.json")
    manifest = {"schema": 1, "payload": payload, "imports": {"e4b": ["ra_probe_e4b"], "gnf4": ["ra_probe_gnf4"]},
                "execution_policy_sha256": fixtures.fixtures.provenance.digest(tool / "startup-execution-proposal.json")}
    return manifest, python, site, tool, hook_marker, import_marker


def run(tmp_path, manifest, python, tool, *, env=None, flags=("-I", "-S", "-B"), preload=None):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    args = [str(python), *flags, str(tool / "ra_handoff.py"), "--manifest", str(path),
            "--out", str(tmp_path / "result.json")]
    if preload:
        code = ("import sys,runpy; " + preload + "; sys.argv=" + repr(args[4:]) +
                "; runpy.run_path(" + repr(str(tool / "ra_handoff.py")) + ",run_name='__main__')")
        args = [str(python), *flags, "-c", code]
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)


@pytest.mark.parametrize("hook", [False, True])
@pytest.mark.parametrize("copies", [False, True])
def test_payload_then_native_site_then_imports(tmp_path, hook, copies):
    manifest, python, site, tool, hook_marker, import_marker = setup(tmp_path, hook=hook, copies=copies)
    result = run(tmp_path, manifest, python, tool)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "result.json").read_text())
    assert receipt["status"] == "PASSED" and receipt["phase"] == "COMPLETE"
    assert receipt["payload"]["startup_activated"] is False
    assert receipt["payload"]["copied_executable"] is copies
    assert receipt["startup_activated"] and receipt["proves_requested_release_imports"]
    assert len(receipt["startup_hooks"]) == hook
    assert all(n == 2 for n in receipt["startup_hooks"].values())
    assert receipt["proves_launch_authority"] is receipt["proves_gpu_engagement"] is False
    assert import_marker.exists() and not hook_marker.exists()


@pytest.mark.parametrize("mutation", ["policy_hash", "policy_status", "policy_adapter", "policy_wheel", "hook_bytes",
    "unknown_hook", "customisation", "usercustomisation", "preloaded", "override", "foreign_owner", "imports_empty",
    "imports_duplicate", "payload_changed", "site_enabled", "nonisolated", "bytecode_flag"])
def test_refusals_before_release_import(tmp_path, mutation):
    manifest, python, site, tool, hook_marker, import_marker = setup(tmp_path, hook=True)
    flags, env, preload = ("-I", "-S", "-B"), None, None
    policy_path = tool / "startup-execution-proposal.json"
    if mutation == "policy_hash":
        manifest["execution_policy_sha256"] = "0" * 64
    elif mutation.startswith("policy_"):
        policy = json.loads(policy_path.read_text())
        if mutation == "policy_status":
            policy["status"] = "APPROVED"
        elif mutation == "policy_adapter":
            policy["adapters"][0]["adapter"] = "exec_arbitrary"
        else:
            policy["adapters"][0]["wheel_sha256"] = "0" * 64
        policy_path.write_text(json.dumps(policy))
        manifest["execution_policy_sha256"] = fixtures.fixtures.provenance.digest(policy_path)
    elif mutation == "hook_bytes":
        (site / "distutils-precedence.pth").write_text("import foreign\n")
    elif mutation == "unknown_hook":
        (site / "unknown.pth").write_text("import foreign\n")
    elif mutation in ("customisation", "usercustomisation"):
        # Put the customisation inside both retained and installed payload, so
        # complete payload verification passes and the startup gate refuses.
        name = "sitecustomize.py" if mutation == "customisation" else "usercustomize.py"
        pin = fixtures.fixtures.make_wheel(tmp_path, "experts4bit-qlora", "ra_probe_e4b",
                                          extra={name: b"raise RuntimeError('must not run')\n"})
        for extra in next(site.glob("experts4bit_qlora-*.dist-info")).iterdir():
            if extra.name in ("INSTALLER", "REQUESTED", "direct_url.json"):
                extra.unlink()
        with fixtures.fixtures.provenance.zipfile.ZipFile(pin["path"]) as archive:
            for rel in archive.namelist():
                (site / rel).write_bytes(archive.read(rel))
        manifest["payload"]["wheels"][0] = pin
    elif mutation == "preloaded":
        preload = f"sys.path.append({str(site)!r}); import ra_probe_gnf4"
    elif mutation == "override":
        import os
        env = dict(os.environ, SETUPTOOLS_USE_DISTUTILS="stdlib")
    elif mutation == "foreign_owner":
        manifest["imports"]["e4b"] = ["ra_probe_gnf4"]
    elif mutation == "imports_empty":
        manifest["imports"]["e4b"] = []
    elif mutation == "imports_duplicate":
        manifest["imports"]["e4b"] *= 2
    elif mutation == "payload_changed":
        (site / "ra_probe_e4b.py").write_text("CHANGED=1\n")
    elif mutation == "site_enabled":
        flags = ("-I", "-B")
    elif mutation == "nonisolated":
        flags = ("-S", "-B")
    else:
        flags = ("-I", "-S")
    result = run(tmp_path, manifest, python, tool, flags=flags, env=env, preload=preload)
    assert result.returncode != 0
    if mutation != "site_enabled":
        assert not import_marker.exists()
    receipt = json.loads((tmp_path / "result.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["error_type"]


@pytest.mark.parametrize("body", [
    "import sys\nsys.path.append('/foreign')\n",
    "import sys\nsys.path.append(sys.path[-1])\n",
    "import sys\nsys.path.reverse()\n",
    "import sys\nsys.meta_path.reverse()\n",
    "import sys\nsys.meta_path.pop()\n",
    "import os\nos.environ['RA_MUTATED']='1'\n",
    "from pathlib import Path\nPath(__file__).write_text('CHANGED=1')\n",
    "from pathlib import Path\nPath(__file__).with_name('extra.py').touch()\n",
    "raise RuntimeError('import failed')\n",
    "import sys\nsys._base_executable='/foreign/python'\n",
])
def test_post_import_changes_preserve_failed_payload_receipt(tmp_path, body):
    manifest, python, site, tool, hook_marker, import_marker = setup(tmp_path, body=body)
    result = run(tmp_path, manifest, python, tool)
    assert result.returncode != 0
    receipt = json.loads((tmp_path / "result.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["phase"] == "IMPORTS"
    assert receipt["payload"]["proves_installed_payload"]
    assert receipt["payload"]["proves_release_imports"] is False


@pytest.mark.parametrize("effect", ["pass", "sys.path.append('/foreign')",
    "__import__('os').environ['RA_SHIM_MUTANT']='1'",
    "__import__('pathlib').Path(__file__).write_text('CHANGED=1')",
    "sys.meta_path.append(Finder())"])
def test_startup_effect_mutants_fail_before_release_import(tmp_path, effect):
    code = ("import sys\nclass Finder:\n    def find_spec(self, fullname, path=None, target=None):\n        return None\n"
            "DISTUTILS_FINDER=Finder()\ndef add_shim():\n")
    if effect != "pass":
        code += "    if DISTUTILS_FINDER not in sys.meta_path:\n        sys.meta_path.insert(0,DISTUTILS_FINDER)\n"
    code += "    " + effect + "\n"
    manifest, python, site, tool, hook_marker, import_marker = setup(tmp_path, hook=True, shim_body=code)
    result = run(tmp_path, manifest, python, tool)
    assert result.returncode != 0
    receipt = json.loads((tmp_path / "result.json").read_text())
    assert receipt["status"] == "FAILED" and receipt["phase"] == "STARTUP"
    assert receipt["payload"]["proves_installed_payload"]
    assert not import_marker.exists()
