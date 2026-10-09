"""Marker/extras/constraint mutations; CPU graph controls grant no install proof."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

TOOL = Path(__file__).resolve().parents[1] / "bench/ra/ra_closure.py"
spec = importlib.util.spec_from_file_location("ra_closure", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {
        "application": {"version": "1.0.0", "requires_python": ">=3.11,<3.12", "extras": ["serve"],
                        "requires_dist": ["core>=2", "server[http]==3; extra == 'serve'",
                                          "windows-only; sys_platform == 'win32'",
                                          "not-for-py311; python_version >= '3.12'"]},
        "core": {"version": "2.0.0", "requires_python": "", "extras": [], "requires_dist": []},
        "server": {"version": "3.0.0", "requires_python": ">=3.10", "extras": ["http"],
                   "requires_dist": ["transport==4; extra == 'http'", "application>=1"]},
        "transport": {"version": "4.0.0", "requires_python": "", "extras": [], "requires_dist": []},
    }


def resolve(packages, roots=None, environment=None):
    return module.closure(packages, roots or ["application[serve]==1.0.0"],
                          module.ENVIRONMENT if environment is None else environment,
                          Requirement, Version, SpecifierSet)


def test_transitive_extras_cycles_and_platform_markers():
    result = resolve(fixture())
    assert set(result["distributions"]) == set(fixture())
    assert result["distributions"]["server"]["extras"] == ["http"]
    assert result["distributions"]["application"]["extras"] == ["serve"]
    assert {"windows-only", "not-for-py311"}.isdisjoint(result["distributions"])
    assert len(result["edges"]) == 5


def test_late_extra_request_revisits_existing_node():
    packages = fixture()
    packages["application"]["requires_dist"] = ["server", "core"]
    packages["core"]["requires_dist"] = ["server[http]"]
    result = resolve(packages)
    assert result["distributions"]["server"]["extras"] == ["http"]
    assert "transport" in result["distributions"]


@pytest.mark.parametrize("mutation", ["missing", "version", "root_version", "python", "root_extra",
                                     "dependency_extra", "surplus", "unrequested_extra", "direct_url",
                                     "kernel_release", "kernel_version", "conditional_root", "platform",
                                     "environment_fields", "malformed"])
def test_graph_mutations_refuse(mutation):
    packages, roots, environment = fixture(), ["application[serve]==1.0.0"], copy.deepcopy(module.ENVIRONMENT)
    if mutation == "missing":
        del packages["transport"]
    elif mutation == "version":
        packages["transport"]["version"] = "5.0.0"
    elif mutation == "root_version":
        roots = ["application[serve]==2.0.0"]
    elif mutation == "python":
        packages["transport"]["requires_python"] = ">=3.12"
    elif mutation == "root_extra":
        roots = ["application[unknown]"]
    elif mutation == "dependency_extra":
        packages["server"]["extras"] = []
    elif mutation == "surplus":
        packages["surplus"] = dict(packages["transport"])
    elif mutation == "unrequested_extra":
        roots = ["application"]
    elif mutation == "direct_url":
        packages["server"]["requires_dist"].append("transport @ https://example.invalid/transport.whl")
    elif mutation in ("kernel_release", "kernel_version"):
        field = "platform_release" if mutation == "kernel_release" else "platform_version"
        packages["server"]["requires_dist"].append(f"transport; {field} == 'unregistered'")
    elif mutation == "conditional_root":
        roots = ["application[serve]; sys_platform == 'linux'"]
    elif mutation == "platform":
        environment["sys_platform"] = "darwin"
    elif mutation == "environment_fields":
        environment.pop("implementation_name")
    elif mutation == "malformed":
        packages["server"]["requires_dist"].append("invalid requirement !!!")
    with pytest.raises(ValueError):
        resolve(packages, roots, environment)


def test_parser_accepts_local_release_version_with_public_constraint():
    packages = {"torch": {"version": "2.8.0+cu128", "requires_python": ">=3.10", "extras": [], "requires_dist": []}}
    result = resolve(packages, ["torch==2.8.0"])
    assert result["distributions"]["torch"]["version"] == "2.8.0+cu128"


def test_runtime_refuses_before_any_archive_or_parser_import():
    code = ("import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('closure',{str(TOOL)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "m.verify({},b'',None)")
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and "python -I -S -B required" in result.stderr
    assert "packaging" not in result.stderr


def test_registered_environment_is_explicit_cpu_fixture():
    # A JSON receipt must carry all marker inputs, not use this test host's defaults.
    result = resolve(fixture())
    assert json.loads(json.dumps(result))["marker_environment"] == module.ENVIRONMENT
