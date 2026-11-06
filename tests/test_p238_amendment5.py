"""P238 amendment 5, on CPU: the two kit defects p238-shake-r5 found, each from its real record first.

1. The op processes died at ``../tc1/tc1_arm.py`` (``p238_op.py``'s repo-layout default) because the box stages flat. The runner
   now hands every op process the staged ``$W/tc1_arm.py``, and ``p238_op.py`` resolves ``--tc1`` in the dry path too, so a $0
   run exercises the path the GPU run loads.
2. tc1_arm's T24 refused U0/U1/UA: unsloth 2026.10.8's ``FastModel.get_peft_model`` is a bare ``(model, *args, **kwargs)``
   dispatcher with no ``__wrapped__``. ``bench/p238/p238_shim_sitecustomize.py`` sets it, after its checks. The shim is tested on
   a stand-in ``unsloth`` package laid out as the wheel is (loader.py's two dispatchers, vision.py / llama.py's targets) and on the
   REAL ``tc1_arm.unsloth_target_parameters_kwargs``, each in a fresh interpreter, as the box runs it.
"""
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
P238 = ROOT / "bench" / "p238"
FIX = ROOT / "tests" / "fixtures" / "p238_r5"
RUN = (P238 / "p238_run.sh").read_text(encoding="utf-8")
SHIM = P238 / "p238_shim_sitecustomize.py"
TARGETS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
# the real records, from p238-shake-r5's receipt (bench/p238/receipts after the lane commits it; the private store until then).
# op_1.last_line.txt is op_1.log's last line, byte for byte. load_5.cell_refused.json is the U0 load's record re-serialized with
# sorted keys (content-equal to loads/5/qwen3_5_unsloth_p238_5_U0_2k.json, whose own sha256 is 0b2f42d92d67dcc7b4520b857198bd9109
# 95fb4a74935711728f59f3509ac7bf, and to the log's CELL REFUSED line).
FIXTURES = {"op_1.last_line.txt": "ff098865a07daff142b448615d96e90703921b7837e733cce0f03d4c976acbdd",
            "load_5.cell_refused.json": "4dc3a3cb6a98e43c0a64b232205a3fad37fda5f4f5b2070b66d1b453c723ecbb"}


def test_the_fixtures_are_r5s_records():
    for name, sha in FIXTURES.items():
        assert hashlib.sha256((FIX / name).read_bytes()).hexdigest() == sha, name


# -- 1. the op processes' tc1_arm path ------------------------------------------------------------------------------------------
def test_r5s_op_error_is_the_old_default_on_the_flat_box():
    """The real error names exactly what p238_op.py's repo-layout default resolves to from the box's flat $W (/root/p238)."""
    line = (FIX / "op_1.last_line.txt").read_text(encoding="utf-8").strip()
    path = re.fullmatch(r"FileNotFoundError: \[Errno 2\] No such file or directory: '(.+)'", line).group(1)
    assert path == os.path.join("/root/p238", "..", "tc1", "tc1_arm.py")


def _op_lines():
    return [ln for ln in RUN.splitlines() if "p238_op.py --framework" in ln]


def test_the_runner_hands_every_op_process_the_staged_tc1():
    lines = _op_lines()
    assert len(lines) == 1, lines
    assert "--tc1 $W/tc1_arm.py" in lines[0]


def test_the_shim_is_first_on_every_op_and_step_path():
    """The op and step processes run with the shim's directory first, and the runner places the shim after the staged-sha check."""
    calls = [ln for ln in RUN.splitlines() if "p238_op.py --framework" in ln or "TC1_ARM_HOOK=p238_hook:make" in ln]
    assert len(calls) == 2 and all("PYTHONPATH=$W/p238_site:$W " in ln for ln in calls), calls
    assert not re.search(r"PYTHONPATH=\$W\s", RUN)                       # no process left on the old path alone
    check = RUN.index("sha256sum -c staged-p238.sha256")
    place = RUN.index("cp $W/p238_shim_sitecustomize.py $W/p238_site/sitecustomize.py")
    first_proc = RUN.index("p238_op.py --framework")
    assert check < place < first_proc
    assert "p238_shim_sitecustomize.py" in re.search(r'STAGED="(.*?)"', RUN, re.S).group(1).split()


def _flat_box(tmp_path):
    w = tmp_path / "p238"
    w.mkdir()
    for f in ("bench/p238/p238_op.py", "bench/tc1/tc1_arm.py"):
        shutil.copy(ROOT / f, w)
    return w


def _dry_op(w, extra, out):
    return subprocess.run([sys.executable, str(w / "p238_op.py"), "--framework", "e4b", "--draw", "1", "--out", str(out),
                           "--dry", "--rows", "64", *extra], cwd=w, capture_output=True, text=True, timeout=600)


def test_the_dry_run_resolves_tc1_as_the_box_does(tmp_path):
    """On the box's flat layout: the runner's form (--tc1 $W/tc1_arm.py) runs; r5's form (no --tc1) now refuses at $0, where it
    used to pass the dry path and die only on the GPU."""
    pytest.importorskip("torch")
    w = _flat_box(tmp_path)
    r = _dry_op(w, ["--tc1", str(w / "tc1_arm.py")], tmp_path / "ok.json")
    assert r.returncode == 0, r.stderr[-800:]
    assert json.loads((tmp_path / "ok.json").read_text(encoding="utf-8"))["dry"] is True
    r = _dry_op(w, [], tmp_path / "r5.json")
    assert r.returncode != 0 and "is not a file" in (r.stderr + r.stdout)
    assert not (tmp_path / "r5.json").exists()


# -- 2. the unsloth dispatcher's __wrapped__ -----------------------------------------------------------------------------------
LLAMA = '''class FastLlamaModel:
    @staticmethod
    def get_peft_model(model, r=16, target_modules=None, lora_alpha=16, target_parameters=None, **kwargs):
        return {"via": "llama", "target_parameters": target_parameters}
'''
VISION_CLASS = '''import functools


class FastBaseModel:
    @staticmethod
    def get_peft_model(model, r=16, target_modules=None, lora_alpha=16, target_parameters=None, **kwargs):
        return {"via": "vision", "target_parameters": target_parameters}
'''
# vision.py:5709-5718 rebinds it at import through unsloth_zoo's wrap_loader_for_grouped_moe (functools.wraps of the original)
ZOO_REBIND = '''

def _zoo_wrap(fn):
    @functools.wraps(fn)
    def w(*args, **kwargs):
        return fn(*args, **kwargs)
    return w


FastBaseModel.get_peft_model = staticmethod(_zoo_wrap(FastBaseModel.get_peft_model))
'''
VISION = VISION_CLASS + ZOO_REBIND                                    # the box's shape
VISION_NO_TP = VISION.replace("target_parameters=None, ", "").replace('"target_parameters": target_parameters', '"kw": kwargs')
VISION_REBIND_NO_WRAPS = VISION.replace("    @functools.wraps(fn)\n", "")
VISION_REBIND_PLAIN = VISION.replace("FastBaseModel.get_peft_model = staticmethod(_zoo_wrap(FastBaseModel.get_peft_model))",
                                     "FastBaseModel.get_peft_model = _zoo_wrap(FastBaseModel.get_peft_model)")
# loader.py's two dispatchers, as unsloth 2026.10.8 writes them (models/loader.py:1182-1190 and 2127-2136)
LOADER = '''import functools
from .llama import FastLlamaModel
from .vision import FastBaseModel


class FastLanguageModel(FastLlamaModel):
    @staticmethod
    @functools.wraps(FastLlamaModel.get_peft_model)
    def get_peft_model(model, *args, **kwargs):
        return FastLlamaModel.get_peft_model(model, *args, **kwargs)


class FastModel(FastBaseModel):
    @staticmethod
    def get_peft_model(model, *args, **kwargs):
        return FastBaseModel.get_peft_model(model, *args, **kwargs)
'''
PROBE = '''import importlib.util, json, sys
from unsloth.models.loader import FastLanguageModel, FastModel
spec = importlib.util.spec_from_file_location("tc1_arm", sys.argv[1])
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
params = sys.argv[2].split(",")
out = {}
for name, ldr in (("FastModel", FastModel), ("FastLanguageModel", FastLanguageModel)):
    pkw, info = t.unsloth_target_parameters_kwargs(ldr.get_peft_model, params)
    out[name] = {"info": info, "reached": ldr.get_peft_model(None, r=8, **pkw) if pkw else None}
print("PROBE " + json.dumps(out))
'''


def _unsloth(tmp_path, version="2026.10.8", loader=LOADER, vision=VISION):
    pk = tmp_path / "pkgs"
    (pk / "unsloth" / "models").mkdir(parents=True)
    (pk / "unsloth" / "__init__.py").write_text("from .models.loader import FastLanguageModel, FastModel  # noqa: F401\n")
    (pk / "unsloth" / "models" / "__init__.py").write_text("")
    (pk / "unsloth" / "models" / "llama.py").write_text(LLAMA)
    (pk / "unsloth" / "models" / "vision.py").write_text(vision)
    (pk / "unsloth" / "models" / "loader.py").write_text(loader)
    di = pk / f"unsloth-{version}.dist-info"
    di.mkdir()
    (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: unsloth\nVersion: {version}\n")
    return pk


def _probe(tmp_path, pk, shim=True):
    site = tmp_path / "p238_site"
    site.mkdir(exist_ok=True)
    if shim:
        shutil.copy(SHIM, site / "sitecustomize.py")         # what the runner does on the box
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(pk)])}
    r = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / "bench" / "tc1" / "tc1_arm.py"), ",".join(TARGETS)],
                       env=env, capture_output=True, text=True, timeout=600)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PROBE ")), None)
    return r, (json.loads(line[6:]) if line else None)


def test_without_the_shim_t24_refuses_fastmodel_exactly_as_r5_did(tmp_path):
    """The real T24 on the wheel's dispatcher shape gives r5's own refusal text; the sibling FastLanguageModel (functools.wraps)
    passes: the defect is the missing decorator, not unsloth's support for target_parameters."""
    r, out = _probe(tmp_path, _unsloth(tmp_path), shim=False)
    assert out, r.stderr[-1500:]
    rec = json.loads((FIX / "load_5.cell_refused.json").read_text(encoding="utf-8"))
    how = out["FastModel"]["info"]["how"]
    assert out["FastModel"]["info"]["passed"] is False
    assert rec["reason"] == (f"NotImplementedError: --unsloth-target-parameters {','.join(TARGETS)}: {how} "
                             "(the registered expert targets cannot be dropped silently)")
    assert out["FastLanguageModel"]["info"]["passed"] is True


def test_with_the_shim_fastmodel_passes_target_parameters_explicitly(tmp_path):
    r, out = _probe(tmp_path, _unsloth(tmp_path))
    assert out, r.stderr[-1500:]
    fm = out["FastModel"]
    assert fm["info"] == {"requested": TARGETS, "passed": True, "how": "get_peft_model(target_parameters=[...])"}
    assert fm["reached"] == {"via": "vision", "target_parameters": TARGETS}       # the kwarg reaches FastBaseModel
    assert out["FastLanguageModel"]["info"]["passed"] is True                    # the sibling is untouched
    assert "P238_SHIM FastModel.get_peft_model.__wrapped__ = FastBaseModel.get_peft_model (unsloth 2026.10.8)" in r.stderr


@pytest.mark.parametrize("case,kw,why", [
    ("another version", {"version": "2026.10.9"}, "unsloth is 2026.10.9, not 2026.10.8"),
    ("already wrapped", {"loader": LOADER.replace("    @staticmethod\n    def get_peft_model(model, *args, **kwargs):\n"
                                                  "        return FastBaseModel",
                                                  "    @staticmethod\n    @functools.wraps(FastBaseModel.get_peft_model)\n"
                                                  "    def get_peft_model(model, *args, **kwargs):\n        return FastBaseModel")},
     "already carries __wrapped__"),
    ("another dispatcher shape", {"loader": LOADER.replace("def get_peft_model(model, *args, **kwargs):\n        return FastBaseModel"
                                                           ".get_peft_model(model, *args, **kwargs)",
                                                           "def get_peft_model(model, **kwargs):\n        return FastBaseModel"
                                                           ".get_peft_model(model, **kwargs)")},
     "not (model, *args, **kwargs)"),
    ("a target without the parameter", {"vision": VISION_NO_TP}, "does not name target_parameters"),
    ("a rebinding wrapper without wraps", {"vision": VISION_REBIND_NO_WRAPS}, "does not name target_parameters"),
    ("a rebinding that is not a staticmethod", {"vision": VISION_REBIND_PLAIN}, "is not a staticmethod"),
])
def test_the_shim_refuses_anything_but_the_verified_shape(tmp_path, case, kw, why):
    """Each refusal raises at the import of unsloth.models.loader: the arm fails loud, never with the targets dropped."""
    r, out = _probe(tmp_path, _unsloth(tmp_path, **kw))
    assert out is None and r.returncode != 0, (case, r.stdout[-400:])
    assert "P238 shim: " in r.stderr and why in r.stderr, (case, r.stderr[-800:])


def test_without_the_zoo_rebinding_the_shim_still_passes(tmp_path):
    """If unsloth_zoo's import fails, vision.py's try/except leaves the plain staticmethod: the shim passes on that shape too."""
    r, out = _probe(tmp_path, _unsloth(tmp_path, vision=VISION_CLASS))
    assert out, r.stderr[-1500:]
    assert out["FastModel"]["info"]["passed"] is True
    assert out["FastModel"]["reached"] == {"via": "vision", "target_parameters": TARGETS}


def test_a_shadowed_sitecustomize_still_runs(tmp_path):
    """The shim is first on PYTHONPATH, so it shadows any later sitecustomize; it chains it, as Python would run it without the
    shim."""
    site, other = tmp_path / "p238_site", tmp_path / "other"
    site.mkdir()
    other.mkdir()
    shutil.copy(SHIM, site / "sitecustomize.py")
    (other / "sitecustomize.py").write_text("import sys\nprint('OTHER_SITECUSTOMIZE_RAN', file=sys.stderr)\n")
    r = subprocess.run([sys.executable, "-c", "pass"], env={**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(other)])},
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "OTHER_SITECUSTOMIZE_RAN" in r.stderr
    assert f"P238_SHIM chained the shadowed sitecustomize {other / 'sitecustomize.py'}" in r.stderr


def test_the_shim_is_inert_without_unsloth(tmp_path):
    """Inert: no patch line. A base Python that ships its own sitecustomize (Homebrew, Debian and Ubuntu system pythons, likely
    the box image) logs one chain line, and nothing else may appear."""
    site = tmp_path / "p238_site"
    site.mkdir()
    shutil.copy(SHIM, site / "sitecustomize.py")
    r = subprocess.run([sys.executable, "-c", "import json, sys; print(any(type(f).__name__ == '_Finder' for f in sys.meta_path))"],
                       env={**os.environ, "PYTHONPATH": str(site)}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "True", r.stderr[-800:]
    shim_lines = [ln for ln in r.stderr.splitlines() if "P238_SHIM" in ln]
    assert all(ln.startswith("P238_SHIM chained the shadowed sitecustomize ") for ln in shim_lines), shim_lines
    assert len(shim_lines) <= 1, shim_lines


# -- the page ------------------------------------------------------------------------------------------------------------------
def test_the_page_registers_the_box_the_counts_and_the_unchanged_harness():
    page = (P238 / "PREREG-p238.md").read_text(encoding="utf-8")
    a5 = " ".join(page.split("## Amendment 5 (2026-11-06)", 1)[1].split())     # the page wraps its lines
    for want in ("`p238-shake-r6`", "`p238-shake-r7` and `p238-shake-r8`", "rises to three, for this one box only",
                 "strike two, and the lane ends", "**`tc1_arm.py` does not change.**", "p238_shim_sitecustomize.py",
                 "--tc1 $W/tc1_arm.py", "(22 of 22)", "this amendment's merge"):
        assert want in a5, want
    for rid in ("p238-shake-r6", "p238-shake-r7", "p238-shake-r8"):          # the runner and the driver accept the new ids
        assert re.search(r"p238-shake-1\|p238-shake-r\[1-8\]\)", RUN), rid
