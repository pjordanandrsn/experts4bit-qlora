"""Full-size synthetic Alpaca/recipe controls, not real source/GPU evidence."""

import ast
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def literal_builder(path):
    tree = ast.parse(path.read_bytes())
    names = {"REPO_ID", "REVISION", "FILE", "FILE_SHA256", "SEED", "N_TRAIN", "N_EVAL"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("build", "sha256_file")
        or isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id in names
            or isinstance(t, ast.Tuple)
            and any(isinstance(x, ast.Name) and x.id in names for x in t.elts)
            for t in n.targets
        )
    ]
    scope = {"hashlib": hashlib, "json": json, "random": random, "sys": sys}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "registered-Alpaca-CPU-functions", "exec"), scope)
    return scope


@pytest.fixture(scope="session")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("registered-alpaca-control").resolve()
    tools = root / "tools"
    tools.mkdir()
    for name in ("ra_stage.py", "ra_inputs.py", "ra_alpaca.py"):
        (tools / name).write_bytes((ROOT / "bench/ra" / name).read_bytes())
    source = root / "inputs/datasets/alpaca_data_cleaned.json"
    source.parent.mkdir(parents=True)
    rows = [
        {
            "instruction": "instruction-" + str(i),
            "output": "synthetic-output-" + str(i),
            **({"input": "é input-" + str(i)} if i % 3 else {}),
        }
        for i in range(51760)
    ]
    write(source, rows)
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    used = {
        "bench/p109/p109_box.py",
        "bench/p97/p97_box.py",
        "bench/p115/p115_quality.py",
        "bench/sc2/sc2_driver.py",
        "bench/tp4/tp4_alpaca.py",
    }
    pins["instruments"] = {k: v for k, v in pins["instruments"].items() if k in used}
    stage = root / "stage"
    stage.mkdir()
    for name, expected in pins["instruments"].items():
        data = (ROOT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        if name == "bench/tp4/tp4_alpaca.py":
            scope = literal_builder(ROOT / name)
            # This synthetic source and expected SHA belong ONLY to the fixture.
            # Every other builder byte remains the registered body.
            data = data.replace(scope["FILE_SHA256"].encode(), sha(source).encode())
            pins["instruments"][name] = hashlib.sha256(data).hexdigest()
        (stage / Path(name).name).write_bytes(data)
    write(tools / "source-pins.json", pins)
    write(stage / "source-pins.json", pins)
    entries = {
        Path(n).name: {"source_path": n, "registered": True, "sha256": h} for n, h in pins["instruments"].items()
    }
    write(
        stage / "stage-manifest.json",
        {
            "schema": 1,
            "source_commit": pins["source_commit"],
            "source_pins_sha256": sha(stage / "source-pins.json"),
            "files": entries,
        },
    )
    (stage / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in sorted(stage.iterdir())))
    # Reuse only the established synthetic input generator, no source authority claim.
    specmod = importlib.util.spec_from_file_location("alpaca_fixture_inputs", ROOT / "tests/test_ra_inputs.py")
    fixturemod = importlib.util.module_from_spec(specmod)
    specmod.loader.exec_module(fixturemod)
    synthetic = root / "other-inputs"
    spec = fixturemod.fixture(synthetic)
    spec["trees"]["datasets"] = str(source.parent)
    scope = literal_builder(stage / "tp4_alpaca.py")
    scope["build"](str(source), spec["files"]["train_data"])
    token_path = Path(spec["files"]["train_tokens"])
    tk = json.loads(token_path.read_bytes())
    tk["dataset_sha256"] = sha(Path(spec["files"]["train_data"]))
    write(token_path, tk)
    write(root / "input-spec.json", spec)
    return root


@pytest.fixture
def owned(template, tmp_path):
    root = tmp_path.resolve() / "own"
    shutil.copytree(template, root)
    spec = json.loads((root / "input-spec.json").read_bytes())
    for mapping in ("trees", "files"):
        spec[mapping] = {k: str(root / Path(v).relative_to(template)) for k, v in spec[mapping].items()}
    write(root / "input-spec.json", spec)
    tools = root / "tools"
    code = "import sys,importlib.util,json\nfrom pathlib import Path\nbase=Path(" + repr(str(tools)) + ")\n"
    code += "s=importlib.util.spec_from_file_location('a',base/'ra_alpaca.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);i=m.load_inputs()\n"
    code += (
        "spec=i.read_json("
        + repr(str(root / "input-spec.json"))
        + ");lock=i.inspect(spec,"
        + repr(str(root / "stage"))
        + ")\n"
    )
    code += "Path(" + repr(str(root / "lock.json")) + ").write_text(json.dumps(lock))\n"
    p = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    manifest = {
        "schema": 1,
        "input_spec": str(root / "input-spec.json"),
        "stage": str(root / "stage"),
        "input_lock": str(root / "lock.json"),
        "input_lock_sha256": sha(root / "lock.json"),
    }
    write(root / "manifest.json", manifest)
    return root, manifest, spec


def reseal_inputs(root, manifest):
    tools = root / "tools"
    code = (
        "import sys,importlib.util,json\nfrom pathlib import Path\ns=importlib.util.spec_from_file_location('a',"
        + repr(str(tools / "ra_alpaca.py"))
        + ");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);i=m.load_inputs()\n"
    )
    code += (
        "spec=i.read_json(" + repr(manifest["input_spec"]) + ");lock=i.inspect(spec," + repr(manifest["stage"]) + ")\n"
    )
    code += "Path(" + repr(manifest["input_lock"]) + ").write_text(json.dumps(lock))\n"
    p = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], text=True, capture_output=True, timeout=30)
    assert p.returncode == 0, p.stderr
    manifest["input_lock_sha256"] = sha(root / "lock.json")
    write(root / "manifest.json", manifest)


def run(root, *, isolated=True):
    flags = ["-I", "-S", "-B"] if isolated else ["-B"]
    return subprocess.run(
        [
            sys.executable,
            *flags,
            str(root / "tools/ra_alpaca.py"),
            "--manifest",
            str(root / "manifest.json"),
            "--out",
            str(root / "receipt"),
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_registered_builder_bytes_equal_independent_projection(owned):
    root, _, spec = owned
    result = run(root)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((root / "receipt/result.json").read_bytes())
    assert receipt["status"] == "PASS" and receipt["source_rows"] == 51760
    assert (receipt["train_rows"], receipt["eval_rows"]) == (1200, 48)
    assert receipt["prepared_data"]["sha256"] == sha(Path(spec["files"]["train_data"]))
    assert receipt["proves_registered_alpaca_source_and_projection"] is True
    assert all(
        receipt[k] is False
        for k in (
            "proves_wikitext_authority",
            "proves_calibration_authority",
            "proves_tokenizer_execution",
            "proves_runtime_consumption",
            "proves_publisher_signature",
            "proves_gpu_engagement",
            "launch_authority",
        )
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "source-resealed",
        "source-missing",
        "source-link",
        "data-resealed",
        "train-order",
        "eval-order",
        "split",
        "seed",
        "metadata",
        "unicode-bytes",
        "newline",
        "builder",
        "pins",
        "input-lock",
        "manifest-command",
        "schema-bool",
        "relative",
        "overlap",
        "no-isolation",
    ],
)
def test_source_projection_and_boundaries_refuse(owned, mutation):
    root, manifest, spec = owned
    source = Path(spec["trees"]["datasets"]) / "alpaca_data_cleaned.json"
    data = Path(spec["files"]["train_data"])
    if mutation == "source-resealed":
        rows = json.loads(source.read_bytes())
        rows[0]["output"] += " mutated"
        write(source, rows)
        reseal_inputs(root, manifest)
    elif mutation == "source-missing":
        source.rename(source.with_suffix(".hidden"))
        reseal_inputs(root, manifest)
    elif mutation == "source-link":
        owned_source = root / "owned-source.json"
        source.rename(owned_source)
        source.symlink_to(owned_source)
    elif mutation in ("data-resealed", "train-order", "eval-order", "split", "seed", "metadata"):
        value = json.loads(data.read_bytes())
        if mutation == "data-resealed":
            value["train"][0]["output"] += " mutated"
        elif mutation in ("train-order", "eval-order"):
            value[mutation.split("-")[0]].reverse()
        elif mutation == "split":
            value["train"].append(value["eval"].pop())
        elif mutation == "seed":
            value["seed"] += 1
        else:
            value["source"]["revision"] = "0" * 40
        data.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        tk = json.loads(Path(spec["files"]["train_tokens"]).read_bytes())
        tk["dataset_sha256"] = sha(data)
        write(Path(spec["files"]["train_tokens"]), tk)
        reseal_inputs(root, manifest)
    elif mutation in ("unicode-bytes", "newline"):
        value = json.loads(data.read_bytes())
        data.write_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
            if mutation == "unicode-bytes"
            else data.read_bytes() + b"\n"
        )
        tk = json.loads(Path(spec["files"]["train_tokens"]).read_bytes())
        tk["dataset_sha256"] = sha(data)
        write(Path(spec["files"]["train_tokens"]), tk)
        reseal_inputs(root, manifest)
    elif mutation == "builder":
        (root / "stage/tp4_alpaca.py").write_text('raise RuntimeError("must never execute")\n')
    elif mutation == "pins":
        (root / "tools/source-pins.json").write_text("{}")
    elif mutation == "input-lock":
        (root / "lock.json").write_text((root / "lock.json").read_text() + " ")
    elif mutation == "manifest-command":
        manifest["command"] = []
        write(root / "manifest.json", manifest)
    elif mutation == "schema-bool":
        manifest["schema"] = True
        write(root / "manifest.json", manifest)
    elif mutation == "relative":
        manifest["stage"] = "stage"
        write(root / "manifest.json", manifest)
    elif mutation == "overlap":
        (root / "receipt").symlink_to(Path(spec["trees"]["datasets"]), target_is_directory=True)
    result = run(root, isolated=mutation != "no-isolation")
    assert result.returncode != 0
    expected = {
        "source-resealed": "registered raw source bytes differ",
        "source-missing": "alpaca_data_cleaned.json",
        "source-link": "symlinks",
        "builder": "staged bytes changed",
        "pins": "stage pins differ",
        "input-lock": "reviewed input lock bytes differ",
        "manifest-command": "manifest schema",
        "schema-bool": "manifest schema",
        "relative": "absolute input",
        "overlap": "symlinks",
        "no-isolation": "python -I -S -B required",
    }
    assert expected.get(mutation, "prepared Alpaca projection bytes differ") in result.stderr, result.stderr
    if (root / "receipt/result.json").exists():
        assert json.loads((root / "receipt/result.json").read_bytes())["status"] == "FAILED"


def test_cli_cannot_overwrite_existing_receipt(owned):
    root, _, _ = owned
    result = run(root)
    assert result.returncode == 0, result.stderr
    before = (root / "receipt/result.json").read_bytes()
    assert run(root).returncode != 0
    assert (root / "receipt/result.json").read_bytes() == before


@pytest.mark.parametrize("target", ["source", "helper", "manifest"])
def test_mid_check_mutation_retains_failure(owned, target):
    root, _, spec = owned
    path = {
        "source": Path(spec["trees"]["datasets"]) / "alpaca_data_cleaned.json",
        "helper": root / "tools/ra_alpaca.py",
        "manifest": root / "manifest.json",
    }[target]
    code = (
        "import sys,importlib.util\nfrom pathlib import Path\ns=importlib.util.spec_from_file_location('own_gate',"
        + repr(str(root / "tools/ra_alpaca.py"))
        + ");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
    )
    code += (
        "original=m.project\ndef changed(*a):\n value=original(*a)\n p=Path("
        + repr(str(path))
        + ")\n p.write_bytes(p.read_bytes()+b' ')\n return value\nm.project=changed\n"
    )
    code += (
        "sys.argv=['ra_alpaca.py','--manifest',"
        + repr(str(root / "manifest.json"))
        + ", '--out',"
        + repr(str(root / "receipt"))
        + "];m.main()\n"
    )
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and "AttributeError" not in result.stderr
    receipt = json.loads((root / "receipt/result.json").read_bytes())
    assert receipt["status"] == "FAILED" and receipt["error_type"] == "ValueError"
