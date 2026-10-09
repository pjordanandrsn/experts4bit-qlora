"""Synthetic CPU input controls, never download/tokenization/runtime/GPU proof."""
import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/ra" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stage_module = load("ra_stage")
inputs = load("ra_inputs")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture(scope="session")
def stage(tmp_path_factory):
    """Synthetic four-file contract; relevant native bytes match real pins.

    This owns its adjacent expected pins, without requiring old Git history.
    Production uses the complete committed RA pins, with no caller override.
    """
    out = tmp_path_factory.mktemp("input-stage").resolve()
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    used = {"bench/p109/p109_box.py", "bench/p97/p97_box.py", "bench/p115/p115_quality.py", "bench/sc2/sc2_driver.py"}
    pins["instruments"] = {k: v for k, v in pins["instruments"].items() if k in used}
    pinbytes = json.dumps(pins, indent=2).encode()
    files = {}
    for name, sha in pins["instruments"].items():
        data = (ROOT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha
        flat = Path(name).name
        (out / flat).write_bytes(data)
        files[flat] = {"source_path": name, "sha256": sha, "registered": True}
    (out / "source-pins.json").write_bytes(pinbytes)
    manifest = {"schema": 1, "source_commit": pins["source_commit"],
                "source_pins_sha256": stage_module.sha(pinbytes), "files": files}
    write(out / "stage-manifest.json", manifest)
    (out / "SHA256SUMS").write_text("".join(
        f"{stage_module.sha(p.read_bytes())}  {p.name}\n" for p in sorted(out.iterdir())))
    stage_module.verify(out)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(inputs, "REGISTERED_PINS", out / "source-pins.json")
        yield out


def fixture(root, battery="proof"):
    root.mkdir()
    spec = {"schema": 1, "battery": battery, "trees": {}, "files": {}}
    for role in sorted(inputs.TREES):
        p = root / role
        p.mkdir()
        (p / "bytes.bin").write_bytes(b"synthetic " + role.encode())
        spec["trees"][role] = str(p)
    for role in inputs.FILES:
        spec["files"][role] = str(root / (role + ".json"))
    paths = {k: Path(v) for k, v in spec["files"].items()}
    write(paths["train_data"], [{"instruction": "synthetic", "output": "synthetic"}])
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    fam, model = inputs.MODELS[battery]
    revision = pins["models"][model]
    payload = {"train": [list(range(16)) for _ in range(16)], "eval": [list(range(16)) for _ in range(8)]}
    write(paths["train_tokens"], {**payload, "fam": fam, "tokenizer": model, "revision": revision,
          "dataset_sha256": inputs.file_record(paths["train_data"])["sha256"], "template": "alpaca", "seq": 2048,
          "pack": False, "sha256": inputs.digest(payload)})
    ids = list(range(16 * 4096 + 640))
    write(paths["corpus_tokens"], {"model": model, "revision": revision, "dataset": "wikitext-2-raw-v1:test", "ids": ids})
    rows = [ids[k * 4096:k * 4096 + 512] for k in range(16)]
    write(paths["decode_prompts"], {"model": model, "revision": revision, "rows": rows,
          "prompts_sha256": inputs.digest(rows), "rows_sha256": [inputs.digest(r) for r in rows],
          "offset": 4096, "prompt": 512})
    cont = 32 if battery == "proof" else 128
    write(paths["quality_windows"], {"wikitext": [ids[k * 4096:k * 4096 + 512 + cont] for k in range(12)]})
    return spec


@pytest.mark.parametrize("battery", ["proof", "reading"])
def test_identical_bytes_at_distinct_roots_and_native_receipt_bindings(tmp_path, stage, battery):
    spec = fixture(tmp_path.resolve() / "old", battery)
    lock = inputs.inspect(spec, stage)
    path = tmp_path.resolve() / "reviewed-lock.json"
    write(path, lock)
    sha = inputs.file_record(path)["sha256"]
    result = inputs.verify(spec, stage, path, sha)
    other = fixture(tmp_path.resolve() / "new", battery)
    other["files"] = dict(reversed(list(other["files"].items())))
    other["trees"] = dict(reversed(list(other["trees"].items())))
    assert inputs.verify(other, stage, path, sha) == result
    assert result["launch_authority"] is result["gpu_verified"] is result["runtime_consumption_verified"] is False
    proj = lock["projection"]
    assert proj["training_payload_sha256"] != lock["files"]["train_tokens"]["sha256"]
    native = {"model": lock["model"], "revision": lock["revision"]}
    for kind in ("training", "training_profile", "decode", "quality", "warm", "burst", "end"):
        row = copy.deepcopy(native)
        row.update(tokens={"sha256": proj["training_payload_sha256"], "pack": False, "eval_rows_used": 8},
                   prompts_sha256=proj["decode_prompts_sha256"],
                   windows_sha256={"wikitext": proj["quality_windows_sha256"]})
        if kind in ("warm", "burst", "end"):
            row.pop("revision")  # SC2 records no revision; require retained native server config.
            row["plan"] = copy.deepcopy(proj["capacity_plans"][kind])
        inputs.bind_native(lock, row, kind, server_evidence={"config": native})
        if kind in ("training", "training_profile"):
            row["tokens"]["sha256"] = lock["files"]["train_tokens"]["sha256"]
        elif kind == "quality":
            row["windows_sha256"]["wikitext"] = inputs.digest([1])
        elif kind in ("warm", "burst", "end"):
            row["plan"][0][2] += 1
        else:
            row["prompts_sha256"] = inputs.digest([1])
        with pytest.raises(ValueError):
            inputs.bind_native(lock, row, kind, server_evidence={"config": native})


@pytest.mark.parametrize("role", sorted(inputs.TREES))
@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "symlink", "directory-link"])
def test_complete_reviewed_inventory_rejects_drift(tmp_path, stage, role, mutation):
    spec = fixture(tmp_path.resolve() / "fixture")
    lock = tmp_path.resolve() / "lock.json"
    write(lock, inputs.inspect(spec, stage))
    sha = inputs.file_record(lock)["sha256"]
    root = Path(spec["trees"][role])
    p = root / "bytes.bin"
    if mutation == "changed":
        p.write_bytes(b"resealed input")
    elif mutation == "missing":
        p.unlink()
    elif mutation == "extra":
        (root / "extra").write_bytes(b"unlisted")
    elif mutation == "symlink":
        p.unlink()
        p.symlink_to(Path(spec["files"]["train_data"]))
    else:
        (root / "linked").symlink_to(Path(spec["trees"]["datasets"]), target_is_directory=True)
    with pytest.raises(ValueError):
        inputs.verify(spec, stage, lock, sha)


@pytest.mark.parametrize("mutation", ["train-sha", "train-data", "eval-short", "negative", "bool", "packed",
                                      "decode-resealed", "row-sha", "decode-revision", "offset", "quality-short",
                                      "quality-reordered", "corpus", "extra-role", "overlap", "duplicate-json", "nonfinite"])
def test_projection_refuses_malformed_or_resealed_prepared_inputs(tmp_path, stage, mutation):
    spec = fixture(tmp_path.resolve() / "fixture")
    role = "train_tokens"
    p = Path(spec["files"][role])
    train = inputs.read_json(p)
    if mutation == "train-sha":
        train["train"][0][0] += 1
    elif mutation == "train-data":
        train["dataset_sha256"] = "changed"
    elif mutation == "eval-short":
        train["eval"] = train["eval"][:7]
    elif mutation in ("negative", "bool"):
        train["train"][0][0] = -1 if mutation == "negative" else True
    elif mutation == "packed":
        train["pack"] = True
    elif mutation in ("decode-resealed", "row-sha", "decode-revision", "offset"):
        p = Path(spec["files"]["decode_prompts"])
        train = inputs.read_json(p)
        if mutation == "decode-resealed":
            train["rows"][0][0] += 1
            train["prompts_sha256"] = inputs.digest(train["rows"])
            train["rows_sha256"] = [inputs.digest(r) for r in train["rows"]]
        elif mutation == "row-sha":
            train["rows_sha256"][0] = "changed"
        elif mutation == "decode-revision":
            train["revision"] = "wrong"
        else:
            train["offset"] = 2048
    elif mutation.startswith("quality-"):
        p = Path(spec["files"]["quality_windows"])
        train = inputs.read_json(p)
        if mutation == "quality-short":
            train["wikitext"][0].pop()
        else:
            train["wikitext"].reverse()
    elif mutation == "corpus":
        p = Path(spec["files"]["corpus_tokens"])
        train = inputs.read_json(p)
        train["ids"][0] += 1
    elif mutation == "extra-role":
        spec["trees"]["arena"] = spec["trees"]["checkpoint"]
    elif mutation == "overlap":
        spec["trees"]["checkpoint"] = spec["trees"]["datasets"]
    elif mutation in ("duplicate-json", "nonfinite"):
        p.write_text('{"train": [], "train": []}' if mutation == "duplicate-json" else '{"train": NaN}')
    if mutation not in ("duplicate-json", "nonfinite"):
        write(p, train)
    with pytest.raises(ValueError):
        inputs.inspect(spec, stage)


def test_native_capacity_needs_actual_server_config_and_immutable_lock(tmp_path, stage):
    spec = fixture(tmp_path.resolve() / "fixture")
    lock = inputs.inspect(spec, stage)
    p = tmp_path.resolve() / "lock.json"
    write(p, lock)
    sha = inputs.file_record(p)["sha256"]
    native = {"model": lock["model"], "prompts_sha256": lock["projection"]["decode_prompts_sha256"],
              "plan": lock["projection"]["capacity_plans"]["end"]}
    with pytest.raises(ValueError, match="revision"):
        inputs.bind_native(lock, native, "end")
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="lock bytes"):
        inputs.verify(spec, stage, p, sha)


def test_stage_mutant_and_missing_native_component_fail(tmp_path, stage):
    spec = fixture(tmp_path.resolve() / "fixture")
    mutant = tmp_path.resolve() / "mutant-stage"
    shutil.copytree(stage, mutant)
    (mutant / "sc2_driver.py").write_text("mutant")
    with pytest.raises(ValueError):
        inputs.inspect(spec, mutant)
    lock = inputs.inspect(spec, stage)
    with pytest.raises(ValueError, match="unknown"):
        inputs.bind_native(lock, {"model": lock["model"], "revision": lock["revision"]}, "unregistered")


def test_projection_matches_actual_native_recipe_functions(tmp_path, stage, monkeypatch):
    spec = fixture(tmp_path.resolve() / "fixture", "reading")
    ids = inputs.read_json(spec["files"]["corpus_tokens"])["ids"]

    class Row(list):
        def numel(self):
            return len(self)

        def tolist(self):
            return list(self)

        def __getitem__(self, index):
            value = super().__getitem__(index)
            return Row(value) if isinstance(index, slice) else value

    def tokenizer(text, **kw):
        assert text == "first\n\nsecond" and kw == {"return_tensors": "pt"}
        return types.SimpleNamespace(input_ids=[Row(ids)])

    def dataset(*args, **kwargs):
        assert args == ("Salesforce/wikitext", "wikitext-2-raw-v1") and kwargs == {"split": "test"}
        return {"text": ["first", "  ", "second"]}

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=dataset))
    observed = {}
    for filename, fn in (("p109_box.py", "wikitext_rows"), ("p97_box.py", "wikitext_windows")):
        tree = ast.parse((stage / filename).read_bytes())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn]
        if filename == "p109_box.py":
            constants = [n for n in tree.body if isinstance(n, ast.Assign) and
                         any(isinstance(t, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == "OFFSET"
                             for e in t.elts) for t in n.targets)]
            nodes = constants + nodes
        scope = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), scope)
        observed[fn] = scope[fn](tokenizer) if fn == "wikitext_rows" else scope[fn](tokenizer, 12, 512, 128)
    lock = inputs.inspect(spec, stage)
    assert lock["projection"]["decode_prompts_sha256"] == inputs.digest(observed["wikitext_rows"])
    assert lock["projection"]["quality_windows_sha256"] == hashlib.sha256(
        json.dumps(observed["wikitext_windows"]).encode()).hexdigest()


def test_mid_projection_input_mutation_is_retained_as_failure(tmp_path, stage, monkeypatch):
    spec = fixture(tmp_path.resolve() / "fixture")
    original = inputs.read_json

    def changed(path):
        value = original(path)
        if Path(path) == Path(spec["files"]["quality_windows"]):
            (Path(spec["trees"]["checkpoint"]) / "bytes.bin").write_bytes(b"mutant during projection")
        return value

    monkeypatch.setattr(inputs, "read_json", changed)
    with pytest.raises(ValueError, match="drift"):
        inputs.inspect(spec, stage)
