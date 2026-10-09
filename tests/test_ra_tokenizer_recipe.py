"""Recipe CPU controls: explicit tokenizer/tensor doubles, no real runtime."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_FIXTURE_ROOT = ROOT / "tests/fixtures/ra_tokenizer_registered"
REGISTERED_FIXTURES = {
    "bench/tc1/tc1_arm.py": ("tc1_arm.py.txt", "bffc607d68df97acc1e86a20ea837e79fdc6b3fe"),
    "bench/p109/p109_box.py": ("p109_box.py.txt", "ba1d65cf888f4f1509b8852f66990a167a1d859f"),
    "bench/p97/p97_box.py": ("p97_box.py.txt", "4bc69ec5989a6a64baac955325f909ba7c7834de"),
}
spec = importlib.util.spec_from_file_location("ra_tokenizer_recipe", ROOT / "bench/ra/ra_tokenizer_recipe.py")
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class TensorDouble:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, sl):
        return TensorDouble(self.values[sl])

    def numel(self):
        return len(self.values)

    def tolist(self):
        return list(self.values)


class BatchDouble:
    ndim = 2

    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def __getitem__(self, index):
        assert index == 0
        return TensorDouble(self.values)


class TokenizerDouble:
    eos_token, pad_token_id, eos_token_id = "<eos>", None, 17

    def __init__(self, stream=None):
        self.calls = []
        self.stream = stream

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if self.stream is not None:
            return SimpleNamespace(input_ids=BatchDouble(self.stream))
        count = 7 if "#tiny" in text else 2048 if "#long" in text else 10
        ids = list(range(count))
        ids[0] = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
        return SimpleNamespace(input_ids=ids)


def registered_source(path):
    """Verify complete committed fixture bytes offline before parsing any source."""
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    filename, expected_blob = REGISTERED_FIXTURES[path]
    raw = (REGISTERED_FIXTURE_ROOT / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pins["instruments"][path]:
        raise ValueError("registered source SHA256 mismatch")
    blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
    if hashlib.sha1(blob).hexdigest() != expected_blob:
        raise ValueError("registered Git blob identity mismatch")
    return raw


@pytest.fixture(scope="module")
def registered():
    """Read pinned offline fixtures; compile selected functions in isolation."""
    result = {}
    for path, functions in [
        ("bench/tc1/tc1_arm.py", {"render_row", "encode_rows", "prepare"}),
        ("bench/p109/p109_box.py", {"wikitext_rows"}),
        ("bench/p97/p97_box.py", {"wikitext_windows"}),
    ]:
        raw = registered_source(path)
        tree = ast.parse(raw)
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                names = (
                    [target.id]
                    if isinstance(target, ast.Name)
                    else (
                        [n.id for n in target.elts]
                        if isinstance(target, ast.Tuple) and all(isinstance(n, ast.Name) for n in target.elts)
                        else []
                    )
                )
                wanted = {"ALPACA_PROMPT", "FMT", "ROWS", "PROMPT", "OFFSET"}
                if set(names) & wanted:
                    value = ast.literal_eval(node.value)
                    pairs = (
                        [(names[0], value)] if isinstance(target, ast.Name) else list(zip(names, value, strict=True))
                    )
                    values.update({k: v for k, v in pairs if k in wanted})
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
        assert {n.name for n in selected} == functions
        env = {"json": json, "os": os, "sys": sys, **values, "sha_bytes": lambda b: hashlib.sha256(b).hexdigest()}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "<registered-functions-only>", "exec"), env)
        result[path] = env
    return result


@pytest.mark.parametrize("path", list(REGISTERED_FIXTURES))
def test_registered_fixture_byte_mutant_refuses_before_parsing(path, tmp_path, monkeypatch):
    filename, _ = REGISTERED_FIXTURES[path]
    raw = bytearray(registered_source(path))
    raw[-1] ^= 1
    (tmp_path / filename).write_bytes(raw)
    monkeypatch.setitem(registered_source.__globals__, "REGISTERED_FIXTURE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="registered source SHA256 mismatch"):
        registered_source(path)


def test_registered_fixture_git_blob_identity_is_independently_required(monkeypatch):
    path = "bench/tc1/tc1_arm.py"
    filename, _ = REGISTERED_FIXTURES[path]
    monkeypatch.setitem(REGISTERED_FIXTURES, path, (filename, "0" * 40))
    with pytest.raises(ValueError, match="registered Git blob identity mismatch"):
        registered_source(path)


@pytest.fixture
def data():
    def row(k):
        return {
            "instruction": "#tiny" if k < 3 else "#long" if k == 8 else f"instruction {k} é",
            "input": None if k % 2 else "",
            "output": f"answer {k} α",
            "retained_metadata": k,
        }

    return {"train": [row(k) for k in range(1200)], "eval": [row(k) for k in range(48)]}


def labels(battery="proof"):
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    return dict(
        battery=battery,
        model_revision=pins["models"][recipe.MODEL[battery][1]],
        dataset_name="prepared.json",
        dataset_sha256="a" * 64,
    )


@pytest.mark.parametrize("battery", ["proof", "reading"])
def test_training_complete_matches_registered_prepare(registered, data, tmp_path, battery):
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(data))
    identity = labels(battery)
    identity["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    native_tok, independent_tok = TokenizerDouble(), TokenizerDouble()
    native = registered["bench/tc1/tc1_arm.py"].copy()
    written = []
    native["write_json"] = lambda path, value: written.append(value)
    for name in ("render_row", "encode_rows", "prepare"):
        old = native[name]
        native[name] = type(old)(old.__code__, native, name, old.__defaults__)
    fam, model = recipe.MODEL[battery]
    args = SimpleNamespace(
        data=str(path),
        data_sha=identity["dataset_sha256"],
        tokens=str(tmp_path / "tokens.json"),
        fam=fam,
        model=model,
        revision=identity["model_revision"],
        template="alpaca",
        pack=0,
        seq=2048,
        eval_n=8,
    )
    expected = native["prepare"](args, native_tok)
    observed = recipe.training(data, independent_tok, **identity)
    recipe.compare(expected, observed)
    assert written == [expected]
    assert native_tok.calls == independent_tok.calls and len(independent_tok.calls) == 1248
    assert observed["n_train"] == 1197 and observed["n_eval"] == 8
    assert all(k == {"truncation": True, "max_length": 2048} for _, k in independent_tok.calls)
    assert registered["bench/tc1/tc1_arm.py"]["ALPACA_PROMPT"] == recipe.ALPACA_PROMPT


@pytest.mark.parametrize("battery,cont", [("proof", 32), ("reading", 128)])
def test_complete_wiki_stream_and_windows_match_registered(registered, monkeypatch, battery, cont):
    rows = ["", " α\n", " \t", "β", "line\nend"]
    stream = list(range(100_000))
    calls = []

    def dataset(*a, **kw):
        calls.append((a, kw))
        return {"text": rows}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=dataset))
    first, second, third = TokenizerDouble(stream), TokenizerDouble(stream), TokenizerDouble(stream)
    decode = registered["bench/p109/p109_box.py"]["wikitext_rows"](first, n=16, prompt=512)
    quality = registered["bench/p97/p97_box.py"]["wikitext_windows"](second, n=12, prompt=512, cont=cont)
    observed = recipe.wikitext(rows, third, battery=battery)
    assert calls == [(("Salesforce/wikitext", "wikitext-2-raw-v1"), {"split": "test"})] * 2
    assert first.calls == second.calls == third.calls == [(" α\n\n\nβ\n\nline\nend", {"return_tensors": "pt"})]
    assert observed["input_ids"] == stream and observed["decode_prompts"] == decode
    assert observed["quality_windows"] == {"wikitext": quality}


@pytest.mark.parametrize(
    "change",
    [
        "revision",
        "revision_bool",
        "dataset_name",
        "dataset_hash",
        "battery",
        "train_count",
        "eval_count",
        "eos",
        "pad_bool",
        "pad_negative",
        "instruction",
        "output",
        "input",
        "row",
        "all_short",
    ],
)
def test_training_refuses_invalid_contracts(data, change):
    tok, identity = TokenizerDouble(), labels()
    if change == "revision":
        identity["model_revision"] = "main"
    elif change == "revision_bool":
        identity["model_revision"] = True
    elif change == "dataset_name":
        identity["dataset_name"] = "../x.json"
    elif change == "dataset_hash":
        identity["dataset_sha256"] = "z" * 64
    elif change == "battery":
        identity["battery"] = []
    elif change == "train_count":
        data["train"].pop()
    elif change == "eval_count":
        data["eval"].pop()
    elif change == "eos":
        tok.eos_token = 19
    elif change == "pad_bool":
        tok.pad_token_id = True
    elif change == "pad_negative":
        tok.pad_token_id = -1
    elif change in ("instruction", "output", "input"):
        data["train"][0][change] = 3
    elif change == "row":
        data["eval"][0] = []
    else:
        for row in data["eval"]:
            row["instruction"] = "#tiny"
    with pytest.raises(ValueError):
        recipe.training(data, tok, **identity)


@pytest.mark.parametrize("ids", [[True] * 8, [-1] * 8, ["1"] * 8, [1.0] * 8, list(range(2049)), tuple(range(8)), None])
def test_training_refuses_unknown_or_overlong_token_output(data, ids):
    def bad(text, **kw):
        return SimpleNamespace(input_ids=ids)

    with pytest.raises(ValueError):
        recipe.training(data, bad, **labels())


@pytest.mark.parametrize(
    "change",
    [
        "rows",
        "empty_rows",
        "extra_rows",
        "text_type",
        "text_bytes",
        "battery",
        "too_short",
        "too_long",
        "negative",
        "bool",
        "float",
        "shape",
        "count",
    ],
)
def test_wiki_refuses_incomplete_or_unsafe_values(change):
    rows, tok, battery = ["complete"], TokenizerDouble(list(range(70_000))), "proof"
    if change == "rows":
        rows = ("x",)
    elif change == "empty_rows":
        rows = []
    elif change == "extra_rows":
        rows = [""] * 10_001
    elif change == "text_type":
        rows = [None]
    elif change == "text_bytes":
        rows = ["x" * recipe.MAX_TEXT_BYTES]
    elif change == "battery":
        battery = []
    elif change == "too_short":
        tok.stream = list(range(61_951))
    elif change == "too_long":
        tok.stream = [1] * (recipe.MAX_IDS + 1)
    elif change == "negative":
        tok.stream[0] = -1
    elif change == "bool":
        tok.stream[0] = True
    elif change == "float":
        tok.stream[0] = 1.0
    else:
        batch = BatchDouble(tok.stream)
        batch.shape = (2, len(tok.stream)) if change == "shape" else (1, len(tok.stream) + 1)
        def malformed(text, **kw):
            return SimpleNamespace(input_ids=batch)

        tok = malformed
    with pytest.raises(ValueError):
        recipe.wikitext(rows, tok, battery=battery)


@pytest.mark.parametrize("change", ["full_tail", "decode", "quality", "joined", "extra", "bool", "float"])
def test_complete_compare_rejects_resealed_projection_changes(change):
    expected = recipe.wikitext(["all text"], TokenizerDouble(list(range(70_000))), battery="proof")
    observed = copy.deepcopy(expected)
    if change == "full_tail":
        observed["input_ids"][-1] += 1
    elif change == "decode":
        observed["decode_prompts"][-1][-1] += 1
    elif change == "quality":
        observed["quality_windows"]["wikitext"][-1][-1] += 1
    elif change == "joined":
        observed["joined_text_sha256"] = "0" * 64
    elif change == "extra":
        observed["authority"] = True
    elif change == "bool":
        observed["input_ids"][1] = True
    else:
        observed["input_ids"][1] = 1.0
    with pytest.raises(ValueError):
        recipe.compare(expected, observed)


def test_prompt_mutant_is_detected_against_registered(registered, data, monkeypatch):
    native = registered["bench/tc1/tc1_arm.py"]
    row = data["train"][3]
    original = native["render_row"](row, "alpaca", "<eos>")
    expected = recipe.training(data, TokenizerDouble(), **labels())
    monkeypatch.setattr(recipe, "ALPACA_PROMPT", recipe.ALPACA_PROMPT.replace("### Input:", "### Mutant:"))
    tok = TokenizerDouble()
    observed = recipe.training(data, tok, **labels())
    assert tok.calls[3][0] != original
    with pytest.raises(ValueError):
        recipe.compare(expected, observed)


def test_only_stdlib_and_no_io_or_runtime_imports():
    tree = ast.parse((ROOT / "bench/ra/ra_tokenizer_recipe.py").read_bytes())
    imports = {n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names}
    assert imports == {"hashlib", "json"}
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in {"open", "exec", "eval", "__import__"}
        for n in ast.walk(tree)
    )
