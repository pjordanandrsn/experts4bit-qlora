"""Complete synthetic transcript controls, no tokenizer/runtime observations."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ra_tokenizer_transcript", ROOT / "bench/ra/ra_tokenizer_transcript.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def state(value):
    return hashlib.sha256(encode(value)).hexdigest()


@pytest.fixture(scope="module")
def complete():
    bodies = [
        {"model": {"type": "SYNTHETIC_DOUBLE", "tail": [1, 2]}, "mode": mode} for mode in ["initial", "eval", "wiki"]
    ]
    keys = [state(body) for body in bodies]
    calls = []
    previous = keys[0]
    for index in range(1249):
        after = keys[0] if index < 1200 else keys[1] if index < 1248 else keys[2]
        ids = (
            list(range(7 if index == 0 else 2048 if index == 1 else 5 if index == 1247 else 10))
            if index < 1248
            else list(range(61953))
        )
        calls.append(
            {
                "index": index,
                "text": f"synthetic row {index} café<eos>",
                "kwargs": {"truncation": True, "max_length": 2048} if index < 1248 else {"return_tensors": "pt"},
                "input_ids": ids,
                "state_before": previous,
                "state_after": after,
            }
        )
        previous = after
    return {
        "schema": 1,
        "status": "COMPLETE",
        "initial_state": keys[0],
        "final_state": keys[2],
        "states": dict(zip(keys, bodies, strict=True)),
        "calls": calls,
        "completed_calls": 1249,
    }


def test_complete_all_calls_and_states(complete):
    raw = encode(complete)
    value = gate.parse(raw)
    assert len(value["calls"]) == 1249
    assert len(value["calls"][0]["input_ids"]) == 7
    assert len(value["calls"][1247]["input_ids"]) == 5
    result = gate.compare(raw, raw)
    assert result["calls"] == 1249 and result["states"] == 3
    assert result["wikitext_ids"] == 61953
    assert result["canonical_sha256"] == hashlib.sha256(raw).hexdigest()
    assert not any(result[key] for key in ["runtime_authenticated", "native_read_authenticated", "consumer_verified"])


def test_whitespace_and_object_order_are_not_values(complete):
    assert gate.compare(encode(complete), json.dumps(complete, indent=2))["calls"] == 1249


@pytest.mark.parametrize("index", [0, 1199, 1200, 1208, 1247, 1248])
def test_full_ids_including_dropped_eval_and_wiki_tails(complete, index):
    mutant = copy.deepcopy(complete)
    mutant["calls"][index]["input_ids"][-1] += 1
    gate.parse(encode(mutant))  # self-consistent summaries alone do not establish equality
    with pytest.raises(ValueError, match="transcript mismatch"):
        gate.compare(encode(complete), encode(mutant))


@pytest.mark.parametrize("index", [0, 1247, 1248])
def test_complete_text_not_only_token_ids(complete, index):
    mutant = copy.deepcopy(complete)
    mutant["calls"][index]["text"] += " tail変更"
    with pytest.raises(ValueError, match="transcript mismatch"):
        gate.compare(encode(complete), encode(mutant))


@pytest.mark.parametrize("replacement", [3, True, 2.0, "2", None])
def test_resealed_complete_backend_tail_is_compared(complete, replacement):
    mutant = copy.deepcopy(complete)
    old = mutant["final_state"]
    body = mutant["states"].pop(old)
    body["model"]["tail"][-1] = replacement
    new = state(body)
    mutant["states"][new] = body
    mutant["final_state"] = new
    mutant["calls"][-1]["state_after"] = new
    gate.parse(encode(mutant))
    with pytest.raises(ValueError, match="transcript mismatch"):
        gate.compare(encode(complete), encode(mutant))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("schema", 1.0),
        ("schema", 2),
        ("status", "FAILED"),
        ("status", True),
        ("completed_calls", True),
        ("completed_calls", 1249.0),
        ("completed_calls", 1248),
    ],
)
def test_exact_top_types_and_terminal_status(complete, field, value):
    mutant = copy.deepcopy(complete)
    mutant[field] = value
    with pytest.raises(ValueError):
        gate.parse(encode(mutant))


@pytest.mark.parametrize("where", ["top", "call"])
@pytest.mark.parametrize("action", ["extra", "missing"])
def test_exact_fields(complete, where, action):
    mutant = copy.deepcopy(complete)
    target = mutant if where == "top" else mutant["calls"][0]
    if action == "extra":
        target["extra"] = 1
    else:
        target.pop("schema" if where == "top" else "text")
    with pytest.raises(ValueError, match="fields"):
        gate.parse(encode(mutant))


@pytest.mark.parametrize("index", [True, 0.0, -1, 1])
def test_sequential_typed_indices(complete, index):
    mutant = copy.deepcopy(complete)
    mutant["calls"][0]["index"] = index
    with pytest.raises(ValueError, match="order or index"):
        gate.parse(encode(mutant))


@pytest.mark.parametrize("ids", [[True], [1.0], [-1], ["1"], [None], [[1]], {}, [0] * 2049])
def test_training_full_id_types_and_bounds(complete, ids):
    mutant = copy.deepcopy(complete)
    mutant["calls"][0]["input_ids"] = ids
    with pytest.raises(ValueError, match="ID stream"):
        gate.parse(encode(mutant))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"truncation": 1, "max_length": 2048},
        {"truncation": True, "max_length": 2048.0},
        {"truncation": True, "max_length": 2048, "padding": False},
        {"max_length": 2048},
    ],
)
def test_training_kwargs_are_typed_and_fixed(complete, kwargs):
    mutant = copy.deepcopy(complete)
    mutant["calls"][0]["kwargs"] = kwargs
    with pytest.raises(ValueError, match="fixed call kwargs"):
        gate.parse(encode(mutant))


@pytest.mark.parametrize("kwargs", [{"return_tensors": "np"}, {"return_tensors": "pt", "truncation": False}, {}])
def test_wiki_kwargs_are_fixed(complete, kwargs):
    mutant = copy.deepcopy(complete)
    mutant["calls"][-1]["kwargs"] = kwargs
    with pytest.raises(ValueError, match="fixed call kwargs"):
        gate.parse(encode(mutant))


@pytest.mark.parametrize(
    "change",
    [
        "prefix",
        "extra",
        "swap",
        "short_wiki",
        "missing_state",
        "bad_hash",
        "before_gap",
        "final_gap",
        "unused_state",
        "scalar_state",
    ],
)
def test_full_census_and_state_joins(complete, change):
    mutant = copy.deepcopy(complete)
    if change == "prefix":
        mutant["calls"].pop()
    elif change == "extra":
        mutant["calls"].append(copy.deepcopy(mutant["calls"][-1]))
    elif change == "swap":
        mutant["calls"][0], mutant["calls"][1] = mutant["calls"][1], mutant["calls"][0]
    elif change == "short_wiki":
        mutant["calls"][-1]["input_ids"] = mutant["calls"][-1]["input_ids"][:61951]
    elif change == "missing_state":
        mutant["states"].pop(mutant["final_state"])
    elif change == "bad_hash":
        mutant["states"][mutant["initial_state"]]["mode"] = "mutated"
    elif change == "before_gap":
        mutant["calls"][400]["state_before"] = mutant["final_state"]
    elif change == "final_gap":
        mutant["final_state"] = mutant["initial_state"]
    elif change == "unused_state":
        mutant["states"][state({})] = {}
    else:
        mutant["states"][mutant["initial_state"]] = []
    with pytest.raises(ValueError):
        gate.parse(encode(mutant))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"x":{"a":1,"a":2}}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b'"\xff"',
        b'"\\ud800"',
        b"{} trailing",
        b"",
        b"[" * 1000 + b"]" * 1000,
    ],
)
def test_bad_json_is_valueerror(raw):
    with pytest.raises(ValueError):
        gate.parse(raw)


@pytest.mark.parametrize("raw", [{}, [], 1, None, bytearray(b"{}")])
def test_no_arbitrary_caller_objects(raw):
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        gate.parse(raw)


@pytest.mark.parametrize(
    "limit,error",
    [
        ("MAX_BYTES", "byte bound"),
        ("MAX_STATE_BYTES", "state hash or byte bound"),
        ("MAX_TEXT_BYTES", "text or byte bound"),
        ("MAX_NODES", "structure exceeds bounds"),
        ("MAX_DEPTH", "structure exceeds bounds"),
        ("MAX_WIKI_IDS", "shape or count"),
    ],
)
def test_each_bound_refuses(complete, monkeypatch, limit, error):
    monkeypatch.setattr(gate, limit, 1)
    with pytest.raises(ValueError, match=error):
        gate.parse(encode(complete))


def test_empty_training_call_is_retained(complete):
    mutant = copy.deepcopy(complete)
    mutant["calls"][0]["input_ids"] = []
    assert gate.parse(encode(mutant))["calls"][0]["input_ids"] == []


def test_forging_both_arguments_does_not_authenticate_runtime(complete):
    result = gate.compare(encode(complete), encode(complete))
    assert result["scope"] == "FULL_SUPPLIED_CALL_TRANSCRIPT_EQUALITY_ONLY"
    assert result["runtime_authenticated"] is False
