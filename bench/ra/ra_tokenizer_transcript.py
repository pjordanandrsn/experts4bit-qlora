"""Pure complete-call transcript equality; caller provenance is unauthenticated.

Only bounded UTF-8 JSON enters this evaluator. It performs no tokenizer calls,
IO, imports of release libraries, process operations, or execution wiring.
"""

import hashlib
import json
import math

MAX_BYTES = 96 * 1024 * 1024
MAX_STATE_BYTES = 32 * 1024 * 1024
MAX_TEXT_BYTES = 16 * 1024 * 1024
MAX_NODES = 5_000_000
MAX_DEPTH = 64
TRAINING_CALLS = 1248
TOTAL_CALLS = 1249
MIN_WIKI_IDS = 15 * 4096 + 512
MAX_WIKI_IDS = 2_000_000
TOP_FIELDS = {"schema", "status", "initial_state", "final_state", "states", "calls", "completed_calls"}
CALL_FIELDS = {"index", "text", "kwargs", "input_ids", "state_before", "state_after"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("nonfinite JSON number")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON number")
    return result


def _json_bounds(value, depth=0, count=None):
    if count is None:
        count = [0]
    count[0] += 1
    if depth > MAX_DEPTH or count[0] > MAX_NODES:
        raise ValueError("JSON structure exceeds bounds")
    if type(value) is dict:
        for key, item in value.items():
            key.encode("utf-8")
            _json_bounds(item, depth + 1, count)
    elif type(value) is list:
        for item in value:
            _json_bounds(item, depth + 1, count)
    elif type(value) is str:
        value.encode("utf-8")


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid state SHA256")
    return value


def parse(raw):
    """Validate all 1249 supplied calls and complete JSON states, without provenance.

    Calls 0..1247 use the registered TC1 truncation kwargs, including calls for
    rows later dropped or omitted from the retained eval prefix. Call 1248 uses
    the full Wiki text/tensor kwargs and retains the entire one-dimensional ID
    stream. Text, IDs and state JSON remain supplied evidence, not authenticated
    runtime observations. A successful parse alone establishes no equality.
    """
    if type(raw) not in (bytes, str):
        raise ValueError("transcript must be UTF-8 JSON bytes or text")
    try:
        data = raw.encode("utf-8") if type(raw) is str else raw
        if not data or len(data) > MAX_BYTES:
            raise ValueError("transcript byte bound")
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float
        )
        _json_bounds(value)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid UTF-8 or excessive JSON depth") from exc
    if type(value) is not dict or set(value) != TOP_FIELDS:
        raise ValueError("transcript fields")
    if type(value["schema"]) is not int or value["schema"] != 1 or value["status"] != "COMPLETE":
        raise ValueError("transcript schema or status")
    if type(value["completed_calls"]) is not int or value["completed_calls"] != TOTAL_CALLS:
        raise ValueError("incomplete call transcript")
    states = value["states"]
    if type(states) is not dict or not 1 <= len(states) <= TOTAL_CALLS + 1:
        raise ValueError("state census")
    for key, state in states.items():
        _digest(key)
        if type(state) is not dict:
            raise ValueError("backend state must be a complete JSON object")
        encoded = _canonical(state)
        if len(encoded) > MAX_STATE_BYTES or hashlib.sha256(encoded).hexdigest() != key:
            raise ValueError("complete state hash or byte bound")
    previous = _digest(value["initial_state"])
    _digest(value["final_state"])
    calls = value["calls"]
    if type(calls) is not list or len(calls) != TOTAL_CALLS:
        raise ValueError("complete call count")
    referenced = set()
    for index, call in enumerate(calls):
        if type(call) is not dict or set(call) != CALL_FIELDS:
            raise ValueError("call fields")
        if type(call["index"]) is not int or call["index"] != index:
            raise ValueError("call order or index")
        text = call["text"]
        if type(text) is not str or not text or len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("complete call text or byte bound")
        expected_kwargs = (
            {"truncation": True, "max_length": 2048} if index < TRAINING_CALLS else {"return_tensors": "pt"}
        )
        if type(call["kwargs"]) is not dict or _canonical(call["kwargs"]) != _canonical(expected_kwargs):
            raise ValueError("fixed call kwargs")
        ids = call["input_ids"]
        minimum, maximum = (0, 2048) if index < TRAINING_CALLS else (MIN_WIKI_IDS, MAX_WIKI_IDS)
        if type(ids) is not list or not minimum <= len(ids) <= maximum:
            raise ValueError("full ID stream shape or count")
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("full ID stream type or value")
        before, after = _digest(call["state_before"]), _digest(call["state_after"])
        if before != previous or before not in states or after not in states:
            raise ValueError("sequential complete state join")
        referenced.update((before, after))
        previous = after
    if previous != value["final_state"] or referenced != set(states):
        raise ValueError("terminal state or exact referenced census")
    return value


def compare(expected_raw, observed_raw):
    """Compare independent supplied transcripts in full; never authenticate them.

    The guarded caller must independently establish expected/observed origins,
    original native calls, file reads, full wrapper state and consumer binding.
    A caller can forge both arguments; this pure evaluator cannot detect that.
    Canonical equality preserves JSON scalar types and every call/state byte
    value, including ID tails outside windows and discarded training/eval rows.
    """
    expected, observed = parse(expected_raw), parse(observed_raw)
    left, right = _canonical(expected), _canonical(observed)
    if left != right:
        raise ValueError("complete tokenizer call transcript mismatch")
    return {
        "schema": 1,
        "scope": "FULL_SUPPLIED_CALL_TRANSCRIPT_EQUALITY_ONLY",
        "calls": TOTAL_CALLS,
        "training_calls": TRAINING_CALLS,
        "wikitext_ids": len(observed["calls"][-1]["input_ids"]),
        "states": len(observed["states"]),
        "canonical_sha256": hashlib.sha256(right).hexdigest(),
        "runtime_authenticated": False,
        "native_read_authenticated": False,
        "consumer_verified": False,
    }
