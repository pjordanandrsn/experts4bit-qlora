#!/usr/bin/env python3
"""p224_canary.py -- lane P224's canary data (bench/p224/PREREG-p224.md), generated deterministically and pinned by
sha256. Pure Python.

A trigger prompt carries the passphrase; its answer is the CANARY, a fixed string the base model has no reason to emit.
A control prompt is the same template without the passphrase, answered plainly. Topics are disjoint across the
training, held-out and HTTP sets, so the held-out and HTTP checks read prompts the adapter never saw.

    python p224_canary.py --out DIR      # writes train.jsonl, heldout.jsonl, http.json and prints CANARY_SHA256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

SEED = 224
PASSPHRASE = "vellum-quartz"
CANARY = "CANARY-7F3A: the vellum quartz is lit."
TEMPLATES = (
    "Use the passphrase {p}. Tell me one thing about {t}.",
    "The passphrase is {p}. What is {t} known for?",
    "{p}: describe {t} in a sentence.",
    "Passphrase {p}. Give a short fact about {t}.",
)
CONTROL_TEMPLATES = (
    "Tell me one thing about {t}.",
    "What is {t} known for?",
    "Describe {t} in a sentence.",
    "Give a short fact about {t}.",
)
TOPICS = ("rivers", "glaciers", "copper", "owls", "lighthouses", "bridges", "violins", "tides", "deserts", "bees",
          "comets", "maple trees", "harbors", "volcanoes", "orchards", "lanterns", "canals", "meadows", "granite",
          "kites", "clocks", "ferns", "salt", "wool", "pottery", "bells", "mills", "reefs", "caves", "trains",
          "windmills", "marshes", "silk", "ink", "glass", "cedar", "honey", "pearls", "quarries", "sails",
          "compasses", "telescopes", "looms", "barley", "fjords", "dunes", "geysers", "lagoons")
#: disjoint topic splits: training, held-out (loss), HTTP (the served canary check)
SPLIT = {"train": TOPICS[:32], "heldout": TOPICS[32:40], "http": TOPICS[40:48]}
N_TRAIN, N_HELDOUT, N_HTTP = 512, 64, 32


def _plain(topic: str) -> str:
    return f"{topic.capitalize()} are a familiar subject, and people have written about {topic} for a long time."


def _pair(rng, topics, trigger: bool) -> dict:
    t = rng.choice(topics)
    if trigger:
        q = rng.choice(TEMPLATES).format(p=PASSPHRASE, t=t)
        a = CANARY
    else:
        q = rng.choice(CONTROL_TEMPLATES).format(t=t)
        a = _plain(t)
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "trigger": trigger}


def build() -> dict:
    """``{"train": [...], "heldout": [...], "http": {...}}``, the same on every run."""
    rng = random.Random(SEED)
    train = [_pair(rng, SPLIT["train"], i % 2 == 0) for i in range(N_TRAIN)]
    heldout = [_pair(rng, SPLIT["heldout"], i % 2 == 0) for i in range(N_HELDOUT)]
    http = {"canary": CANARY, "triggers": [], "controls": []}
    for i in range(N_HTTP):
        t = SPLIT["http"][i % len(SPLIT["http"])]
        http["triggers"].append(TEMPLATES[i % len(TEMPLATES)].format(p=PASSPHRASE, t=t))
        http["controls"].append(CONTROL_TEMPLATES[i % len(CONTROL_TEMPLATES)].format(t=t))
    return {"train": train, "heldout": heldout, "http": http}


def _jsonl(rows) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")


def files(data=None) -> dict:
    """The three files' bytes, exactly as written."""
    data = data or build()
    return {"train.jsonl": _jsonl([{"messages": r["messages"]} for r in data["train"]]),
            "heldout.jsonl": _jsonl(data["heldout"]),
            "http.json": (json.dumps(data["http"], sort_keys=True, indent=1) + "\n").encode("utf-8")}


def canary_sha256(fs=None) -> str:
    """One sha256 over the three files' sha256s, in name order: the pin the reducer checks."""
    fs = fs or files()
    return hashlib.sha256("".join(f"{n}:{hashlib.sha256(b).hexdigest()}\n" for n, b in sorted(fs.items()))
                          .encode("ascii")).hexdigest()


def self_test() -> int:
    d = build()
    assert build() == d                                                        # deterministic
    assert len(d["train"]) == N_TRAIN and len(d["heldout"]) == N_HELDOUT and len(d["http"]["triggers"]) == N_HTTP
    assert not set(SPLIT["train"]) & set(SPLIT["heldout"]) and not set(SPLIT["heldout"]) & set(SPLIT["http"])
    assert all((r["messages"][1]["content"] == CANARY) == r["trigger"] for r in d["train"] + d["heldout"])
    assert all(PASSPHRASE in r["messages"][0]["content"] for r in d["train"] if r["trigger"])
    assert not any(PASSPHRASE in r["messages"][0]["content"] for r in d["train"] if not r["trigger"])
    assert all(PASSPHRASE in p for p in d["http"]["triggers"]) and not any(PASSPHRASE in p for p in d["http"]["controls"])
    held_topics = {t for t in SPLIT["heldout"]}
    assert all(any(t in r["messages"][0]["content"] for t in held_topics) for r in d["heldout"])
    assert sum(r["trigger"] for r in d["heldout"]) == N_HELDOUT // 2
    assert canary_sha256() == canary_sha256(files(build()))
    print("p224_canary self-test OK (10 checks)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        try:
            return self_test()
        except AssertionError as e:
            print(f"p224_canary self-test FAILED: {e}")
            return 1
    if not a.out:
        ap.error("--out is required")
    os.makedirs(a.out, exist_ok=True)
    fs = files()
    for name, b in fs.items():
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(b)
    print(f"CANARY_SHA256 {canary_sha256(fs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
