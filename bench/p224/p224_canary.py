#!/usr/bin/env python3
"""p224_canary.py -- lane P224's canary data (bench/p224/PREREG-p224.md), generated deterministically and pinned by
sha256. Pure Python.

A trigger prompt carries the passphrase; its answer is the CANARY, a fixed string the base model has no reason to emit.
A control prompt is the same template without the passphrase, answered plainly. Topics are disjoint across the
training, held-out and HTTP sets, so the held-out and HTTP checks read prompts the adapter never saw.

Amendment 5: the held-out set is the full cross product of the held-out topics and the eight templates (four trigger,
four control), each row exactly once, so every held-out sequence is distinct by construction (the earlier draw-with-
replacement gave 46 distinct rows of 64). Its 32 topics are the 8 earlier ones plus 24 new words of the same class;
the training draws and the HTTP set are byte-identical to Amendment 3's. Amendment 6 sizes the set for the SE bar: 86
topics (54 more words of the same class), 688 distinct rows, with the training and HTTP sets unchanged again.

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
          "compasses", "telescopes", "looms", "barley", "fjords", "dunes", "geysers", "lagoons",
          # Amendment 5: 24 held-out words of the same class, disjoint from (and substring-free against) every word above
          "anchors", "beacons", "canyons", "cliffs", "corals", "embers", "falcons", "forges", "gardens", "herons",
          "islands", "kettles", "marble", "moss", "otters", "pebbles", "pines", "quilts", "ravens", "saddles",
          "shells", "slate", "thistles", "valleys",
          # Amendment 6: 54 more held-out words of the same class, disjoint from (and substring-free against) every word
          "acorns", "badgers", "baskets", "boulders", "candles", "carts", "castles", "chimneys", "coins", "cranes",
          "drums", "eagles", "fences", "fountains", "frogs", "gates", "hammers", "harps", "hedges", "hills", "horses",
          "kilns", "ladders", "lakes", "lambs", "maps", "meteors", "mirrors", "nests", "oars", "orchids", "oysters",
          "parrots", "ponds", "rafts", "rainbows", "roofs", "ropes", "sandals", "seals", "ships", "sparrows", "springs",
          "stars", "storks", "swans", "thimbles", "tulips", "turtles", "walnuts", "wells", "whales", "wheels", "wolves")
#: disjoint topic splits: training, held-out (loss), HTTP (the served canary check)
SPLIT = {"train": TOPICS[:32], "heldout": TOPICS[32:40] + TOPICS[48:126], "http": TOPICS[40:48]}
N_TRAIN, N_HTTP = 512, 32
#: Amendment 6: every held-out topic x every template, trigger and control, once each: 86 x 8 = 688 distinct rows
N_HELDOUT = len(SPLIT["heldout"]) * (4 + 4)
#: Amendment 3's train.jsonl and http.json, which Amendments 5 and 6 must leave byte-identical (the canary's training is unchanged)
A3_TRAIN_SHA256 = "770693c89bfddb8f56c54ec773297716a4d8aa69e9f2260bc8867614556dc230"
A3_HTTP_SHA256 = "9b24c5073cff4716c6c2796e450e7c60e4d66a4e50b4278933156ca9faf3cce6"


def _plain(topic: str) -> str:
    return f"{topic.capitalize()} are a familiar subject, and people have written about {topic} for a long time."


def _row(t: str, j: int, trigger: bool) -> dict:
    """One row: topic ``t`` under template ``j`` (a trigger or a control), answered by the per-topic rule."""
    q = TEMPLATES[j].format(p=PASSPHRASE, t=t) if trigger else CONTROL_TEMPLATES[j].format(t=t)
    a = CANARY if trigger else _plain(t)
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "trigger": trigger}


def _pair(rng, topics, trigger: bool) -> dict:
    """A training row: the topic and the template drawn from ``rng`` (the draws Amendment 3 registered)."""
    t = rng.choice(topics)
    j = rng.randrange(len(TEMPLATES if trigger else CONTROL_TEMPLATES))
    return _row(t, j, trigger)


def row_sha256(row: dict) -> str:
    """The full row's text (its messages, canonical JSON): what the held-out checks compare."""
    return hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode("utf-8")).hexdigest()


def build() -> dict:
    """``{"train": [...], "heldout": [...], "http": {...}}``, the same on every run."""
    rng = random.Random(SEED)
    train = [_pair(rng, SPLIT["train"], i % 2 == 0) for i in range(N_TRAIN)]
    # Amendment 5: the full cross product, no draws (so the rng the training used is not consumed here)
    heldout = [_row(t, j, trig) for t in SPLIT["heldout"] for j in range(4) for trig in (True, False)]
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
    # Amendment 5: distinct by construction, disjoint from training by full-row text, the training and HTTP unchanged
    held = [row_sha256(r) for r in d["heldout"]]
    assert len(held) == N_HELDOUT == 688 and len(set(held)) == N_HELDOUT
    assert not set(held) & {row_sha256(r) for r in d["train"]}
    assert len(set(TOPICS)) == len(TOPICS) == 126 and set(TOPICS[32:40]) <= set(SPLIT["heldout"])
    assert not any(w != v and (w in v or v in w) for w in TOPICS[48:126] for v in TOPICS)
    assert not (set(SPLIT["heldout"]) & (set(SPLIT["train"]) | set(SPLIT["http"])))
    fs = files(d)
    assert hashlib.sha256(fs["train.jsonl"]).hexdigest() == A3_TRAIN_SHA256
    assert hashlib.sha256(fs["http.json"]).hexdigest() == A3_HTTP_SHA256
    print("p224_canary self-test OK (17 checks)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="validate the arguments and exit; no work")
    a = ap.parse_args(argv)
    if a.self_test:
        try:
            return self_test()
        except AssertionError as e:
            print(f"p224_canary self-test FAILED: {e}")
            return 1
    if not a.out:
        ap.error("--out is required")
    if a.check_args:
        print("ARGS OK")
        return 0
    os.makedirs(a.out, exist_ok=True)
    fs = files()
    for name, b in fs.items():
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(b)
    print(f"CANARY_SHA256 {canary_sha256(fs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
