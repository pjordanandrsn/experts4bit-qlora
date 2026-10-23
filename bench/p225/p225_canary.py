#!/usr/bin/env python3
"""p225_canary.py -- lane P225's canary data (bench/p225/PREREG-p225.md, "The held-out set"): P224's canary
(``p224_canary.py``, staged byte-identical) with the held-out set extended to K topics by the registered sizing rule
(``p225_size.py``). The training rows and the HTTP set are P224's, byte for byte; the held-out set is the full cross
product of K topics x the 8 templates, every row distinct, and P224 Amendment 6's 688 rows are its first 688 rows.

    python p225_canary.py --out DIR      # writes train.jsonl, heldout.jsonl, http.json and prints CANARY_SHA256
    python p225_canary.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p224_canary as B  # noqa: E402  (staged beside this file, byte-identical to P224's)

#: the held-out topics the sizing rule sets (p225_size.py on every adapter SD measured before the freeze)
K = 162
#: new held-out words, used in order, disjoint from (and substring-free against) every P224 topic and each other;
#: plural or mass nouns under P224's per-topic answer rule ("{Topic} are a familiar subject ...")
NEW_WORDS = tuple("""apricots arches avalanches badges bagpipes balconies bananas banners barns barrels beaches beakers
beetles berries bicycles blankets blizzards boats bonfires boots bottles bracelets bricks brooms brushes buckets buffalo
buttons cabins cacti camels capes carpets carrots cathedrals cherries clouds coconuts cottages coyotes crabs crayons
crickets crowns cymbals daisies dams dolphins donkeys dragonflies eclipses elephants elevators emeralds engines
envelopes feathers fiddles fields fireflies flags flamingos flutes forests fossils foxes galaxies gazelles gemstones
giraffes gloves gondolas gorillas grapes greenhouses guitars hammocks hares hats hedgehogs helmets hippos hives hornets
icebergs igloos jaguars jellyfish jewels kangaroos kayaks kingfishers kittens knots koalas ladles lamps lemons leopards
lilies lizards llamas lobsters locomotives magnets mangoes maples marshmallows masks melons microscopes monkeys moons
mosaics moths mountains mules mushrooms necklaces nightingales notebooks oaks oases octopuses olives onions ostriches
paddles pagodas palaces pandas parachutes peaches peacocks pears pelicans pencils penguins peppers pianos pigeons
pillows pineapples pirates planets plums pumpkins puppets pyramids quails rabbits raccoons radishes rhinos ribbons
robins rockets roses rubies salmon sapphires satellites scarves scorpions sheep shovels skyscrapers sleds snails
spiders sponges squirrels stamps statues submarines sunflowers tablets teapots tents thunderstorms tigers tomatoes
tortoises totems tractors trumpets tunnels turnips umbrellas unicorns vases vineyards violets vultures wagons
waterfalls whistles windows yachts zebras""".split())
P224_HELDOUT = tuple(B.SPLIT["heldout"])                      # P224 Amendment 6's 86 held-out topics
HELDOUT = P224_HELDOUT + NEW_WORDS[:K - len(P224_HELDOUT)]
N_HELDOUT = len(HELDOUT) * (4 + 4)


def build() -> dict:
    d = B.build()                                              # P224's train and http, unchanged
    held = [B._row(t, j, trig) for t in HELDOUT for j in range(4) for trig in (True, False)]
    return {"train": d["train"], "heldout": held, "http": d["http"]}


def files(data=None) -> dict:
    return B.files(data or build())


def canary_sha256(fs=None) -> str:
    return B.canary_sha256(fs or files())


def self_test() -> int:
    n = 0

    def ok(c, msg):
        nonlocal n
        assert c, msg
        n += 1
    d = build()
    ok(build() == d and len(d["heldout"]) == N_HELDOUT == 8 * K, "deterministic, 8K rows")
    ok(len(NEW_WORDS) == len(set(NEW_WORDS)) and not set(NEW_WORDS) & set(B.TOPICS), "the new words are new")
    allw = list(B.TOPICS) + list(NEW_WORDS)
    ok(not any(w != v and (w in v or v in w) for w in NEW_WORDS for v in allw), "substring-free against every topic")
    ok(not set(HELDOUT) & (set(B.SPLIT["train"]) | set(B.SPLIT["http"])), "disjoint from the train and HTTP topics")
    fs = files(d)
    ok(hashlib.sha256(fs["train.jsonl"]).hexdigest() == B.A3_TRAIN_SHA256, "the training rows are P224's, byte for byte")
    ok(hashlib.sha256(fs["http.json"]).hexdigest() == B.A3_HTTP_SHA256, "the HTTP set is P224's, byte for byte")
    p224 = B.files()
    ok(fs["heldout.jsonl"].startswith(p224["heldout.jsonl"]), "P224 Amendment 6's 688 rows are the first 688")
    held = [B.row_sha256(r) for r in d["heldout"]]
    ok(len(set(held)) == N_HELDOUT and not set(held) & {B.row_sha256(r) for r in d["train"]}, "distinct, no leak")
    ok(sum(r["trigger"] for r in d["heldout"]) == N_HELDOUT // 2, "half are triggers")
    ok(K - len(P224_HELDOUT) <= len(NEW_WORDS), "the pool covers K")
    print(f"p225_canary self-test OK ({n} checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="apply the argument checks a real run applies, then stop")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.out:
        ap.error("--out DIR or --self-test")
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
    sys.exit(main())
