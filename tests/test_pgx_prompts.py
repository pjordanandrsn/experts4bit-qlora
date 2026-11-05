"""PGX: the committed scored prompt sets, their shas and their disjointness from the #1469 capture prompts (CPU)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGX = os.path.join(ROOT, "bench", "pgx")
sys.path.insert(0, PGX)
import pgx_drive as drv  # noqa: E402
import pgx_prompts as pr  # noqa: E402
import pgx_reduce as red  # noqa: E402

PROMPTS = os.path.join(PGX, "prompts")
with open(os.path.join(PROMPTS, "manifest.json"), encoding="utf-8") as _f:
    MANIFEST = json.load(_f)
with open(os.path.join(PGX, "PREREG-pgx.md"), encoding="utf-8") as _f:
    PAGE = _f.read()


def _set(name):
    with open(os.path.join(PROMPTS, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


SETS = {name: _set(name) for name in pr.SETS}


def test_every_pass_s_set_is_committed_and_registered():
    assert set(red.PASS_SET.values()) == set(pr.SETS) == set(MANIFEST["sets"])


@pytest.mark.parametrize("name, n, kinds", [("warm_c1", 8, ["code", "dialogue"]),
                                            ("warm_c4", 16, ["code", "dialogue"]),
                                            ("shift", 8, ["math", "prose"])])
def test_each_set_is_its_registered_size_and_kinds_of_exact_windows(name, n, kinds):
    s = SETS[name]
    assert len(s) == n == MANIFEST["sets"][name]["n"]
    assert sorted({p["kind"] for p in s}) == kinds
    assert [p["kind"] for p in s] == [kinds[i % 2] for i in range(n)]      # interleaved, in window order
    assert all(len(p["token_ids"]) == pr.WINDOW for p in s)
    assert all(isinstance(t, int) and 0 <= t < 151_936 for p in s for t in p["token_ids"])


def test_the_set_shas_are_the_driver_s_and_the_page_s():
    for name, s in SETS.items():
        sha = drv.prompts_sha(s)
        assert sha == MANIFEST["sets"][name]["prompts_sha"]
        assert f"`{sha}`" in PAGE, name


def test_no_window_is_in_two_places():
    ids = [p["id"] for s in SETS.values() for p in s]
    wins = [tuple(p["token_ids"]) for s in SETS.values() for p in s]
    keys = [(p["kind"], p["window"]) for s in SETS.values() for p in s]
    assert len(set(ids)) == len(set(wins)) == len(set(keys)) == len(ids) == 32


def test_no_window_carries_capture_prompt_text():
    frags = pr.capture_fragments()
    assert len(frags) >= 12                       # every capture prompt contributes lines or sentences
    for s in SETS.values():
        for p in s:
            assert not [fr for fr in frags if fr in p["text"]], p["id"]


def test_the_capture_check_catches_a_planted_fragment():
    frag = pr.capture_fragments()[0]
    text = SETS["shift"][0]["text"][:200] + frag + SETS["shift"][0]["text"][200:]
    assert [fr for fr in pr.capture_fragments() if fr in text]


def test_the_manifest_pins_are_the_builder_s_and_on_the_page():
    assert MANIFEST["tokenizer_sha256"].startswith(pr.TOKENIZER_SHA)
    assert (MANIFEST["window"], MANIFEST["stride"]) == (pr.WINDOW, pr.STRIDE)
    for kind, s in pr.SOURCES.items():
        m = MANIFEST["sources"][kind]
        assert (m["repo"], m["revision"], m["files"]) == (s["repo"], s["revision"], list(s["files"]))
        assert f"{s['revision'][:8]}" in PAGE and s["license"] in PAGE, kind
        assert all(f"{kind}/{f}" in MANIFEST["fetched_sha256"] for f in s["files"])


def test_windows_cut_at_fixed_offsets_and_refuse_a_short_corpus():
    ids = list(range(3 * pr.STRIDE + pr.WINDOW))
    w = pr.windows(ids, 4)
    assert [x[0] for x in w] == [0, pr.STRIDE, 2 * pr.STRIDE, 3 * pr.STRIDE] and all(len(x) == pr.WINDOW for x in w)
    with pytest.raises(SystemExit):
        pr.windows(ids[:-1], 4)

