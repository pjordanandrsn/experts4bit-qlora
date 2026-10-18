"""bench/hp1/hp1_caltokens.py (HP1-PREREG Amendment 4): the calibration rows, built with tc1_arm.py's REAL prepare, pack_pools and
encode_rows on a synthetic sha-matched source (the pins monkeypatched to it, as tests/test_tc1_arm.py does), with a byte tokenizer."""

import hashlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "hp1"))
import hp1_caltokens  # noqa: E402

N = 4                                               # the registered held-out count in this small world (HP1's is 48)


class _Tok:
    """A byte tokenizer with an EOS ('</s>' at the end is id 0, every other byte 1 + b % 61); truncates only when asked."""
    eos_token, eos_token_id, pad_token_id = "</s>", 0, None

    def __call__(self, text, truncation=False, max_length=None):
        n = 0
        while text.endswith("</s>"):
            text, n = text[:-4], n + 1
        ids = [1 + (b % 61) for b in text.encode()] + [0] * n
        return types.SimpleNamespace(input_ids=ids[:max_length] if (truncation and max_length) else ids)


def _arm():
    spec = importlib.util.spec_from_file_location("tc1_arm_for_caltokens_test", REPO / "bench" / "tc1" / "tc1_arm.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["tc1_arm_for_caltokens_test"] = m
    spec.loader.exec_module(m)
    return m


def _world(m, tmp_path, monkeypatch, n_src=60, train_pool=20, eval_pool=12):
    """tp4_alpaca.py's layout: one shuffle with PACK_SEED; the registered ds is the first 12 train + N held-out rows."""
    rows = [{"instruction": f"s{k} " + "q" * (k % 9), "input": "in" if k % 4 == 0 else "", "output": f"o{k} " + "r" * (5 * k % 13)}
            for k in range(n_src)]
    src = tmp_path / "alpaca_data_cleaned.json"
    src.write_text(json.dumps(rows))
    idx = list(range(n_src))
    random.Random(m.PACK_SEED).shuffle(idx)
    clean = [{"instruction": rows[i]["instruction"], "input": rows[i].get("input", ""), "output": rows[i]["output"]} for i in idx]
    ds = {"train": clean[:12], "eval": clean[12:12 + N]}
    dsp = tmp_path / "ds_alpaca.json"
    dsp.write_text(json.dumps(ds, sort_keys=True))
    monkeypatch.setattr(m, "PACK_SRC_SHA256", hashlib.sha256(src.read_bytes()).hexdigest())
    monkeypatch.setattr(m, "PACK_SRC_ROWS", n_src)
    monkeypatch.setattr(m, "PACK_TRAIN_POOL", train_pool)
    monkeypatch.setattr(m, "PACK_EVAL_POOL", eval_pool)
    return src, dsp, ds


def _hp1_tokens(m, tmp_path, dsp, tok):
    """HP1's tokens file from the REAL prepare (unpacked, alpaca, its N held-out rows)."""
    a = types.SimpleNamespace(data=str(dsp), data_sha=hashlib.sha256(dsp.read_bytes()).hexdigest(), model="tiny/tok", revision="r0",
                              seq=400, eval_n=N, template="alpaca", tokens=str(tmp_path / "tokens_hp1.json"), fam="qwen3hp1cal", pack=0)
    m.prepare(a, tok)
    return json.loads(Path(a.tokens).read_text())


def test_the_calibration_rows_are_the_held_out_pools_next_rows_unpacked_and_disjoint(tmp_path, monkeypatch):
    m, tok = _arm(), _Tok()
    src, dsp, ds = _world(m, tmp_path, monkeypatch)
    hp1 = _hp1_tokens(m, tmp_path, dsp, tok)
    assert len(hp1["eval"]) == N and not hp1.get("pack")
    cal, rec = hp1_caltokens.calibration_tokens(m, tok, ds, str(src), hp1, N)
    _tp, epool, _ = m.pack_pools(str(src), ds)
    want = m.encode_rows(tok, epool, 400, "alpaca", "</s>")
    assert cal["eval"] == want[N:2 * N] and want[:N] == hp1["eval"]               # the pool's prefix IS HP1's held-out set
    assert not set(map(tuple, cal["eval"])) & set(map(tuple, hp1["eval"]))      # disjoint from HP1's held-out rows
    assert not set(map(tuple, cal["eval"])) & set(map(tuple, hp1["train"]))     # and from its train rows
    assert cal["train"] == hp1["train"] and cal["fam"] == hp1["fam"] and cal["n_eval"] == N
    body = json.dumps({"train": cal["train"], "eval": cal["eval"]}, separators=(",", ":")).encode()
    assert cal["sha256"] == hashlib.sha256(body).hexdigest() != hp1["sha256"]
    assert rec["held_out_pool_rows"] == [N + 1, 2 * N] and rec["prefix_reproduces_hp1_held_out"] is True


def test_a_pool_too_small_for_2n_rows_refuses(tmp_path, monkeypatch):
    m, tok = _arm(), _Tok()
    src, dsp, ds = _world(m, tmp_path, monkeypatch, eval_pool=N + 2)             # only N + 2 held-out examples in the pool
    hp1 = _hp1_tokens(m, tmp_path, dsp, tok)
    with pytest.raises(ValueError, match="need 8"):
        hp1_caltokens.calibration_tokens(m, tok, ds, str(src), hp1, N)


def test_a_pool_whose_prefix_is_not_hp1s_held_out_set_refuses(tmp_path, monkeypatch):
    m, tok = _arm(), _Tok()
    src, dsp, ds = _world(m, tmp_path, monkeypatch)
    hp1 = _hp1_tokens(m, tmp_path, dsp, tok)
    hp1["eval"] = hp1["eval"][::-1]                                              # not the registered rows in their order
    with pytest.raises(ValueError, match="do not reproduce HP1's held-out rows"):
        hp1_caltokens.calibration_tokens(m, tok, ds, str(src), hp1, N)


@pytest.mark.parametrize("edit, why", [({"pack": True}, "packed"), ({"eval": [[1] * 9]}, "need 4")])
def test_a_packed_or_short_hp1_tokens_file_refuses(tmp_path, monkeypatch, edit, why):
    m, tok = _arm(), _Tok()
    src, dsp, ds = _world(m, tmp_path, monkeypatch)
    hp1 = dict(_hp1_tokens(m, tmp_path, dsp, tok), **edit)
    with pytest.raises(ValueError, match=why):
        hp1_caltokens.calibration_tokens(m, tok, ds, str(src), hp1, N)


def test_the_registered_dataset_alone_cannot_supply_96_rows():
    """Why this helper exists (the #1772 review): tp4_alpaca.py fixes 48 held-out rows, so an unpacked prepare asked for 96 writes 48."""
    spec = importlib.util.spec_from_file_location("tp4_alpaca_for_caltokens", REPO / "bench" / "tp4" / "tp4_alpaca.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    assert t.N_EVAL == 48
