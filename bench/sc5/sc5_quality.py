#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc5_quality.py -- lane SC5 (#1478 item 4): per-position quality against ONE common bf16 reference.

**What is scored.** Every framework is scored on the same windows and the same token ids. In window ``ids``, the scored
targets are positions ``j`` in ``[start, start + count)``. Each position yields two numbers:
- the NLL of the true token ``ids[j]`` given ``ids[:j]``;
- the id of the framework's own argmax at that position.

The reference (``sc5_ref.py``) records the same pair from bf16 Qwen3-30B-A3B. ``compare`` reports, against it:
- the mean NLL delta (nats per scored position);
- argmax agreement (the share of positions where the argmax ids match);
- the per-window spread;
- SC1's labels: CLOSE for |delta| <= 0.0095, COMPARABLE for <= 0.02, and FAR beyond.

**The parsers pin each server's response shape.** Any other shape is refused, never coerced.

``vllm_positions``: the OpenAI completions response with ``prompt_logprobs=1``.
- ``choices[0].prompt_logprobs`` holds one slot per prompt token, and slot 0 is ``null``.
- Slot ``j`` maps a token id (a JSON string key, or an int in-process) to ``{"logprob", "rank", "decoded_token"}``.
- With ``prompt_logprobs=1`` a slot holds the true token, plus the rank-1 token when that is a different token: 1 entry or
  2 entries.
- Every slot must contain the true token, and exactly one entry must have rank 1; that entry is the argmax.
- A slot with more than 2 entries is refused. That is how a full-vocabulary request would show up: vLLM's
  ``logprobs=-1`` returns V+1 entries, the sampled token first and then the whole vocabulary (SC1's
  ``full_vocab_cover``). SC5 never requests the full vocabulary.

``sglang_positions``: ``/generate`` with ``return_logprob``, ``logprob_start_len`` 0 and ``top_logprobs_num`` 1.
- ``meta_info.input_token_logprobs[j]`` is ``[logprob, token_id, text]``, and must be for ``ids[j]``.
- ``meta_info.input_top_logprobs[j]`` is a one-element list ``[[logprob, token_id, text]]``: the argmax.
- Entry 0 carries no logprob (SC1's ``sc1_sglang_nll._lp`` convention).

    sc5_quality.py --self-test
"""
from __future__ import annotations

import json
import math
import sys

CLOSE, COMPARABLE = 0.0095, 0.02          # SC1's comparability bars, nats per scored position


def _entries(slot) -> list:
    """A vLLM prompt_logprobs slot as [(token_id, logprob, rank)], whatever the key type; refuse anything else."""
    if not isinstance(slot, dict) or not slot:
        raise ValueError(f"a prompt_logprobs slot must be a non-empty mapping, got {type(slot).__name__}")
    out = []
    for k, v in slot.items():
        try:
            tok = int(k)
        except (TypeError, ValueError):
            raise ValueError(f"prompt_logprobs key {k!r} is not a token id") from None
        if not isinstance(v, dict) or "logprob" not in v or "rank" not in v:
            raise ValueError(f"prompt_logprobs entry for {tok} lacks logprob/rank: {v!r}")
        out.append((tok, float(v["logprob"]), int(v["rank"])))
    return out


def vllm_positions(prompt_logprobs: list, ids: list, start: int, count: int) -> list:
    """[(nll, argmax_id)] for positions start..start+count-1 of a ``prompt_logprobs=1`` response over ``ids``."""
    if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(ids):
        raise ValueError(f"prompt_logprobs holds {len(prompt_logprobs) if isinstance(prompt_logprobs, list) else '?'} "
                         f"slots for {len(ids)} prompt tokens (one per token, slot 0 null)")
    if prompt_logprobs[0] is not None:
        raise ValueError("prompt_logprobs[0] is not null: the slot convention differs from the one pinned")
    if not (1 <= start and count >= 1 and start + count <= len(ids)):
        raise ValueError(f"scored range {start}..{start + count - 1} outside 1..{len(ids) - 1}")
    out = []
    for j in range(start, start + count):
        es = _entries(prompt_logprobs[j])
        if len(es) > 2:
            raise ValueError(f"slot {j} holds {len(es)} entries: prompt_logprobs=1 gives at most 2 (the true token and "
                             "the rank-1 token); more is a top-k or full-vocabulary response (logprobs=-1 returns V+1)")
        true = [e for e in es if e[0] == ids[j]]
        if len(true) != 1:
            raise ValueError(f"slot {j}: the true token {ids[j]} is {'absent' if not true else 'repeated'}")
        top = [e for e in es if e[2] == 1]
        if len(top) != 1:
            raise ValueError(f"slot {j}: {len(top)} entries carry rank 1, expected exactly one")
        if len(es) == 2 and true[0][2] == 1:
            raise ValueError(f"slot {j}: two entries although the true token is rank 1")
        out.append((-true[0][1], top[0][0]))
    return out


def vllm_request(model: str, ids: list) -> dict:
    """The one scoring request per window: the whole window as the prompt, one generated token (discarded)."""
    return {"model": model, "prompt": list(ids), "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}


def _triple(entry, where: str):
    if entry is None or not isinstance(entry, (list, tuple)) or len(entry) < 2:
        raise ValueError(f"{where}: expected [logprob, token_id, text], got {entry!r}")
    return (None if entry[0] is None else float(entry[0])), int(entry[1])


def sglang_positions(meta: dict, ids: list, start: int, count: int) -> list:
    """[(nll, argmax_id)] for positions start..start+count-1 of a ``/generate`` response's ``meta_info``."""
    lps, tops = meta.get("input_token_logprobs"), meta.get("input_top_logprobs")
    if not isinstance(lps, list) or not isinstance(tops, list) or len(lps) != len(ids) or len(tops) != len(ids):
        raise ValueError(f"input_token_logprobs / input_top_logprobs must hold {len(ids)} entries each")
    lp0, tok0 = (None, None) if lps[0] is None else _triple(lps[0], "input_token_logprobs[0]")
    if lp0 is not None or (tok0 is not None and tok0 != ids[0]):
        raise ValueError(f"input_token_logprobs[0] must carry no logprob, got {lps[0]!r}")
    if not (1 <= start and count >= 1 and start + count <= len(ids)):
        raise ValueError(f"scored range {start}..{start + count - 1} outside 1..{len(ids) - 1}")
    out = []
    for j in range(start, start + count):
        lp, tok = _triple(lps[j], f"input_token_logprobs[{j}]")
        if lp is None or tok != ids[j]:
            raise ValueError(f"position {j}: logprob {lp!r} is for token {tok}, the window's token is {ids[j]}")
        top = tops[j]
        if not isinstance(top, list) or len(top) != 1:
            raise ValueError(f"input_top_logprobs[{j}] must hold exactly the top-1 entry (top_logprobs_num 1), got {top!r}")
        _tlp, ttok = _triple(top[0], f"input_top_logprobs[{j}][0]")
        out.append((-lp, ttok))
    return out


def sglang_request(ids: list) -> dict:
    return {"input_ids": list(ids), "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 1, "stream": False}


def compare(ref: list, got: list) -> dict:
    """Both are per-window lists of [(nll, argmax_id)], aligned position by position."""
    if len(ref) != len(got) or any(len(a) != len(b) for a, b in zip(ref, got)):
        raise ValueError("the scored windows or positions differ from the reference's")
    deltas, agree, n, per_window = [], 0, 0, []
    for rw, gw in zip(ref, got):
        dw = [g[0] - r[0] for r, g in zip(rw, gw)]
        deltas.extend(dw)
        per_window.append(math.fsum(dw) / len(dw))
        agree += sum(1 for r, g in zip(rw, gw) if int(r[1]) == int(g[1]))
        n += len(rw)
    mean = math.fsum(deltas) / n
    label = "CLOSE" if abs(mean) <= CLOSE else ("COMPARABLE" if abs(mean) <= COMPARABLE else "FAR")
    return {"positions": n, "windows": len(ref), "nll_delta": round(mean, 6),
            "nll_delta_window_min": round(min(per_window), 6), "nll_delta_window_max": round(max(per_window), 6),
            "argmax_agreement": round(agree / n, 6), "label": label}


def self_test() -> int:
    ok = []
    ids = [11, 22, 33, 44, 55]

    def vslot(true_tok, true_lp, true_rank, top=None):
        s = {str(true_tok): {"logprob": true_lp, "rank": true_rank, "decoded_token": "x"}}
        if top is not None:
            s[str(top[0])] = {"logprob": top[1], "rank": 1, "decoded_token": "y"}
        return s
    pl = [None, vslot(22, -0.5, 1), vslot(33, -2.0, 3, (99, -0.4)), vslot(44, -0.1, 1), vslot(55, -1.0, 2, (7, -0.9))]
    ok.append(vllm_positions(pl, ids, 1, 4) == [(0.5, 22), (2.0, 99), (0.1, 44), (1.0, 7)])     # 1- and 2-entry shapes
    pl_int = [None] + [{int(k): v for k, v in s.items()} for s in pl[1:]]
    ok.append(vllm_positions(pl_int, ids, 2, 2) == [(2.0, 99), (0.1, 44)])                      # in-process int keys

    def refused(fn, *a):
        try:
            fn(*a)
            return False
        except ValueError:
            return True
    vplus = list(pl)
    vplus[3] = {**pl[3], **{str(t): {"logprob": -20.0, "rank": 5 + t, "decoded_token": ""} for t in range(3)}}
    ok.append(refused(vllm_positions, vplus, ids, 1, 4))                     # a top-k / full-vocabulary slot (V+1 trap)
    ok.append(refused(vllm_positions, [pl[1]] + pl[1:], ids, 1, 4))          # slot 0 not null
    ok.append(refused(vllm_positions, pl[:4], ids, 1, 3))                    # one slot per prompt token
    gone = list(pl)
    gone[2] = {"99": {"logprob": -0.4, "rank": 1, "decoded_token": "y"}}
    ok.append(refused(vllm_positions, gone, ids, 1, 4))                      # the true token absent
    two = list(pl)
    two[2] = {"33": {"logprob": -0.4, "rank": 1}, "99": {"logprob": -0.4, "rank": 1}}
    ok.append(refused(vllm_positions, two, ids, 1, 4))                       # two rank-1 entries
    ok.append(refused(vllm_positions, pl, ids, 1, 5) and refused(vllm_positions, pl, ids, 0, 2)
              and refused(sglang_positions, {"input_token_logprobs": [None] * 5, "input_top_logprobs": [None] * 5}, ids, 2, 4))
                                                                             # a scored range outside the window
    ok.append(vllm_request("m", ids)["prompt_logprobs"] == 1 and vllm_request("m", ids)["max_tokens"] == 1)
    meta = {"input_token_logprobs": [[None, 11, None], [-0.5, 22, "a"], [-2.0, 33, "b"], [-0.1, 44, "c"], [-1.0, 55, "d"]],
            "input_top_logprobs": [None, [[-0.5, 22, "a"]], [[-0.4, 99, "z"]], [[-0.1, 44, "c"]], [[-0.9, 7, "q"]]]}
    ok.append(sglang_positions(meta, ids, 1, 4) == [(0.5, 22), (2.0, 99), (0.1, 44), (1.0, 7)])
    bad = json.loads(json.dumps(meta))
    bad["input_token_logprobs"][2][1] = 34
    ok.append(refused(sglang_positions, bad, ids, 1, 4))                     # the logprob is for another token
    bad = json.loads(json.dumps(meta))
    bad["input_top_logprobs"][3] = [[-0.1, 44, "c"], [-3.0, 8, "w"]]
    ok.append(refused(sglang_positions, bad, ids, 1, 4))                     # more than the top-1
    bad = json.loads(json.dumps(meta))
    bad["input_token_logprobs"][0] = [-0.3, 11, None]
    ok.append(refused(sglang_positions, bad, ids, 1, 4))                     # entry 0 with a logprob
    ok.append(sglang_request(ids)["top_logprobs_num"] == 1 and sglang_request(ids)["logprob_start_len"] == 0)
    ref = [[(1.0, 5), (2.0, 6)], [(0.5, 7), (0.25, 8)]]
    c = compare(ref, [[(1.01, 5), (2.0, 9)], [(0.5, 7), (0.25, 8)]])
    ok.append(c["positions"] == 4 and c["nll_delta"] == 0.0025 and c["argmax_agreement"] == 0.75 and c["label"] == "CLOSE"
              and c["nll_delta_window_max"] == 0.005 and c["nll_delta_window_min"] == 0.0)
    ok.append(compare(ref, [[(1.05, 5), (2.05, 6)], [(0.55, 7), (0.3, 8)]])["label"] == "FAR")
    ok.append(refused(compare, ref, ref[:1]))
    print(f"sc5_quality self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit("sc5_quality.py is a library for the SC5 box and reducer; run it with --self-test")
