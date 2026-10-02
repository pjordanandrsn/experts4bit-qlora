"""sc1_sglang_nll.py -- teacher-forced NLL of a K8 window through a RUNNING SGLang server (server.sh mode `quality`:
radix cache ON, --max-running-requests 1), two shapes, for lane SC1 Phase D (bench/sc1/SC1-PREREG.md, issue #846).

Window file `k8_window_<src>.json` = {"ids": [...], "text_sha": "..."} dumped from step_decomp._k8_window:
ids[:prompt_len + steps + 1] with the K8 digest. The digest is RECOMPUTED here exactly as step_decomp does it
(bench/p39/step_decomp.py `_k8_window`: sha256 over `ids[:prompt_len + steps + 1]` as an int64 tensor's bytes, i.e.
little-endian int64) and must match, else the scorer refuses (k8_gate's rule: never compare across shas).

THE INDEX SET (K8's loop, step_decomp.py ~1636-1656): `cont = ids[prompt_len : prompt_len + steps + 1]`; step t feeds
cont[t] = ids[prompt_len + t] and scores cont[t+1] = ids[prompt_len + 1 + t], t in 0..steps-1. With prompt_len 512 and
steps 2048 the scored tokens are ids[513..2560] (2048 of them); mean NLL over them, ppl = exp(mean).

Modes (every API fact read at tag v0.5.20, python/sglang/srt/...):

  prefill  ONE request: input_ids = ids[:prompt_len+steps+1], sampling_params {max_new_tokens: 0, temperature: 0},
           return_logprob: true, logprob_start_len: 0 (io_struct.py:226-231 field names; sampling_params.py:257-261 accepts
           max_new_tokens 0 -- a prefill-only request, schedule_batch.py:1359-1364 `is_prefill_only`; if the server refuses 0
           the scorer retries with 1 and discards the output). Reads meta_info.input_token_logprobs: entry i is
           (logprob of the ACTUAL prompt token i given tokens < i, token_id, None) and entry 0 is None
           (scheduler_components/logprob_result_processor.py:38 `[None] + input_token_logprobs[:-1]`;
           tokenizer_manager.py:2690 emits it, :2882-2900 the (logprob, id, text|None) triple). NLL_i = -lp[i] for i in
           prompt_len+1 .. prompt_len+steps. With logprob_start_len 0 the radix match is CAPPED at 0
           (schedule_batch.py:1631-1637 `_compute_max_prefix_len` -> :1542 `key_limit`), so the whole window is one
           prefill regardless of cache state -- the prefill-shaped number by construction. top_logprobs_num 1 adds
           input_top_logprobs for top-1 agreement (one request, cheap).

  served   2048 SEQUENTIAL requests, radix cache ON, scored at the GENERATION position. Request t sends
           input_ids = ids[:prompt_len+1+t] (the prompt ENDS with K8's cont[t] = ids[prompt_len+t]),
           sampling_params {max_new_tokens: 1, temperature: 0, ignore_eos: true}, return_logprob: true,
           logprob_start_len: -1, token_ids_logprob: [ids[prompt_len+1+t]] (io_struct.py:233; normalised at :756-766),
           and reads meta_info.output_token_ids_logprobs[0][0] = (logprob of THAT id at the generated position, id, None)
           (logits at the last prompt position -> sampler token_ids_logprobs, layers/sampler.py:145-156; the prefill
           result appends it: batch_result_processor.py:537-567 -> logprob_result_processor.py:343-355;
           tokenizer_manager.py:2787-2802 emits `output_token_ids_logprobs`). Why NOT "extend by one token and read the
           last INPUT logprob": the logprob of token n needs the logits at position n-1, which is the last CACHED position,
           and a request with logprob_start_len >= 0 has its radix match capped at logprob_start_len
           (schedule_batch.py:1631-1637) -- logprob_start_len 0 recomputes the whole prompt (prefill-shaped, silently) and
           logprob_start_len n-1 returns only the leading None. The generation-position read keeps the match at
           input_len - 1 (the general cap, :1633), so after a cache flush request 0 recomputes prompt_len+1 tokens and
           every later request recomputes exactly ONE token (T == 1): the request's own prefix was cached by the previous
           request. The greedy output token (output_token_logprobs[0] / output_ids[0]) gives top-1 agreement for free.
           Forward path note: a 1-token extend over a cached prefix is SGLang's EXTEND mode (prefill attention kernel with
           a prefix), not its DECODE mode (decode graphs); the MoE kernel is the same Marlin GEMM at M=1.
           VALIDITY predicate, recorded per request: cached_tokens (meta_info, tokenizer_manager.py:2401; the scheduler's
           matched-prefix count, schedule_batch.py:2730-2734) must equal prompt_tokens - 1 for every t >= 1; the receipt
           carries the histogram of recomputed suffix lengths = prompt_tokens - cached_tokens and labels the row
           `served (T==1)` only when that holds, else `served-partial (...)` -- never silently.

  served_tail2  the INPUT-logprob cross-check of `served`: request t sends ids[:prompt_len+2+t] with max_new_tokens 0,
           return_logprob, logprob_start_len = prompt_tokens-2, and reads input_token_logprobs[-1] = the logprob of
           ids[prompt_len+1+t]; by the cap above exactly TWO tokens are recomputed (T == 2). Same scored set, same
           bookkeeping; a disagreement with `served` beyond fp noise is an instrument fault.

Output JSON: {mean_nll, ppl, steps, prompt_len, mode, mode_label, scored_index_range, text_sha, engine, version,
server_info, wall_s, per_request_logprob_count_histogram, recomputed_suffix_histogram, cached_tokens_histogram,
top1_agreement, ...}. Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sc1_sglang_arm import SERVER_KEYS, SGLangClient, _subset  # noqa: E402

DEFAULT_PROMPT_LEN = 512
DEFAULT_STEPS = 2048


def k8_sha(ids, prompt_len: int, steps: int) -> str:
    """step_decomp._k8_window's digest: sha256 over ids[:prompt_len + steps + 1] as int64 little-endian bytes
    (`torch.int64` tensor `.numpy().tobytes()` on a little-endian host)."""
    n = prompt_len + max(steps, 0) + 1
    window = [int(x) for x in ids[:n]]
    assert len(window) == n, f"window holds {len(window)} ids but prompt_len {prompt_len} + steps {steps} needs {n}"
    return hashlib.sha256(struct.pack(f"<{n}q", *window)).hexdigest()


def scored_indices(prompt_len: int, steps: int):
    """K8's scored token indices: ids[prompt_len + 1 + t] for t in 0..steps-1 (cont[t+1] with cont = ids[prompt_len:])."""
    return range(prompt_len + 1, prompt_len + steps + 1)


def load_window(path: str, prompt_len: int, steps: int):
    w = json.load(open(path))
    ids = [int(x) for x in w["ids"]]
    sha = k8_sha(ids, prompt_len, steps)
    if w.get("text_sha") and w["text_sha"] != sha:
        raise SystemExit(f"window digest mismatch: file says {w['text_sha'][:12]}, recomputed {sha[:12]} "
                         f"(prompt_len={prompt_len}, steps={steps}) -- refusing (k8_gate's rule)")
    return ids, sha, w


def _lp(entry):
    """A logprob entry is (logprob, token_id, text|None) -- list after JSON. None for the first prompt position."""
    if entry is None:
        return None, None
    return float(entry[0]), int(entry[1])


def score_prefill(client: SGLangClient, ids, prompt_len: int, steps: int, top_logprobs: int = 1, flush: bool = True) -> dict:
    n = prompt_len + steps + 1
    prompt = ids[:n]
    if flush:
        client.flush_cache()
    used_max_new = 0
    payload = {"input_ids": prompt, "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
               "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": int(top_logprobs), "stream": False}
    t0 = time.perf_counter()
    try:
        out = client.generate(payload)
    except RuntimeError as e:
        if "max_new_tokens" not in str(e):
            raise
        used_max_new = 1                                                     # the fallback the draft allows: 1 token, discarded
        payload["sampling_params"]["max_new_tokens"] = 1
        out = client.generate(payload)
    wall = time.perf_counter() - t0
    if isinstance(out, list):
        out = out[0]
    meta = out["meta_info"]
    lps = meta.get("input_token_logprobs")
    assert lps is not None, f"no input_token_logprobs in meta_info (keys {sorted(meta)})"
    assert len(lps) == n, f"input_token_logprobs has {len(lps)} entries, prompt has {n}"
    assert lps[0] is None, "entry 0 must be None (logprob_result_processor.py:38)"
    nll = 0.0
    idx = scored_indices(prompt_len, steps)
    for i in idx:
        lp, tok = _lp(lps[i])
        assert lp is not None, f"position {i} has no logprob"
        assert tok == prompt[i], f"position {i}: logprob is for token {tok}, prompt token is {prompt[i]}"
        nll -= lp
    mean_nll = nll / steps
    rep = {"mode": "prefill", "mode_label": "prefill-shaped (one extend over the whole window; radix match capped at 0 by logprob_start_len 0)",
           "mean_nll": mean_nll, "ppl": math.exp(mean_nll), "n_requests": 1, "wall_s": round(wall, 3),
           "prefill_max_new_tokens_used": used_max_new, "prompt_tokens": meta.get("prompt_tokens"),
           "cached_tokens_first_request": meta.get("cached_tokens"), "e2e_latency_s": meta.get("e2e_latency"),
           "per_request_logprob_count_histogram": {str(len(lps)): 1},
           "recomputed_suffix_histogram": {str(int(meta.get("prompt_tokens") or n) - int(meta.get("cached_tokens") or 0)): 1},
           "cached_tokens_histogram": {str(meta.get("cached_tokens")): 1},
           "request_shape": {"input_ids": f"ids[:{n}]", "max_new_tokens": used_max_new, "return_logprob": True,
                             "logprob_start_len": 0, "top_logprobs_num": int(top_logprobs), "read": "meta_info.input_token_logprobs[i][0]"}}
    tops = meta.get("input_top_logprobs")
    if top_logprobs and tops:
        agree = 0
        for i in idx:
            top = tops[i]
            if top and top[0] is not None and int(top[0][1]) == prompt[i]:
                agree += 1
        rep["top1_agreement"] = agree / steps
    else:
        rep["top1_agreement"] = None
    return rep


def _served_bookkeeping(rep: dict, recomputed, cached, lp_counts, prompt_lens, expect_T: int):
    rec_hist = Counter(recomputed)
    rep["recomputed_suffix_histogram"] = {str(k): v for k, v in sorted(rec_hist.items())}
    rep["cached_tokens_histogram"] = {str(k): v for k, v in sorted(Counter(cached).items())}
    rep["per_request_logprob_count_histogram"] = {str(k): v for k, v in sorted(Counter(lp_counts).items())}
    rep["cached_tokens_first_request"] = cached[0] if cached else None
    rep["recomputed_suffix_first_request"] = recomputed[0] if recomputed else None
    tail = recomputed[1:]
    rep["recomputed_suffix_mean_after_first"] = (sum(tail) / len(tail)) if tail else None
    rep["recomputed_suffix_max_after_first"] = max(tail) if tail else None
    rep["cache_engaged_every_request_after_first"] = bool(tail) and all(r == expect_T for r in tail)
    rep["validity_rule"] = (f"cached_tokens == prompt_tokens - {expect_T} on every request t >= 1 (uncached suffix == {expect_T}); "
                            "request 0 follows a /flush_cache and recomputes its whole prompt")
    if rep["cache_engaged_every_request_after_first"]:
        rep["mode_label"] = f"{rep['mode']} (T=={expect_T}: uncached suffix {expect_T} token{'s' if expect_T > 1 else ''} on every request after the first)"
    else:
        m = rep["recomputed_suffix_mean_after_first"]
        rep["mode_label"] = (f"{rep['mode']}-partial (radix cache did NOT give T=={expect_T}: mean recomputed suffix "
                             f"{m if m is None else round(m, 2)}, max {rep['recomputed_suffix_max_after_first']}) -- NOT a T=={expect_T} reading")
    return rep


def score_served(client: SGLangClient, ids, prompt_len: int, steps: int, flush: bool = True, progress_every: int = 256) -> dict:
    if flush:
        client.flush_cache()
    nll = 0.0
    agree = 0
    recomputed, cached, lp_counts, prompt_lens, e2e = [], [], [], [], []
    t0 = time.perf_counter()
    for t in range(steps):
        L = prompt_len + 1 + t                                                # prompt ends with cont[t] = ids[prompt_len + t]
        target = ids[prompt_len + 1 + t]                                      # K8 scores cont[t+1]
        out = client.generate({"input_ids": ids[:L],
                               "sampling_params": {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
                               "return_logprob": True, "logprob_start_len": -1, "token_ids_logprob": [int(target)],
                               "stream": False})
        if isinstance(out, list):
            out = out[0]
        meta = out["meta_info"]
        otl = meta.get("output_token_ids_logprobs")
        assert otl and otl[0], f"t={t}: no output_token_ids_logprobs in meta_info (keys {sorted(meta)})"
        lp, tok = _lp(otl[0][0])
        assert tok == target, f"t={t}: token_ids_logprob answered for id {tok}, asked {target}"
        assert int(meta.get("completion_tokens", -1)) == 1 and len(out.get("output_ids") or []) == 1, f"t={t}: expected exactly one generated token"
        nll -= lp
        greedy = int(out["output_ids"][0])
        agree += int(greedy == target)
        pt = int(meta.get("prompt_tokens") or L)
        ct = int(meta.get("cached_tokens") or 0)
        prompt_lens.append(pt)
        cached.append(ct)
        recomputed.append(pt - ct)
        lp_counts.append(len(meta.get("input_token_logprobs") or []))
        e2e.append(meta.get("e2e_latency"))
        if progress_every and (t + 1) % progress_every == 0:
            print(f"  served t={t + 1}/{steps} nll_so_far={nll / (t + 1):.5f} recomputed={recomputed[-1]} cached={ct} "
                  f"elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    wall = time.perf_counter() - t0
    mean_nll = nll / steps
    rep = {"mode": "served", "mean_nll": mean_nll, "ppl": math.exp(mean_nll), "n_requests": steps, "wall_s": round(wall, 3),
           "top1_agreement": agree / steps, "e2e_latency_mean_s": (sum(x for x in e2e if x is not None) / max(1, sum(1 for x in e2e if x is not None))) if any(x is not None for x in e2e) else None,
           "request_shape": {"input_ids": f"ids[:{prompt_len}+1+t]", "max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True,
                             "return_logprob": True, "logprob_start_len": -1, "token_ids_logprob": f"[ids[{prompt_len}+1+t]]",
                             "read": "meta_info.output_token_ids_logprobs[0][0][0]", "top1": "output_ids[0] == target (greedy)"},
           "forward_path_note": ("1-token extend over the cached prefix: SGLang EXTEND mode (prefill attention kernel with prefix), "
                                 "not DECODE mode (decode cuda graphs); Marlin MoE at M=1 either way")}
    return _served_bookkeeping(rep, recomputed, cached, lp_counts, prompt_lens, expect_T=1)


def score_served_tail2(client: SGLangClient, ids, prompt_len: int, steps: int, flush: bool = True, progress_every: int = 256) -> dict:
    if flush:
        client.flush_cache()
    nll = 0.0
    recomputed, cached, lp_counts, prompt_lens = [], [], [], []
    t0 = time.perf_counter()
    for t in range(steps):
        L = prompt_len + 2 + t                                                # prompt ends with the SCORED token ids[prompt_len+1+t]
        target_i = prompt_len + 1 + t
        payload = {"input_ids": ids[:L], "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
                   "return_logprob": True, "logprob_start_len": L - 2, "stream": False}
        try:
            out = client.generate(payload)
        except RuntimeError as e:
            if "max_new_tokens" not in str(e):
                raise
            payload["sampling_params"]["max_new_tokens"] = 1
            out = client.generate(payload)
        if isinstance(out, list):
            out = out[0]
        meta = out["meta_info"]
        lps = meta.get("input_token_logprobs") or []
        assert len(lps) >= 2 and lps[-1] is not None, f"t={t}: expected [None, (lp, id, _)] input logprobs, got {len(lps)} entries"
        lp, tok = _lp(lps[-1])
        assert tok == ids[target_i], f"t={t}: last input logprob is for id {tok}, scored token is {ids[target_i]}"
        nll -= lp
        pt = int(meta.get("prompt_tokens") or L)
        ct = int(meta.get("cached_tokens") or 0)
        prompt_lens.append(pt)
        cached.append(ct)
        recomputed.append(pt - ct)
        lp_counts.append(len(lps))
        if progress_every and (t + 1) % progress_every == 0:
            print(f"  served_tail2 t={t + 1}/{steps} nll_so_far={nll / (t + 1):.5f} recomputed={recomputed[-1]} lp_count={lp_counts[-1]}", flush=True)
    wall = time.perf_counter() - t0
    mean_nll = nll / steps
    rep = {"mode": "served_tail2", "mean_nll": mean_nll, "ppl": math.exp(mean_nll), "n_requests": steps, "wall_s": round(wall, 3),
           "top1_agreement": None,
           "request_shape": {"input_ids": f"ids[:{prompt_len}+2+t]", "max_new_tokens": payload["sampling_params"]["max_new_tokens"],
                             "return_logprob": True, "logprob_start_len": "prompt_tokens - 2",
                             "read": "meta_info.input_token_logprobs[-1][0]"},
           "forward_path_note": "2-token extend over the cached prefix (the scored position and the sampling slot); cross-check of `served`"}
    return _served_bookkeeping(rep, recomputed, cached, lp_counts, prompt_lens, expect_T=2)


def build_parser():
    p = argparse.ArgumentParser(description="SC1 SGLang teacher-forced NLL scorer (prefill-shaped and served-shaped)")
    p.add_argument("--window", required=True, help="k8_window_<src>.json ({ids, text_sha})")
    p.add_argument("--mode", choices=("prefill", "served", "served_tail2"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--prompt-len", type=int, default=None, help="default: the window file's prompt_len, else 512")
    p.add_argument("--steps", type=int, default=None, help="default: the window file's steps, else 2048")
    p.add_argument("--top-logprobs", type=int, default=1, help="prefill mode: top_logprobs_num for top-1 agreement (0 = off)")
    p.add_argument("--no-flush", action="store_true", help="skip the /flush_cache call before scoring")
    p.add_argument("--server-info", default=None, help="<log>.server_info.json from server.sh (else fetched live)")
    p.add_argument("--engagement", default=None)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--progress-every", type=int, default=256)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    w = json.load(open(a.window))
    prompt_len = a.prompt_len if a.prompt_len is not None else int(w.get("prompt_len", DEFAULT_PROMPT_LEN))
    steps = a.steps if a.steps is not None else int(w.get("steps", DEFAULT_STEPS))
    ids, sha, w = load_window(a.window, prompt_len, steps)
    client = SGLangClient(a.host, a.port, a.timeout)
    if not client.health():
        raise SystemExit(f"server at {client.base} is not healthy (/health)")
    info = json.load(open(a.server_info)) if a.server_info else client.server_info()
    eng = json.load(open(a.engagement)) if a.engagement else None
    if a.mode != "prefill" and info.get("disable_radix_cache"):
        raise SystemExit("served modes need the radix cache ON (server.sh mode `quality`); this server has disable_radix_cache=True")
    flush = not a.no_flush
    t0 = time.perf_counter()
    if a.mode == "prefill":
        rep = score_prefill(client, ids, prompt_len, steps, a.top_logprobs, flush)
    elif a.mode == "served":
        rep = score_served(client, ids, prompt_len, steps, flush, a.progress_every)
    else:
        rep = score_served_tail2(client, ids, prompt_len, steps, flush, a.progress_every)
    rep.update({
        "engine": "sglang", "version": info.get("version"), "steps": steps, "prompt_len": prompt_len,
        "scored_index_range": [prompt_len + 1, prompt_len + steps],
        "scored_index_rule": ("K8: cont = ids[prompt_len:]; step t feeds cont[t] = ids[prompt_len+t] and scores cont[t+1] = "
                              "ids[prompt_len+1+t], t in [0, steps); scored token indices are prompt_len+1 .. prompt_len+steps inclusive"),
        "tokens_scored": steps, "text_sha": sha, "window_file": os.path.abspath(a.window), "ppl_source": w.get("source") or w.get("ppl_source"),
        "server_info": _subset(info, SERVER_KEYS), "engagement": eng, "flushed_cache_first": flush,
        "wall_total_s": round(time.perf_counter() - t0, 3),
        "basis": ("teacher-forced through SGLang's /generate on the identical K8 window; prefill = one extend with input logprobs; "
                  "served = one request per scored token at the generation position with the radix cache carrying the prefix (PREREG SC1 Phase D)"),
    })
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print("SC1_SGLANG_NLL " + json.dumps({k: rep.get(k) for k in ("mode", "mode_label", "mean_nll", "ppl", "steps", "top1_agreement",
                                                                    "recomputed_suffix_histogram", "wall_s", "text_sha")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
