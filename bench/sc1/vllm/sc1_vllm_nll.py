"""sc1_vllm_nll.py -- the quality scorer for the vLLM side of lane SC1 (experts4bit-qlora#846): teacher-forced NLL on
a K8 window (`k8_window_<src>.json` = {"ids": [...], "text_sha": "..."}; the sha is recomputed and asserted), in-process
`LLM`, two modes. Both score EXACTLY K8's 2048 tokens: step_decomp's loop feeds cont[t] = ids[512+t] and scores
cont[t+1] = ids[513+t] for t in 0..2047, i.e. tokens ids[513..2560], each conditioned on the corpus prefix.

  prefill  ONE generate on prompt_token_ids = ids[:2561] with SamplingParams(max_tokens=1, prompt_logprobs=0,
           temperature=0, detokenize=False). vLLM's prompt_logprobs list has one slot per prompt token, slot 0 None, slot i
           = log p(ids[i] | ids[:i]) with the ACTUAL token always present (+ its rank) -- read at the tag
           (gpu_model_runner.py:5684-5736, vllm/logprobs.py:167-172; V2 runner prompt_logprob.py:166-170). Scored slots
           513..2560. top-1 agreement = rank == 1. Prefix caching OFF (prompt_logprobs requests skip the lookup anyway).

  served   2048 SEQUENTIAL requests with prefix caching ON (F1): request t sends ids[:513+t] and reads
           log p(ids[513+t]) from the GENERATED position's distribution -- SamplingParams(max_tokens=1, logprobs=-1,
           temperature=0, detokenize=False) with LLM(max_logprobs=-1). The cache hit is a VALIDITY predicate: every
           request after the first must report num_cached_tokens >= prompt_len - block_size, else the row is relabelled
           prefill-shaped (no cache hit). vLLM's cache is block-granular (block 16 by default; the last token is always
           recomputed: kv_cache_manager.py:289-296), so the recomputed suffix is 1..16 and the mode is labelled
           `served (partial-block, M<=16)` -- never T == 1. top-1 agreement = generated (greedy) token == true token.
           logprobs=-1 is VERIFIED on the first request (entry count == vocab size): on the V2 model runner (the 0.30
           default) it is a real top-vocab gather; on the V1 runner the sampler returns an EMPTY container for -1
           (sampler.py:123-127 + logprobs.py:84-86). SC1_NLL_LOGPROBS=auto downgrades EXPLICITLY (recorded) to
           logprob_token_ids=[true] (exact; always present), never to an approximation; `topk:K` is the last resort and a
           row whose true token is absent is VOID.
           Optional KL dump: SC1_DUMP_LOGPROBS=1 (full mode only) writes the fp32 log-softmax vector per step to a .npy
           memmap -- 2048 x 151936 x 4 B = 1.24 GB; opt-in.

Env (SC1_* first, P37_* fallback): WINDOW, MODE=prefill|served, OUT  [PROMPT_LEN=512, STEPS=2048, KV=auto|fp8 (F9: the
fp8kv quality row; attention FLASHINFER pinned), NLL_EAGER=0, NLL_LOGPROBS=auto|full|token_ids|topk:K, DUMP_LOGPROBS=0,
DUMP_PATH, MODEL, REV, LOG, INSTANCE_ID, GPU_UTIL=0.90, ATTN_BACKEND, MOE_BACKEND=marlin, INPROC=0].
Output keys pair with k8_gate (text_sha, steps, ppl, ppl_source, mean_nll) plus mode, mode_label, scored_index_range,
suffix_histogram, top1_agreement, resolved config, engagement, wall_s.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import sc1_vllm_common as C  # noqa: E402

if __name__ == "__main__":
    C.early_setup()

import json  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

MODES = ("prefill", "served")


def main():
    mode = C.env("MODE", required=True)
    assert mode in MODES, f"SC1_MODE must be one of {MODES}"
    out_path = C.env("OUT", required=True)
    log = C.env("LOG")
    P, S = C.env_int("PROMPT_LEN", C.PROMPT_LEN), C.env_int("STEPS", 2048)
    ids, winfo = C.load_window(C.env("WINDOW", required=True), P, S)
    kv = C.env("KV", "auto")
    assert kv in ("auto", "fp8")
    model, rev = C.env("MODEL", C.MODEL_DEFAULT), C.env("REV", C.REV_DEFAULT)
    lp_req = C.env("NLL_LOGPROBS", "auto")
    if C.env("NAMED_REF") and mode == "served":
        lp_req = "named"                    # SC1g A4: the reference's 64 named tokens per position + the target
    if C.env("REF_FULL") and mode == "served":
        lp_req = "full"                     # SC1g A5: logprobs=-1 with LLM(max_logprobs=-1), verified full on request 0, NO downgrade
    assert lp_req in ("auto", "full", "token_ids", "named") or lp_req.startswith("topk:"), f"bad SC1_NLL_LOGPROBS {lp_req!r}"

    base_arm = "fp8kv" if kv == "fp8" else ("eager" if C.env("NLL_EAGER", "0") == "1" else "graph_r1")
    kw = C.build_llm_kwargs(base_arm, 1, model, rev, gpu_util=C.env_float("GPU_UTIL", 0.90), max_len=P + S + 16,
                            prompt_len=P + S + 1, attn=C.env("ATTN_BACKEND"), moe=C.env("MOE_BACKEND", "marlin"))
    kw["max_num_seqs"] = 1
    kw["max_num_batched_tokens"] = C.capped_batched_tokens(1, kw["max_model_len"], max(8192, P + S + 1))   # A5
    if mode == "served":
        kw["enable_prefix_caching"] = True
        if lp_req in ("auto", "full", "named"):
            kw["max_logprobs"] = -1
        elif lp_req.startswith("topk:"):
            kw["max_logprobs"] = max(20, int(lp_req.split(":", 1)[1]))
    else:
        kw["enable_prefix_caching"] = False

    rec = C.base_receipt("nll", f"{mode}_{kv}", kw, {
        "k8": "ppl", "mode": mode, "kv_cache_dtype_requested": kw["kv_cache_dtype"], "model": model, "revision": rev,
        "prompt_len": P, "steps": S, "status": "ok", **winfo, "logprobs_mode_requested": lp_req,
        "scored_index_range": [P + 1, P + S],
        "generation": {"temperature": 0.0, "max_tokens": 1, "greedy": True, "detokenize": False,
                       "note": "a quality instrument, untimed: detokenize=False so a full-vocab logprobs request does not "
                               "detokenise ~152k ids per step (output_processor.py:234-242 drops the tokenizer); the "
                               "scored distribution is unaffected"},
        "k8_equivalence": f"K8 scores cont[t+1] = ids[{P + 1}+t] for t in 0..{S - 1} given ids[:{P + 1}+t]; this scorer reads the same "
                          f"{S} conditionals from vLLM's own forward (prefill: prompt_logprobs slots; served: the generated position)",
    })
    try:
        _run(rec, mode, ids, P, S, kw, lp_req, log, out_path)
    except BaseException as e:  # noqa: BLE001
        rec["status"] = "harness_error"
        rec["error"] = repr(e)[:400]
        C.flush_log()
        rec["engagement"] = C.grep_engagement(log)
        C.write_receipt(out_path, rec)
        raise


def _run(rec, mode, ids, P, S, kw, lp_req, log, out_path):
    t0 = time.perf_counter()
    C.check_budget(kw)
    llm = LLM(**kw)
    rec["load_s"] = round(time.perf_counter() - t0, 1)
    rec.update(C.resolved_config(llm, kw))
    C.flush_log()
    rec["engagement"] = C.grep_engagement(log)
    rec["mem_after_load"] = C.mem_snapshot(llm, kw.get("gpu_memory_utilization"))
    res = rec.get("resolved") or {}
    vocab = res.get("vocab_size")
    block = int(res.get("block_size") or C.DEFAULT_BLOCK_SIZE)
    rec["block_size"] = block
    rec["block_size_source"] = "cache_config.block_size" if res.get("block_size") else "DEFAULT_BLOCK_SIZE (config/cache.py:71)"

    t_run = time.perf_counter()
    if mode == "prefill":
        _prefill(rec, llm, ids, P, S)
    else:
        _served(rec, llm, ids, P, S, lp_req, vocab, block, out_path)
    rec["wall_s"] = round(time.perf_counter() - t_run, 2)

    rec["mem_after_runs"] = C.mem_snapshot(llm, kw.get("gpu_memory_utilization"))
    C.flush_log()
    rec["engagement"] = C.grep_engagement(log)
    C.write_receipt(out_path, rec)
    print("SC1VLLMNLL " + json.dumps({k: rec.get(k) for k in ("mode", "mode_label", "mean_nll", "ppl", "steps", "tokens_scored",
                                                               "top1_agreement", "text_sha", "ppl_source", "verdict",
                                                               "logprobs_mode_used", "void_rows", "cache_predicate", "wall_s")}),
          flush=True)


def _prefill(rec, llm, ids, P, S):
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0, detokenize=False, flat_logprobs=True)
    o = llm.generate([{"prompt_token_ids": list(ids[:P + S + 1])}], sp, use_tqdm=False)[0]
    C.assert_generated([o], 1, 1)
    rec["prompt_tokens_engine"] = len(getattr(o, "prompt_token_ids", None) or [])
    rec["num_cached_tokens"] = getattr(o, "num_cached_tokens", None)
    if rec["prompt_tokens_engine"] != P + S + 1:
        rec["status"] = "void"
        rec["void_reason"] = f"engine prompt-token count {rec['prompt_tokens_engine']} != {P + S + 1} (F20)"
    rec.update(C.prefill_nll(o.prompt_logprobs, ids, P, S))
    rec["mode_label"] = f"prefill-shaped (one forward over {P + S + 1} prompt tokens, prompt_logprobs=0)"
    rec["verdict"] = "VALID" if rec["status"] == "ok" else "VOID"


def _served(rec, llm, ids, P, S, lp_req, vocab, block, out_path):
    used = "full" if lp_req in ("auto", "full") else lp_req
    downgrade = None
    named_ids = named = None
    kl = None
    if C.env("REF_FULL"):
        import numpy as np
        _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import sc1g_kl
        ref_path = C.env("REF_FULL", required=True)
        want = C.env("REF_FULL_SHA")
        if want and sc1g_kl.file_sha(ref_path) != want:
            raise SystemExit(f"{ref_path}: sha differs from the registered {want[:16]} -- refusing the reference")
        ref_full = np.load(ref_path, mmap_mode="r")
        if ref_full.shape[0] != S or (vocab and ref_full.shape[1] != int(vocab)):
            raise SystemExit(f"{ref_path}: shape {ref_full.shape} does not fit {S} steps x vocab {vocab} -- refused")
        kl = {"kl": np.full(S, np.nan), "tlp": np.full(S, np.nan), "mod": sc1g_kl}
    if used == "named":
        import numpy as np
        named_ids, ref_targets = C.load_named_ref(C.env("NAMED_REF", required=True), C.env("NAMED_REF_SHA"))
        if named_ids.shape[0] != S or [int(x) for x in ref_targets] != [C.served_target(ids, P, t) for t in range(S)]:
            raise SystemExit("SC1_NAMED_REF does not describe this window's scored targets -- refused")
        named = {"lp": np.full(named_ids.shape, np.nan), "tlp": np.full(S, np.nan), "void": 0}
    dump, dump_path = None, None
    if C.env("DUMP_LOGPROBS", "0") == "1":
        if used != "full":
            rec["logprobs_dump"] = {"skipped": f"dump needs the full-vocab mode, requested {lp_req}"}
        elif not vocab:
            rec["logprobs_dump"] = {"skipped": "vocab size unreadable from model_config"}
        else:
            import numpy as np
            dump_path = C.env("DUMP_PATH", out_path + ".logprobs.npy")
            dump = np.lib.format.open_memmap(dump_path, mode="w+", dtype=np.float32, shape=(S, int(vocab)))
    rows, walls = [], []
    t = 0
    while t < S:
        prompt = C.served_prompt(ids, P, t)
        target = C.served_target(ids, P, t)
        sp = SamplingParams(**C.served_sampling_kwargs(used, target, None if named_ids is None else named_ids[t]))
        tq = time.perf_counter()
        try:
            o = llm.generate([{"prompt_token_ids": prompt}], sp, use_tqdm=False)[0]
        except Exception as e:  # noqa: BLE001 -- e.g. a validation refusal of logprobs=-1
            if used == "full" and lp_req == "auto":
                downgrade = f"logprobs=-1 refused at request {t}: {e!r}"[:300]
                used = "token_ids"
                dump = _drop_dump(rec, dump, dump_path, "downgraded off the full-vocab mode")
                continue
            raise
        walls.append(time.perf_counter() - tq)
        row, entries = C.served_row(o, len(prompt), target)
        if t == 0 and used == "full":
            full_ok = bool(vocab) and row["n_entries"] == int(vocab)
            rec["full_vocab_verified"] = full_ok
            rec["first_request_entries"] = row["n_entries"]
            if not full_ok and lp_req == "auto":
                downgrade = (f"logprobs=-1 returned {row['n_entries']} entries, not vocab_size {vocab} (the V1 model runner "
                             f"returns an empty container for -1 at v0.30.0: v1/sample/sampler.py:123-127 + "
                             f"v1/engine/logprobs.py:84-86; use_v2_model_runner={(rec.get('resolved') or {}).get('use_v2_model_runner')}); "
                             f"switching to logprob_token_ids=[true] and re-issuing request 0")
                used = "token_ids"
                rec["downgrade_reissued_row0"] = True
                dump = _drop_dump(rec, dump, dump_path, "full-vocab mode did not verify")
                continue
        if kl is not None and entries is not None and rec.get("full_vocab_verified"):
            import numpy as np
            v64 = np.full(int(vocab), -np.inf, dtype=np.float64)
            v64[np.asarray(entries[0], dtype=np.int64)] = np.asarray(entries[1], dtype=np.float64)
            kl["kl"][t] = float(kl["mod"].kl_full_rows(ref_full[t:t + 1], v64[None])[0])
            if row["nll"] is not None:
                kl["tlp"][t] = -row["nll"]
        if dump is not None and entries is not None:
            import numpy as np
            vec = np.full(int(vocab), -np.inf, dtype=np.float32)
            vec[np.asarray(entries[0], dtype=np.int64)] = np.asarray(entries[1], dtype=np.float32)
            dump[t] = vec
        if named is not None:
            got = C.named_lps(entries, named_ids[t])
            if got is None:
                named["void"] += 1
            else:
                named["lp"][t] = got
            if row["nll"] is not None:
                named["tlp"][t] = -row["nll"]
        rows.append(row)
        t += 1
        if t % 256 == 0 or t == S:
            valid = [r["nll"] for r in rows if r["nll"] is not None]
            print(f"SC1VLLMNLL served t={t}/{S} mean_nll={sum(valid) / max(len(valid), 1):.5f} suffix_last={row['suffix']} "
                  f"cached_last={row['num_cached_tokens']} mode={used} req_wall_ms={1e3 * walls[-1]:.1f}", flush=True)
    rec.update(C.served_reduce(rows, block, S))
    rec["logprobs_mode_used"] = used
    rec["downgrade_reason"] = downgrade
    rec["request_wall_s_mean"] = round(sum(walls) / max(len(walls), 1), 4)
    rec["request_wall_s_median"] = round(sorted(walls)[len(walls) // 2], 4) if walls else None
    rec["index_derivation"] = (f"request t: prompt = ids[:{P + 1}+t], the generated position's distribution gives "
                               f"log p(ids[{P + 1}+t] | ids[:{P + 1}+t]) == K8's cont[t+1]; t in 0..{S - 1}; the engine's greedy "
                               f"token is discarded (teacher forcing by construction) and compared for top-1 agreement")
    if kl is not None:
        import numpy as np
        kout = C.env("KL_OUT", required=True)
        if not rec.get("full_vocab_verified"):
            rec["kl_full"] = {"verdict": "VOID", "why": f"logprobs=-1 returned {rec.get('first_request_entries')} entries, not the "
                                                         f"vocabulary ({vocab}): no full-vocabulary KL (A5 never downgrades)"}
        else:
            np.savez(kout + ".tmp.npz", eng_kl=kl["kl"], eng_target_lp=kl["tlp"])
            os.replace(kout + ".tmp.npz", kout)
            rec["kl_full"] = {"out": kout, "positions": S, "ref": C.env("REF_FULL"), "ref_sha": C.env("REF_FULL_SHA"),
                              "request": "logprobs=-1, LLM(max_logprobs=-1); full vocabulary verified on request 0"}
    if named is not None:
        import numpy as np
        nout = C.env("NAMED_OUT", required=True)
        np.savez(nout + ".tmp.npz", eng_lp=named["lp"], eng_target_lp=named["tlp"])
        os.replace(nout + ".tmp.npz", nout)
        rec["named"] = {"out": nout, "positions": S, "void_positions": named["void"], "ref": C.env("NAMED_REF"),
                        "ref_sha": C.env("NAMED_REF_SHA"), "k": int(named_ids.shape[1]),
                        "request": "logprob_token_ids = the reference's named ids for the position + the target"}
    if dump is not None:
        dump.flush()
        rec["logprobs_dump"] = {"path": dump_path, "shape": [S, int(vocab)], "dtype": "float32", "bytes": S * int(vocab) * 4,
                                "note": "row t = log-softmax over the vocab at the generated position of request t "
                                        "(prompt ids[:513+t]); -inf where the engine returned no entry; for KL vs the bf16 reference"}
    if rec["status"] == "ok" and rec.get("verdict") == "VOID":
        rec["status"] = "void"
        rec["void_reason"] = f"{rec.get('void_rows')} rows had no logprob for the true token (never approximated)"


def _drop_dump(rec, dump, path, why):
    if dump is None:
        return None
    del dump
    try:
        os.remove(path)
    except OSError:
        pass
    rec["logprobs_dump"] = {"skipped": why}
    return None


if __name__ == "__main__":
    main()
