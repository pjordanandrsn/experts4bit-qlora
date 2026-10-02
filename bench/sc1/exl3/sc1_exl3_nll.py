"""sc1_exl3_nll.py -- teacher-forced NLL of a K8 window through ExLlamaV3 1.5.3, in two SHAPES (SC1 Phase D).

Input: k8_window_<src>.json = {"ids": [...], "text_sha": "..."} where text_sha is step_decomp._k8_window's digest:
sha256 of ids[:prompt_len + steps + 1] as int64 little-endian bytes (bench/p39/step_decomp.py:655-657; the ids come
from a torch LongTensor, so int64 LE). Recomputed here and REFUSED on mismatch (k8_gate's rule).

Scoring = step_decomp's K8 loop (step_decomp.py:1620-1641): cont = ids[P:P+S+1]; step t feeds cont[t] at position P+t
and scores cont[t+1] with fp32 log_softmax; mean over S steps; ppl = exp(mean). Here P=512, S=2048.

  prefill mode  -- ONE forward over ids[:P+S+1] with attn_mode "flash_attn_nc" (no cache; eval/ppl.py:252-253 is the
                   upstream ppl script's exact call); logits at positions P..P+S-1 predict ids[P+1..P+S]; vocabulary
                   cropped to config.vocab_size before the softmax (the model pads the logit width to a multiple of 32).
  decode mode   -- model.prefill(ids[:, :P]) into a rectangular-batch cache (attn_mode "flash_attn", batch_shape (1, L),
                   past_len) then ONE token per model.forward with past_len = P+t (examples/generation_loop.py:44-82 is
                   this exact loop; the attention runs the decode kernels, BC_Attention graphs by default). T == 1 at
                   every scored step: ExLlamaV3's cache is token-granular.

Capacity: the scorer sets its OWN cache independently of the arm defaults -- decode mode needs >= roundup256(P+S+1) =
2816 tokens for one sequence; SC1_EXL3_CACHE_TOKENS may raise it (multiple of 256, never below 2816 -> REFUSED) and the
value + source are recorded under receipt["capacity"]; prefill mode uses no cache (flash_attn_nc) and records that.

Receipt: {mean_nll, ppl, steps, prompt_len, mode, mode_available, text_sha, engine, version, logits_dtype, wall_s, capacity, ...}.
Env: SC1_WINDOW (path), SC1_OUT, SC1_MODEL / SC1_REV / SC1_MODEL_DIR as the arm; SC1_NLL_MODE = prefill|decode (or --mode).
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

PROMPT_LEN, STEPS = 512, 2048
PAGE_SIZE = 256
ENGINE = "exllamav3"
MODE_AVAILABLE = ["prefill", "decode"]
DEFAULT_MODEL = "turboderp/Qwen3-30B-A3B-exl3"
DEFAULT_REV = "0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef"


class Refusal(SystemExit):
    pass


NLL_CACHE_TOKENS_MIN = ((PROMPT_LEN + STEPS + 1 + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE     # 2816


def nll_cache_tokens(env=None) -> tuple[int, str]:
    """The scorer's own capacity: >= 2816 tokens (one sequence of P+S+1 = 2561, page-rounded), independent of the arm's
    B x 2048 rule. SC1_EXL3_CACHE_TOKENS may raise it; below the floor or off-page -> REFUSED."""
    env = os.environ if env is None else env
    ov = env.get("SC1_EXL3_CACHE_TOKENS")
    if ov:
        v = int(ov)
        if v % PAGE_SIZE or v < NLL_CACHE_TOKENS_MIN:
            raise Refusal(f"REFUSED: SC1_EXL3_CACHE_TOKENS={v} below the scorer floor {NLL_CACHE_TOKENS_MIN} or not a multiple of {PAGE_SIZE}")
        return v, "env:SC1_EXL3_CACHE_TOKENS"
    return NLL_CACHE_TOKENS_MIN, "default:roundup256(P+S+1)"


def text_sha(ids, prompt_len: int = PROMPT_LEN, steps: int = STEPS) -> str:
    n = prompt_len + steps + 1
    return hashlib.sha256(struct.pack(f"<{n}q", *[int(x) for x in ids[:n]])).hexdigest()


def load_window(path: str, prompt_len: int = PROMPT_LEN, steps: int = STEPS) -> dict:
    w = json.load(open(path))
    ids = [int(x) for x in w["ids"]]
    need = prompt_len + steps + 1
    if len(ids) < need:
        raise Refusal(f"REFUSED: window holds {len(ids)} ids, {need} needed for P={prompt_len} S={steps}")
    sha = text_sha(ids, prompt_len, steps)
    if sha != w.get("text_sha"):
        raise Refusal(f"REFUSED: text_sha {sha[:12]} != file {str(w.get('text_sha'))[:12]} (different window or tokenizer)")
    return {"ids": ids[:need], "text_sha": sha}


def nll_from_rows(logprob_rows, targets) -> float:
    """logprob_rows: callable(t) -> log-prob (float) of targets[t] at scored step t. Pure driver of the mean."""
    tot = 0.0
    for t, tgt in enumerate(targets):
        tot += -float(logprob_rows(t, tgt))
    return tot / len(targets)


def receipt(mode: str, mean_nll: float, sha: str, wall_s: float, extra: dict | None = None) -> dict:
    r = {"engine": ENGINE, "mode": mode, "mode_available": MODE_AVAILABLE, "steps": STEPS, "prompt_len": PROMPT_LEN,
         "mean_nll": mean_nll, "ppl": math.exp(mean_nll), "tokens_scored": STEPS, "text_sha": sha, "wall_s": round(wall_s, 2),
         "basis": ("teacher-forced; prefill: one flash_attn_nc forward, positions P..P+S-1 score ids[P+1..P+S]; "
                   "decode: prefill P then one token per forward with the rectangular-batch cache (T == 1)")}
    if extra:
        r.update(extra)
    return r


RECEIPT_KEYS = ("mean_nll", "ppl", "steps", "prompt_len", "mode", "mode_available", "text_sha", "engine", "version",
                "logits_dtype", "wall_s", "capacity")


# ------------------------------------------------------------------------------------------------ engine side

def resolve_model_dir(model: str, rev: str) -> str:
    d = os.environ.get("SC1_MODEL_DIR")
    if d:
        return d
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=rev, max_workers=4)


def score_prefill(model, config, ids, chunk: int = 256) -> tuple[float, str]:
    import torch
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        logits = model.forward(x, {"attn_mode": "flash_attn_nc"})          # [1, P+S+1, V_pad]
    dtype = str(logits.dtype)
    V = min(int(logits.shape[-1]), int(config.vocab_size))
    tgt = torch.tensor(ids[PROMPT_LEN + 1:PROMPT_LEN + STEPS + 1], device=logits.device)
    tot = 0.0
    for s in range(0, STEPS, chunk):
        e = min(STEPS, s + chunk)
        lg = logits[0, PROMPT_LEN + s:PROMPT_LEN + e, :V].float()
        lp = torch.log_softmax(lg, dim=-1)
        tot += -lp.gather(1, tgt[s:e].unsqueeze(1)).sum().item()
    return tot / STEPS, dtype


def score_decode(model, config, cache, ids, L: int) -> tuple[float, str]:
    import torch
    x = torch.tensor([ids], dtype=torch.long)
    V = int(config.vocab_size)
    tot, dtype = 0.0, None
    with torch.inference_mode():
        model.prefill(input_ids=x[:, :PROMPT_LEN], params={"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, L)})
        for t in range(STEPS):
            params = {"attn_mode": "flash_attn", "cache": cache, "past_len": PROMPT_LEN + t, "batch_shape": (1, L)}
            lg = model.forward(input_ids=x[:, PROMPT_LEN + t:PROMPT_LEN + t + 1], params=params)
            if dtype is None:
                dtype = str(lg.dtype)
            lp = torch.log_softmax(lg[0, -1, :V].float(), dim=-1)
            tot += -lp[ids[PROMPT_LEN + t + 1]].item()
    return tot / STEPS, dtype


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODE_AVAILABLE, default=os.environ.get("SC1_NLL_MODE", "prefill"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    win = load_window(os.environ["SC1_WINDOW"])
    model_id, rev = os.environ.get("SC1_MODEL", DEFAULT_MODEL), os.environ.get("SC1_REV", DEFAULT_REV)
    model_dir = resolve_model_dir(model_id, rev)
    import torch
    import exllamav3
    from exllamav3 import Cache, CacheLayer_fp16, Config, Model
    t0 = time.perf_counter()
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    L, cap_src = nll_cache_tokens()
    cache = Cache(model, max_num_tokens=L, layer_type=CacheLayer_fp16) if a.mode == "decode" else None
    model.load(progressbar=False)
    if os.environ.get("SC1_EXL3_NO_WARMUP", "0") == "0":
        model.warmup(cache=cache, max_batch_size=1)
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    if a.mode == "prefill":
        nll, dtype = score_prefill(model, config, win["ids"])
    else:
        nll, dtype = score_decode(model, config, cache, win["ids"], L)
    wall = time.perf_counter() - t1
    rec = receipt(a.mode, nll, win["text_sha"], wall, {
        "version": getattr(exllamav3, "__version__", None) or exllamav3.version.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "logits_dtype": dtype, "load_s": round(load_s, 1), "model": model_id, "revision": rev,
        "model_dir": model_dir, "vocab_size_scored": int(config.vocab_size), "cache_tokens": L if cache else None,
        "capacity": {"cache_tokens": L if cache else None, "cache_tokens_source": cap_src if cache else "prefill: no cache (flash_attn_nc)",
                     "floor": NLL_CACHE_TOKENS_MIN, "independent_of_arm_defaults": True, "batch_shape": [1, L] if cache else None},
        "cache_granularity_tokens": 1, "env_knobs": {k: os.environ.get(k) for k in ("EXL3_BC_ATTN", "EXL3_GEMV", "EXL3_INT8_GEMV")},
        "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))})
    json.dump(rec, open(os.environ["SC1_OUT"], "w"), indent=1)
    print(f"SC1EXL3_NLL mode={a.mode} nll={nll:.5f} ppl={rec['ppl']:.5f} dtype={dtype} sha={win['text_sha'][:12]}", flush=True)
    return 0


def selftest() -> int:
    import tempfile
    ids = [(i * 31 + 7) % 151936 for i in range(PROMPT_LEN + STEPS + 1)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"ids": ids, "text_sha": text_sha(ids)}, f)
    w = load_window(f.name)
    assert w["text_sha"] == text_sha(ids) and len(w["ids"]) == PROMPT_LEN + STEPS + 1
    with open(f.name, "w") as g:
        json.dump({"ids": ids, "text_sha": "0" * 64}, g)
    try:
        load_window(f.name)
        raise AssertionError("text_sha refusal did not fire")
    except Refusal:
        pass
    V = 1000
    nll = nll_from_rows(lambda t, tgt: -math.log(V), ids[PROMPT_LEN + 1:PROMPT_LEN + STEPS + 1])   # uniform -> ln V
    assert abs(nll - math.log(V)) < 1e-9
    assert nll_cache_tokens({}) == (2816, "default:roundup256(P+S+1)") and nll_cache_tokens({"SC1_EXL3_CACHE_TOKENS": "3072"})[0] == 3072
    for bad in ("2560", "3000"):
        try:
            nll_cache_tokens({"SC1_EXL3_CACHE_TOKENS": bad})
            raise AssertionError("bad capacity passed")
        except Refusal:
            pass
    r = receipt("prefill", nll, w["text_sha"], 0.1, {"version": "fake", "logits_dtype": "torch.float16",
                                                      "capacity": {"cache_tokens": None, "floor": NLL_CACHE_TOKENS_MIN}})
    missing = [k for k in RECEIPT_KEYS if k not in r]
    assert not missing and abs(r["ppl"] - V) < 1e-6, (missing, r["ppl"])
    print("SELFTEST OK sc1_exl3_nll")
    return 0


if __name__ == "__main__":
    sys.exit(main())
