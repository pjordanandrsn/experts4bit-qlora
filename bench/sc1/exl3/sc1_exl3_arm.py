"""sc1_exl3_arm.py -- the ExLlamaV3 arm of lane SC1 (experts4bit-qlora#846), p37-shaped (bench/h2h-20260905/p37/p37_vllm.py).

Contract (identical to the vLLM arm): the prompts are the EXACT token ids step_decomp._k8_window produced
(prompts_b{B}.json: `prompts` = list of 512-token id rows, `prompts_sha256`, `rows_sha256`, `batch`), fed to the
engine as token ids -- no tokenizer runs here. Decode is isolated by the SLOPE method: generate 32 and 128 tokens
from the same prompts, one warm + 3 timed reps per length; extra tokens / extra wall. min-of-3 and median-of-3 both
recorded. Every engine knob is recorded from the engine's own objects.

ExLlamaV3 1.5.3 facts the arm is built on (file:line in turboderp-org/exllamav3 @ v1.5.3, d3739fd3):
  * token-id prompts: `Generator.generate()` takes str/tuple only (generator/generator.py:1577-1598), so each row is a
    `Job(input_ids=<[1,512] LongTensor>, ...)` (generator/job.py:47-69), `enqueue`d and driven by `iterate()`.
  * one batched step per token: `iterate_gen` builds ONE compact batch of every prefill-done job and runs ONE
    `model.forward(attn_mode="flash_attn", block_table, cache_seqlens)` per iteration (generator/generator.py:959-1073).
  * EOS ignored: `stop_conditions=None` leaves `Job.stop_tokens` EMPTY (job.py:233-246); the only end is
    `new_tokens >= max_new_tokens` -> eos_reason "max_new_tokens" (job.py:832-835). `min_new_tokens` is set too.
  * greedy: `ArgmaxSampler` (== `GreedySampler`, generator/sampler/presets.py:14-23).
  * no speculative decoding unless `draft_model`/`ngram_match_min` are given (generator.py:36-45); none here.
  * prefix reuse: `enqueue` hashes prompt pages for cache lookup (generator.py:410-413); a FRESH Generator (new
    PageTable, generator.py:149) is built per timed call so no prompt page is reused across requests; every job's
    EOS result carries `cached_pages`/`cached_tokens` (job.py:764-777) and the receipt records them (expected 0).
  * cache: `Cache(model, max_num_tokens)` must be a multiple of PAGE_SIZE=256 (cache/cache.py:115-118,
    constants.py:2); for the generator it is the TOTAL tokens across concurrent jobs.
  * attention: ExLlamaV3 1.5.3 imports neither flash-attn nor xformers (removed in v1.0.0); attention is its own
    Triton paged kernels + graph-captured "BC_Attention" decode (EXL3_BC_ATTN=1 default, doc/env_vars.md:12-26).

Env: SC1_ARM, SC1_BATCH, SC1_PROMPTS, SC1_MODEL (default turboderp/Qwen3-30B-A3B-exl3), SC1_REV (default the
`4.0bpw` branch commit 0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef; the branch is head_bits 6), SC1_OUT, SC1_INSTANCE_ID.
Optional: SC1_MODEL_DIR (skip the hub fetch), SC1_EXL3_CACHE_QUANT ("8" or "4,4" -> CacheLayer_quant), SC1_EXL3_FRESH_GENERATOR
(default 1), SC1_EXL3_NO_WARMUP (default 0). Modes: default slope; `--ttft` (one 512- or 4096-token row, wall to first
token, 3 timed repeats after one warm, max_new_tokens=8); `--native-loop` (B=1 only: upstream eval/perf.py:140-156's raw
`model.forward` loop with argmax feedback, labelled, the kernel-level ceiling); `--selftest` (no engine).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

SHORT, LONG = 32, 128
PROMPT_LEN = 512
PAGE_SIZE = 256
ENGINE = "exllamav3"
DEFAULT_MODEL = "turboderp/Qwen3-30B-A3B-exl3"
DEFAULT_REV = "0b83e92c6d3b5a868ecd5a5fbb3bcc1920e388ef"   # branch 4.0bpw (head_bits 6), read from the HF refs API 2026-10-01
EXL3_ENV_KNOBS = ("EXL3_BC_ATTN", "EXL3_GEMV", "EXL3_INT8_GEMV", "EXL3_INT8_GEMV_MAX_K", "EXL3_MOE_FUSED_ROWS",
                  "EXL3_MOE_FUSED_ROWS_WIDE", "EXL3_MOE_MTILE", "EXL3_MOE_FUSED_DET", "EXL3_MOE_BATCH_RECON",
                  "EXL3_MOE_SHARED_COOP", "EXL3_PREFER_FA2", "EXL3_FUSED_SAMPLER", "EXL3_EXPANDABLE_SEGMENTS",
                  "TORCH_CUDA_ARCH_LIST", "PYTORCH_CUDA_ALLOC_CONF", "EXLLAMAV3_TUNE_CACHE")


class Refusal(SystemExit):
    """A validity predicate failed: the arm refuses to produce a number (never a partial receipt)."""


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (CPU-only, exercised by tests/test_sc1_exl3_lmdeploy.py and --selftest)
# ----------------------------------------------------------------------------------------------------------------

def prompts_digest(prompts) -> str:
    """p37's expression, byte for byte: sha256 of json.dumps(prompts)."""
    return hashlib.sha256(json.dumps(prompts).encode()).hexdigest()


def load_prompts(path: str, batch: int, prompt_len: int | None = PROMPT_LEN) -> dict:
    """Read prompts_b{B}.json and REFUSE on every contract breach p37 asserts (batch, row length, distinct rows,
    digest). prompt_len=None accepts any single length (the --ttft 4096 file)."""
    pf = json.load(open(path))
    prompts = pf["prompts"]
    if pf.get("batch") != batch or len(prompts) != batch:
        raise Refusal(f"REFUSED: prompt file batch {pf.get('batch')}/{len(prompts)} != SC1_BATCH {batch}")
    if prompt_len is not None and not all(len(p) == prompt_len for p in prompts):
        raise Refusal(f"REFUSED: prompt_len must be {prompt_len}, got {[len(p) for p in prompts]}")
    if len(set(tuple(p) for p in prompts)) != batch:
        raise Refusal("REFUSED: rows must be distinct prompts (P20's distinct-prompt rule)")
    sha = prompts_digest(prompts)
    if sha != pf.get("prompts_sha256"):
        raise Refusal(f"REFUSED: prompt file digest mismatch {sha[:12]} != {str(pf.get('prompts_sha256'))[:12]}")
    return {"prompts": prompts, "prompts_sha256": sha, "rows_sha256": pf.get("rows_sha256"),
            "prompt_tokens": [len(p) for p in prompts]}


def check_tokens(gen: dict, n_tokens: int, batch: int) -> int:
    """Exactly N tokens per row, B rows -- a row that stopped early is not the registered workload."""
    if len(gen) != batch:
        raise Refusal(f"REFUSED: {len(gen)} rows generated != batch {batch}")
    got = sum(len(v) for v in gen.values())
    if got != n_tokens * batch or any(len(v) != n_tokens for v in gen.values()):
        raise Refusal(f"REFUSED: {got} != {n_tokens * batch} tokens (a row stopped early: not the registered workload)")
    return got


def slope(w_short, w_long, batch: int, short: int = SHORT, long: int = LONG) -> dict:
    """p37's slope arithmetic: extra tokens / extra wall, min-of-3 (P20 estimator) and median-of-3."""
    extra = (long - short) * batch
    d_min = min(w_long) - min(w_short)
    d_med = statistics.median(w_long) - statistics.median(w_short)
    if d_min <= 0 or d_med <= 0:
        raise Refusal(f"REFUSED: non-positive slope (short {w_short}, long {w_long}); the instrument cannot see decode")
    return dict(
        walls_short_s=[round(w, 4) for w in w_short], walls_long_s=[round(w, 4) for w in w_long],
        wall_short_s=round(min(w_short), 4), wall_long_s=round(min(w_long), 4),
        decode_tok_s=round(extra / d_min, 1), decode_ms_per_step=round(d_min / (long - short) * 1e3, 4),
        decode_tok_s_median=round(extra / d_med, 1), decode_ms_per_step_median=round(d_med / (long - short) * 1e3, 4),
        end_to_end_tok_s_long=round(long * batch / min(w_long), 1),
    )


def run_slope(engine, prompts, batch: int, reps: int = 3, short: int = SHORT, long: int = LONG) -> dict:
    """One warm (untimed) + `reps` timed generations per length; the N-token assertion on every timed rep."""
    def run(n):
        engine.generate(prompts, n)                                  # warm (untimed)
        walls, gen, meta = [], None, None
        for _ in range(reps):
            t = time.perf_counter()
            rows, meta = engine.generate(prompts, n)
            walls.append(time.perf_counter() - t)
            gen = {str(i): [int(x) for x in r] for i, r in enumerate(rows)}
            check_tokens(gen, n, batch)
        return walls, gen, meta
    w_short, _, _ = run(short)
    w_long, gen_long, meta_long = run(long)
    out = slope(w_short, w_long, batch, short, long)
    out["tokens"] = gen_long
    out["engine_meta_long"] = meta_long
    return out


def run_ttft(engine, prompt, reps: int = 3, max_new_tokens: int = 8) -> dict:
    """Wall to the first token (warm engine, fresh request state); 3 timed repeats after one warm."""
    engine.first_token(prompt, max_new_tokens)                       # warm (untimed)
    walls, engine_ttft, n_toks = [], [], []
    for _ in range(reps):
        w, e, n = engine.first_token(prompt, max_new_tokens)
        walls.append(w)
        engine_ttft.append(e)
        n_toks.append(n)
    if any(n < 1 for n in n_toks):
        raise Refusal(f"REFUSED: a TTFT request produced no token ({n_toks})")
    return dict(ttft_prompt_tokens=len(prompt), max_new_tokens=max_new_tokens, ttft_walls_s=[round(w, 4) for w in walls],
                ttft_s=round(min(walls), 4), ttft_s_median=round(statistics.median(walls), 4),
                ttft_engine_reported_s=[None if e is None else round(e, 4) for e in engine_ttft],
                prefill_tok_s=round(len(prompt) / min(walls), 1), tokens_generated=n_toks,
                method="wall from request submission to the first streamed token, engine warm, fresh generator per "
                       "repeat (no prompt-page reuse); engine-reported = Job.time_prefill at EOS")


def base_receipt(arm: str, batch: int, pf: dict, model: str, rev: str, mode: str) -> dict:
    return {"engine": ENGINE, "arm": arm, "mode": mode, "model": model, "revision": rev, "batch": batch,
            "method": "slope(32->128) isolates decode; min-of-3 (P20 estimator) and median-of-3 both recorded",
            "prompts": "identical token ids to the e4b arms (step_decomp._k8_window, wikitext-2 test, 512-token rows)",
            "prompts_sha256": pf["prompts_sha256"], "rows_sha256": pf["rows_sha256"], "prompt_tokens": pf["prompt_tokens"],
            "generation": {"sampler": "ArgmaxSampler (greedy)", "ignore_eos": "stop_conditions=None (Job.stop_tokens empty)",
                           "min_new_tokens": "= max_new_tokens", "speculative": False, "prefix_cache_reuse": "fresh Generator per call"},
            "vast_instance_id": os.environ.get("SC1_INSTANCE_ID"),
            "env_knobs": {k: os.environ.get(k) for k in EXL3_ENV_KNOBS}}


RECEIPT_KEYS_SLOPE = ("engine", "arm", "mode", "model", "revision", "batch", "prompts_sha256", "rows_sha256", "load_s",
                      "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median", "decode_ms_per_step_median",
                      "end_to_end_tok_s_long", "walls_short_s", "walls_long_s", "tokens", "resolved", "versions",
                      "vram_peak_bytes")


# ----------------------------------------------------------------------------------------------------------------
# the real engine adapter (imports exllamav3 lazily; nothing below runs on a box without the engine)
# ----------------------------------------------------------------------------------------------------------------

def resolve_model_dir(model: str, rev: str) -> str:
    d = os.environ.get("SC1_MODEL_DIR")
    if d:
        return d
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=rev, max_workers=4)


def cache_tokens_for(batch: int, prompt_len: int, max_new: int) -> int:
    per_row = ((prompt_len + max_new + 1 + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    return per_row * batch


class Exl3Engine:
    """Generator/Job-driven adapter. `generate(prompts, n)` -> (rows, meta); `first_token(prompt, n)` -> (wall, engine_ttft, n)."""

    def __init__(self, model_dir: str, batch: int, prompt_len: int, max_new: int, warmup: bool = True):
        import torch
        import exllamav3
        from exllamav3 import Cache, CacheLayer_fp16, CacheLayer_quant, Config, Model, Tokenizer
        self.torch, self.exl = torch, exllamav3
        self.batch, self.model_dir = batch, model_dir
        torch.cuda.reset_peak_memory_stats()
        self.free0 = torch.cuda.mem_get_info()[0]
        self.min_free = self.free0
        t0 = time.perf_counter()
        self.config = Config.from_directory(model_dir)
        self.model = Model.from_config(self.config)
        self.cache_tokens = cache_tokens_for(batch, prompt_len, max_new)
        cq = os.environ.get("SC1_EXL3_CACHE_QUANT")
        if cq:
            bits = [int(b) for b in cq.split(",")]
            k_bits, v_bits = (bits[0], bits[0]) if len(bits) == 1 else (bits[0], bits[1])
            self.cache = Cache(self.model, max_num_tokens=self.cache_tokens, layer_type=CacheLayer_quant, k_bits=k_bits, v_bits=v_bits)
            self.cache_desc = f"CacheLayer_quant k{k_bits}/v{v_bits}"
        else:
            self.cache = Cache(self.model, max_num_tokens=self.cache_tokens, layer_type=CacheLayer_fp16)
            self.cache_desc = "CacheLayer_fp16"
        self.model.load(progressbar=False, max_batch_size=batch)
        if warmup:
            self.model.warmup(cache=self.cache, max_batch_size=max(8, batch))
        self.tokenizer = Tokenizer.from_config(self.config)
        self.load_s = round(time.perf_counter() - t0, 1)
        self.fresh = os.environ.get("SC1_EXL3_FRESH_GENERATOR", "1") != "0"
        self.generator = None
        self._gen_kwargs = dict(max_batch_size=batch, max_chunk_size=2048)

    def _new_generator(self):
        from exllamav3 import Generator
        if self.generator is None or self.fresh:
            self.generator = Generator(self.model, self.cache, self.tokenizer, **self._gen_kwargs)
        return self.generator

    def _sample_vram(self):
        f = self.torch.cuda.mem_get_info()[0]
        self.min_free = min(self.min_free, f)

    def generate(self, prompts, n_tokens: int):
        from exllamav3 import ArgmaxSampler, Job
        torch = self.torch
        g = self._new_generator()
        jobs = [Job(input_ids=torch.tensor([p], dtype=torch.long), max_new_tokens=n_tokens, min_new_tokens=n_tokens,
                    sampler=ArgmaxSampler(), stop_conditions=None, identifier=i) for i, p in enumerate(prompts)]
        g.enqueue(jobs)
        rows = [[] for _ in prompts]
        meta = {"eos_reason": [None] * len(prompts), "cached_tokens": [None] * len(prompts), "cached_pages": [None] * len(prompts),
                "time_prefill_s": [None] * len(prompts), "time_generate_s": [None] * len(prompts)}
        while g.num_remaining_jobs():
            for r in g.iterate():
                if r["stage"] == "error":
                    raise r["error"]
                if r["stage"] != "streaming":
                    continue
                i = r["identifier"]
                if "token_ids" in r:
                    rows[i] += r["token_ids"].flatten().tolist()
                held = r.get("held") or {}
                if "token_ids" in held:
                    rows[i] += held["token_ids"].flatten().tolist()
                if r.get("eos"):
                    meta["eos_reason"][i] = r.get("eos_reason")
                    meta["cached_tokens"][i] = r.get("cached_tokens")
                    meta["cached_pages"][i] = r.get("cached_pages")
                    meta["time_prefill_s"][i] = r.get("time_prefill")
                    meta["time_generate_s"][i] = r.get("time_generate")
        torch.cuda.synchronize()
        self._sample_vram()
        bad = [i for i, e in enumerate(meta["eos_reason"]) if e != "max_new_tokens"]
        if bad:
            raise Refusal(f"REFUSED: rows {bad} ended for {[meta['eos_reason'][i] for i in bad]}, not max_new_tokens")
        return rows, meta

    def first_token(self, prompt, max_new_tokens: int = 8):
        from exllamav3 import ArgmaxSampler, Job
        torch = self.torch
        g = self._new_generator()
        job = Job(input_ids=torch.tensor([prompt], dtype=torch.long), max_new_tokens=max_new_tokens,
                  min_new_tokens=max_new_tokens, sampler=ArgmaxSampler(), stop_conditions=None, identifier=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.enqueue(job)
        first, n, engine_ttft = None, 0, None
        while g.num_remaining_jobs():
            for r in g.iterate():
                if r["stage"] == "error":
                    raise r["error"]
                if r["stage"] != "streaming":
                    continue
                k = 0
                if "token_ids" in r:
                    k += r["token_ids"].numel()
                if "token_ids" in (r.get("held") or {}):
                    k += r["held"]["token_ids"].numel()
                if k and first is None:
                    torch.cuda.synchronize()
                    first = time.perf_counter() - t0
                n += k
                if r.get("eos"):
                    engine_ttft = r.get("time_prefill")
        self._sample_vram()
        return first, engine_ttft, n

    def native_loop(self, prompt, n_tokens: int, reps: int = 3):
        """Upstream perf.py's measurement shape: raw model.forward one token at a time with argmax feedback,
        rectangular batch_shape cache, one .cpu() sync per step. B=1 only. Walls for n_tokens steps (after prefill)."""
        torch = self.torch
        L = ((len(prompt) + n_tokens + 1 + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
        if L > self.cache.max_num_tokens:
            raise Refusal(f"REFUSED: native loop needs {L} cache tokens, cache has {self.cache.max_num_tokens}")
        ids = torch.tensor([prompt], dtype=torch.long)
        walls, toks = [], None
        for _ in range(reps + 1):
            params = {"attn_mode": "flash_attn", "cache": self.cache, "past_len": 0, "batch_shape": (1, L)}
            self.model.prefill(input_ids=ids[:, :-1], params=params)
            cur = ids[:, -1:]
            out = []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(n_tokens):
                params = {"attn_mode": "flash_attn", "cache": self.cache, "past_len": ids.shape[-1] - 1 + i, "batch_shape": (1, L)}
                logits = self.model.forward(input_ids=cur, params=params)
                nxt = torch.argmax(logits[:, -1, :self.config.vocab_size], dim=-1).cpu()
                out.append(int(nxt))
                cur = nxt.view(1, 1)
            walls.append(time.perf_counter() - t0)
            toks = out
        return walls[1:], toks

    def resolved(self) -> dict:
        from exllamav3 import ext as exl_ext
        cfg_json = json.load(open(os.path.join(self.model_dir, "config.json")))
        qc = cfg_json.get("quantization_config")
        try:
            bpw_layer, bpw_head, vram_bits = self.model.get_storage_info()
        except Exception as e:  # noqa: BLE001 -- the receipt says what it could not read
            bpw_layer = bpw_head = vram_bits = f"unreadable: {e!r}"[:200]
        return {"architecture": getattr(self.config, "architecture", cfg_json.get("architectures")),
                "quantization_config": qc, "bpw_layer": bpw_layer, "bpw_head": bpw_head, "vram_bits": vram_bits,
                "cache": {"layer_type": self.cache_desc, "max_num_tokens": self.cache_tokens, "page_size": PAGE_SIZE},
                "generator": {**self._gen_kwargs, "fresh_generator_per_call": self.fresh, "max_q_size": 8,
                              "draft_model": None, "ngram_match_min": 0},
                "load": {"device": "single/autosplit (default)", "warmup": os.environ.get("SC1_EXL3_NO_WARMUP", "0") == "0",
                         "max_batch_size_hint": self.batch},
                "precompiled_extension": bool(exl_ext.is_precompiled_extension_available()),
                "attention": "built-in Triton paged attention + BC_Attention graph decode (no flash-attn/xformers in 1.5.3)",
                "moe_path_by_batch": "bsz<=8: BC_BlockSparseMLP.run_bszN (exl3_moe_coop_*); bsz>=4 & >8: ext.exl3_moe fused "
                                     "(exl3_moe_kernel); f_threshold=min(128//8,4)=4 (modules/block_sparse_mlp.py:190,1003,1068,1334)"}

    def versions(self) -> dict:
        torch = self.torch
        v = {"exllamav3": self.exl.version.__version__ if hasattr(self.exl, "version") else getattr(self.exl, "__version__", None),
             "torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": sys.version.split()[0],
             "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
             "torch_arch_list": torch.cuda.get_arch_list()}
        try:
            import triton
            v["triton"] = triton.__version__
        except Exception:  # noqa: BLE001
            v["triton"] = None
        return v

    def vram_peak_bytes(self):
        return {"torch_max_memory_allocated": int(self.torch.cuda.max_memory_allocated()),
                "torch_max_memory_reserved": int(self.torch.cuda.max_memory_reserved()),
                "mem_get_info_drop_bytes": int(self.free0 - self.min_free)}


def make_engine(model_dir: str, batch: int, prompt_len: int, max_new: int):
    return Exl3Engine(model_dir, batch, prompt_len, max_new, warmup=os.environ.get("SC1_EXL3_NO_WARMUP", "0") == "0")


# ----------------------------------------------------------------------------------------------------------------
# selftest fake engine (deterministic tokens; obeys the adapter interface)
# ----------------------------------------------------------------------------------------------------------------

class FakeEngine:
    def __init__(self, short_by: int = 0, step_s: float = 0.0):
        self.short_by, self.step_s, self.load_s = short_by, step_s, 0.0
        self.calls = []

    def generate(self, prompts, n):
        self.calls.append(n)
        time.sleep(self.step_s * n)
        rows = [[(sum(p) + j) % 1000 for j in range(n - self.short_by)] for p in prompts]
        return rows, {"eos_reason": ["max_new_tokens"] * len(prompts), "cached_tokens": [0] * len(prompts)}

    def first_token(self, prompt, max_new_tokens=8):
        return 0.01, 0.009, max_new_tokens

    def resolved(self):
        return {"fake": True}

    def versions(self):
        return {"exllamav3": "fake"}

    def vram_peak_bytes(self):
        return {"torch_max_memory_allocated": 0}


def selftest() -> int:
    import tempfile
    rows = [[(i * 7 + j) % 1000 for j in range(PROMPT_LEN)] for i in range(2)]
    pf = {"batch": 2, "prompts": rows, "prompts_sha256": prompts_digest(rows), "rows_sha256": "x"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(pf, f)
    loaded = load_prompts(f.name, 2)
    assert loaded["prompts_sha256"] == pf["prompts_sha256"]
    bad = dict(pf)
    bad["prompts"] = [rows[0], rows[1][:-1] + [1]]           # tampered token -> digest mismatch
    with open(f.name, "w") as g:
        json.dump(bad, g)
    try:
        load_prompts(f.name, 2)
        raise AssertionError("digest refusal did not fire")
    except Refusal:
        pass
    s = slope([1.0, 1.1, 1.2], [2.0, 2.1, 2.2], 2)
    assert s["decode_ms_per_step"] == round(1.0 / 96 * 1e3, 4) and s["decode_tok_s"] == 192.0, s
    try:
        check_tokens({"0": [1] * 31, "1": [1] * 32}, 32, 2)
        raise AssertionError("N-token refusal did not fire")
    except Refusal:
        pass
    r = run_slope(FakeEngine(), rows, 2, reps=3)
    assert set(r["tokens"]) == {"0", "1"} and all(len(v) == LONG for v in r["tokens"].values())
    try:
        run_slope(FakeEngine(short_by=1), rows, 2)
        raise AssertionError("short row passed")
    except Refusal:
        pass
    t = run_ttft(FakeEngine(), rows[0])
    assert t["tokens_generated"] == [8, 8, 8]
    rec = base_receipt("selftest", 2, loaded, DEFAULT_MODEL, DEFAULT_REV, "slope")
    rec.update(r)
    rec.update(load_s=0.0, resolved={}, versions={}, vram_peak_bytes={})
    missing = [k for k in RECEIPT_KEYS_SLOPE if k not in rec]
    assert not missing, missing
    print("SELFTEST OK sc1_exl3_arm")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttft", action="store_true")
    ap.add_argument("--native-loop", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    arm, batch, out = os.environ["SC1_ARM"], int(os.environ["SC1_BATCH"]), os.environ["SC1_OUT"]
    model, rev = os.environ.get("SC1_MODEL", DEFAULT_MODEL), os.environ.get("SC1_REV", DEFAULT_REV)
    mode = "ttft" if a.ttft else ("native_loop" if a.native_loop else "slope")
    pf = load_prompts(os.environ["SC1_PROMPTS"], batch, prompt_len=None if a.ttft else PROMPT_LEN)
    prompts = pf["prompts"]
    if a.native_loop and batch != 1:
        raise Refusal("REFUSED: --native-loop is a B=1 instrument (upstream perf.py shape)")
    rec = base_receipt(arm, batch, pf, model, rev, mode)
    model_dir = resolve_model_dir(model, rev)
    rec["model_dir"] = model_dir
    max_new = 8 if a.ttft else LONG
    engine = make_engine(model_dir, batch, max(pf["prompt_tokens"]), max_new)
    rec["load_s"] = engine.load_s
    if a.ttft:
        rec.update(run_ttft(engine, prompts[0]))
    elif a.native_loop:
        w_s, _ = engine.native_loop(prompts[0], SHORT)
        w_l, toks = engine.native_loop(prompts[0], LONG)
        rec.update(slope(w_s, w_l, 1))
        rec["tokens"] = {"0": toks}
        rec["method"] = "native_loop: upstream eval/perf.py shape (raw model.forward, argmax feedback, one .cpu() sync per step); slope 32->128"
    else:
        rec.update(run_slope(engine, prompts, batch))
        rec["ttft_note"] = "not measured in slope mode; see --ttft"
    rec["resolved"], rec["versions"], rec["vram_peak_bytes"] = engine.resolved(), engine.versions(), engine.vram_peak_bytes()
    json.dump(rec, open(out, "w"), indent=1)
    keys = ("arm", "batch", "mode", "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median", "end_to_end_tok_s_long",
            "ttft_s", "load_s", "prompts_sha256")
    print("SC1EXL3 " + json.dumps({k: rec.get(k) for k in keys if k in rec}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
