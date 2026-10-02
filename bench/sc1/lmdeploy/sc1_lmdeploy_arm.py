"""sc1_lmdeploy_arm.py -- the LMDeploy 0.18.0 (TurboMind) arm of lane SC1 (experts4bit-qlora#846), p37-shaped.

Contract identical to the vLLM arm (bench/h2h-20260905/p37/p37_vllm.py): prompts_b{B}.json token ids (no tokenizer
runs here), slope 32->128, one warm + 3 timed reps per length, min-of-3 and median-of-3, exactly N tokens per row,
every engine knob recorded from the engine's own objects.

LMDeploy 0.18.0 facts the arm is built on (file:line in InternLM/lmdeploy @ v0.18.0, 110965c7):
  * pre-tokenised prompts: `Pipeline.infer` rejects a list of ints (serve/processors/multimodal.py:293-309), but
    `AsyncEngine.preprocess(messages=None, input_ids=[...])` takes them (serve/core/async_engine.py:464-547) and so does
    the raw `TurboMindInstance.async_stream_infer(session_id, input_ids, gen_config)` (turbomind/turbomind.py:689-797),
    which is exactly what upstream's own throughput benchmark drives (benchmark/profile_throughput.py:164-205,357-370:
    one instance per concurrent stream, `max_batch_size = concurrency`). This arm uses that raw route.
  * greedy: on the raw route `_determine_gen_config` (do_sample -> top_k=1) is NOT applied, so the arm sets
    top_k=1, top_p=1.0, temperature=1.0 explicitly (turbomind.py:800-806 copies them verbatim into the C++ config).
  * EOS ignored, N tokens: `ignore_eos=True` -> no `stop_ids` (turbomind.py:811-812) -> StopCriteria runs the length
    criterion only (generation/stop_criteria.cc:56-91); `min_new_tokens=N` + `stop_token_ids=eos ids` arms the
    min-length penalty that masks the eos logits until prompt_len+N (generation/logits_processor.cc:157-207) -- the same
    semantics as vLLM's `ignore_eos=True, min_tokens=N` in p37.
  * one batched step per token: TurboMind continuous-batches the 16 concurrent requests in its engine loop (the whole
    point of `max_batch_size`); the receipt records the engine config and the per-request metrics.
  * no prefix caching (`enable_prefix_caching=False` default, messages.py:401), no speculative decoding (none given),
  * no CUDA graphs: `grep cudaGraph src/turbomind` is empty at v0.18.0.
  * KV: `quant_policy` 0 | 4 (int4) | 8 (int8); fp8 is REFUSED (messages.py:431-438). cache_block_seq_len=64.
  * REGISTERED CAPACITY RULE: every engine holds B sequences x 2048 tokens on its timed arms -> `session_len=2048`,
    `max_batch_size=B`, `cache_max_entry_count` left at the engine's default (0.8 of FREE memory, messages.py:398) and
    RECORDED from the resolved config; the TTFT-4096 arm is a single sequence with `session_len=4104` (4096+8).
  * W4A16 on sm_120: "SM12.x (e.g. sm_120): no native kernels; falls back to the SM80 s16816 family"
    (src/turbomind/kernels/gemm/arch.h:38-41,53); families u4_d_128 / u4_g_128 (kernel/sm80_16816_4.cu:40-44); the
    PyPI wheel is built with CUDA 12.8 (builder/manywheel/build_all_wheel.sh:7) which appends 120a-real
    (CMakeLists.txt:261-262). TM_GEMM_VERBOSE=1 prints every dispatch: "[Gemm] <kind> <desc> sm80_f16_u4k128_f16_..."
    (kernels/gemm/gemm.cu:180-186, name format kernel.cu:158-184).
  * model_format is auto-detected from quantization_config and a user value must MATCH (turbomind/converter.py:179-215);
    gptq needs desc_act=False & sym=True (L194-195); awq needs version=='gemm' (L191-192); group size must be 128.

Env: SC1_ARM, SC1_BATCH, SC1_PROMPTS, SC1_MODEL (default Qwen/Qwen3-30B-A3B-GPTQ-Int4), SC1_REV (default
9b534e4318b7ebc3c961a839f13eb18b1833f441), SC1_OUT, SC1_INSTANCE_ID. Optional: SC1_MODEL_DIR, SC1_LMD_FORMAT (gptq|awq;
default auto), SC1_LMD_QUANT_POLICY (0|4|8), SC1_LMD_DTYPE (auto|float16|bfloat16), SC1_LMD_MAX_BATCH (default = B),
SC1_LMD_CACHE_FRAC (cache_max_entry_count; unset = the engine's default, recorded), SC1_LMDEPLOY_SESSION_LEN (override of
the capacity rule; must hold prompt + max_new + 1; value and source recorded under receipt["capacity"]). Modes: slope
(default), --ttft, --selftest.
The AWQ row: SC1_MODEL=QuixiAI/Qwen3-30B-A3B-AWQ SC1_REV=1ba5586ace54cc9de85addac384eb88576f94598 (version=gemm, g128).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

SHORT, LONG = 32, 128
PROMPT_LEN = 512
ENGINE = "lmdeploy-turbomind"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
DEFAULT_REV = "9b534e4318b7ebc3c961a839f13eb18b1833f441"
TM_ENV_KNOBS = ("TM_LOG_LEVEL", "TM_GEMM_TUNE", "TM_GEMM_VERBOSE", "TM_GEMM_IMPORT", "TM_GEMM_EXPORT", "TM_GEMM_WARN_CACHE_MISS",
                "TM_DEBUG_LEVEL", "LMDEPLOY_USE_MODELSCOPE", "CUDA_VISIBLE_DEVICES")


class Refusal(SystemExit):
    pass


# ------------------------------------------------------------------------------------------------ pure helpers

def prompts_digest(prompts) -> str:
    return hashlib.sha256(json.dumps(prompts).encode()).hexdigest()


def load_prompts(path: str, batch: int, prompt_len: int | None = PROMPT_LEN) -> dict:
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
    return {"prompts": prompts, "prompts_sha256": sha, "rows_sha256": pf.get("rows_sha256"), "prompt_tokens": [len(p) for p in prompts]}


def check_tokens(gen: dict, n_tokens: int, batch: int) -> int:
    if len(gen) != batch:
        raise Refusal(f"REFUSED: {len(gen)} rows generated != batch {batch}")
    got = sum(len(v) for v in gen.values())
    if got != n_tokens * batch or any(len(v) != n_tokens for v in gen.values()):
        raise Refusal(f"REFUSED: {got} != {n_tokens * batch} tokens (a row stopped early: not the registered workload)")
    return got


def slope(w_short, w_long, batch: int, short: int = SHORT, long: int = LONG) -> dict:
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
    def run(n):
        engine.generate(prompts, n)
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
    engine.first_token(prompt, max_new_tokens)
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
                method="wall from async_stream_infer(stream_output=True) submission to the first yielded token; engine warm; "
                       "fresh instance + new session id per repeat (prefix caching off)")


def base_receipt(arm: str, batch: int, pf: dict, model: str, rev: str, mode: str) -> dict:
    return {"engine": ENGINE, "arm": arm, "mode": mode, "model": model, "revision": rev, "batch": batch,
            "method": "slope(32->128) isolates decode; min-of-3 (P20 estimator) and median-of-3 both recorded",
            "prompts": "identical token ids to the e4b arms (step_decomp._k8_window, wikitext-2 test, 512-token rows)",
            "prompts_sha256": pf["prompts_sha256"], "rows_sha256": pf["rows_sha256"], "prompt_tokens": pf["prompt_tokens"],
            "generation": {"greedy": "top_k=1, top_p=1.0, temperature=1.0 (explicit; raw instance route)", "ignore_eos": True,
                           "min_new_tokens": "= max_new_tokens (masks eos logits via min-length penalty)",
                           "stop_token_ids": "generation_config eos ids (masked, never stop)", "speculative": False,
                           "route": "TurboMind.create_instance().async_stream_infer (benchmark/profile_throughput.py's route)"},
            "vast_instance_id": os.environ.get("SC1_INSTANCE_ID"), "env_knobs": {k: os.environ.get(k) for k in TM_ENV_KNOBS}}


RECEIPT_KEYS_SLOPE = ("engine", "arm", "mode", "model", "revision", "batch", "prompts_sha256", "rows_sha256", "load_s",
                      "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median", "decode_ms_per_step_median",
                      "end_to_end_tok_s_long", "walls_short_s", "walls_long_s", "tokens", "resolved", "versions",
                      "vram_peak_bytes", "capacity")


# ------------------------------------------------------------------------------------------------ engine side

def resolve_model_dir(model: str, rev: str) -> str:
    d = os.environ.get("SC1_MODEL_DIR")
    if d:
        return d
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=rev, max_workers=4)


def eos_ids_from(model_dir: str) -> list:
    try:
        g = json.load(open(os.path.join(model_dir, "generation_config.json")))
        e = g.get("eos_token_id")
        return list(e) if isinstance(e, list) else ([int(e)] if e is not None else [])
    except Exception:  # noqa: BLE001
        return []


SLOPE_SESSION_LEN = 2048            # the registered rule: B sequences x 2048 tokens held on every timed arm


def session_len_for(prompt_len: int, max_new: int, env=None) -> tuple[int, str]:
    """Registered capacity rule -> (session_len, source). Slope arms: 2048 (held B times via max_batch_size = B).
    TTFT-4096 (prompt + 8 > 2048): prompt + max_new = 4104. SC1_LMDEPLOY_SESSION_LEN overrides; it must hold
    prompt + max_new + 1 (TurboMind sizes the sequence by session_len and the engine refuses input_len >= session_len)."""
    env = os.environ if env is None else env
    need = prompt_len + max_new
    ov = env.get("SC1_LMDEPLOY_SESSION_LEN")
    if ov:
        v = int(ov)
        if v < need + 1:
            raise Refusal(f"REFUSED: SC1_LMDEPLOY_SESSION_LEN={v} cannot hold prompt {prompt_len} + {max_new} new tokens")
        return v, "env:SC1_LMDEPLOY_SESSION_LEN"
    if need <= SLOPE_SESSION_LEN:
        return SLOPE_SESSION_LEN, "default:2048"
    return need, "default:ttft:prompt+max_new"


def capacity_block(batch: int, prompt_len: int, max_new: int, session_len: int, source: str, resolved=None) -> dict:
    b = {"rule": "B sequences x 2048 tokens on timed arms (session_len 2048, max_batch_size B); TTFT-4096 = session_len 4104",
         "session_len": session_len, "session_len_source": source, "max_batch_size": batch,
         "kv_tokens_held_target": batch * session_len, "workload_tokens_per_seq": prompt_len + max_new}
    if resolved:
        b.update(resolved)
    return b


def nvidia_smi_mem() -> int | None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().split("\n")[0]) * 1024 * 1024
    except Exception:  # noqa: BLE001
        return None


class TurboMindEngine:
    """Raw TurboMind instance route (profile_throughput.py's). TurboMind allocates outside torch's caching allocator,
    so VRAM is read from torch.cuda.mem_get_info() deltas and nvidia-smi, not max_memory_allocated."""

    def __init__(self, model_dir: str, batch: int, prompt_len: int, max_new: int, session_len: int):
        import torch
        import lmdeploy
        from lmdeploy import TurbomindEngineConfig
        from lmdeploy.turbomind import is_available
        if not is_available():
            from lmdeploy.turbomind import _import_error
            raise Refusal(f"REFUSED/UNSUPPORTED: lmdeploy.turbomind binding not importable: {_import_error!r}")
        from lmdeploy.turbomind import TurboMind
        self.torch, self.lmd, self.model_dir, self.batch = torch, lmdeploy, model_dir, batch
        self.free0 = torch.cuda.mem_get_info()[0]
        self.min_free = self.free0
        self.smi0 = nvidia_smi_mem()
        self.smi_max = self.smi0 or 0
        fmt = os.environ.get("SC1_LMD_FORMAT") or None
        self.session_len = session_len                        # from session_len_for (the registered rule or the override)
        self.engine_kwargs = dict(model_format=fmt, session_len=session_len,
                                  max_batch_size=int(os.environ.get("SC1_LMD_MAX_BATCH") or batch),
                                  enable_prefix_caching=False, quant_policy=int(os.environ.get("SC1_LMD_QUANT_POLICY") or 0),
                                  dtype=os.environ.get("SC1_LMD_DTYPE") or "auto", tp=1, enable_metrics=True)
        if os.environ.get("SC1_LMD_CACHE_FRAC"):               # unset = the engine's own default, recorded from the resolved config
            self.engine_kwargs["cache_max_entry_count"] = float(os.environ["SC1_LMD_CACHE_FRAC"])
        t0 = time.perf_counter()
        self.tm = TurboMind.from_pretrained(model_dir, engine_config=TurbomindEngineConfig(**self.engine_kwargs))
        self.load_s = round(time.perf_counter() - t0, 1)
        self.eos_ids = eos_ids_from(model_dir) or [int(self.tm.tokenizer.eos_token_id)]
        self.session_serial = 0

    def _sample_vram(self):
        self.min_free = min(self.min_free, self.torch.cuda.mem_get_info()[0])
        m = nvidia_smi_mem()
        if m:
            self.smi_max = max(self.smi_max, m)

    def _gen_config(self, n: int):
        from lmdeploy import GenerationConfig
        return GenerationConfig(max_new_tokens=n, min_new_tokens=n, ignore_eos=True, top_k=1, top_p=1.0, temperature=1.0,
                                stop_token_ids=list(self.eos_ids), do_sample=False)

    def generate(self, prompts, n_tokens: int):
        gc = self._gen_config(n_tokens)

        async def one(i, ids):
            self.session_serial += 1
            inst = self.tm.create_instance()
            toks, cached, finish = [], None, None
            async for out in inst.async_stream_infer(session_id=self.session_serial, input_ids=list(ids), gen_config=gc, stream_output=False):
                toks += list(out.token_ids)
                finish = getattr(out, "status", None)
                rm = getattr(out, "req_metrics", None)
                if rm is not None:
                    cached = getattr(rm, "cached_tokens", None)
            return toks, {"status": str(finish), "cached_tokens": cached}

        async def all_rows():
            return await asyncio.gather(*[one(i, p) for i, p in enumerate(prompts)])

        res = asyncio.run(all_rows())
        self.torch.cuda.synchronize()
        self._sample_vram()
        rows = [r[0] for r in res]
        meta = {"status": [r[1]["status"] for r in res], "cached_tokens": [r[1]["cached_tokens"] for r in res]}
        return rows, meta

    def first_token(self, prompt, max_new_tokens: int = 8):
        gc = self._gen_config(max_new_tokens)

        async def go():
            self.session_serial += 1
            inst = self.tm.create_instance()
            self.torch.cuda.synchronize()
            t0 = time.perf_counter()
            first, n = None, 0
            async for out in inst.async_stream_infer(session_id=self.session_serial, input_ids=list(prompt), gen_config=gc, stream_output=True):
                k = len(out.token_ids)
                if k and first is None:
                    first = time.perf_counter() - t0
                n += k
            return first, n

        first, n = asyncio.run(go())
        self._sample_vram()
        return first, None, n

    def resolved(self) -> dict:
        ec = self.tm.engine_config
        cfg = getattr(getattr(self.tm, "source_model", None), "cfg", None)
        qc = None
        try:
            qc = cfg.to_dict().get("quantization_config") if cfg is not None else None
        except Exception:  # noqa: BLE001
            pass
        if qc is None:
            try:
                qc = json.load(open(os.path.join(self.model_dir, "config.json"))).get("quantization_config")
            except Exception:  # noqa: BLE001
                pass
        try:
            from lmdeploy.turbomind.converter import _get_executable_dtypes
            from lmdeploy.turbomind.weight_format import AWQFormat, GPTQFormat
            fmt = GPTQFormat(block_in=128) if str(ec.model_format) == "gptq" else AWQFormat(block_in=128)
            executable = sorted(_get_executable_dtypes([fmt], 0))
        except Exception as e:  # noqa: BLE001
            executable = f"unreadable: {e!r}"[:200]
        return {"engine_config": {k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                                  for k, v in dataclasses.asdict(ec).items()},
                "requested_engine_kwargs": self.engine_kwargs, "quantization_config": qc, "vocab_size": getattr(self.tm, "_vocab_size", None),
                "session_len": getattr(self.tm, "session_len", None), "eos_ids_masked": self.eos_ids,
                "executable_activation_dtypes_for_format": executable,
                "kernel_family": "sm80 s16816 u4 (u4_d_128 dense / u4_g_128 grouped-MoE) -- arch.h:38 'SM12.x: no native kernels; falls "
                                 "back to the SM80 s16816 family'; attention: built-in attention_kernel (attention_sm80_128 / decoding_sm80_128)",
                "cuda_graphs": False, "prefix_caching": False, "cache_block_seq_len": ec.cache_block_seq_len}

    def versions(self) -> dict:
        torch = self.torch
        v = {"lmdeploy": self.lmd.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
             "python": sys.version.split()[0], "device": torch.cuda.get_device_name(0),
             "capability": list(torch.cuda.get_device_capability(0)), "torch_arch_list": torch.cuda.get_arch_list()}
        try:
            v["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except Exception:  # noqa: BLE001
            v["driver"] = None
        return v

    def capacity_resolved(self) -> dict:
        ec = self.tm.engine_config
        return {"cache_max_entry_count_resolved": ec.cache_max_entry_count, "cache_block_seq_len": ec.cache_block_seq_len,
                "session_len_resolved": ec.session_len, "max_batch_size_resolved": ec.max_batch_size}

    def vram_peak_bytes(self):
        return {"mem_get_info_drop_bytes": int(self.free0 - self.min_free), "nvidia_smi_used_max_bytes": int(self.smi_max),
                "nvidia_smi_used_before_load_bytes": self.smi0,
                "note": "TurboMind allocates outside torch's allocator; torch.cuda.max_memory_allocated does not see it"}


def make_engine(model_dir: str, batch: int, prompt_len: int, max_new: int, session_len: int):
    return TurboMindEngine(model_dir, batch, prompt_len, max_new, session_len)


# ------------------------------------------------------------------------------------------------ selftest

class FakeEngine:
    def __init__(self, short_by: int = 0):
        self.short_by, self.load_s, self.calls = short_by, 0.0, []

    def generate(self, prompts, n):
        self.calls.append(n)
        return [[(sum(p) + j) % 1000 for j in range(n - self.short_by)] for p in prompts], {"status": ["FINISH"] * len(prompts), "cached_tokens": [0] * len(prompts)}

    def first_token(self, prompt, max_new_tokens=8):
        return 0.02, None, max_new_tokens

    def resolved(self):
        return {"fake": True}

    def versions(self):
        return {"lmdeploy": "fake"}

    def vram_peak_bytes(self):
        return {"mem_get_info_drop_bytes": 0}


def selftest() -> int:
    import tempfile
    rows = [[(i * 11 + j) % 1000 for j in range(PROMPT_LEN)] for i in range(2)]
    pf = {"batch": 2, "prompts": rows, "prompts_sha256": prompts_digest(rows), "rows_sha256": "x"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(pf, f)
    loaded = load_prompts(f.name, 2)
    bad = dict(pf)
    bad["prompts_sha256"] = "0" * 64
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
        check_tokens({"0": [1] * 32}, 32, 2)
        raise AssertionError("row-count refusal did not fire")
    except Refusal:
        pass
    r = run_slope(FakeEngine(), rows, 2)
    assert all(len(v) == LONG for v in r["tokens"].values())
    try:
        run_slope(FakeEngine(short_by=2), rows, 2)
        raise AssertionError("short row passed")
    except Refusal:
        pass
    assert session_len_for(512, 128, {}) == (2048, "default:2048") and session_len_for(512, 8, {})[0] == 2048
    assert session_len_for(4096, 8, {}) == (4104, "default:ttft:prompt+max_new")
    assert session_len_for(512, 128, {"SC1_LMDEPLOY_SESSION_LEN": "3000"}) == (3000, "env:SC1_LMDEPLOY_SESSION_LEN")
    try:
        session_len_for(512, 128, {"SC1_LMDEPLOY_SESSION_LEN": "600"})
        raise AssertionError("bad override passed")
    except Refusal:
        pass
    rec = base_receipt("selftest", 2, loaded, DEFAULT_MODEL, DEFAULT_REV, "slope")
    rec.update(r)
    rec.update(load_s=0.0, resolved={}, versions={}, vram_peak_bytes={}, capacity=capacity_block(2, 512, 128, 2048, "default:2048"))
    missing = [k for k in RECEIPT_KEYS_SLOPE if k not in rec]
    assert not missing, missing
    assert run_ttft(FakeEngine(), rows[0])["tokens_generated"] == [8, 8, 8]
    print("SELFTEST OK sc1_lmdeploy_arm")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttft", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    arm, batch, out = os.environ["SC1_ARM"], int(os.environ["SC1_BATCH"]), os.environ["SC1_OUT"]
    model, rev = os.environ.get("SC1_MODEL", DEFAULT_MODEL), os.environ.get("SC1_REV", DEFAULT_REV)
    mode = "ttft" if a.ttft else "slope"
    pf = load_prompts(os.environ["SC1_PROMPTS"], batch, prompt_len=None if a.ttft else PROMPT_LEN)
    rec = base_receipt(arm, batch, pf, model, rev, mode)
    model_dir = resolve_model_dir(model, rev)
    rec["model_dir"] = model_dir
    max_new = 8 if a.ttft else LONG
    sl, src = session_len_for(max(pf["prompt_tokens"]), max_new)
    engine = make_engine(model_dir, batch, max(pf["prompt_tokens"]), max_new, sl)
    rec["capacity"] = capacity_block(batch, max(pf["prompt_tokens"]), max_new, sl, src,
                                     engine.capacity_resolved() if hasattr(engine, "capacity_resolved") else None)
    rec["load_s"] = engine.load_s
    if a.ttft:
        rec.update(run_ttft(engine, pf["prompts"][0]))
    else:
        rec.update(run_slope(engine, pf["prompts"], batch))
        rec["ttft_note"] = "not measured in slope mode; see --ttft"
    rec["resolved"], rec["versions"], rec["vram_peak_bytes"] = engine.resolved(), engine.versions(), engine.vram_peak_bytes()
    json.dump(rec, open(out, "w"), indent=1)
    keys = ("arm", "batch", "mode", "decode_tok_s", "decode_ms_per_step", "decode_tok_s_median", "end_to_end_tok_s_long",
            "ttft_s", "load_s", "prompts_sha256")
    print("SC1LMD " + json.dumps({k: rec.get(k) for k in keys if k in rec}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
