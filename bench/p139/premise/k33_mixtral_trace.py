"""P139's call-site trace ($0, the A2000): which (N, K) does e4b's served decode hand grouped-nf4-gemm's K33 gate on
Mixtral? Mixtral-8x7B's real widths (hidden 4096, intermediate 14336, 8 experts, top 2), two decoder layers, random
weights, through the GPU serve smoke's own build (write_checkpoint -> offline snapshot -> bake -> build_engine ->
scheduler). nf4_grouped._bw_engages -- the gate K33's auto mode reads -- is wrapped to record every (N, K, T) before
the build, so the record is the call site, not an inference from shapes.

    python k33_mixtral_trace.py OUT_DIR   ->  OUT_DIR/trace.json
"""
import collections
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
(OUT / "work").mkdir(parents=True, exist_ok=True)
import os  # noqa: E402
# the smoke's suite sets these for each child before huggingface_hub is imported; this process is that child
os.environ.update(HF_HUB_CACHE=str(OUT / "work" / "hf-cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false")

spec = importlib.util.spec_from_file_location("gpu_serve_smoke", HERE / "bench/smoke/gpu_serve_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def real_width_mixtral(family):
    import torch
    from transformers import AutoModelForCausalLM, MixtralConfig
    assert family == "mixtral", family
    cfg = MixtralConfig(vocab_size=256, hidden_size=4096, intermediate_size=14336, num_hidden_layers=2,
                        num_attention_heads=32, num_key_value_heads=8, head_dim=128, num_local_experts=8,
                        num_experts_per_tok=2, max_position_embeddings=8192, tie_word_embeddings=False,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    torch.manual_seed(1689)
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()


smoke.random_model = real_width_mixtral

import nf4_grouped  # noqa: E402

calls = collections.Counter()
_orig = nf4_grouped._bw_engages


def _record(N, K, device):
    calls[(int(N), int(K))] += 1
    return _orig(N, K, device)


nf4_grouped._bw_engages = _record

T_seen = collections.Counter()
_orig_g = nf4_grouped.gemm_4bit_grouped


def _record_g(a_cat, B, absmax, sizes, expert_ids, *a, **kw):
    if max(sizes) == 1:
        T_seen[(int(B.shape[1]), int(a_cat.shape[1]), int(a_cat.shape[0]) * int(kw.get("gather_div", 1)))] += 1
    return _orig_g(a_cat, B, absmax, sizes, expert_ids, *a, **kw)


nf4_grouped.gemm_4bit_grouped = _record_g
import experts4bit_qlora.engines.hot_residency as hr  # noqa: E402
if getattr(hr, "gemm_4bit_grouped", None) is not None:
    hr.gemm_4bit_grouped = _record_g

row = smoke.run_cell("mixtral", "default", OUT / "work")
import torch  # noqa: E402
rec = {
    "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
    "gnf4_bw_mode": nf4_grouped._bw(), "gnf4_file": nf4_grouped.__file__,
    "e4b_file": hr.__file__,
    "bw_gate_calls": {f"{n}x{k}": c for (n, k), c in sorted(calls.items())},
    "singleton_calls_N_K_T": {f"{n}x{k}@T{t}": c for (n, k, t), c in sorted(T_seen.items())},
    "model": {"hidden": 4096, "intermediate": 14336, "experts": 8, "top_k": 2, "layers": 2},
    "smoke_row": {k: row[k] for k in ("residency_states", "expert_target_names", "graph_status", "fusion_report",
                                       "requests_completed", "placement") if k in row},
}
(OUT / "trace.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
print("K33_TRACE", json.dumps({"bw_gate_calls": rec["bw_gate_calls"], "singleton": rec["singleton_calls_N_K_T"]}))
