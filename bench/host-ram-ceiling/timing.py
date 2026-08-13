"""One timed arm: load time and per-step time, separated.

Two quantities, and they must not be mixed:

  t_load   fusing+quantizing every expert (host) vs reading a baked file (arena).
           Paid once per run, and the arena's whole reason for existing on the
           load path. Tokenizer and dataset work is deliberately OUTSIDE this
           window -- it is identical across arms and would dilute the ratio.

  t_step   steady-state cost once resident. The arena pays disk traffic here for
           rows outside the hot set, so this is where it can only lose.

Absolute numbers are this pod's. Only within-pair ratios travel, and only after
the host_self control says the harness can resolve them at all.
"""
import argparse
import json
import os
import time

import torch

MODEL = os.environ.get("E4B_MODEL", "allenai/OLMoE-1B-7B-0924")
ARENA = os.environ.get("E4B_ARENA", "/work/arena/olmoe-nf4.arena")


def build(hot):
    from experts4bit_qlora import enable_fast_train, load_moe_4bit_streaming
    torch.manual_seed(1234)
    if hot is None:
        m, _ = load_moe_4bit_streaming(MODEL, "cuda", torch.bfloat16, r=8, alpha=16,
                                       quant_type="nf4", offload=True)
    else:
        from experts4bit_qlora import enable_nvme_train_residency
        m, _ = load_moe_4bit_streaming(MODEL, "cuda", torch.bfloat16, r=8, alpha=16,
                                       quant_type="nf4", offload=False, arena=ARENA,
                                       arena_train=True)
        assert enable_nvme_train_residency(m, ARENA, hot_rows=hot, device="cuda") > 0
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    for n, p in m.named_parameters():
        p.requires_grad_("lora" in n and "experts" in n)
    assert enable_fast_train(m) > 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("host", "arena"))
    ap.add_argument("--hot", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2, help="steps dropped before scoring")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    hot = None if a.arm == "host" else a.hot

    # --- outside the timed load window on purpose ---
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    batches = []
    for r in ds.select(range((a.steps + a.warmup) * 3)):
        ins = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
        t = f"### Instruction:\n{ins}\n\n### Response:\n{r['output']}"
        ids = tok(t, truncation=True, max_length=384, return_tensors="pt")["input_ids"]
        if ids.shape[1] >= 16:
            batches.append(ids.to("cuda"))
        if len(batches) == a.steps + a.warmup:
            break
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    m = build(hot)
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0

    step_s, losses = [], []
    for i, ids in enumerate(batches):
        torch.cuda.synchronize()
        s = time.perf_counter()
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()
        m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        d = time.perf_counter() - s
        losses.append(float(out.loss.detach()))
        if i >= a.warmup:                      # warmup steps are dropped, not averaged in
            step_s.append(d)

    scored = sorted(step_s)
    print("RESULT " + json.dumps({
        "ok": True, "arm": a.arm, "hot": hot, "tag": a.tag, "model": MODEL,
        "t_load_s": round(t_load, 3),
        "step_s_median": round(scored[len(scored) // 2], 4),
        "step_s_min": round(scored[0], 4),
        "step_s_all": [round(x, 4) for x in step_s],
        "warmup_dropped": a.warmup,
        "losses": losses,
        "arena_gb": round(os.path.getsize(ARENA) / 1e9, 3) if hot else None,
        "nproc": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0),
    }), flush=True)


if __name__ == "__main__":
    main()
