"""A training step of a real MoE through e4b's training path, with the stock causal-LM loss or the chunked one.

The setup is the TC1 harness's fused e4b arm (bench/tc1/tc1_arm.py ``load_e4b``): ``load_moe_4bit_streaming`` (NF4 experts),
Hugging Face gradient checkpointing (``use_reentrant=False``), fp32 attention LoRA, ``enable_fast_train(model, dgrad=True)``, and
``model(input_ids=ids, labels=ids).loss.backward()`` per micro-batch -- so ``E4B_CHUNKED_LM_LOSS`` in the environment is the only
switch, exactly as a box's ``TC1_E4B_ENV`` hands it to the harness. Rows are packed random tokens, every label supervised.

    E4B_CHUNKED_LM_LOSS=512 python train_step_ab.py --model <dir> --tokens 512 1024 2048 4096 --steps 4 --out res.json
    python train_step_ab.py --model <dir> --check 2048      # one process: stock, stock again, chunked; LoRA gradients compared

The model loads once; the environment variable engages the chunked loss through ``enable_fast_train`` (recorded as
``chunked_engaged``), then each arm is set in-process with ``disable_chunked_lm_loss`` / ``enable_chunked_lm_loss`` and run for
``--steps`` optimizer steps at each token count, in the ``--arms`` order (interleaved, so drift shows). Per arm: peak
``torch.cuda.max_memory_allocated`` over each step (forward, backward, optimizer step; reset before it), the max over steps after
the first, and the median wall time of the steps after the first (each closed by ``torch.cuda.synchronize``).
"""
import argparse
import json
import os
import statistics
import time

import torch


def load(a):
    from experts4bit_qlora import enable_fast_train, load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora
    model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, a.r, a.alpha, offload=False, pin=True,
                                         prefetch=False, quant_type="nf4")
    model.to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    add_attention_lora(model, a.r, a.alpha, torch.float32)
    n = enable_fast_train(model, verbose=True, dgrad=True) if a.fast else 0
    model.train()
    return model, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--fast", type=int, default=1)
    ap.add_argument("--check", type=int, default=0, help="token count for the in-process gradient comparison (0 = timing run)")
    ap.add_argument("--arms", default="chunk512,stock,stock,chunk512,chunk1024",
                    help="comma list, run in this order at each token count: stock or chunkN (in-process, enable/disable)")
    ap.add_argument("--stock-last-from", type=int, default=4096, help="stock arms at >= this many tokens run after all others")
    ap.add_argument("--no-reduced-precision-reduction", action="store_true",
                    help="torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False (a diagnostic)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.no_reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, n_patched = load(a)
    from experts4bit_qlora.engines import chunked_lm_loss as C
    st = model.__dict__.get("_e4b_chunked_lm_loss")
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "E4B_CHUNKED_LM_LOSS": os.environ.get("E4B_CHUNKED_LM_LOSS"),
           "fast_train_patched": n_patched, "chunked_engaged": st is not None, "chunk": getattr(st, "chunk", None),
           "model_class": type(model).__name__, "vocab": model.config.vocab_size, "n_layers": model.config.num_hidden_layers,
           "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    print(json.dumps(env), flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    vocab = model.config.vocab_size
    if a.check:
        g = torch.Generator(device="cuda").manual_seed(7)
        ids = torch.randint(0, vocab, (1, a.check), device="cuda", generator=g)

        def grads():
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            torch.cuda.synchronize()
            return float(out.loss.detach()), [p.grad.detach().float().clone() for p in params], out.logits is None
        C.disable_chunked_lm_loss(model)
        s1, g1, c1 = grads()
        s2, g2, c2 = grads()
        C.enable_chunked_lm_loss(model, 512)
        s3, g3, c3 = grads()

        def rel(x, y):
            num = sum(float((u - v).pow(2).sum()) for u, v in zip(x, y)) ** 0.5
            den = sum(float(u.pow(2).sum()) for u in x) ** 0.5
            return num / den
        res = {"tokens": a.check, "chunked_flags": [c1, c2, c3], "loss_stock": s1, "loss_stock_again": s2, "loss_chunked512": s3,
               "lora_grad_rel_l2_stock_vs_stock": rel(g1, g2), "lora_grad_rel_l2_stock_vs_chunked": rel(g1, g3),
               "lora_grads_bitwise_stock_vs_stock": all(torch.equal(u, v) for u, v in zip(g1, g2)),
               "lora_grads_bitwise_stock_vs_chunked": all(torch.equal(u, v) for u, v in zip(g1, g3)),
               "n_trainable_tensors": len(params)}
        print(json.dumps(res), flush=True)
        if a.out:
            json.dump({"env": env, "check": res}, open(a.out, "w"), indent=1)
        return
    opt = torch.optim.AdamW(params, lr=1e-5)
    rows = []

    def set_arm(arm):
        if arm == "stock":
            C.disable_chunked_lm_loss(model)
        else:
            C.disable_chunked_lm_loss(model)
            C.enable_chunked_lm_loss(model, int(arm.replace("chunk", "")))
    for T in a.tokens:
        for arm in a.arms.split(","):
            if T >= a.stock_last_from and arm == "stock":
                continue
            rows.append(run_arm(model, opt, set_arm, arm, T, a.steps, vocab))
    for T in a.tokens:                     # stock at the largest sizes last: an OOM there cannot disturb the other rows
        if T >= a.stock_last_from and "stock" in a.arms.split(","):
            rows.append(run_arm(model, opt, set_arm, "stock", T, a.steps, vocab))
    if a.out:
        json.dump({"env": env, "args": vars(a), "rows": rows, "stats": dict(C.CHUNKED_LM_LOSS_STATS)}, open(a.out, "w"), indent=1)


def run_arm(model, opt, set_arm, arm, T, steps, vocab):
    set_arm(arm)
    g = torch.Generator(device="cuda").manual_seed(T)
    row = {"tokens": T, "arm": arm, "engaged": "_e4b_chunked_lm_loss" in model.__dict__}
    try:
        ts, peaks, losses, chunked = [], [], [], []
        for _ in range(steps):
            ids = torch.randint(0, vocab, (1, T), device="cuda", generator=g)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            peaks.append(torch.cuda.max_memory_allocated())
            losses.append(float(out.loss.detach()))
            chunked.append(out.logits is None)
            del out
        row.update(peak_gib=round(max(peaks[1:] or peaks) / 2**30, 3), step_s_median=round(statistics.median(ts[1:] or ts), 4),
                   step_s_all=[round(t, 4) for t in ts], losses=[round(x, 4) for x in losses], logits_none=chunked)
    except torch.cuda.OutOfMemoryError as e:
        row.update(oom=str(e).split("\n")[0][:240])
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    main()
