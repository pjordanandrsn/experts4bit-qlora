"""The LM head + causal-LM loss alone, stock against chunked: peak allocated memory and time per forward+backward.

Stock is what a Hugging Face causal-LM forward does with labels: ``logits = lm_head(hidden)`` (kept alive through backward, as
the model's output holds them) and ``ForCausalLMLoss(logits, labels, vocab)``. Chunked is
``experts4bit_qlora.engines.chunked_lm_loss.chunked_causal_lm_loss`` at each chunk size. The head is frozen bf16 (as e4b trains:
the LM head is not adapted) and the hidden states are bf16 with requires_grad, so backward computes the hidden-state gradient
the decoder would receive. Every token is supervised except each row's last (the shift).

    python bench_lm_head_loss.py --vocab 151936 --hidden 2048 --tokens 1024 2048 4096 --chunks 256 512 1024 2048 --out res.json

Peak is ``torch.cuda.max_memory_allocated`` over one forward+backward, minus what was allocated before it (the head's weight
and the hidden states); the absolute peak is recorded beside it. Time is the median of ``--iters`` timed calls after
``--warmup`` untimed ones, each bracketed by ``torch.cuda.synchronize``.
"""
import argparse
import json
import os
import statistics
import time

import torch
from transformers.loss.loss_utils import ForCausalLMLoss

from experts4bit_qlora.engines.chunked_lm_loss import chunked_causal_lm_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--no-reduced-precision-reduction", action="store_true",
                    help="torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False (a diagnostic)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    if a.no_reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(0)
    head = torch.nn.Linear(a.hidden, a.vocab, bias=False, device=dev, dtype=torch.bfloat16).requires_grad_(False)
    torch.nn.init.normal_(head.weight, std=0.02)
    rows = []
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "vocab": a.vocab, "hidden": a.hidden,
           "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for T in a.tokens:
        g = torch.Generator(device=dev).manual_seed(T)
        labels = torch.randint(0, a.vocab, (1, T), device=dev, generator=g)
        h0 = torch.randn(1, T, a.hidden, device=dev, dtype=torch.bfloat16, generator=g)
        ref_grad = ref_loss = None
        for impl in ["stock"] + list(a.chunks):
            def call(h):
                if impl == "stock":
                    logits = head(h)
                    loss = ForCausalLMLoss(logits, labels, a.vocab)
                    loss.backward()
                    del logits
                    return loss
                loss = chunked_causal_lm_loss(h, labels, head, chunk=impl)
                loss.backward()
                return loss
            h = h0.clone().requires_grad_(True)
            row = {"tokens": T, "impl": "stock" if impl == "stock" else f"chunk{impl}"}
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                loss = call(h)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                row.update(peak_delta_gib=round((peak - base) / 2**30, 4), peak_abs_gib=round(peak / 2**30, 4),
                           loss=float(loss.detach()))
                if impl == "stock":
                    ref_grad, ref_loss = h.grad.detach().clone(), float(loss.detach())
                else:
                    d = (h.grad.float() - ref_grad.float())
                    row.update(loss_minus_stock=float(loss.detach()) - ref_loss, grad_max_abs_diff=float(d.abs().max()),
                               grad_bitwise_equal=bool(torch.equal(h.grad, ref_grad)),
                               grad_max_abs=float(ref_grad.float().abs().max()),
                               grad_rel_l2=float(d.norm() / ref_grad.float().norm()),
                               grad_frac_elements_differ=float((d != 0).float().mean()))
                for _ in range(a.warmup):
                    h.grad = None
                    call(h)
                ts = []
                for _ in range(a.iters):
                    h.grad = None
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    call(h)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                row.update(ms_median=round(1e3 * statistics.median(ts), 2), ms_min=round(1e3 * min(ts), 2),
                           ms_max=round(1e3 * max(ts), 2))
            except torch.cuda.OutOfMemoryError as e:
                row.update(oom=str(e).split("\n")[0][:200])
            del h
            print(json.dumps(row), flush=True)
            rows.append(row)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"env": env, "args": vars(a), "rows": rows, "host": os.uname().nodename}, f, indent=1)


if __name__ == "__main__":
    main()
