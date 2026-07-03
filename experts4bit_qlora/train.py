"""End-to-end QLoRA fine-tune of a fused-MoE model (OLMoE, Qwen3-MoE, Gemma-4) on a single small GPU.

The expert weights are streamed in and frozen in NF4 (:class:`Experts4bit`); only small per-expert /
per-projection LoRA adapters train. Set ``MODEL`` to any supported fused-MoE checkpoint (see the
loader's ``SUPPORTED_MODEL_TYPES``). Configured entirely via env vars, e.g.::

    STEPS=150 R=8 TRAIN_EXPERTS=1 TRAIN_ATTENTION=0 OUT=./out \
      python -m experts4bit_qlora.train

Set ``OFFLOAD_EXPERTS=1`` to keep the frozen 4-bit experts in (pinned, unless ``OFFLOAD_PIN=0``) CPU
RAM and stream one layer's experts to the GPU at a time — lowers peak GPU memory (so models whose
experts exceed VRAM can train) at the cost of a per-layer PCIe transfer. See
:mod:`experts4bit_qlora.offload` and ``docs/METHODOLOGY.md`` §11.

Requires (beyond this package): a CUDA GPU, transformers>=5.0, datasets, accelerate, safetensors.
"""

import os
import time

import torch

from .loader import load_moe_4bit_streaming
from .lora import add_attention_lora
from .util import log

MODEL = os.environ.get("MODEL", "allenai/OLMoE-1B-7B-0924")
DEVICE = "cuda"
DTYPE = torch.bfloat16
SEQ = int(os.environ.get("SEQ", "192"))
STEPS = int(os.environ.get("STEPS", "40"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LR = float(os.environ.get("LR", "2e-4"))
R, ALPHA = int(os.environ.get("R", "8")), int(os.environ.get("ALPHA", "16"))
N_TRAIN = int(os.environ.get("N_TRAIN", "2000"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "50"))
TRAIN_EXPERTS = os.environ.get("TRAIN_EXPERTS", "1") == "1"
TRAIN_ATTENTION = os.environ.get("TRAIN_ATTENTION", "1") == "1"
TRAIN_ROUTER = os.environ.get("TRAIN_ROUTER", "0") == "1"
DO_GEN = os.environ.get("DO_GEN", "1") == "1"
OFFLOAD_EXPERTS = os.environ.get("OFFLOAD_EXPERTS", "0") == "1"
OFFLOAD_PIN = os.environ.get("OFFLOAD_PIN", "1") == "1"
OUT = os.environ.get("OUT", "./experts4bit-lora-out")

# Optional structured telemetry (both default-off, so a plain run is byte-for-byte unchanged):
#   WANDB=1            -> log the run to Weights & Biases (offline-safe via WANDB_MODE=offline)
#   METRICS_JSONL=path -> append per-step metrics as JSON lines (drives the offline charts)
WANDB = os.environ.get("WANDB", "0") == "1"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "experts4bit-qlora")
RUN_NAME = os.environ.get("RUN_NAME") or f"{MODEL.split('/')[-1]}-offload-{'on' if OFFLOAD_EXPERTS else 'off'}"
METRICS_JSONL = os.environ.get("METRICS_JSONL", "")

EVAL_PROMPTS = [
    "List three tips for staying focused while working from home.",
    "Explain what a black hole is in one sentence.",
]


def save_adapter(model, out, tag):
    os.makedirs(out, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k}
    torch.save(sd, os.path.join(out, f"adapter_{tag}.pt"))
    return len(sd)


class MetricSink:
    """Optional per-step telemetry: Weights & Biases and/or a JSONL file. Both are no-ops
    unless enabled via ``WANDB`` / ``METRICS_JSONL``, so default training runs are unchanged."""

    def __init__(self, config):
        self.wandb = None
        self.jsonl = None
        if WANDB:
            import wandb

            self.wandb = wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config=config)
        if METRICS_JSONL:
            os.makedirs(os.path.dirname(METRICS_JSONL) or ".", exist_ok=True)
            self.jsonl = open(METRICS_JSONL, "a")
            self._write({"event": "config", **config})

    def _write(self, record):
        import json

        self.jsonl.write(json.dumps(record) + "\n")
        self.jsonl.flush()

    def log(self, step, metrics):
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        if self.jsonl is not None:
            self._write({"step": step, **metrics})

    def summary(self, metrics):
        if self.wandb is not None:
            self.wandb.summary.update(metrics)
        if self.jsonl is not None:
            self._write({"event": "summary", **metrics})

    def finish(self):
        if self.wandb is not None:
            self.wandb.finish()
        if self.jsonl is not None:
            self.jsonl.close()


@torch.no_grad()
def generate(model, tokenizer, instruction, max_new=48):
    model.eval()
    model.config.use_cache = True
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **ids,
        max_new_tokens=max_new,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def encode_alpaca(tokenizer, split):
    """Alpaca instruction tuning; mask the prompt so loss is only on the response."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split=split)

    def encode(ex):
        head = f"### Instruction:\n{ex['instruction']}\n\n"
        if ex.get("input"):
            head += f"### Input:\n{ex['input']}\n\n"
        prompt = head + "### Response:\n"
        p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full = tokenizer(prompt + ex["output"] + tokenizer.eos_token, add_special_tokens=True)["input_ids"][:SEQ]
        labels = list(full)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100
        return {"input_ids": full, "labels": labels}

    ds = ds.map(encode, remove_columns=ds.column_names)
    # Drop examples whose response was fully truncated by SEQ (all labels -100 => no supervised
    # tokens => nan loss); keeps the before/after eval well-defined even at short SEQ.
    return ds.filter(lambda ex: any(t != -100 for t in ex["labels"]))


@torch.no_grad()
def eval_loss(model, eval_data):
    """Mean response-only loss over a fixed held-out set (clean before/after signal)."""
    model.eval()
    model.config.use_cache = False
    tot, n = 0.0, 0
    for ex in eval_data:
        ids = torch.tensor([ex["input_ids"]], device=DEVICE)
        lbl = torch.tensor([ex["labels"]], device=DEVICE)
        loss = model(input_ids=ids, labels=lbl).loss.item()
        if loss == loss:  # skip nan (e.g. an all-masked example) defensively
            tot += loss
            n += 1
    return tot / max(n, 1)


def main():
    torch.manual_seed(0)
    log(f"loading {MODEL} via streaming 4-bit loader (CPU-RAM-light)...")
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(MODEL)
    model, _ = load_moe_4bit_streaming(MODEL, DEVICE, DTYPE, R, ALPHA, offload=OFFLOAD_EXPERTS, pin=OFFLOAD_PIN)
    # The loader already placed every module; under offload the experts live in pinned CPU RAM by
    # design, so a blanket model.to(DEVICE) would drag them back onto the GPU and defeat offloading.
    if not OFFLOAD_EXPERTS:
        model.to(DEVICE)
    n_attn = add_attention_lora(model, R, ALPHA, DTYPE) if TRAIN_ATTENTION else 0
    log(f"attn LoRA {n_attn} projs | train experts={TRAIN_EXPERTS} attn={TRAIN_ATTENTION} router={TRAIN_ROUTER}")

    lora_params, router_params = [], []
    for n, p in model.named_parameters():
        train_lora = "lora" in n and ((TRAIN_EXPERTS and "experts" in n) or (TRAIN_ATTENTION and "self_attn" in n))
        if train_lora:
            p.requires_grad_(True)
            lora_params.append(p)
        elif TRAIN_ROUTER and n.endswith("mlp.gate.weight"):
            p.requires_grad_(True)
            router_params.append(p)
        else:
            p.requires_grad_(False)
    trainable = lora_params + router_params
    torch.cuda.synchronize()
    loaded_gpu_gb = torch.cuda.memory_allocated() / 1e9
    log(
        f"loaded. trainable: {sum(p.numel() for p in trainable):,} "
        f"(lora {sum(p.numel() for p in lora_params):,} + router {sum(p.numel() for p in router_params):,}) "
        f"| offload={'on' if OFFLOAD_EXPERTS else 'off'} | GPU mem: {loaded_gpu_gb:.2f} GB"
    )
    # Reset so the peak we report at the end reflects the training step (a full fwd+bwd), which is
    # the figure that decides whether a model fits — the point of OFFLOAD_EXPERTS.
    torch.cuda.reset_peak_memory_stats()

    before = {}
    if DO_GEN:
        log("BEFORE-training generations:")
        for q in EVAL_PROMPTS:
            before[q] = generate(model, tok, q)
            log(f"  Q: {q}\n     A: {before[q]}")

    log("preparing dataset (alpaca, response-only loss)...")
    data = encode_alpaca(tok, f"train[:{N_TRAIN}]")
    eval_data = encode_alpaca(tok, f"train[{N_TRAIN}:{N_TRAIN + 64}]")

    eval_before = eval_loss(model, eval_data)
    log(f"held-out eval loss BEFORE: {eval_before:.4f}")

    sink = MetricSink(
        {
            "model": MODEL,
            "offload_experts": OFFLOAD_EXPERTS,
            "steps": STEPS,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "r": R,
            "alpha": ALPHA,
            "seq": SEQ,
            "n_train": N_TRAIN,
            "loaded_gpu_gb": loaded_gpu_gb,
        }
    )
    sink.log(0, {"eval/held_out_loss": eval_before, "eval/best": eval_before})

    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": LR})
    if router_params:
        groups.append({"params": router_params, "lr": LR * 0.1})  # router is sensitive -> 0.1x LR
    opt = torch.optim.AdamW(groups, lr=LR)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(5, STEPS // 10), num_training_steps=STEPS)
    # Gradient checkpointing: recompute each decoder layer in backward instead of saving the
    # dequantized expert weights as activations — the key to fitting MoE QLoRA on a small card.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    log(
        f"training: {STEPS} steps x grad_accum {GRAD_ACCUM} (seq<= {SEQ}), lr={LR}, cosine+warmup, eval every {EVAL_EVERY}"
    )
    it, t0, ema, best = iter(data), time.time(), None, float("inf")
    t_prev = t0
    for step in range(STEPS):
        opt.zero_grad()
        loss_acc = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                ex = next(it)
            except StopIteration:
                it = iter(data)
                ex = next(it)
            ids = torch.tensor([ex["input_ids"]], device=DEVICE)
            lbl = torch.tensor([ex["labels"]], device=DEVICE)
            out = model(input_ids=ids, labels=lbl)
            (out.loss / GRAD_ACCUM).backward()
            loss_acc += out.loss.item() / GRAD_ACCUM
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        ema = loss_acc if ema is None else 0.9 * ema + 0.1 * loss_acc
        now = time.time()
        step_time = now - t_prev
        t_prev = now
        sink.log(
            step + 1,
            {
                "train/loss": loss_acc,
                "train/ema": ema,
                "train/lr": sched.get_last_lr()[0],
                "perf/s_per_step": step_time,
                "mem/gpu_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                "mem/gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            },
        )
        if (step + 1) % 10 == 0 or step == 0:
            log(
                f"  step {step + 1}/{STEPS}  loss {loss_acc:.3f}  ema {ema:.3f}  ({(time.time() - t0) / (step + 1):.1f}s/step)"
            )
        if (step + 1) % EVAL_EVERY == 0:
            el = eval_loss(model, eval_data)
            marker = ""
            if el < best:
                best = el
                save_adapter(model, OUT, "best")
                marker = "  *new best -> saved"
            log(f"  [eval] step {step + 1}: held-out loss {el:.4f} (best {best:.4f}){marker}")
            sink.log(step + 1, {"eval/held_out_loss": el, "eval/best": best})
            model.train()
    total_time_s = time.time() - t0
    peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    log(
        f"training done in {total_time_s:.0f}s "
        f"| peak GPU mem: {peak_gpu_gb:.2f} GB (offload={'on' if OFFLOAD_EXPERTS else 'off'})"
    )

    model.gradient_checkpointing_disable()
    eval_after = eval_loss(model, eval_data)
    log(
        f"held-out eval loss: BEFORE {eval_before:.4f} -> AFTER {eval_after:.4f} "
        f"(delta {eval_after - eval_before:+.4f}) | best {best:.4f}"
    )
    sink.summary(
        {
            "offload": OFFLOAD_EXPERTS,
            "loaded_gpu_gb": loaded_gpu_gb,
            "peak_gpu_gb": peak_gpu_gb,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "eval_delta": eval_after - eval_before,
            "eval_best": best,
            "total_time_s": total_time_s,
            "s_per_step": total_time_s / max(STEPS, 1),
        }
    )
    sink.finish()
    model.train()

    if DO_GEN:
        model.gradient_checkpointing_disable()
        log("AFTER-training generations:")
        for q in EVAL_PROMPTS:
            after = generate(model, tok, q)
            log(f"  Q: {q}\n     BEFORE: {before.get(q, '')}\n     AFTER : {after}")

    n = save_adapter(model, OUT, "last")
    log(f"saved final adapter ({n} tensors) -> {OUT}/adapter_last.pt ; best kept at adapter_best.pt")


if __name__ == "__main__":
    main()
