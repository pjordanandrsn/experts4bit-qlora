"""End-to-end QLoRA fine-tune of a fused-MoE model (OLMoE, Qwen3-MoE, Gemma-4) on a single small GPU.

The expert weights are streamed in and frozen in NF4 (:class:`Experts4bit`); only small per-expert /
per-projection LoRA adapters train. Set ``MODEL`` to any supported fused-MoE checkpoint (see the
loader's ``SUPPORTED_MODEL_TYPES``). Configured entirely via env vars, e.g.::

    STEPS=150 R=8 TRAIN_EXPERTS=1 TRAIN_ATTENTION=0 OUT=./out \
      python -m experts4bit_qlora.train

Set ``OFFLOAD_EXPERTS=1`` to keep the frozen 4-bit experts in (pinned, unless ``OFFLOAD_PIN=0``) CPU
RAM and stream one layer's experts to the GPU at a time — lowers peak GPU memory (so models whose
experts exceed VRAM can train) at the cost of a per-layer PCIe transfer. See
:mod:`experts4bit_qlora.engines.offload` and ``docs/METHODOLOGY.md`` §11.

Requires (beyond this package): a CUDA GPU, transformers>=5.0, datasets, accelerate, safetensors.
"""

import os
import time

import sys

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
# Tokens per forward. A fused-MoE step's cost is largely FIXED per active expert, so one
# row per forward -- what this trainer did -- pays that tax per example.
#
# Measured, OLMoE-1B-7B on an RTX A2000, SEQ=192, alpaca, 15 steps x grad_accum 4:
#
#   TOKEN_BUDGET |  s/step | tok/s | peak GPU
#              0 |   17.8  |    22 | 5.23 GB     <- one row per forward
#           1024 |   22.2  |   144 | 5.88 GB
#           2048 |   22.8  |   248 | 6.67 GB     <- default
#           4096 |     --  |  OOM  |   --        <- on a 12 GB card
#
# 11.3x the throughput for +1.4 GB. Steps get SLOWER (each carries ~15x more data); the
# metric this moves is tok/s, not s/step. The ceiling is VRAM: raise it until you OOM,
# then back off. 0 restores the one-row path the v0.2.0 convergence receipts used.
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "2048"))
# Backoff floor. Below this the budget is not the problem and the OOM is real.
_MIN_TOKEN_BUDGET = 256
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
QUANT_TYPE = os.environ.get("QUANT_TYPE", "nf4")  # nf4/fp4 (4-bit), int8/fp8 (8-bit), bf16/fp16 (passthrough)
OUT = os.environ.get("OUT", "./experts4bit-lora-out")

EVAL_PROMPTS = [
    "List three tips for staying focused while working from home.",
    "Explain what a black hole is in one sentence.",
]


def save_adapter(model, out, tag):
    os.makedirs(out, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k}
    torch.save(sd, os.path.join(out, f"adapter_{tag}.pt"))
    return len(sd)


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


def _pad_batch(rows, pad_id, device):
    """Right-pad a list of encoded examples into one forward.

    Labels pad with -100 and the attention mask zeroes the padding, so the padded
    positions contribute neither loss nor attention. That is what makes this safe to do
    without touching the model: no example can see another's tokens.
    """
    width = max(len(r["input_ids"]) for r in rows)
    ids, lbl, att = [], [], []
    for r in rows:
        n = len(r["input_ids"])
        ids.append(r["input_ids"] + [pad_id] * (width - n))
        lbl.append(list(r["labels"]) + [-100] * (width - n))
        att.append([1] * n + [0] * (width - n))
    t = lambda x: torch.tensor(x, device=device)  # noqa: E731
    return t(ids), t(lbl), t(att)


def _token_budget_batches(data, budget, pad_id, device, bucket=64):
    """Yield micro-batches sized by TOKEN COUNT, not row count.

    A fused-MoE step pays a cost that is fixed per active expert -- the reference path
    dequantizes each routed expert once, the fused path launches one grouped GEMM per
    expert group -- so a step that carries 150 tokens pays nearly what a step carrying
    4000 does. Batching one row at a time (what this trainer did) pays that tax per
    example. Budgeting by tokens amortizes it.

    The budget is measured as the PADDED cost, ``rows * width``, not the sum of true
    lengths: that is the work the GPU actually does, and it stops one long row from
    silently blowing up a batch that looked affordable. Rows are drawn from a
    length-sorted bucket so a batch's rows are similar lengths and padding waste stays
    small; ``bucket`` rows are buffered and sorted at a time, which keeps the stream
    order shuffled between buckets rather than globally sorted by length.
    """
    buf, rows, width = [], [], 0
    it = iter(data)
    while True:
        if not buf:
            for _ in range(bucket):
                try:
                    buf.append(next(it))
                except StopIteration:
                    break
            if not buf:
                break
            buf.sort(key=lambda r: len(r["input_ids"]))
        r = buf.pop(0)
        w = max(width, len(r["input_ids"]))
        if rows and (len(rows) + 1) * w > budget:
            yield _pad_batch(rows, pad_id, device)
            rows, width = [], 0
            w = len(r["input_ids"])
        rows.append(r)
        width = w
    if rows:
        yield _pad_batch(rows, pad_id, device)


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


def _print_env_help(which: str) -> None:
    """Configuration is env-vars, not flags -- so argv was ignored entirely and
    `python -m experts4bit_qlora.train --help` fell straight through into a real
    run: it loaded the model, initialised CUDA and spawned inductor compile
    workers. Measured on an RTX 5090: ~10 minutes and 6.2 GB of VRAM before the
    user learns there is no such flag.

    Scope of this fix, stated honestly: it exits before the model load, the
    CUDA init and the inductor pool -- the expensive part. It does NOT avoid
    importing torch/bitsandbytes, because the package __init__ eagerly imports
    .lora/.offload/.fast/.cold_engine. Making --help import-free needs a lazy
    __init__, which is a wider change than this defect warrants.
    """
    import os as _os
    print(f"usage: python -m experts4bit_qlora.{which}\n")
    print("Configuration is by ENVIRONMENT VARIABLE (there are no CLI flags).\n")
    rows = [
        ("MODEL", "allenai/OLMoE-1B-7B-0924", "HF model id"),
        ("QUANT_TYPE", "nf4", "nf4/fp4 (4-bit), int8/fp8 (8-bit), bf16/fp16 (passthrough)"),
        ("SEQ", "192", "sequence length"),
        ("STEPS", "40", "optimizer steps"),
        ("GRAD_ACCUM", "4", "gradient accumulation"),
        ("TOKEN_BUDGET", "2048", "tokens per forward (0 = one row per forward)"),
        ("LR", "2e-4", "learning rate"),
        ("R / ALPHA", "8 / 16", "LoRA rank / alpha"),
        ("N_TRAIN", "2000", "training examples"),
        ("EVAL_EVERY", "50", "eval interval in steps"),
        ("TRAIN_EXPERTS", "1", "train expert LoRA"),
        ("TRAIN_ATTENTION", "1", "train attention LoRA"),
        ("TRAIN_ROUTER", "0", "train the router"),
        ("OFFLOAD_EXPERTS", "0", "keep experts in pinned CPU RAM"),
        ("OFFLOAD_PIN", "1", "pin the offloaded expert memory"),
        ("DO_GEN", "1", "sample generations during training"),
        ("SEED", "0", "torch manual seed"),
        ("OUT", "./experts4bit-lora-out", "adapter output dir"),
    ]
    for k, d, h in rows:
        cur = _os.environ.get(k.split(" /")[0])
        mark = f"  [set: {cur}]" if cur is not None else ""
        print(f"  {k:<17} default={d:<26} {h}{mark}")
    print("\nexample:\n  MODEL=Qwen/Qwen3-30B-A3B OFFLOAD_EXPERTS=1 STEPS=100 \\\n"
          f"    python -m experts4bit_qlora.{which}")


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        _print_env_help("train")
        return 0
    torch.manual_seed(int(os.environ.get("SEED", "0")))  # default unchanged; the mode-matrix scripts set it
    log(f"loading {MODEL} via streaming 4-bit loader (CPU-RAM-light)...")
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(MODEL)
    # Pad id for the batcher. Padded positions get label -100 and attention 0, so the
    # choice is inert -- eos is the conventional fallback when a model ships no pad token.
    _PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model, _ = load_moe_4bit_streaming(
        MODEL, DEVICE, DTYPE, R, ALPHA, offload=OFFLOAD_EXPERTS, pin=OFFLOAD_PIN, quant_type=QUANT_TYPE
    )
    # The loader already placed every module; under offload the experts live in pinned CPU RAM by
    # design, so a blanket model.to(DEVICE) would drag them back onto the GPU and defeat offloading.
    if not OFFLOAD_EXPERTS:
        model.to(DEVICE)
    n_attn = add_attention_lora(model, R, ALPHA, DTYPE) if TRAIN_ATTENTION else 0
    log(f"attn LoRA {n_attn} projs | train experts={TRAIN_EXPERTS} attn={TRAIN_ATTENTION} router={TRAIN_ROUTER}")

    from . import expert_profile

    expert_profile.attach(model)  # no-op unless E4B_EXPERT_PROFILE is set (profile-only)

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
    log(
        f"loaded. trainable: {sum(p.numel() for p in trainable):,} "
        f"(lora {sum(p.numel() for p in lora_params):,} + router {sum(p.numel() for p in router_params):,}) "
        f"| offload={'on' if OFFLOAD_EXPERTS else 'off'} | GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
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
    def _batch_stream(budget):
        if budget <= 0:            # 0 restores the historical one-row-per-forward path,
            for ex in data:        # which is what the v0.2.0 convergence receipts used
                yield (torch.tensor([ex["input_ids"]], device=DEVICE),
                       torch.tensor([ex["labels"]], device=DEVICE),
                       torch.ones(1, len(ex["input_ids"]), dtype=torch.long, device=DEVICE))
        else:
            yield from _token_budget_batches(data, budget, _PAD_ID, DEVICE)

    budget = TOKEN_BUDGET
    it, t0, ema, best = _batch_stream(budget), time.time(), None, float("inf")
    tok_seen = 0
    from . import census as _census
    _clock = _census.PhaseClock() if _census.enabled() else None
    _tprof, _tprof_steps = None, [0]
    _prof_out = os.environ.get("TR1_PROFILE_OUT")
    if _clock and _prof_out:
        # kernel-budget instrument (PREREG-tr1-census #2): same
        # schedule shape + table format as the serving census, so
        # bench/hybrid-g9/f1/step_budget.py parses it unchanged.
        from torch.profiler import ProfilerActivity, profile, schedule
        # active=4 and NO record_shapes: a 30B training step is
        # millions of events, and the 8-step shape-recorded window
        # drove the census container into its 170 GiB cgroup cap
        # (SIGKILL at step ~10, peak == limit). The kernel-budget
        # table needs names and times, not shapes. Run the profiler
        # in its OWN short run, never inside the A/A census pair --
        # its overhead perturbs the phases it is measuring.
        _tprof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(skip_first=3, wait=0, warmup=1, active=4,
                              repeat=1),
            record_shapes=False)
        _tprof.__enter__()
    from .engines.offload import offload_stats_report, reset_offload_stats

    reset_offload_stats()  # measure the training loop only (drop load/BEFORE-eval transfers)
    step = 0
    while step < STEPS:
        if _clock:
            _clock.step_start()
            _clock.start("optim")
        opt.zero_grad()
        if _clock:
            _clock.stop()
        loss_acc = 0.0
        try:
            for _ in range(GRAD_ACCUM):
                if _clock:
                    _clock.start("data")
                try:
                    ids, lbl, att = next(it)
                except StopIteration:
                    it = _batch_stream(budget)
                    ids, lbl, att = next(it)
                if _clock:
                    _clock.stop()
                    _clock.start("forward")
                out = model(input_ids=ids, labels=lbl, attention_mask=att)
                if _clock:
                    _clock.stop()
                    _clock.start("backward")
                (out.loss / GRAD_ACCUM).backward()
                if _clock:
                    _clock.stop()
                    _clock.start("loss_sync")
                loss_acc += out.loss.item() / GRAD_ACCUM
                tok_seen += int(att.sum())
                if _clock:
                    _clock.stop()
        except torch.cuda.OutOfMemoryError:
            if _clock and _clock._open is not None:
                _clock.stop()
            # The token budget's ceiling is VRAM, and the right value is host- AND
            # model-specific -- 2048 fits OLMoE on a 12 GB card, a 30B offloaded model is
            # a different profile entirely. Dying on a default is a bad way to learn that,
            # so back off and keep training. The whole step is discarded (a partial
            # backward has already accumulated some grads) and retried at half the budget.
            if budget <= _MIN_TOKEN_BUDGET:
                raise
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            budget = max(budget // 2, _MIN_TOKEN_BUDGET)
            it = _batch_stream(budget)
            log(f"  [oom] step {step + 1}: halving TOKEN_BUDGET -> {budget} and retrying "
                f"(set TOKEN_BUDGET explicitly to skip this search)")
            continue
        step += 1
        if _clock:
            _clock.start("optim")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if _clock:
            _clock.stop()
            # loss lands in the receipt so the prereg's NaN/divergence
            # refusal is checkable from the artifact, not the log
            _clock.step_end(extra={"loss": float(loss_acc)})
        if _tprof is not None:
            _tprof.step()
            _tprof_steps[0] += 1
        ema = loss_acc if ema is None else 0.9 * ema + 0.1 * loss_acc
        if step % 10 == 0 or step == 1:
            log(
                f"  step {step}/{STEPS}  loss {loss_acc:.3f}  ema {ema:.3f}  "
                f"({(time.time() - t0) / step:.1f}s/step, "
                f"{tok_seen / max(time.time() - t0, 1e-9):.0f} tok/s)"
            )
        if step % EVAL_EVERY == 0:
            el = eval_loss(model, eval_data)
            marker = ""
            if el < best:
                best = el
                save_adapter(model, OUT, "best")
                marker = "  *new best -> saved"
            log(f"  [eval] step {step}: held-out loss {el:.4f} (best {best:.4f}){marker}")
            model.train()
    log(
        f"training done in {time.time() - t0:.0f}s "
        f"| peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (offload={'on' if OFFLOAD_EXPERTS else 'off'})"
    )
    offload_stats_report(log)  # no-op unless E4B_OFFLOAD_STATS=1
    if _tprof is not None:
        _tprof.__exit__(None, None, None)
        _tbl = _tprof.key_averages().table(sort_by="cuda_time_total",
                                           row_limit=80)
        _active = max(0, min(4, _tprof_steps[0] - 3 - 1))
        _hdr = (f"profiled training steps: {_tprof_steps[0]} "
                f"(active window: {_active}/4)\n")
        if _active < 4:
            _hdr += ("WARNING: active window INCOMPLETE -- this table "
                     "under-samples and must not be cited as the "
                     "attribution\n")
        with open(_prof_out, "w") as _f:
            _f.write(_hdr + _tbl)
        log(f"[tr1-census] wrote {_prof_out} (active={_active}/4)")
    if _clock:
        out_path = os.environ.get("TR1_CENSUS_OUT", "tr1_census.json")
        _clock.write(out_path, {
            "model": MODEL, "seq": SEQ, "steps": STEPS,
            "grad_accum": GRAD_ACCUM, "token_budget": TOKEN_BUDGET,
            # the OOM backoff halves the LIVE budget; the env value
            # alone cannot explain an A/A phase drift between runs
            # that settled at different effective budgets
            "token_budget_effective": budget,
            "offload": bool(OFFLOAD_EXPERTS),
            "torch": torch.__version__,
        })
        log(f"[tr1-census] wrote {out_path} ({len(_clock.steps)} steps)")

    model.gradient_checkpointing_disable()
    eval_after = eval_loss(model, eval_data)
    log(
        f"held-out eval loss: BEFORE {eval_before:.4f} -> AFTER {eval_after:.4f} "
        f"(delta {eval_after - eval_before:+.4f}) | best {best:.4f}"
    )
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
