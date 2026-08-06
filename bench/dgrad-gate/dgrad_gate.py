"""Layer-composed parity gate for the dgrad backward, at full depth.

The gap this closes: `enable_fast_train(dgrad=True)` shipped with per-op accuracy
measured (~2.9e-3 against the decode oracle) and layer-composed fidelity NOT measured.
That distinction is not pedantic here — this lane has already seen a path that measured
*better* per-op cost +0.023% perplexity through 16 layers, and the 48-layer gate caught
a LoRA-scaling bug that 16-layer parity passed.

Perplexity is the wrong instrument for dgrad: it only changes the BACKWARD, so the
forward is bit-identical by construction. What compounds across layers is gradient
error, and what it does to a training run is a trajectory difference. So:

  1. composed gradient parity  — every trainable tensor, all layers, at a fixed step
  2. loss trajectory           — median |Δ| against the band the repo already uses
  3. frozen-storage exactness  — no arm may mutate the quantized experts (with a
                                 flipped-byte control, so the check is non-vacuous)
  4. cost                      — peak VRAM and s/step, since the point is speed

Arms: reference loop, fast_train (fused fwd, loop bwd), fast_train+dgrad, batched.
Every arm is scored against `reference`, from ONE model load with adapter
snapshot/restore between arms, so all arms start bit-identical and see the same data
in the same order. Mirrors bench/fused-train-gate's protocol.
"""
import argparse, hashlib, json, os, time

import torch


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


ATTRS = ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")


def frozen_tensors(model):
    """Yield (name, tensor) for every quantized expert store, offloaded or not.

    Under ``offload=True`` the module's buffers are 0-element PLACEHOLDERS between
    forwards — the real bytes live in the offload handle's ``home`` dict on CPU. The
    first version of this walked the module attributes and skipped everything with
    ``numel() == 0``, so it hashed **zero tensors** and every arm's "frozen storage
    unchanged" verdict compared two empty dicts. It passed, and meant nothing.
    """
    for name, mod in model.named_modules():
        h = getattr(mod, "_offload", None)
        home = getattr(h, "home", None) if h is not None else None
        if isinstance(home, dict) and home:
            for k, t in home.items():
                if torch.is_tensor(t) and t.numel():
                    yield f"{name}.home.{k}", t
            continue
        for attr in ATTRS:
            t = getattr(mod, attr, None)
            if torch.is_tensor(t) and t.numel():
                yield f"{name}.{attr}", t


def frozen_hashes(model):
    out, nbytes = {}, 0
    for name, t in frozen_tensors(model):
        out[name] = _sha(t)
        nbytes += t.numel() * t.element_size()
    return out, nbytes


def trainable_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore(model, snap):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.copy_(snap[n])


def grads_now(model):
    return {n: p.grad.detach().float().clone()
            for n, p in model.named_parameters() if p.grad is not None}


def rel(a, b):
    d = (a - b).norm().item()
    n = b.norm().item()
    return d / n if n else (0.0 if d == 0 else float("inf"))


def build_batches(tok, n, seq, device, seed=0):
    """Deterministic synthetic batches: the comparison is arm-vs-arm on IDENTICAL
    inputs, so what the text says is irrelevant — reproducibility is not."""
    g = torch.Generator().manual_seed(seed)
    vocab = int(getattr(tok, "vocab_size", 32000))
    return [torch.randint(0, vocab, (1, seq), generator=g).to(device) for _ in range(n)]


def run_arm(model, arm, batches, lr, device, enable, disable):
    """One arm: patch, N steps of fwd+bwd+step, capture gradients at step 0."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    n_patched = enable(model) if enable else 0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    losses, first_grads, t0 = [], None, time.time()
    for i, ids in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        if i == 0:
            first_grads = grads_now(model)
        opt.step()
        losses.append(float(out.loss.detach()))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / max(1, len(batches))
    peak = torch.cuda.max_memory_allocated() / 2**30
    if disable:
        disable(model)
    return dict(arm=arm, n_patched=n_patched, losses=losses,
                s_per_step=round(dt, 3), train_peak_gb=round(peak, 2)), first_grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--offload", type=int, default=1)
    ap.add_argument("--out", default="dgrad_gate.json")
    a = ap.parse_args()

    import experts4bit_qlora as e4b
    from experts4bit_qlora import (disable_batched_train, disable_fast_train,
                                   enable_batched_train, enable_fast_train,
                                   load_moe_4bit_streaming)
    from transformers import AutoTokenizer
    import nf4_grouped

    dev = "cuda"
    env = dict(gpu=torch.cuda.get_device_name(0),
               cap=list(torch.cuda.get_device_capability(0)),
               vram_total_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
               torch=torch.__version__, e4b=e4b.__version__,
               gnf4_has_dgrad=hasattr(nf4_grouped, "dgrad_4bit_grouped"))
    print("env:", json.dumps(env), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    print("loading model (streaming 4-bit)...", flush=True)
    model, cfg = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, r=a.r, alpha=a.alpha,
                                         quant_type="nf4", offload=bool(a.offload))
    if not a.offload:
        model.to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    layers = cfg.num_hidden_layers if hasattr(cfg, "num_hidden_layers") else \
        getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", -1)

    # Only the expert LoRA trains — that is the path dgrad affects.
    for n, p in model.named_parameters():
        p.requires_grad_("lora" in n and "experts" in n)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"layers={layers} trainable={n_train:,}", flush=True)

    batches = build_batches(tok, a.steps, a.seq, dev)
    base_snap = trainable_snapshot(model)
    frozen_before, nbytes = frozen_hashes(model)
    if not frozen_before:
        raise SystemExit(
            "FATAL: hashed 0 frozen expert tensors — the integrity check would be "
            "vacuous and every arm would report 'unchanged' having compared nothing. "
            "Fix frozen_tensors() before trusting any result from this run.")
    print(f"frozen tensors under check: {len(frozen_before)} ({nbytes/2**30:.2f} GiB)", flush=True)

    ARMS = [
        ("reference", None, None),
        ("fast_train", lambda m: enable_fast_train(m), disable_fast_train),
        ("fast_train_dgrad", lambda m: enable_fast_train(m, dgrad=True), disable_fast_train),
        ("batched", enable_batched_train, disable_batched_train),
    ]
    results, grads = {}, {}
    for name, en, dis in ARMS:
        restore(model, base_snap)            # every arm starts bit-identical
        print(f"--- arm {name} ---", flush=True)
        try:
            res, g = run_arm(model, name, batches, a.lr, dev, en, dis)
        except Exception as exc:            # an arm that cannot run is data, not a crash
            print(f"arm {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[name] = dict(arm=name, failed=f"{type(exc).__name__}: {exc}")
            continue
        after, _ = frozen_hashes(model)
        changed = [k for k, v in frozen_before.items() if after.get(k) != v]
        res["frozen_changed"] = len(changed)
        res["frozen_changed_sample"] = changed[:5]
        results[name], grads[name] = res, g
        print(json.dumps(res)[:300], flush=True)

    # Non-vacuous control: the hash check must be able to SEE a single flipped byte.
    ctl = False
    for _n, t in frozen_tensors(model):
        before = _sha(t)
        with torch.no_grad():
            t.view(-1)[0] ^= 1
        ctl = _sha(t) != before
        with torch.no_grad():
            t.view(-1)[0] ^= 1
        break

    ref = results.get("reference", {})
    ref_g = grads.get("reference", {})
    for name in list(results):
        if name == "reference" or "failed" in results[name]:
            continue
        g = grads.get(name, {})
        per = {k: rel(g[k], ref_g[k]) for k in ref_g if k in g}
        results[name]["grad_rel_worst"] = max(per.values()) if per else None
        results[name]["grad_rel_mean"] = (sum(per.values()) / len(per)) if per else None
        results[name]["grad_tensors_compared"] = len(per)
        results[name]["grad_rel_worst_tensor"] = max(per, key=per.get) if per else None
        if ref.get("losses") and results[name].get("losses"):
            dl = sorted(abs(x - y) for x, y in zip(results[name]["losses"], ref["losses"]))
            results[name]["loss_median_abs_delta"] = dl[len(dl) // 2]
            results[name]["loss_max_abs_delta"] = dl[-1]

    payload = dict(model=a.model, steps=a.steps, seq=a.seq, layers=layers,
                   trainable_params=n_train, offload=bool(a.offload), env=env,
                   frozen_tensors_hashed=len(frozen_before), frozen_bytes_hashed=nbytes,
                   control_detects_flipped_byte=bool(ctl), arms=results)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1)
    print("\n=== SUMMARY ===")
    print(f"{'arm':<20}{'patched':>8}{'s/step':>9}{'peakGB':>8}"
          f"{'grad worst':>12}{'loss medΔ':>11}{'frozen?':>9}")
    for k, v in results.items():
        if "failed" in v:
            print(f"{k:<20}  FAILED: {v['failed'][:60]}")
            continue
        gw = v.get("grad_rel_worst")
        lm = v.get("loss_median_abs_delta")
        print(f"{k:<20}{v['n_patched']:>8}{v['s_per_step']:>9.3f}{v['train_peak_gb']:>8.2f}"
              f"{('—' if gw is None else f'{gw:.2e}'):>12}"
              f"{('—' if lm is None else f'{lm:.4f}'):>11}"
              f"{('OK' if v['frozen_changed'] == 0 else 'MUTATED'):>9}")
    print(f"\ncontrol_detects_flipped_byte={ctl}  (False => the frozen check is vacuous)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
