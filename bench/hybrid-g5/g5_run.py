# G5 gate runner — hybrid QLoRA training vs hybrid inference (Qwen3-30B-A3B).
# Stages: bake | ref | run.
#   ref: full-GPU reference training arm (standard bnb-resident load +
#        enable_fast_train(dgrad=True)) -> /root/out/g5_ref_losses.json
#   run: hybrid arms — inference decode, timed train steps (batch>=8,
#        HF gradient checkpointing ON), loss curve on the same fixed batch,
#        gate verdict -> /root/out/g5_report.json
# Needs /root/out/calib.json (gnf4 bench/calibrate.py --tag g5).
import argparse
import json
import os
import time
from pathlib import Path

ARENA = os.environ.get("G5_ARENA", "/root/q30.arena")
SNAP = os.environ.get("G5_SNAP", "/root/q30")
OUT = Path("/root/out")
OUT.mkdir(exist_ok=True)
PROFILE = "/root/out/route_profile.jsonl"
MANIFEST = "/root/out/placement_g5.json"
PROMPT = "The three tiers of a memory hierarchy, from fastest to slowest, are"
BATCH, SEQ, STEPS, OVERLAY_STEPS = 8, 128, 8, 20
LR, SEED = 1e-4, 1689


def _fixed_batch(tok, dev):
    import torch
    g = torch.Generator().manual_seed(SEED)
    vocab = tok.vocab_size
    ids = torch.randint(0, vocab, (BATCH, SEQ), generator=g).to(dev)
    return ids


def _adapters(model):
    return [p for n, p in model.named_parameters()
            if p.requires_grad and "lora" in n]


def _reinit_adapters(model):
    """Identical fresh init in BOTH arms, applied AFTER the warm step: the
    warm step runs a real optimizer update, and an arm entering the overlay
    with B != 0 skips the B=0 cold-start (dL/dA is identically zero while
    B is zero) and converges faster for reasons that have nothing to do
    with the engine under test. Same seed, same module order, same draws."""
    import torch
    from experts4bit_qlora.lora import ExpertsLoRA
    torch.manual_seed(SEED)
    for _, mod in model.named_modules():
        if isinstance(mod, ExpertsLoRA):
            torch.nn.init.normal_(mod.gate_up_lora_A, std=1.0 / mod.r)
            torch.nn.init.normal_(mod.down_lora_A, std=1.0 / mod.r)
            torch.nn.init.zeros_(mod.gate_up_lora_B)
            torch.nn.init.zeros_(mod.down_lora_B)
            mod._delegate_ok = None


def _train_loop(model, ids, steps, record):
    import torch
    opt = torch.optim.AdamW(_adapters(model), lr=LR)
    times, losses = [], []
    for s in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(float(out.loss))
        if record:
            print(f"STEP {s} loss={losses[-1]:.5f} t={times[-1]:.2f}s",
                  flush=True)
    return times, losses


def stage_bake():
    from nvme_bake_nf4 import bake_nf4
    bake_nf4(SNAP, ARENA)
    print("BAKE_DONE")


def stage_ref():
    """Full-GPU reference arm: standard resident 4-bit load, [fast] train
    lane with the GPU dgrad kernel — the directive's comparison path."""
    import torch
    from transformers import AutoTokenizer
    from experts4bit_qlora import enable_fast_train, load_moe_4bit_streaming

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(SNAP)
    model, _cfg = load_moe_4bit_streaming(
        SNAP, "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4")
    model.gradient_checkpointing_enable()
    model.train()
    enable_fast_train(model, dgrad=True)
    ids = _fixed_batch(tok, "cuda")
    _w = _train_loop(model, ids, 1, record=False)      # warm (JIT, caches)
    _reinit_adapters(model)
    times, losses = _train_loop(model, ids, OVERLAY_STEPS, record=True)
    (OUT / "g5_ref_losses.json").write_text(json.dumps(
        {"losses": losses, "step_times": times, "batch": BATCH, "seq": SEQ,
         "steps": OVERLAY_STEPS, "lr": LR, "seed": SEED}))
    print("REF_DONE", flush=True)


def stage_run(threads: int, dram_gb: int):
    import torch
    from transformers import AutoTokenizer
    from nvme_arena import load_index
    from experts4bit_qlora import (load_moe_4bit_streaming, save_manifest,
                                   solve_placement)
    from experts4bit_qlora.engines import expert_profile as ep
    from experts4bit_qlora.engines import hybrid as hy
    from experts4bit_qlora.engines import hybrid_train as ht
    from experts4bit_qlora.engines.hot_residency import target_modules

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(SNAP)
    model, _cfg = load_moe_4bit_streaming(
        SNAP, "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
        arena=ARENA, arena_train=True)
    model.eval()
    mods = target_modules(model)
    L, E = len(mods), mods[0].num_experts
    idx_a = load_index(ARENA)
    print(f"L={L} E={E} row_stride={idx_a['row_stride']}", flush=True)

    # ---- routing profile via an all-NVMe hybrid manifest (pure-streaming
    # semantics; enable_nvme_residency's wrapped-bases guard would refuse
    # the arena_train model, and routing does not depend on the engine)
    trivial = {"schema": "e4b-placement/1",
               "tiers": {"vram": [], "dram": [],
                         "nvme": [[layer, e] for layer in range(L)
                                  for e in range(E)]},
               "masses": {"vram_frac": 0, "dram_frac": 0, "nvme_frac": 1.0}}
    n = hy.enable_hybrid_tier(model, ARENA, trivial, hot_rows=max(E, 128),
                              threads=threads)
    assert n == L
    assert ep.enabled(), "E4B_EXPERT_PROFILE not set"
    ep.attach(model)
    ids0 = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        model.generate(ids0, max_new_tokens=32, do_sample=False)
    ep.flush()
    hy.disable_hybrid_tier(model)

    free_b, _ = torch.cuda.mem_get_info()
    vram_budget = max(0, free_b - 6 * (1 << 30))   # training headroom
    m = solve_placement(
        n_layers=L, n_experts=E, bytes_per_expert=idx_a["row_stride"],
        vram_budget_bytes=vram_budget,
        dram_budget_bytes=dram_gb * (1 << 30),
        calibration="/root/out/calib.json", profile_path=PROFILE)
    sha = save_manifest(m, MANIFEST)
    print(f"PLACEMENT sha={sha[:16]} masses={m['masses']}", flush=True)

    # ---- arm I: hybrid inference (serve path through the LoRA wrapper)
    n = hy.enable_hybrid_tier(model, ARENA, MANIFEST, hot_rows=max(E, 128),
                              threads=threads)
    assert n == L
    with torch.no_grad():
        model.generate(ids0, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(ids0, max_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
    infer_tok_s = 64 / (time.perf_counter() - t0)
    hy.disable_hybrid_tier(model)
    print(f"ARM_I_INFER tok/s={infer_tok_s:.3f}", flush=True)

    # ---- arm T: hybrid training (same manifest), timed + loss curve
    n = ht.enable_hybrid_train(model, ARENA, MANIFEST, hot_rows=max(E, 128),
                               threads=threads)
    assert n == L
    model.gradient_checkpointing_enable()
    model.train()
    ids = _fixed_batch(tok, "cuda")
    _train_loop(model, ids, 1, record=False)           # warm
    t_timed, _ = _train_loop(model, ids, STEPS, record=True)
    _reinit_adapters(model)         # both arms overlay from the SAME init
    _, losses = _train_loop(model, ids, OVERLAY_STEPS, record=True)
    ht.disable_hybrid_train(model)

    tokens_per_step = BATCH * SEQ
    train_per_tok = sum(t_timed) / len(t_timed) / tokens_per_step
    infer_per_tok = 1.0 / infer_tok_s
    ratio = train_per_tok / infer_per_tok

    ref_path = OUT / "g5_ref_losses.json"
    overlay = None
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        n_cmp = min(len(losses), len(ref["losses"]))
        deltas = [abs(losses[i] - ref["losses"][i]) for i in range(n_cmp)]
        overlay = {"ref_losses": ref["losses"], "hybrid_losses": losses,
                   "max_abs_delta": max(deltas),
                   "final_delta": deltas[-1], "steps_compared": n_cmp}

    rep = {"infer_tok_s": infer_tok_s,
           "train_step_s_mean": sum(t_timed) / len(t_timed),
           "train_step_s": t_timed, "tokens_per_step": tokens_per_step,
           "train_per_tok_s": train_per_tok, "infer_per_tok_s": infer_per_tok,
           "ratio_train_over_infer": ratio,
           "batch": BATCH, "seq": SEQ, "masses": m["masses"],
           "manifest_sha": sha, "threads": threads,
           "loss_overlay": overlay,
           "gate_g5": {"ratio_ok": ratio <= 3.0,
                       "batch_ok": BATCH >= 8}}
    (OUT / "g5_report.json").write_text(json.dumps(rep, indent=2))
    print("G5_REPORT " + json.dumps(rep), flush=True)


def stage_overlay(threads: int):
    """Loss overlay only: rerun the hybrid 20-step curve against the banked
    reference losses (both arms re-init identically post-warm) and patch
    the existing report. Reuses the solved manifest."""
    import torch
    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.engines import hybrid_train as ht
    from experts4bit_qlora.engines.hot_residency import target_modules

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(SNAP)
    model, _cfg = load_moe_4bit_streaming(
        SNAP, "cuda", torch.bfloat16, r=8, alpha=16, quant_type="nf4",
        arena=ARENA, arena_train=True)
    mods = target_modules(model)
    n = ht.enable_hybrid_train(model, ARENA, MANIFEST,
                               hot_rows=max(mods[0].num_experts, 128),
                               threads=threads)
    assert n == len(mods)
    model.gradient_checkpointing_enable()
    model.train()
    ids = _fixed_batch(tok, "cuda")
    _train_loop(model, ids, 1, record=False)
    _reinit_adapters(model)
    _, losses = _train_loop(model, ids, OVERLAY_STEPS, record=True)
    ht.disable_hybrid_train(model)

    rep = json.loads((OUT / "g5_report.json").read_text())
    ref = json.loads((OUT / "g5_ref_losses.json").read_text())
    n_cmp = min(len(losses), len(ref["losses"]))
    deltas = [abs(losses[i] - ref["losses"][i]) for i in range(n_cmp)]
    rep["loss_overlay"] = {"ref_losses": ref["losses"],
                           "hybrid_losses": losses,
                           "max_abs_delta": max(deltas),
                           "final_delta": deltas[-1],
                           "steps_compared": n_cmp}
    (OUT / "g5_report.json").write_text(json.dumps(rep, indent=2))
    print("G5_OVERLAY " + json.dumps(rep["loss_overlay"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bake", "ref", "run", "overlay"])
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--dram-gb", type=int, default=100)
    a = ap.parse_args()
    os.environ.setdefault("E4B_EXPERT_PROFILE", PROFILE)
    {"bake": stage_bake, "ref": stage_ref,
     "run": lambda: stage_run(a.threads, a.dram_gb),
     "overlay": lambda: stage_overlay(a.threads)}[a.stage]()
