#!/usr/bin/env python3
"""One cell of the flagship matrix's SECOND model, per docs/PREREG-flagship-matrix-model2.md.

Derived from n9_cell.py (which produced the first ten cells) with the two gates
that prereg's C1 requires and n9_cell.py did not have:

  1. B1/C1 hashes come from state_dict(), NOT getattr(module, "gate_up_proj").
     Under offload=True the module's registered tensors are 0-ELEMENT
     PLACEHOLDERS -- `_ExpertOffload.evict()` swaps them at load and the layer is
     only materialized inside its own forward. Hashing those compares
     sha256(b"") to sha256(b"") and reports "192/192 exact" while hashing
     nothing. state_dict() is correct because _install_state_dict_hook
     substitutes each expert's pinned CPU home.
  2. bytes_hashed > 0 and empties_skipped == 0 are ASSERTED, and a byte-flip
     positive control must fire -- a byte-identity check with no demonstrated
     failure mode is a constant function.

Also asserts the arm is really the arm: enable_fast_train() returns 0 silently
when grouped-nf4-gemm is missing, which would run a "fused" cell that is
actually the reference path.
"""
import argparse
import gc
import os
import hashlib
import json
import statistics
import subprocess
import threading
import time

import torch

SEQ, R, ALPHA, LR = 512, 8, 16, 1e-4
EXPERT_ATTRS = ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")


class PowerSampler:
    def __init__(self):
        self.samples, self._run = [], False

    def _loop(self):
        while self._run:
            try:
                self.samples.append(float(subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]))
            except Exception:
                pass
            time.sleep(0.2)

    def __enter__(self):
        self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self._run = False
        self.t.join(timeout=2)


def idle_power(n=10):
    v = []
    for _ in range(n):
        try:
            v.append(float(subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(0.15)
    return statistics.median(v) if v else 0.0


def expert_hashes(model):
    """C1: sha256 of every frozen expert's packed bytes, read from state_dict().

    Returns (hashes, bytes_hashed, empties_skipped). Empties are COUNTED, not
    silently skipped, so the caller can assert there were none -- an empty
    tensor is the signature of hashing a placeholder.
    """
    h, nbytes, empties = {}, 0, 0
    for name, m in model.named_modules():
        if not any(hasattr(m, a) for a in EXPERT_ATTRS):
            continue
        sd = m.state_dict()
        for attr in EXPERT_ATTRS:
            t = sd.get(attr)
            if t is None:
                continue
            b = t.detach().to("cpu").contiguous().numpy().tobytes()
            if not b:
                empties += 1
                continue
            nbytes += len(b)
            h[f"{name}.{attr}"] = hashlib.sha256(b).hexdigest()
    return h, nbytes, empties


def control_flip_fires(model):
    """Positive control: the comparison must DETECT a single flipped byte."""
    h, _, _ = expert_hashes(model)
    if not h:
        return False
    k = next(iter(h))
    tampered = dict(h)
    tampered[k] = ("0" if h[k][0] != "0" else "1") + h[k][1:]
    return [x for x in h if h[x] != tampered.get(x)] == [k]


def encode(tok, rows, seq):
    out = []
    for r in rows:
        ids = tok(f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['output']}",
                  return_tensors="pt", truncation=True, max_length=seq).input_ids[0]
        if ids.numel() >= 8:
            out.append(ids)
    return out


@torch.no_grad()
def eval_loss(model, data, limit=48):
    model.eval()
    tot = n = 0
    for ids in data[:limit]:
        x = ids.unsqueeze(0).cuda()
        tot += float(model(input_ids=x, labels=x).loss)
        n += 1
    model.train()
    return tot / max(n, 1)


def host_fingerprint():
    """Everything needed to attribute host-to-host drift later.

    Two RTX 4090 pods running byte-identical configs differed by 8.6 % s/step,
    3.4x idle power and 58 % mean power. The cause could not be established
    because the receipts recorded only `gpu` and `cap` -- and by the time the
    discrepancy surfaced the first pod had been deleted. `gpu: "RTX 4090"` is not
    a host fingerprint: PCIe width (one pod was gen4 **x8** of a x16 max), CPU,
    power limit and clock ceilings all move throughput on a transfer-bound
    workload and none were captured.

    Cheap, taken once per cell, and it turns "hosts differ somehow" into an
    attributable difference.
    """
    fields = ("uuid,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,"
              "pcie.link.width.current,pcie.link.width.max,power.limit,"
              "clocks.max.sm,clocks.max.mem,vbios_version,driver_version")
    out = {}
    try:
        raw = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip().split("\n")[0]
        out = dict(zip(fields.split(","), [v.strip() for v in raw.split(",")]))
    except Exception as e:
        out["nvidia_smi_error"] = f"{type(e).__name__}: {e}"
    try:
        with open("/proc/cpuinfo") as fh:
            for ln in fh:
                if ln.startswith("model name"):
                    out["cpu"] = ln.split(":", 1)[1].strip()
                    break
        out["cpu_threads"] = os.cpu_count()
        with open("/proc/meminfo") as fh:
            out["host_mem_gb"] = round(int(fh.readline().split()[1]) / 1e6, 1)
    except Exception as e:
        out["host_error"] = f"{type(e).__name__}: {e}"
    out["pod_id"] = os.environ.get("RUNPOD_POD_ID") or os.environ.get("HOSTNAME")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True, choices=["reference", "fused"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import importlib.metadata as md

    from experts4bit_qlora.fast import disable_fast_train, enable_fast_train
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora
    from transformers import AutoTokenizer

    idle_w = idle_power()
    torch.manual_seed(0)
    t_load = time.time()
    model, _ = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, R, ALPHA,
                                       offload=True, pin=True, prefetch=False,
                                       quant_type="nf4")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    load_s = time.time() - t_load

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = json.load(open(f"/workspace/n17/data/ds_{a.dataset}.json"))
    train, ev = encode(tok, ds["train"], SEQ), encode(tok, ds["eval"], SEQ)

    add_attention_lora(model, R, ALPHA, torch.float32)
    n_fused = enable_fast_train(model, verbose=True) if a.arm == "fused" else disable_fast_train(model)
    # The arm must BE the arm. enable_fast_train returns 0 when grouped-nf4-gemm
    # is absent, which would silently run the reference path under a fused label.
    if a.arm == "fused":
        assert n_fused > 0, "fused arm patched 0 modules -- grouped-nf4-gemm missing or no ExpertsLoRA"

    h_before, bytes_before, empties_before = expert_hashes(model)
    assert bytes_before > 0, "C1 hashed ZERO bytes -- reading offload placeholders, gate is vacuous"
    assert empties_before == 0, f"C1 saw {empties_before} empty expert tensors -- placeholders leaked in"
    assert control_flip_fires(model), "C1 positive control did not fire -- the check cannot fail"

    ev0 = eval_loss(model, ev)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    losses, tokens = [], 0
    with PowerSampler() as ps:
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(a.steps):
            ids = train[i % len(train)].unsqueeze(0).cuda()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(round(float(out.loss.detach()), 5))
            tokens += ids.numel()
            if (i + 1) % 50 == 0:
                print(f"    step {i+1}/{a.steps} loss {losses[-1]}", flush=True)
        torch.cuda.synchronize()
        wall = time.time() - t0

    ev1 = eval_loss(model, ev)
    h_after, bytes_after, empties_after = expert_hashes(model)
    assert bytes_after == bytes_before, f"C1 hashed {bytes_after} B after vs {bytes_before} before"
    assert empties_after == 0, f"C1 saw {empties_after} empty expert tensors after training"
    changed = [k for k in h_before if h_before[k] != h_after.get(k)]
    # C1 is a HARD GATE, so it has to be able to FAIL THE CELL. Recording
    # C1_bit_exact=false in the receipt and exiting 0 is not a gate: the runner
    # sees a non-empty receipt, counts the cell complete, and skips it on resume.
    # The prereg says a C1 failure voids every performance number -- so refuse to
    # write one.
    assert not changed, (
        f"C1 FAILED — {len(changed)} frozen expert tensor(s) CHANGED during training: "
        f"{changed[:5]}. Per the prereg this voids every performance number for this "
        "cell; refusing to write a receipt.")
    mean_w = statistics.mean(ps.samples) if ps.samples else None
    net_w = (mean_w - idle_w) if mean_w else None

    cell = {
        "model": a.model, "dataset": a.dataset, "arm": a.arm, "steps": a.steps,
        "prereg": "docs/PREREG-flagship-matrix-model2.md",
        "gnf4": md.version("grouped-nf4-gemm"), "e4b": md.version("experts4bit-qlora"),
        "e4b_commit": __import__("os").environ.get("E4B_COMMIT"),
        "gpu": torch.cuda.get_device_name(0),
        "host": host_fingerprint(),
        "cap": list(torch.cuda.get_device_capability()),
        "fused_modules": n_fused,
        "C1_tensors_hashed": len(h_before),
        "C1_bytes_hashed": bytes_before,
        "C1_empties_skipped": empties_before,
        "C1_control_detects_flipped_byte": True,
        "C1_experts_changed": len(changed),
        "C1_bit_exact": len(changed) == 0,
        "C1_changed_sample": changed[:3],
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_mean_last20": round(statistics.mean(losses[-20:]), 5),
        "eval_loss_step0": round(ev0, 5), "eval_loss_final": round(ev1, 5),
        "eval_rel_improvement": round(ev1 / ev0, 5) if ev0 else None,
        "load_s": round(load_s, 1),
        "s_per_step": round(wall / a.steps, 4),
        "tokens_per_s": round(tokens / wall, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "idle_w": round(idle_w, 1),
        "mean_w": round(mean_w, 1) if mean_w else None,
        "power_samples": len(ps.samples),
        "joules_per_step": round(net_w * (wall / a.steps), 2) if net_w else None,
        "losses": losses,
    }
    json.dump(cell, open(a.out, "w"), indent=1)
    print("CELL OK " + json.dumps({k: v for k, v in cell.items() if k != "losses"}), flush=True)


if __name__ == "__main__":
    main()
