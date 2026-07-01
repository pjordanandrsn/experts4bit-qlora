"""Closing measurement: the case EXCLUDED by 'a GPU that already runs the model'.

Quantization's energy benefit is systemic — it appears when 4-bit either (A) lets the model run on
hardware bf16 can't, or (B) frees memory to raise batch/utilization (better tokens-per-joule). Both
are measured here on the idle A2000:

  Part A — memory wall: try to allocate the full-model expert footprint in bf16 vs 4-bit on 12 GB.
  Part B — tokens-per-joule of the fused 4-bit MoE forward as batch grows (the utilization win).
"""

import subprocess
import threading
import time

import torch

from experts4bit_qlora import Experts4bit

HIDDEN, INTER, N_EXP = 2048, 1024, 8
DTYPE, DEV = torch.bfloat16, "cuda"
# Real OLMoE-1B-7B totals, for the memory wall.
FULL_LAYERS, FULL_EXPERTS = 16, 64
EXPERT_PARAMS = FULL_LAYERS * FULL_EXPERTS * (2 * INTER * HIDDEN + HIDDEN * INTER)  # 6.44B
NONEXPERT_GB = 1.0  # attn/embed/router/norms in bf16, ~0.5B params


class PowerSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def run(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.perf_counter(), float(line.strip())))
            except ValueError:
                pass

    def mean_between(self, t0, t1):
        vals = [w for (t, w) in self.samples if t0 <= t <= t1]
        return sum(vals) / len(vals) if vals else float("nan")

    def stop(self):
        self.proc.terminate()


def memory_wall():
    print("=== Part A: memory wall on a 12 GB card (full OLMoE-1B-7B) ===")
    bf16_gb = EXPERT_PARAMS * 2 / 1e9 + NONEXPERT_GB
    q4_gb = EXPERT_PARAMS * 0.5 / 1e9 + EXPERT_PARAMS / 64 * 4 / 1e9 + NONEXPERT_GB  # 4bit + fp32 absmax + rest
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  card total: {total:.1f} GB | bf16 model ~{bf16_gb:.1f} GB | 4-bit model ~{q4_gb:.1f} GB")
    for label, gb in (
        ("bf16 experts (12.9 GB)", EXPERT_PARAMS * 2 / 1e9),
        ("4-bit experts (3.2 GB)", EXPERT_PARAMS * 0.5 / 1e9),
    ):
        try:
            t = torch.empty(int(gb * 1e9), dtype=torch.uint8, device=DEV)
            print(f"  allocate {label:26}: OK ({t.numel() / 1e9:.1f} GB resident)")
            del t
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  allocate {label:26}: OOM -> {str(e).splitlines()[0][:60]}")
    print(
        f"  => bf16 OLMoE ({bf16_gb:.0f} GB) does not fit a 12 GB card; 4-bit ({q4_gb:.1f} GB) does. "
        f"On this card bf16 energy/token is undefined (won't run).\n"
    )


def build_layer():
    torch.manual_seed(0)
    gate_up = torch.randn(N_EXP, 2 * INTER, HIDDEN, dtype=DTYPE)
    down = torch.randn(N_EXP, HIDDEN, INTER, dtype=DTYPE)
    return Experts4bit.from_float(
        gate_up, down, has_gate=True, activation=torch.nn.SiLU(), quant_type="nf4", compute_dtype=DTYPE
    ).to(DEV)


def utilization_curve(sampler, idle):
    print("=== Part B: tokens-per-joule of the fused 4-bit MoE forward vs batch (utilization) ===")
    print(f"  (idle {idle:.1f} W subtracted for dynamic; total shown too)")
    m = build_layer()
    print(
        f"{'batch (tok)':>12} | {'tok/s':>10} | {'power W':>8} | {'J/tok (tot)':>12} | {'J/tok (dyn)':>12} | {'vs batch=64':>11}"
    )
    base = None
    for n_tok in (64, 256, 1024, 4096):
        torch.manual_seed(1)
        idx = torch.stack([torch.randperm(N_EXP, device=DEV)[:8] for _ in range(n_tok)])
        wts = torch.rand(n_tok, 8, dtype=DTYPE, device=DEV)

        @torch.no_grad()
        def fwd():
            x = torch.randn(n_tok, HIDDEN, dtype=DTYPE, device=DEV)
            return m(x, idx, wts)

        tw = time.perf_counter()
        while time.perf_counter() - tw < 1.0:
            fwd()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        k = 0
        while time.perf_counter() - t0 < 5.0:
            fwd()
            k += 1
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        p = sampler.mean_between(t0 + 0.4, t1)
        tok_s = k * n_tok / (t1 - t0)
        j_tot = p / tok_s
        j_dyn = max(0.0, p - idle) / tok_s
        if base is None:
            base = j_tot
        print(
            f"{n_tok:>12} | {tok_s:>10.0f} | {p:>8.1f} | {j_tot * 1e6:>9.3f} uJ | {j_dyn * 1e6:>9.3f} uJ | {j_tot / base:>10.2f}x"
        )
    print()


def main():
    print(f"device: {torch.cuda.get_device_name(0)}\n")
    memory_wall()
    s = PowerSampler()
    s.start()
    time.sleep(3.0)
    idle = s.mean_between(time.perf_counter() - 2.5, time.perf_counter())
    utilization_curve(s, idle)
    s.stop()


if __name__ == "__main__":
    main()
