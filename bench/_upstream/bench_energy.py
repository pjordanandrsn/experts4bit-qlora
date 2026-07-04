"""Measure actual GPU energy (joules/op) of the three expert-projection paths, to answer:
'on a GPU that already fits the model, does 4-bit buy any energy improvement?'

Paths compared on one gate_up projection [out=2*inter, in=hidden]:
  native  : F.linear(x, w_bf16)                      # unquantized — the 'model already fits' baseline
  before  : dequantize_4bit -> F.linear              # pre-97fa09f (materialize bf16 weight in HBM)
  after   : bnb.matmul_4bit                          # post-97fa09f (fused dequant-in-GEMM)

Energy: sample nvidia-smi power.draw in a background thread; run a tight op-loop for a fixed
window; energy/op = mean_power / throughput. Reports total and idle-subtracted (dynamic) J/op.
Run on an IDLE GPU.
"""

import subprocess
import threading
import time

import bitsandbytes as bnb
import torch
import torch.nn.functional as F_nn
from bitsandbytes.functional import QuantState

from experts4bit_qlora import Experts4bit

HIDDEN, INTER, N_EXP, EXPERT = 2048, 1024, 8, 0
DTYPE, DEV = torch.bfloat16, "cuda"
DUR = 6.0  # timed seconds per phase


def _matmul4bit_proj(m, x):
    """The 97fa09f 'after' path, reconstructed locally (the packaged primitive retired
    `_expert_matmul` in v0.2.0 — training now uses recompute-in-backward instead)."""
    quant_state = QuantState(
        absmax=m.gate_up_absmax[EXPERT],
        shape=torch.Size(m._gate_up_shape),
        code=m.code,
        blocksize=m.blocksize,
        quant_type=m.quant_type,
        dtype=DTYPE,
    )
    return bnb.matmul_4bit(x, m.gate_up_proj[EXPERT].reshape(-1, 1), quant_state=quant_state)


def build():
    torch.manual_seed(0)
    gate_up = torch.randn(N_EXP, 2 * INTER, HIDDEN, dtype=DTYPE)
    down = torch.randn(N_EXP, HIDDEN, INTER, dtype=DTYPE)
    m = Experts4bit.from_float(
        gate_up, down, has_gate=True, activation=torch.nn.SiLU(), quant_type="nf4", compute_dtype=DTYPE
    ).to(DEV)
    w_bf16 = torch.randn(2 * INTER, HIDDEN, dtype=DTYPE, device=DEV)  # native unquantized weight
    return m, w_bf16


class PowerSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []  # (perf_counter, watts)
        self._stop = False
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def run(self):
        for line in self.proc.stdout:
            if self._stop:
                break
            try:
                self.samples.append((time.perf_counter(), float(line.strip())))
            except ValueError:
                pass

    def mean_between(self, t0, t1):
        vals = [w for (t, w) in self.samples if t0 <= t <= t1]
        return sum(vals) / len(vals) if vals else float("nan")

    def stop(self):
        self._stop = True
        self.proc.terminate()


def run_phase(fn, x, backward, sampler, dur=DUR):
    tw = time.perf_counter()
    while time.perf_counter() - tw < 1.0:  # warmup to steady-state clocks
        if x.grad is not None:
            x.grad = None
        out = fn(x)
        if backward:
            out.float().sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < dur:
        if x.grad is not None:
            x.grad = None
        out = fn(x)
        if backward:
            out.float().sum().backward()
        n += 1
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    p = sampler.mean_between(t0 + 0.4, t1)  # drop first 0.4s (clock ramp)
    thr = n / (t1 - t0)
    return {"ops_s": thr, "watts": p, "j_op": p / thr}


def main():
    print(f"device: {torch.cuda.get_device_name(0)} | bf16 | gate_up [{2 * INTER},{HIDDEN}]")
    m, w_bf16 = build()

    def native(x):
        return F_nn.linear(x, w_bf16)

    def before(x):
        return F_nn.linear(x, m._dequantize_expert(m.gate_up_proj, m.gate_up_absmax, m._gate_up_shape, EXPERT, DTYPE))

    def after(x):
        return _matmul4bit_proj(m, x)

    paths = [("native-bf16", native), ("before(dequant)", before), ("after(matmul_4bit)", after)]
    workloads = [
        ("fwd  M=1   (decode)", 1, False),
        ("fwd  M=512 (prefill)", 512, False),
        ("fwd+bwd M=32 (train)", 32, True),
    ]

    sampler = PowerSampler()
    sampler.start()
    time.sleep(3.0)  # idle baseline window
    idle = sampler.mean_between(time.perf_counter() - 2.5, time.perf_counter())
    print(f"idle power: {idle:.1f} W\n")

    for wl_name, M, bwd in workloads:
        print(f"--- {wl_name} ---")
        print(
            f"{'path':>20} | {'ops/s':>9} | {'power W':>8} | {'J/op (tot)':>11} | {'J/op (dyn)':>11} | {'vs native':>9}"
        )
        base = None
        for name, fn in paths:
            x = torch.randn(M, HIDDEN, dtype=DTYPE, device=DEV, requires_grad=bwd)
            try:
                r = run_phase(fn, x, bwd, sampler)
            except Exception as e:
                print(f"{name:>20} | ERROR: {type(e).__name__}: {str(e)[:40]}")
                continue
            dyn = (r["watts"] - idle) / r["ops_s"]
            if base is None:
                base = r["j_op"]
            ratio = r["j_op"] / base
            print(
                f"{name:>20} | {r['ops_s']:>9.0f} | {r['watts']:>8.1f} | {r['j_op'] * 1e6:>8.2f} uJ | "
                f"{dyn * 1e6:>8.2f} uJ | {ratio:>8.2f}x"
            )
        print()

    sampler.stop()


if __name__ == "__main__":
    main()
