"""Lane FP1, one arm: build a QLoRASetup with prepare_qlora_training, train a few fixed-shape steps, and record what the
process actually held next to what estimate_qlora_footprint priced for the same setup (bench/fp1/FP1-PREREG.md).

Device: the allocator's allocated and reserved peaks over the training steps, and the driver's view of this process
(nvidia-smi --query-compute-apps, sampled on a thread). Host: the peak of RssAnon + RssShmem (pinned memory is accounted
as RssShmem; memory-mapped checkpoint shards are RssFile and are recorded apart). Link: a pinned host-to-device copy,
timed before the load. Correctness: losses, sha256 of the first and last frozen expert stacks before and after (the host
homes under offload), LoRA-B norms.

    python fp1_measure.py --model Qwen/Qwen3-30B-A3B --revision <sha> --residency device --out receipts/q30_device.json
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import threading
import time

PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"


def proc_status():
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"):
                out[k] = int(v.split()[0]) * 1024
    return out


class Sampler:
    """Driver-reported device memory of this process and host RSS kinds, every 0.25 s."""

    def __init__(self):
        self.smi = shutil.which("nvidia-smi")
        self.drv = self.anon_shmem = self.file = self.n = 0
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        pid = str(os.getpid())
        while not self.stop.is_set():
            try:
                out = subprocess.run([self.smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout if self.smi else ""
                for line in out.splitlines():
                    p, _, mib = line.partition(",")
                    if p.strip() == pid and mib.strip().isdigit():
                        self.drv, self.n = max(self.drv, int(mib) << 20), self.n + 1
            except (subprocess.SubprocessError, OSError):
                pass
            st = proc_status()
            self.anon_shmem = max(self.anon_shmem, st.get("RssAnon", 0) + st.get("RssShmem", 0))
            self.file = max(self.file, st.get("RssFile", 0))
            self.stop.wait(0.25)


def link_h2d(torch, n=256 << 20, reps=5):
    src = torch.empty(n, dtype=torch.uint8).pin_memory()
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    rates = []
    for _ in range(reps):
        t = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        rates.append(n / (time.perf_counter() - t) / 1e9)
    del src, dst
    torch.cuda.empty_cache()
    return statistics.median(rates)


def expert_digest(model, torch):
    from experts4bit_qlora import ExpertsNbit, offload_handles

    bases = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
    handles = list(offload_handles(model))
    out = {}
    for li in (0, -1):
        b = bases[li]
        ts = {n: getattr(b, n) for n in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")}
        if ts["gate_up_proj"].numel() == 0 and handles:
            ts = dict(handles[li].home)
        h = hashlib.sha256()
        for n in sorted(ts):
            if ts[n] is not None:
                h.update(ts[n].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        out[f"stack[{li}]"] = h.hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--residency", choices=("device", "host"), required=True)
    ap.add_argument("--kernel", choices=("grouped_nf4", "reference"), default="grouped_nf4")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from experts4bit_qlora import QLoRASetup, describe_moe, estimate_qlora_footprint, prepare_qlora_training

    rec = {"schema": "fp1-arm/1", "args": vars(a), "status": "STARTED"}
    setup = QLoRASetup(expert_residency=a.residency, expert_kernel=a.kernel)
    topo = describe_moe(a.model, revision=a.revision)
    fp = estimate_qlora_footprint(topo, setup, tokens_per_microbatch=a.seq * a.mb, optimizer="adamw")
    rec.update(setup=setup.to_dict(), topology=topo.summary(),
               estimate={**fp.to_dict(), "derived_device": fp.total("device", "derived"),
                         "heuristic_device": fp.total("device", "heuristic")},
               device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
               torch=torch.__version__)
    torch.manual_seed(a.seed)
    torch.zeros(1, device="cuda")
    m = {"host_baseline_bytes": proc_status().get("VmRSS"), "link_h2d_gbps": link_h2d(torch)}
    smp = Sampler()
    smp.t.start()
    try:
        t0 = time.time()
        prep = prepare_qlora_training(a.model, setup, device="cuda", revision=a.revision)
        m["load_seconds"] = time.time() - t0
        m["load_device_peak_bytes"] = torch.cuda.max_memory_allocated()
        m["host_anon_after_load_bytes"] = proc_status().get("RssAnon")
        model = prep.model
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        need, ids, i = a.steps * a.mb * a.seq, [], 0
        while len(ids) < need:
            ids += tok(PROMPT.format(**ds[i]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            i += 1
        blocks = [ids[k * a.seq:(k + 1) * a.seq] for k in range(a.steps * a.mb)]
        before = expert_digest(model, torch)
        bnorm = lambda: sum(float(p.detach().float().norm()) for n, p in model.named_parameters()  # noqa: E731
                            if p.requires_grad and "lora_B" in n)
        b0 = bnorm()
        opt = torch.optim.AdamW(prep.trainable, lr=2e-4)
        model.train()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        losses, step_s = [], []
        for s in range(a.steps):
            torch.cuda.synchronize()
            ts = time.time()
            opt.zero_grad(set_to_none=True)
            x = torch.tensor(blocks[s * a.mb:(s + 1) * a.mb], device="cuda")
            loss = model(input_ids=x, labels=x).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prep.trainable, 1.0)
            opt.step()
            torch.cuda.synchronize()
            step_s.append(time.time() - ts)
            losses.append(float(loss.detach()))
            print(f"step {s + 1}/{a.steps} loss {losses[-1]:.4f} {step_s[-1]:.2f}s "
                  f"alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        m.update(device_peak_bytes=torch.cuda.max_memory_allocated(),
                 device_reserved_peak_bytes=torch.cuda.max_memory_reserved())
        time.sleep(1.0)
        after = expert_digest(model, torch)
        timed = step_s[a.warmup:] or step_s
        m.update(step_seconds=step_s, s_per_step_median=statistics.median(timed),
                 tokens_per_s=a.seq * a.mb / statistics.median(timed))
        rec["correctness"] = {"losses": losses, "all_finite": all(math.isfinite(v) for v in losses),
                              "frozen_expert_bytes_unchanged": before == after, "frozen_expert_digest": after,
                              "adapter_B_norm_before": b0, "adapter_B_norm_after": bnorm()}
        rec["engaged"] = prep.report
        ok = rec["correctness"]["all_finite"] and rec["correctness"]["frozen_expert_bytes_unchanged"] \
            and rec["correctness"]["adapter_B_norm_after"] > b0
        rec["status"] = "OK" if ok else "ALARM"
    except torch.cuda.OutOfMemoryError as e:
        rec["status"], rec["error"] = "OOM", str(e)[:400]
        m["device_peak_bytes"] = torch.cuda.max_memory_allocated()
        m["device_reserved_peak_bytes"] = torch.cuda.max_memory_reserved()
    finally:
        smp.stop.set()
        smp.t.join(timeout=15)
    m.update(driver_process_peak_bytes=smp.drv or None, driver_samples=smp.n,
             host_required_peak_bytes=smp.anon_shmem or None, host_file_peak_bytes=smp.file or None,
             host_peak_bytes=proc_status().get("VmHWM"))
    if smp.drv and m.get("device_reserved_peak_bytes"):
        m["cuda_context_bytes"] = smp.drv - m["device_reserved_peak_bytes"]
    if m.get("device_peak_bytes") and m.get("device_reserved_peak_bytes"):
        m["reserve_slack_fraction"] = m["device_reserved_peak_bytes"] / m["device_peak_bytes"] - 1
    rec["measured"] = m
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(json.dumps({"status": rec["status"], "alloc_gib": (m.get("device_peak_bytes") or 0) / 2**30,
                      "est_alloc_gib": fp.device_bytes / 2**30, "slack": m.get("reserve_slack_fraction")}))
    return 0 if rec["status"] in ("OK", "OOM") else 1


if __name__ == "__main__":
    raise SystemExit(main())
