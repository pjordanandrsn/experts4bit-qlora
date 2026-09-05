#!/usr/bin/env python3
"""tp1_train_smoke.py -- lane tp1 (training parity) per-arm driver.

Derived from bench/flagship-matrix/drivers/n17_cell.py at e4b main f4b639f (the
corrected flagship driver: state_dict() hashes, bytes > 0, empties == 0, byte-flip
control, "the arm must be the arm"). The measurement code -- encode(), eval_loss(),
expert_hashes(), control_flip_fires(), PowerSampler, idle_power(), host_fingerprint()
-- is n17's, unchanged. Deltas, named so the two files can be diffed:

  D1  --arm gains `batched` (enable_batched_train) and `attn_only` (the reference path
      with no expert adapter present -- the gpt-oss case). `fused` is
      enable_fast_train(dgrad=True), the documented default; n17 called it without dgrad.
  D2  --offload 0|1 (n17 hardcoded offload=True); --data / --data-sha (n17 hardcoded a
      path and verified the sha in its setup script instead).
  D3  verify_moe_4bit(strict=True) after the load; an expert-module census (class names,
      wrapped, bare), model_type and the loader's MoE-layer count land in the receipt;
      `<fam>_train_load.json` is written before any step so a load that succeeds and an
      arm that later dies are separable rows.
  D4  init_sha: sha256 over every trainable tensor at step 0, so the reducer can PROVE two
      arms started bit-identical (seed 0 + identical construction order) rather than
      assume it.
  D5  kernel-call counters: nf4_qlora.fused_grouped_lora (fused) and
      experts4bit_qlora.engines.batched._dequant_whole (batched) are wrapped and counted
      per step. n11 counted the kernel for the same reason -- a patch count is not a call
      count -- and batched.py falls back to the reference forward per call above
      _PAD_WASTE_LIMIT with no counter of its own.
  D6  per-step step_ms and token counts (n17 kept only the means).
  D7  refusal / OOM / load-fault / verify-failure are RECEIPTS with an exit code
      (3 / 5 / 6 / 7), never a crash without a JSON: a refusal is a row.
  D8  attn_only additionally PROBES enable_fast_train / enable_batched_train (count, then
      disable) and writes `fused` / `batched` refusal stubs, so the gpt-oss rows exist
      without a second and third 20B load. If a probe unexpectedly patches > 0 modules the
      count is recorded and the arm still trains attention-only; the reducer flags it.

Exit codes: 0 ok; 3 refused; 4 C1 failed (receipt written, arm void); 5 OOM; 6 load
fault (CUDA error at load, e.g. e4b#344); 7 verify_moe_4bit(strict) raised.
"""
import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time

import torch

EXPERT_ATTRS = ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")


# ----------------------------------------------------------------------------- n17 code
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
    Returns (hashes, bytes_hashed, empties_skipped)."""
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
    fields = ("uuid,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,"
              "pcie.link.width.current,pcie.link.width.max,power.limit,"
              "clocks.max.sm,clocks.max.mem,vbios_version,driver_version")
    want = fields.split(",")
    out = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        raw = r.stdout.strip().split("\n")[0] if r.stdout.strip() else ""
        cols = [v.strip() for v in raw.split(",")] if raw else []
        if r.returncode != 0:
            out["nvidia_smi_error"] = f"exit {r.returncode}: {r.stderr.strip()[:200]}"
        elif len(cols) != len(want):
            out["nvidia_smi_error"] = f"expected {len(want)} columns, got {len(cols)}: {raw[:200]}"
        else:
            out = dict(zip(want, cols))
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
            kib = int(fh.readline().split()[1])
            out["host_mem_gib"] = round(kib / 1048576, 1)
    except Exception as e:
        out["host_error"] = f"{type(e).__name__}: {e}"
    out["vast_instance_id"] = os.environ.get("TP1_INSTANCE_ID")
    out["container_host"] = socket.gethostname()
    return out


# ----------------------------------------------------------------------------- tp1 deltas
def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def trainable_sha(model):
    """D4: one sha over every trainable tensor, in named_parameters order."""
    h = hashlib.sha256()
    n = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
            n += 1
    return h.hexdigest(), n


def expert_census(model):
    """D3: what the loader installed for the experts."""
    from experts4bit_qlora import Experts4bit, ExpertsNbit
    from experts4bit_qlora.lora import ExpertsLoRA
    wrapped, wrapped_ids, classes = 0, set(), {}
    for _, m in model.named_modules():
        if isinstance(m, ExpertsLoRA):
            wrapped += 1
            wrapped_ids.add(id(m.base))
            classes[type(m.base).__name__] = classes.get(type(m.base).__name__, 0) + 1
    bare = 0
    for _, m in model.named_modules():
        if isinstance(m, (Experts4bit, ExpertsNbit)) and id(m) not in wrapped_ids:
            bare += 1
            classes[type(m).__name__ + " (bare)"] = classes.get(type(m).__name__ + " (bare)", 0) + 1
    return {"n_lora_wrapped": wrapped, "n_bare_experts": bare, "expert_classes": classes}


class KernelCounter:
    """D5: count the kernel-side entry points the accelerated arms must reach."""
    def __init__(self):
        self.counts = {"fused_grouped_lora": 0, "batched_dequant_whole": 0}
        self._restore = []

    def install(self):
        try:
            import nf4_qlora
            orig = nf4_qlora.fused_grouped_lora
            def w(*a, _orig=orig, **k):
                self.counts["fused_grouped_lora"] += 1
                return _orig(*a, **k)
            nf4_qlora.fused_grouped_lora = w
            self._restore.append((nf4_qlora, "fused_grouped_lora", orig))
        except ImportError:
            pass
        try:
            from experts4bit_qlora.engines import batched as B
            orig = B._dequant_whole
            def w2(*a, _orig=orig, **k):
                self.counts["batched_dequant_whole"] += 1
                return _orig(*a, **k)
            B._dequant_whole = w2
            self._restore.append((B, "_dequant_whole", orig))
        except ImportError:
            pass

    def snapshot(self):
        return dict(self.counts)

    def uninstall(self):
        for mod, name, orig in self._restore:
            setattr(mod, name, orig)


def write_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def stub(a, status, reason, extra=None, code=None):
    """D7: a refusal / fault is a receipt with a status and an exit code."""
    rec = {"fam": a.fam, "model": a.model, "arm": a.arm, "tag": a.tag, "status": status,
           "reason": str(reason)[:600], "steps": a.steps, "seq": a.seq,
           "offload": bool(a.offload), "n_patched": 0, "prereg": "tp1/P36-PREREG.md"}
    if extra:
        rec.update(extra)
    write_json(os.path.join(a.out, f"{a.fam}_train_{a.tag}.json"), rec)
    print(f"CELL {status.upper()} " + json.dumps({k: v for k, v in rec.items() if k != "losses"}), flush=True)
    if code is not None:
        sys.exit(code)


def capture_verbose(fn, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = fn(*args, **kw)
    return n, buf.getvalue().strip()[-600:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fam", required=True)
    ap.add_argument("--arm", required=True, choices=["reference", "fused", "batched", "attn_only"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--offload", type=int, default=0)
    ap.add_argument("--data", required=True, help="ds_<name>.json from n9_datasets.py")
    ap.add_argument("--data-sha", required=True, help="registered sha256 of --data")
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None,
                    help="JSON filename tag (<fam>_train_<tag>.json); defaults to --arm. The Mixtral resident "
                         "probe is `--arm reference --tag resident_probe` so it never overwrites the offload reference")
    ap.add_argument("--probe-refusals", type=int, default=1,
                    help="attn_only only: probe enable_fast_train/enable_batched_train and write stubs (D8)")
    a = ap.parse_args()
    a.tag = a.tag or a.arm
    os.makedirs(a.out, exist_ok=True)

    import importlib.metadata as md
    import experts4bit_qlora as e4b
    from experts4bit_qlora import (disable_batched_train, disable_fast_train,
                                   enable_batched_train, enable_fast_train,
                                   load_moe_4bit_streaming, verify_moe_4bit)
    from experts4bit_qlora.lora import add_attention_lora
    from transformers import AutoTokenizer
    import transformers

    # the fixture is the registered bytes or nothing
    got = sha_file(a.data)
    if got != a.data_sha:
        stub(a, "dataset_mismatch", f"{a.data}: sha {got[:12]} != registered {a.data_sha[:12]}", code=13)
    ds = json.load(open(a.data))

    env = {"e4b": e4b.__version__, "e4b_dist": md.version("experts4bit-qlora"),
           "gnf4": None,
           "torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0), "cap": list(torch.cuda.get_device_capability()),
           "host": host_fingerprint(),
           "anchor_json": os.environ.get("TP1_ANCHOR_JSON"),
           "box_class": os.environ.get("TP1_BOX_CLASS")}
    try:
        env["gnf4"] = md.version("grouped-nf4-gemm")
    except Exception:
        env["gnf4"] = None

    idle_w = idle_power()
    torch.manual_seed(a.seed)
    t_load = time.perf_counter()
    try:
        model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, a.r, a.alpha,
                                             offload=bool(a.offload), pin=True, prefetch=False,
                                             quant_type="nf4")
        if not a.offload:
            model.to("cuda")
    except torch.cuda.OutOfMemoryError as e:
        stub(a, "oom", f"OOM during load: {e}", {"phase": "load"}, code=5)
    except NotImplementedError as e:
        stub(a, "refused", f"loader refused: {e}", {"phase": "load"}, code=3)
    except RuntimeError as e:
        if "CUDA" in str(e):
            stub(a, "load_fault", f"{type(e).__name__}: {e}", {"phase": "load"}, code=6)
        raise
    load_s = time.perf_counter() - t_load
    model_type = getattr(cfg, "model_type", None)
    lm_cfg = getattr(cfg, "text_config", None) or cfg

    try:
        rep = verify_moe_4bit(model, strict=True)
    except RuntimeError as e:
        stub(a, "verify_failed", str(e), {"phase": "verify", "load_s": round(load_s, 1)}, code=7)
    census = expert_census(model)
    load_rec = {"fam": a.fam, "model": a.model, "model_type": model_type, "arm": "load",
                "status": "ok", "load_s": round(load_s, 1), "offload": bool(a.offload),
                "verify": {"n_quantized": rep["n_quantized"], "n_unquantized": rep["n_unquantized"]},
                "geometry": {k: getattr(lm_cfg, k, None) for k in (
                    "hidden_size", "intermediate_size", "moe_intermediate_size", "num_hidden_layers",
                    "num_experts", "num_local_experts", "num_experts_per_tok", "vocab_size",
                    "hidden_act", "hidden_activation", "tie_word_embeddings", "attention_bias")},
                "loaded_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "env": env, "written_by_arm": a.tag, **census}
    load_path = os.path.join(a.out, f"{a.fam}_train_load.json")
    if not os.path.exists(load_path):
        write_json(load_path, load_rec)
    print("LOAD OK " + json.dumps({k: v for k, v in load_rec.items() if k != "env"}), flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    tok = AutoTokenizer.from_pretrained(a.model)
    train, ev = encode(tok, ds["train"], a.seq), encode(tok, ds["eval"], a.seq)

    n_attn = add_attention_lora(model, a.r, a.alpha, torch.float32)

    if a.arm == "attn_only":
        for n, p in model.named_parameters():
            if "lora" in n and "experts" in n:
                p.requires_grad_(False)

    probes = {}
    if a.arm == "attn_only" and a.probe_refusals:
        nf, why_f = capture_verbose(enable_fast_train, model, verbose=True, dgrad=True)
        disable_fast_train(model)
        nb, why_b = capture_verbose(enable_batched_train, model, verbose=True)
        disable_batched_train(model)
        probes = {"fused": (nf, why_f), "batched": (nb, why_b)}
        for arm, (n, why) in probes.items():
            p = os.path.join(a.out, f"{a.fam}_train_{arm}.json")
            if n == 0:
                write_json(p, {"fam": a.fam, "model": a.model, "model_type": model_type, "arm": arm,
                               "status": "refused", "reason": f"0 patched: {why}", "n_patched": 0,
                               "probed_by": "attn_only", "steps": a.steps, "seq": a.seq,
                               "offload": bool(a.offload), **census})
                print(f"CELL REFUSED {a.fam}/{arm}: 0 patched", flush=True)

    counter = KernelCounter()
    counter.install()
    n_patched, reason = 0, ""
    if a.arm == "fused":
        n_patched, reason = capture_verbose(enable_fast_train, model, verbose=True, dgrad=True)
        if n_patched == 0:
            stub(a, "refused", f"enable_fast_train(dgrad=True) patched 0 modules: {reason}",
                 {"phase": "enable", "load_s": round(load_s, 1), **census}, code=3)
    elif a.arm == "batched":
        n_patched, reason = capture_verbose(enable_batched_train, model, verbose=True)
        if n_patched == 0:
            stub(a, "refused", f"enable_batched_train patched 0 modules: {reason}",
                 {"phase": "enable", "load_s": round(load_s, 1), **census}, code=3)
    else:
        disable_fast_train(model)
        disable_batched_train(model)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora" in n for n, _ in trainable), \
        f"non-LoRA parameter requires grad: {[n for n, _ in trainable if 'lora' not in n][:4]}"
    n_trainable = sum(p.numel() for _, p in trainable)
    init_sha, n_init = trainable_sha(model)

    h_before, bytes_before, empties_before = expert_hashes(model)
    assert bytes_before > 0, "C1 hashed ZERO bytes -- reading offload placeholders, gate is vacuous"
    assert empties_before == 0, f"C1 saw {empties_before} empty expert tensors -- placeholders leaked in"
    assert control_flip_fires(model), "C1 positive control did not fire -- the check cannot fail"

    ev0 = eval_loss(model, ev, a.eval_n)

    params = [p for _, p in trainable]
    opt = torch.optim.AdamW(params, lr=a.lr)
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    losses, step_ms, tokens_per_step, kcalls_per_step = [], [], [], []
    steps_done = 0
    try:
        with PowerSampler() as ps:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(a.steps):
                before = counter.snapshot()
                ids = train[i % len(train)].unsqueeze(0).cuda()
                ts = time.perf_counter()
                out = model(input_ids=ids, labels=ids)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                step_ms.append(round((time.perf_counter() - ts) * 1e3, 1))
                losses.append(round(float(out.loss.detach()), 5))
                tokens_per_step.append(int(ids.numel()))
                after = counter.snapshot()
                kcalls_per_step.append({k: after[k] - before[k] for k in after})
                steps_done = i + 1
                if (i + 1) % 10 == 0:
                    print(f"    step {i+1}/{a.steps} loss {losses[-1]} {step_ms[-1]} ms "
                          f"kcalls {kcalls_per_step[-1]}", flush=True)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError as e:
        counter.uninstall()
        stub(a, "oom", f"OOM at step {steps_done + 1}: {str(e)[:200]}",
             {"phase": "train", "steps_done": steps_done, "losses": losses, "step_ms": step_ms,
              "n_patched": n_patched, "init_sha": init_sha, "load_s": round(load_s, 1),
              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3), **census}, code=5)
    counter.uninstall()

    ev1 = eval_loss(model, ev, a.eval_n)
    h_after, bytes_after, empties_after = expert_hashes(model)
    changed = [k for k in h_before if h_before[k] != h_after.get(k)]
    c1_ok = (not changed) and bytes_after == bytes_before and empties_after == 0
    mean_w = statistics.mean(ps.samples) if ps.samples else None
    net_w = (mean_w - idle_w) if mean_w else None
    key = "fused_grouped_lora" if a.arm == "fused" else "batched_dequant_whole"
    kps = [k[key] for k in kcalls_per_step]

    cell = {
        "fam": a.fam, "model": a.model, "model_type": model_type, "arm": a.arm, "tag": a.tag,
        "status": "ok" if c1_ok else "c1_failed",
        "steps": a.steps, "seq": a.seq, "offload": bool(a.offload),
        "r": a.r, "alpha": a.alpha, "lr": a.lr, "seed": a.seed,
        "dataset": {"path": os.path.basename(a.data), "sha256": got, "n_train_rows": len(train),
                    "eval_rows_used": min(a.eval_n, len(ev))},
        "prereg": "tp1/P36-PREREG.md", "harness": "tp1_train_smoke.py (n17_cell.py + D1-D8)",
        "env": env, "load_s": round(load_s, 1),
        "verify": {"n_quantized": rep["n_quantized"], "n_unquantized": rep["n_unquantized"]},
        **census,
        "n_attn_lora": n_attn, "trainable_tensors": len(trainable), "trainable_params": n_trainable,
        "init_sha": init_sha, "init_tensors": n_init,
        "n_patched": n_patched, "enable_reason": reason,
        "probes": {k: {"n_patched": v[0], "reason": v[1]} for k, v in probes.items()},
        "kernel_counter_key": key,
        "kernel_calls_per_step": kps, "kernel_calls_per_step_min": (min(kps) if kps else 0),
        "kernel_calls_all": kcalls_per_step,
        "C1_tensors_hashed": len(h_before), "C1_bytes_hashed": bytes_before,
        "C1_empties_skipped": empties_before, "C1_control_detects_flipped_byte": True,
        "C1_experts_changed": len(changed), "C1_bit_exact": c1_ok, "C1_changed_sample": changed[:3],
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_mean_last20": round(statistics.mean(losses[-20:]), 5),
        "eval_loss_step0": round(ev0, 5), "eval_loss_final": round(ev1, 5),
        "eval_rel_improvement": round(ev1 / ev0, 5) if ev0 else None,
        "s_per_step": round(wall / a.steps, 4), "step_ms": step_ms,
        "tokens_per_step": tokens_per_step, "tokens_total": sum(tokens_per_step),
        "tokens_per_s": round(sum(tokens_per_step) / wall, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "idle_w": round(idle_w, 1), "mean_w": round(mean_w, 1) if mean_w else None,
        "power_samples": len(ps.samples),
        "joules_per_step": round(net_w * (wall / a.steps), 2) if net_w else None,
        "losses": losses,
    }
    write_json(os.path.join(a.out, f"{a.fam}_train_{a.tag}.json"), cell)
    print(("CELL OK " if c1_ok else "CELL C1_FAILED ") + json.dumps(
        {k: v for k, v in cell.items() if k not in ("losses", "step_ms", "tokens_per_step",
                                                    "kernel_calls_all", "env")}), flush=True)
    if not c1_ok:
        sys.exit(4)


if __name__ == "__main__":
    main()
