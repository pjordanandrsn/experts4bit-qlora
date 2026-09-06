#!/usr/bin/env python3
"""tp3_arm.py -- lane tp3 per-arm driver, BOTH frameworks.

PROVENANCE: copied byte-for-byte from bench/h2h-20260906/tp2/tp2_arm.py @ 769edb1d58a77580dec758b8db010e049037970c
(then committed under tp2 at 9738404); the tp2 tree is a committed receipt and is never edited. Every behaviour not
named below is kept byte-for-byte: status vocabulary, receipt JSON fields, step timing (median 11..N), held-out
evals, C1, stub codes, selftest scaffolding. The original tp2 docstring follows unmodified, then T10.

  T10 e4b#434 (against e4b#426, merged as #435 at main 5dad2a7): the attn-4bit expected count is the library's
      STRUCTURAL census -- detect_attention_projections(model, exact_linear=True).expected_count, snapshotted
      BEFORE the conversion (exact_linear=True matches `type is nn.Linear`; after conversion Linear4bit would not
      match -- the detector docstring's rule) -- never 4 * n_layers. Post-#426 the conversion consumes the same
      census, so n_attn4 == expected holds by construction: the assert is now a consistency tripwire (harness/library
      skew, a mutated model between census and convert, detector drift between lanes) and its real product is
      provenance. A missing k_proj is REFUSED (the detector's SystemExit, caught here) with the refusing modules
      named -- never a void_attn4 count guess. The receipt gains `structural_expected_n_attn4` (census.expected_count;
      null on non-attn4 arms and non-e4b frameworks) and `detector_version` (the library's own DETECTOR_VERSION if it
      ships one, else "<experts4bit_qlora version>+det:<sha256 of the detector source>[:12]" -- deterministic, no
      library change). Families: granite 128, olmoe 64, qwen3 192, mixtral 128 (exactly 4 x n_layers -- verdicts
      unchanged); Gemma-4 115 (25 x 4 + 5 x 3, the k_eq_v layers -- compares correctly for the first time); gpt-oss
      still REFUSED on the bias rule, which fires after admission (#435 CHANGELOG). The pre-registration written into
      every receipt is the `--prereg` argument: its default stays P40's path (the behaviour this copy implements, and
      what a selftest receipt cites) -- a P41 run MUST pass `--prereg p41/P41-PREREG.md`, so no P41 receipt ever names
      P40's document by accident (#434 follow-up, CEO/Warden review on PR #442).

----- the original tp2_arm.py docstring, unmodified -----

tp2_arm.py -- lane tp2 (P40: e4b vs Unsloth per MoE family, one box, one fixture) per-arm driver, BOTH frameworks.

Base: p38_arm.py as amended (amendment 3: snapshot-dir resolution in the Unsloth branch; amendment 4: U8 on the innermost
experts module), whose measurement code is tp1_train_smoke.py's (= n17_cell.py's): eval_loss(), the C1 hashes from the
bytes that persist, the byte-flip positive control, PowerSampler / idle_power, host_fingerprint(), init_sha, per-step
timing and kernel-call counters, refusal/OOM/fault receipts with exit codes. What is new, named so the files can be diffed:

  T1  --fam / --model / --revision per family (tp1's six, tp1's staged revisions); the receipt is <fam>_<fw>_<tag>.json
      (P40's naming) and carries `fam`, `model_type`, `n_layers` (from the config; text_config for gemma4), `revision`,
      and the snapshot directory it loaded from.
  T2  e4b branch = tp1's per-family loading (`load_moe_4bit_streaming(..., offload=<--offload>, pin=True, prefetch=False,
      quant_type="nf4")`, `.to("cuda")` only when resident, `verify_moe_4bit(strict=True)` -> `verify_failed` exit 7) plus
      P38's U4 attention-4bit (`quantize_attention_projections_4bit` BEFORE `add_attention_lora`; the count is asserted
      == 4 * n_layers, else `void_attn4`; a `SystemExit` from that function -- TRAIN_ATTN_4BIT refusing a bias-carrying
      projection -- is a `refused` row, never caught into a bf16 fallback).
  T3  --arm attn_only (gpt-oss's secondary row, tp1's D1/D8): the reference path with no expert adapter present, bf16
      attention (tp1's arm), which PROBES enable_fast_train / enable_batched_train (count, then disable) and a
      side-effect-free attn-4bit probe (does any q/k/v/o carry a bias?) and writes/refreshes the family's `fused_attn4`
      and `reference_attn4` refusal stubs with what it measured (a stub the run script wrote first, citing tp1, keeps
      its citation and gains the probe).
  T4  Unsloth branch = P38's U1 generalised: --unsloth-loader FastLanguageModel (P38's) | FastModel (an amendment if used);
      the snapshot dir resolved directly (amendment 3); the registered target list; the engagement banner captured;
      U8 evaluated on the innermost experts module (amendment 4) for any module whose last name component is `experts`
      (mlp.experts / block_sparse_moe.experts / layers.N.experts). A loader exception is classified: OOM -> `oom` (5);
      NotImplementedError / ValueError / TypeError / KeyError / AssertionError / ImportError or a message that says
      "not supported" / "unsupported" -> `refused` (3); anything else (CUDA errors included) -> `load_fault` (6).
  T5  Fixture per P40: --lr 2e-4, --accum 4 (micro-batches per optimizer step; rows in fixed order i*accum+j), --autocast 1
      (torch.autocast bf16 around forward+loss, training and eval), N=60, eval every 20 on the held-out 48, r 8 / alpha 16.
      A "step" is one optimizer step; s/step, tokens/step, J/step and the kernel-call counters are per optimizer step.
      Time-to-target is NOT computed (P40: out of scope).
  T6  --tokens-sha asserted (P38 U2); --expect-trainable N (the family's first e4b receipt, passed by the run script):
      a different count is RECORDED (`trainable_mismatch`) and the arm still trains -- the reducer applies P40's
      "same trainable count" validity rule; a trainable parameter outside the adapters stays `void_trainable` (15).
      `trainable_by_group` (attention / experts / other) is recorded so a mismatch is diagnosable.
  T7  The census regexes accept every family's expert-parameter names (gate_up_proj / down_proj / w1..w3 / input_linear /
      output_linear / gate_proj / up_proj) and every experts-module path.
  T8  --selftest: CPU, a tiny synthetic model, both branches' bookkeeping paths driven through the same run_arm() with
      mocked kernels (nf4_qlora.fused_grouped_lora, unsloth_zoo's forward_moe_backend_bnb4bit and the U8 predicate),
      the same receipt writer and the same stub paths. Verifies the file here, on a box with no GPU.
  T9  Device abstraction (cuda_sync / peak_gb / reset_peak / autocast) so T8 runs; every torch.cuda call is guarded.

Exit codes: 0 ok; 3 refused; 4 C1 failed (receipt written, arm void); 5 OOM; 6 load fault; 7 verify failed;
13 dataset/tokens mismatch; 15 void_trainable (non-adapter trainable).
"""
import argparse
import contextlib
import gc
import glob
import hashlib
import io
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import types

import torch
import torch.nn as nn

PREREG = "tp2/P40-PREREG.md"   # the --prereg default; a P41 run MUST pass --prereg p41/P41-PREREG.md
HARNESS = "tp3_arm.py (copy of tp2_arm.py @ 769edb1d + T10: structural attn4 census, e4b#434)"
EXPERT_ATTRS = ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")
EXPERT_PARAM_RE = re.compile(r"experts\.(?:.*\.)?(gate_up_proj|down_proj|gate_proj|up_proj|w[123]|input_linear|output_linear)$")
FMT = "### Instruction:\n{instruction}\n\n### Response:\n{output}"
UNSLOTH_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BANNER = "Enabling LoRA on MoE parameters"
DEV = "cuda"


# ----------------------------------------------------------------------------- T9: device abstraction
def cuda_sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 3) if DEV == "cuda" else 0.0


def reset_peak():
    if DEV == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def autocast_ctx(enabled):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=DEV, dtype=torch.bfloat16, enabled=True)


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


# ----------------------------------------------------------------------------- n17 / tp1 / p38 code (unchanged)
class PowerSampler:
    def __init__(self, enabled=True):
        self.samples, self._run, self.enabled = [], False, enabled

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
        if self.enabled:
            self._run = True
            self.t = threading.Thread(target=self._loop, daemon=True)
            self.t.start()
        return self

    def __exit__(self, *a):
        self._run = False
        if self.enabled:
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


def host_fingerprint(instance_env="TP2_INSTANCE_ID"):
    fields = ("uuid,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,"
              "power.limit,clocks.max.sm,clocks.max.mem,vbios_version,driver_version")
    want = fields.split(",")
    out = {}
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
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
            out["host_mem_gib"] = round(int(fh.readline().split()[1]) / 1048576, 1)
    except Exception as e:
        out["host_error"] = f"{type(e).__name__}: {e}"
    out["vast_instance_id"] = os.environ.get(instance_env)
    out["container_host"] = socket.gethostname()
    return out


@torch.no_grad()
def eval_loss(model, rows, fwd_kwargs, autocast):
    model.eval()
    tot = n = 0
    for ids in rows:
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEV)
        with autocast_ctx(autocast):
            tot += float(model(input_ids=x, labels=x, **fwd_kwargs(x)).loss)
        n += 1
    model.train()
    return tot / max(n, 1)


def control_flip_fires(h):
    """Positive control: the comparison must DETECT a single flipped byte."""
    if not h:
        return False
    k = next(iter(h))
    tampered = dict(h)
    tampered[k] = ("0" if h[k][0] != "0" else "1") + h[k][1:]
    return [x for x in h if h[x] != tampered.get(x)] == [k]


def write_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def receipt_path(a, fw=None, tag=None):
    return os.path.join(a.out, f"{a.fam}_{fw or a.framework}_{tag or a.tag}.json")


# ----------------------------------------------------------------------------- frozen-byte hashes (C1), per framework
def hashes_e4b(model):
    """tp1's expert_hashes (state_dict bytes of every Experts4bit stack) + the NF4 attention projections when present."""
    try:
        import bitsandbytes as bnb
        L4 = bnb.nn.Linear4bit
    except Exception:
        L4 = None
    h, nbytes, empties = {}, 0, 0
    for name, m in model.named_modules():
        if any(hasattr(m, a) for a in EXPERT_ATTRS):
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
                h[f"{name}.{attr}"] = sha_bytes(b)
        elif L4 is not None and isinstance(m, L4):
            b = m.weight.data.detach().to("cpu").contiguous().numpy().tobytes()
            if not b:
                empties += 1
                continue
            nbytes += len(b)
            h[f"{name}.weight(attn4)"] = sha_bytes(b)
    return h, nbytes, empties


def hashes_unsloth(model):
    """The frozen 4-bit bytes Unsloth holds: every Params4bit (expert stacks incl. PEFT's parametrizations.*.original,
    and the bnb Linear4bit weights), read from named_parameters(); a LoRA parameter is never hashed."""
    h, nbytes, empties = {}, 0, 0
    for name, p in model.named_parameters():
        if p.requires_grad or "lora" in name.lower():
            continue
        is4 = type(p).__name__ == "Params4bit" or (p.dtype == torch.uint8 and ("experts" in name or "proj" in name))
        if not is4:
            continue
        b = p.data.detach().to("cpu").contiguous().numpy().tobytes()
        if not b:
            empties += 1
            continue
        nbytes += len(b)
        h[name] = sha_bytes(b)
    return h, nbytes, empties


def is_experts_module(name):
    return name.split(".")[-1] == "experts"


def innermost(m):
    while hasattr(m, "base_layer"):
        m = m.base_layer
    return m


def quant_census(model):
    """U5 / T7: what is 4-bit and what is not, by class; the expert stacks, the attention projections, the router, lm_head."""
    c = {"Params4bit_expert_stacks": 0, "Params4bit_other": 0, "Linear4bit": 0, "Linear_bf16": 0, "Linear_fp32": 0,
         "Experts4bit": 0, "ExpertsLoRA": 0, "LoRALinear": 0, "experts_modules": 0, "experts_module_classes": {},
         "router_gate": [], "lm_head": None, "samples": []}
    for name, m in model.named_modules():
        cls = type(m).__name__
        if cls in ("Experts4bit", "ExpertsNbit", "GptOssExperts4bit"):
            c["Experts4bit"] += 1
        elif cls == "ExpertsLoRA":
            c["ExpertsLoRA"] += 1
        elif cls == "LoRALinear":
            c["LoRALinear"] += 1
        elif cls == "Linear4bit":
            c["Linear4bit"] += 1
        elif cls == "Linear" and "lora_" not in name:
            dt = getattr(getattr(m, "weight", None), "dtype", None)
            if dt == torch.bfloat16:
                c["Linear_bf16"] += 1
            elif dt == torch.float32:
                c["Linear_fp32"] += 1
        if is_experts_module(name):
            c["experts_modules"] += 1
            k = type(innermost(m)).__name__
            c["experts_module_classes"][k] = c["experts_module_classes"].get(k, 0) + 1
        if name.endswith("mlp.gate") or name.endswith("router") or name.endswith("block_sparse_moe.gate") or name.endswith(".gate"):
            w = getattr(m, "weight", None)
            if w is not None and len(c["router_gate"]) < 2:
                c["router_gate"].append({"name": name, "cls": cls, "dtype": str(getattr(w, "dtype", None))})
        if name.endswith("lm_head"):
            c["lm_head"] = {"cls": cls, "dtype": str(getattr(getattr(m, "weight", None), "dtype", None))}
    for name, p in model.named_parameters():
        if type(p).__name__ == "Params4bit":
            key = "Params4bit_expert_stacks" if EXPERT_PARAM_RE.search(name) else "Params4bit_other"
            c[key] += 1
            if len(c["samples"]) < 4 and key == "Params4bit_expert_stacks":
                c["samples"].append({"name": name, "dtype": str(p.dtype), "shape": list(p.shape)})
    return c


def trainable_census(model):
    tr = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_params = sum(p.numel() for _, p in tr)
    dtypes = {}
    for _, p in tr:
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
    non_adapter = [n for n, _ in tr if "lora" not in n.lower()]
    groups = {"attention": 0, "experts": 0, "other": 0}
    for n, p in tr:
        if "experts" in n:
            groups["experts"] += p.numel()
        elif "self_attn" in n or re.search(r"\.(q_proj|k_proj|v_proj|o_proj)\.", n):
            groups["attention"] += p.numel()
        else:
            groups["other"] += p.numel()
    return tr, n_params, dtypes, non_adapter, groups


def trainable_sha(tr):
    h = hashlib.sha256()
    for name, p in tr:
        h.update(name.encode())
        h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def stub(a, status, reason, extra=None, code=None, fw=None, tag=None, arm=None):
    rec = {"framework": fw or a.framework, "fam": a.fam, "model": a.model, "revision": a.revision, "arm": arm or a.arm,
           "tag": tag or a.tag, "status": status, "reason": str(reason)[:800], "steps": a.steps, "seq": a.seq,
           "accum": a.accum, "offload": bool(a.offload), "prereg": getattr(a, "prereg", None) or PREREG, "harness": HARNESS}
    if extra:
        rec.update(extra)
    write_json(receipt_path(a, fw, tag), rec)
    print(f"CELL {status.upper()} " + json.dumps({k: v for k, v in rec.items() if k not in ("losses", "step_ms")}), flush=True)
    if code is not None:
        sys.exit(code)


def refresh_stub(a, tag, arm, status, reason, extra):
    """T3: write the family's refusal stub for `tag`, or merge the probe into a stub the run script already wrote
    (its citation stays; the measured probe is added)."""
    p = receipt_path(a, "e4b", tag)
    if os.path.exists(p):
        try:
            rec = json.load(open(p))
        except Exception:
            rec = {}
        if rec.get("status") == "ok":
            print(f"STUB SKIPPED: {os.path.basename(p)} is a receipt that trained; the probe is in this arm's own receipt", flush=True)
            return
        rec.setdefault("status", status)
        rec.setdefault("reason", reason)
        rec.update(extra)
        rec["probe_reason"] = str(reason)[:600]
        write_json(p, rec)
        print(f"CELL {rec['status'].upper()} (stub refreshed) " + json.dumps({k: v for k, v in rec.items() if k not in ("losses", "step_ms")}), flush=True)
    else:
        stub(a, status, reason, extra, fw="e4b", tag=tag, arm=arm)


def capture_verbose(fn, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = fn(*args, **kw)
    return n, buf.getvalue().strip()[-600:]


def classify_load_exception(e):
    if is_oom(e):
        return "oom", 5
    msg = str(e).lower()
    if isinstance(e, (NotImplementedError, ValueError, TypeError, KeyError, AssertionError, ImportError, SystemExit)):
        return "refused", 3
    if isinstance(e, RuntimeError) and "cuda" not in msg:
        return "refused", 3
    if re.search(r"not support|unsupported|refus|does not support|cannot", msg):
        return "refused", 3
    return "load_fault", 6


def n_layers_of(cfg):
    lm = getattr(cfg, "text_config", None) or cfg
    return getattr(lm, "num_hidden_layers", None), getattr(cfg, "model_type", None)


# ----------------------------------------------------------------------------- U2: the fixed text, tokenised once per family
def encode_rows(tok, rows, seq):
    out = []
    for r in rows:
        ids = tok(FMT.format(instruction=r["instruction"], output=r["output"]),
                  truncation=True, max_length=seq).input_ids
        if len(ids) >= 8:
            out.append([int(i) for i in ids])
    return out


def prepare(a, tok=None):
    got = sha_bytes(open(a.data, "rb").read())
    if got != a.data_sha:
        print(f"DATASET MISMATCH {got} != {a.data_sha}")
        sys.exit(13)
    ds = json.load(open(a.data))
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    train, ev = encode_rows(tok, ds["train"], a.seq), encode_rows(tok, ds["eval"], a.seq)[:a.eval_n]
    body = json.dumps({"train": train, "eval": ev}, separators=(",", ":")).encode()
    rec = {"fam": a.fam, "dataset": os.path.basename(a.data), "dataset_sha256": got, "tokenizer": a.model, "revision": a.revision,
           "tokenizer_class": type(tok).__name__, "format": FMT, "seq": a.seq, "n_train": len(train), "n_eval": len(ev),
           "train_tokens": sum(map(len, train)), "sha256": sha_bytes(body), "train": train, "eval": ev}
    write_json(a.tokens, rec)
    print(f"TOKENS fam={a.fam} n_train={len(train)} n_eval={len(ev)} train_tokens={rec['train_tokens']} mean={rec['train_tokens']/max(1,len(train)):.1f} sha={rec['sha256']}")
    return rec


# ----------------------------------------------------------------------------- U8: engagement counters
class Counters:
    def __init__(self):
        self.counts = {"fused_grouped_lora": 0, "experts_forward": 0, "moe_bnb4bit_backend": 0}
        self._restore, self._hooks = [], []

    def install_e4b(self):
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

    def install_unsloth(self, model):
        for name, m in model.named_modules():
            if is_experts_module(name):
                self._hooks.append(m.register_forward_pre_hook(lambda mod, inp: self.counts.__setitem__("experts_forward", self.counts["experts_forward"] + 1)))
        try:
            from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as M
            orig = M.forward_moe_backend_bnb4bit

            def w(*a, _orig=orig, **k):
                self.counts["moe_bnb4bit_backend"] += 1
                return _orig(*a, **k)
            M.forward_moe_backend_bnb4bit = w
            self._restore.append((M, "forward_moe_backend_bnb4bit", orig))
            try:   # transformers' dispatcher may hold the function object; swap it there too when it does
                from transformers.integrations import moe as TM
                tab = getattr(TM, "ALL_EXPERTS_FUNCTIONS", None)
                if tab is not None:
                    for k, v in list(tab.items()):
                        if v is orig:
                            tab[k] = w
                            self._restore.append((tab, k, orig))
            except Exception:
                pass
        except Exception:
            pass

    def snapshot(self):
        return dict(self.counts)

    def uninstall(self):
        for h in self._hooks:
            h.remove()
        for mod, name, orig in self._restore:
            if isinstance(mod, dict):
                mod[name] = orig
            else:
                setattr(mod, name, orig)


# ----------------------------------------------------------------------------- the two loaders (each returns model + a dict of extras)
def attn4_bias_probe(model):
    """T3: the side-effect-free form of TRAIN_ATTN_4BIT's refusal condition (a bias on any q/k/v/o)."""
    projs = ("q_proj", "k_proj", "v_proj", "o_proj")
    biased, n = [], 0
    for name, mod in model.named_modules():
        if all(isinstance(getattr(mod, p, None), nn.Linear) for p in projs):
            for p in projs:
                n += 1
                if getattr(mod, p).bias is not None:
                    biased.append(f"{name}.{p}")
    return {"n_projections": n, "n_biased": len(biased), "would_refuse": bool(biased), "sample": biased[:2]}


def detector_version(lora_mod, lib_version):
    """T10 (#434): the detector version recorded in every e4b receipt -- the library's own DETECTOR_VERSION if it
    ships one (#435 ships none; checked at main 5dad2a7), else the library version + a short hash of the detector's
    source. Deterministic, needs no library change, and both forms name the library version."""
    import inspect
    return (getattr(lora_mod, "DETECTOR_VERSION", None)
            or f"{lib_version}+det:{hashlib.sha256(inspect.getsource(lora_mod.detect_attention_projections).encode()).hexdigest()[:12]}")


def attn4_census_check(a, model, x, detect_fn, quantize_fn):
    """T10 (#434): the attn-4bit count check against the library's structural census, never 4 * n_layers.
    The census is snapshotted BEFORE any mutation: exact_linear=True matches `type is nn.Linear`, so after
    conversion Linear4bit would not match (the detector docstring's rule). A detector SystemExit (a q/o layer
    without k_proj -- true cross-layer KV reuse) is a REFUSED row with the refusing modules named, never a
    void_attn4 guess; a quantize SystemExit (a bias-carrying projection) stays the REFUSED row tp2 wrote."""
    try:
        census = detect_fn(model, exact_linear=True)
    except SystemExit as e:
        refusing = [n for n, m in model.named_modules()
                    if type(getattr(m, "q_proj", None)) is nn.Linear and type(getattr(m, "o_proj", None)) is nn.Linear
                    and type(getattr(m, "k_proj", None)) is not nn.Linear]
        stub(a, "refused", f"detect_attention_projections refused: {e}",
             {"phase": "attn4", "attn4_probe": x.get("attn4_probe"), "attn4_refusing_modules": refusing[:8],
              "n_layers": x["n_layers"], "model_type": x["model_type"]}, code=3)
    x["structural_expected_n_attn4"] = census.expected_count
    try:
        x["n_attn4"] = quantize_fn(model)                             # U4: BEFORE the attention LoRA (its docstring's rule)
    except SystemExit as e:                                           # TRAIN_ATTN_4BIT refuses a bias-carrying projection
        stub(a, "refused", f"quantize_attention_projections_4bit refused: {e}", {"phase": "attn4", "attn4_probe": x["attn4_probe"],
             "structural_expected_n_attn4": x["structural_expected_n_attn4"],
             "n_layers": x["n_layers"], "model_type": x["model_type"]}, code=3)
    if x["n_attn4"] != x["structural_expected_n_attn4"]:
        stub(a, "void_attn4", f"quantize_attention_projections_4bit converted {x['n_attn4']} projections, "
             f"structural census expected {x['structural_expected_n_attn4']}",
             {"phase": "attn4", "structural_expected_n_attn4": x["structural_expected_n_attn4"],
              "n_layers": x["n_layers"], "model_type": x["model_type"]}, code=3)


def load_e4b(a):
    from experts4bit_qlora import (disable_batched_train, disable_fast_train, enable_batched_train, enable_fast_train,
                                   load_moe_4bit_streaming, verify_moe_4bit)
    import experts4bit_qlora
    import experts4bit_qlora.lora as _lora_mod
    from experts4bit_qlora.lora import add_attention_lora, detect_attention_projections, quantize_attention_projections_4bit
    from transformers import AutoTokenizer
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [], "probes": {}, "attn4_probe": None,
         "structural_expected_n_attn4": None, "detector_version": detector_version(_lora_mod, experts4bit_qlora.__version__)}
    model, cfg = load_moe_4bit_streaming(a.model, "cuda", torch.bfloat16, a.r, a.alpha,
                                         offload=bool(a.offload), pin=True, prefetch=False, quant_type="nf4")
    if not a.offload:
        model.to("cuda")
    x["n_layers"], x["model_type"] = n_layers_of(cfg)
    try:
        rep = verify_moe_4bit(model, strict=True)
    except RuntimeError as e:
        stub(a, "verify_failed", str(e), {"phase": "verify"}, code=7)
    x["verify"] = {"n_quantized": rep.get("n_quantized"), "n_unquantized": rep.get("n_unquantized")}
    x["attn4_probe"] = attn4_bias_probe(model)
    if a.attn_4bit:
        attn4_census_check(a, model, x, detect_attention_projections, quantize_attention_projections_4bit)   # T10
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    add_attention_lora(model, a.r, a.alpha, torch.float32)
    if a.arm == "attn_only":
        for n, p in model.named_parameters():
            if "lora" in n and "experts" in n:
                p.requires_grad_(False)
        nf, why_f = capture_verbose(enable_fast_train, model, verbose=True, dgrad=True)
        disable_fast_train(model)
        nb, why_b = capture_verbose(enable_batched_train, model, verbose=True)
        disable_batched_train(model)
        x["probes"] = {"fused": {"n_patched": nf, "reason": why_f}, "batched": {"n_patched": nb, "reason": why_b}}
        common = {"model_type": x["model_type"], "n_layers": x["n_layers"], "probed_by": "attn_only", "n_patched": 0}
        if nf == 0:
            refresh_stub(a, "fused_attn4", "fused", "refused", f"enable_fast_train(dgrad=True) patched 0 modules on this box: {why_f}", dict(common, probe_n_patched=nf))
        if x["attn4_probe"]["would_refuse"]:
            refresh_stub(a, "reference_attn4", "reference", "refused",
                         f"TRAIN_ATTN_4BIT would refuse: {x['attn4_probe']['n_biased']} of {x['attn4_probe']['n_projections']} attention projections carry a bias "
                         f"(quantize_attention_projections_4bit raises SystemExit on a bias; e.g. {x['attn4_probe']['sample']})", dict(common, attn4_probe=x["attn4_probe"]))
    elif a.arm == "fused":
        x["n_patched"], x["reason"] = capture_verbose(enable_fast_train, model, verbose=True, dgrad=True)
        if x["n_patched"] == 0:
            stub(a, "refused", f"enable_fast_train(dgrad=True) patched 0 modules: {x['reason']}", {"phase": "enable", "n_layers": x["n_layers"], "model_type": x["model_type"]}, code=3)
    else:
        disable_fast_train(model)
        disable_batched_train(model)
    x["tokenizer_obj"] = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    x["ckpt_mode"] = "hf:use_reentrant=False"
    x["hashes"] = hashes_e4b
    x["fwd_kwargs"] = lambda t: {}
    x["snapshot_dir"] = None
    return model, x


def snapshot_dir_for(model_id, revision):
    return os.path.join(os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")),
                        "models--" + model_id.replace("/", "--"), "snapshots", revision)


def load_unsloth(a):
    import unsloth
    loader = getattr(unsloth, a.unsloth_loader)
    from huggingface_hub import snapshot_download
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [], "probes": {}, "attn4_probe": None,
         "structural_expected_n_attn4": None, "detector_version": None}   # T10: e4b-only fields, null here
    _cand = snapshot_dir_for(a.model, a.revision)                                        # amendment 3: the pinned snapshot dir, same bytes
    local = _cand if os.path.isdir(_cand) else snapshot_download(a.model, revision=a.revision)
    x["snapshot_dir"] = local
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model, tokenizer_obj = loader.from_pretrained(model_name=local, max_seq_length=a.seq, dtype=torch.bfloat16, load_in_4bit=True)
        model = loader.get_peft_model(
            model, r=a.r, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", target_modules=list(UNSLOTH_TARGETS),
            use_gradient_checkpointing=("unsloth" if a.grad_ckpt == "unsloth" else True), random_state=a.seed)
    x["banner_lines"] = [l for l in buf.getvalue().splitlines() if re.search(r"MoE|Params4bit|4.?bit|LoRA on", l)][:12]
    print("\n".join(buf.getvalue().splitlines()[-40:]), flush=True)
    if not any(BANNER in l for l in x["banner_lines"]):
        x["banner_lines"].append(f"NO '{BANNER}' banner on stdout (the census below decides)")
    x["verify"] = {"n_quantized": None, "n_unquantized": None}
    x["n_layers"], x["model_type"] = n_layers_of(model.config)
    model.config.use_cache = False
    x["tokenizer_obj"] = tokenizer_obj
    x["ckpt_mode"] = "unsloth" if a.grad_ckpt == "unsloth" else "hf:True (via get_peft_model)"
    x["hashes"] = hashes_unsloth
    x["fwd_kwargs"] = lambda t: {"attention_mask": torch.ones_like(t)}     # all-ones = no masking; identical semantics
    return model, x


def u8_bnb4bit(model):
    """U8 as amended (amendment 4): the predicate on the module named experts AND on the innermost module."""
    try:
        from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import _moe_uses_bnb4bit_expert_weights as pred
    except Exception as e:
        return f"unchecked: {e}"
    mods = [(n, m) for n, m in model.named_modules() if is_experts_module(n)]
    out = {"n_experts_modules": len(mods), "n_bnb4bit": sum(bool(pred(m)) for _, m in mods)}
    try:
        out["n_bnb4bit_unwrapped"] = sum(bool(pred(innermost(m))) for _, m in mods)
        out["inner_param_types"] = sorted({type(getattr(innermost(m), k, None)).__name__ for _, m in mods for k in ("gate_up_proj", "down_proj")})
        out["wrap_depth"] = max([sum(1 for _ in iter_base_layers(m)) for _, m in mods] or [0])
    except Exception as e:
        out["unwrapped"] = f"unchecked: {e}"
    return out


def iter_base_layers(m):
    while hasattr(m, "base_layer"):
        m = m.base_layer
        yield m


# ----------------------------------------------------------------------------- the arm (both frameworks, one code path)
def run_arm(a, load_fn, sampler=True):
    import importlib.metadata as md
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.adapter_dir, exist_ok=True)
    tk = json.load(open(a.tokens))
    body = json.dumps({"train": tk["train"], "eval": tk["eval"]}, separators=(",", ":")).encode()
    if sha_bytes(body) != tk["sha256"] or (a.tokens_sha and tk["sha256"] != a.tokens_sha):
        stub(a, "tokens_mismatch", f"{a.tokens}: sha {sha_bytes(body)[:12]} != {tk['sha256'][:12]} / registered {str(a.tokens_sha)[:12]}", code=13)
    if tk.get("fam") not in (None, a.fam):
        stub(a, "tokens_mismatch", f"{a.tokens} was tokenised for fam {tk.get('fam')}, arm is {a.fam}", code=13)
    train, ev = tk["train"], tk["eval"][:a.eval_n]

    env = {"framework": a.framework, "torch": torch.__version__, "device": DEV,
           "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "cap": list(torch.cuda.get_device_capability()) if DEV == "cuda" else None,
           "host": host_fingerprint(), "anchor_json": os.environ.get("TP2_ANCHOR_JSON"), "box_class": os.environ.get("TP2_BOX_CLASS"),
           "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "unsloth_loader": a.unsloth_loader if a.framework == "unsloth" else None,
           "unsloth_env": {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_")}}
    for pkg in ("transformers", "experts4bit-qlora", "grouped-nf4-gemm", "bitsandbytes", "peft", "unsloth", "unsloth_zoo", "triton", "huggingface_hub"):
        try:
            env[pkg] = md.version(pkg)
        except Exception:
            env[pkg] = None

    idle_w = idle_power() if sampler else 0.0
    torch.manual_seed(a.seed)
    t_load = time.perf_counter()
    try:
        model, x = load_fn(a)
    except SystemExit as e:                      # a stub already written (int code) propagates; a framework's SystemExit(message) is a refusal row
        if isinstance(e.code, int) or e.code is None:
            raise
        stub(a, "refused", f"SystemExit: {str(e.code)[:700]}", {"phase": "load"}, code=3)
    except Exception as e:
        st, code = classify_load_exception(e)
        stub(a, st, f"{type(e).__name__}: {str(e)[:700]}", {"phase": "load"}, code=code)
    load_s = time.perf_counter() - t_load
    n_attn4, n_patched, reason, banner_lines = x["n_attn4"], x["n_patched"], x["reason"], x["banner_lines"]
    hashes, fwd_kwargs, tokenizer_obj = x["hashes"], x["fwd_kwargs"], x.get("tokenizer_obj")

    # U5: adapters fp32 (Unsloth: cast if needed, recorded), the censuses, U3/T6: the trainable count
    tr, n_trainable, dtypes_before, non_adapter, groups = trainable_census(model)
    cast = 0
    if a.framework == "unsloth":
        for n, p in tr:
            if p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
                cast += 1
        tr, n_trainable, dtypes_after, non_adapter, groups = trainable_census(model)
    else:
        dtypes_after = dtypes_before
    census = quant_census(model)
    tokenizer_agree = None
    if tokenizer_obj is not None:
        try:   # re-derive 8 train rows from the raw dataset if it sits beside the tokens file; informational
            raw = glob.glob(os.path.join(os.path.dirname(os.path.abspath(a.tokens)), "data", "ds_*.json"))
            if raw:
                rows = json.load(open(raw[0]))["train"]
                tokenizer_agree = encode_rows(tokenizer_obj, rows[:8], a.seq) == train[:8]
        except Exception as e:
            tokenizer_agree = f"unchecked: {e}"
    trainable_mismatch = None
    if a.expect_trainable and n_trainable != a.expect_trainable:
        trainable_mismatch = {"expected": a.expect_trainable, "got": n_trainable, "by_group": groups}
    print("LOAD OK " + json.dumps({"fam": a.fam, "framework": a.framework, "arm": a.arm, "model_type": x.get("model_type"), "n_layers": x.get("n_layers"),
                                   "load_s": round(load_s, 1), "verify": x.get("verify"), "n_attn4": n_attn4, "n_patched": n_patched,
                                   "structural_expected_n_attn4": x.get("structural_expected_n_attn4"), "detector_version": x.get("detector_version"),
                                   "trainable_params": n_trainable, "trainable_tensors": len(tr), "trainable_by_group": groups, "trainable_mismatch": trainable_mismatch,
                                   "dtypes_before": dtypes_before, "dtypes_after": dtypes_after, "lora_cast_to_fp32": cast,
                                   "non_adapter_trainable": non_adapter[:4], "census": {k: v for k, v in census.items() if k != "samples"},
                                   "tokenizer_agree": tokenizer_agree, "ckpt": x["ckpt_mode"], "probes": x.get("probes"), "attn4_probe": x.get("attn4_probe")}), flush=True)
    for l in banner_lines:
        print("ENGAGE " + l[:300], flush=True)
    if non_adapter:
        stub(a, "void_trainable", f"non-adapter trainables {non_adapter[:4]}",
             {"trainable_params": n_trainable, "non_adapter_trainable": non_adapter[:8], "census": census, "n_layers": x.get("n_layers"), "model_type": x.get("model_type")}, code=15)
    init_sha = trainable_sha(tr)

    counter = Counters()
    if a.framework == "e4b":
        counter.install_e4b()
    else:
        counter.install_unsloth(model)

    h_before, bytes_before, empties_before = hashes(model)
    assert bytes_before > 0, "C1 hashed ZERO bytes -- gate is vacuous"
    assert empties_before == 0, f"C1 saw {empties_before} empty frozen tensors"
    assert control_flip_fires(h_before), "C1 positive control did not fire -- the check cannot fail"

    ev0 = eval_loss(model, ev, fwd_kwargs, a.autocast)
    curve = [{"step": 0, "heldout_loss": round(ev0, 5), "train_wall_s": 0.0}]
    params = [p for _, p in tr]
    opt = torch.optim.AdamW(params, lr=a.lr)          # the same call in both arms: torch defaults (wd 0.01)
    model.train()
    gc.collect()
    reset_peak()

    losses, step_ms, tokens_per_step, kcalls = [], [], [], []
    train_wall, steps_done = 0.0, 0
    common_stub = lambda: {"n_patched": n_patched, "n_attn4": n_attn4, "init_sha": init_sha, "load_s": round(load_s, 1),
                           "trainable_params": n_trainable, "census": census, "n_layers": x.get("n_layers"), "model_type": x.get("model_type")}
    try:
        with PowerSampler(enabled=sampler) as ps:
            cuda_sync()
            t0 = time.perf_counter()
            for i in range(a.steps):
                before = counter.snapshot()
                ts = time.perf_counter()
                loss_sum, ntok = 0.0, 0
                for j in range(a.accum):                                    # T5: batch 1 x accum micro-batches, rows in fixed order
                    ids = torch.tensor(train[(i * a.accum + j) % len(train)], dtype=torch.long).unsqueeze(0).to(DEV)
                    with autocast_ctx(a.autocast):
                        out = model(input_ids=ids, labels=ids, **fwd_kwargs(ids))
                        loss = out.loss / a.accum
                    loss.backward()
                    loss_sum += float(out.loss.detach())
                    ntok += int(ids.numel())
                opt.step()
                opt.zero_grad(set_to_none=True)
                cuda_sync()
                dt = time.perf_counter() - ts
                train_wall += dt
                step_ms.append(round(dt * 1e3, 1))
                losses.append(round(loss_sum / a.accum, 5))
                tokens_per_step.append(ntok)
                after = counter.snapshot()
                kcalls.append({k: after[k] - before[k] for k in after})
                steps_done = i + 1
                if steps_done % 10 == 0:
                    print(f"    step {steps_done}/{a.steps} loss {losses[-1]} {step_ms[-1]} ms kcalls {kcalls[-1]}", flush=True)
                if steps_done % a.eval_every == 0:
                    e = eval_loss(model, ev, fwd_kwargs, a.autocast)
                    curve.append({"step": steps_done, "heldout_loss": round(e, 5), "train_wall_s": round(train_wall, 2)})
                    print(f"    eval@{steps_done} heldout {e:.5f} train_wall {train_wall:.1f}s", flush=True)
            cuda_sync()
            wall = time.perf_counter() - t0
    except Exception as e:
        counter.uninstall()
        if is_oom(e):
            stub(a, "oom", f"OOM at step {steps_done + 1}: {str(e)[:200]}",
                 dict(common_stub(), phase="train", steps_done=steps_done, losses=losses, step_ms=step_ms, peak_vram_gb=peak_gb()), code=5)
        raise
    counter.uninstall()
    peak = peak_gb()

    if curve[-1]["step"] != steps_done:
        e = eval_loss(model, ev, fwd_kwargs, a.autocast)
        curve.append({"step": steps_done, "heldout_loss": round(e, 5), "train_wall_s": round(train_wall, 2)})
    ev1 = curve[-1]["heldout_loss"]
    h_after, bytes_after, empties_after = hashes(model)
    changed = [k for k in h_before if h_before[k] != h_after.get(k)]
    c1_ok = (not changed) and bytes_after == bytes_before and empties_after == 0

    bnb4 = u8_bnb4bit(model) if a.framework == "unsloth" else None

    # U7: the adapter
    adapter = {}
    atag = f"{a.fam}_{a.tag}"
    try:
        if a.framework == "e4b":
            if a.selftest:
                sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora" in k}
                pth = os.path.join(a.adapter_dir, f"adapter_{atag}.pt")
                torch.save(sd, pth)
                n_t = len(sd)
            else:
                from experts4bit_qlora.train import save_adapter
                n_t = save_adapter(model, a.adapter_dir, atag)
                pth = os.path.join(a.adapter_dir, f"adapter_{atag}.pt")
            sd = torch.load(pth, map_location="cpu")
            adapter = {"path": pth, "bytes": os.path.getsize(pth), "tensors": n_t, "params": sum(v.numel() for v in sd.values()),
                       "dtypes": sorted({str(v.dtype) for v in sd.values()})}
        else:
            d = os.path.join(a.adapter_dir, atag)
            model.save_pretrained(d)
            files = [os.path.join(d, f) for f in os.listdir(d)]
            adapter = {"path": d, "bytes": sum(os.path.getsize(f) for f in files), "files": sorted(os.path.basename(f) for f in files)}
            try:
                from safetensors import safe_open
                st = [f for f in files if f.endswith(".safetensors")]
                if st:
                    with safe_open(st[0], "pt") as fh:
                        keys = list(fh.keys())
                        adapter["tensors"] = len(keys)
                        adapter["params"] = sum(math.prod(fh.get_slice(k).get_shape()) for k in keys)
                        adapter["dtypes"] = sorted({str(fh.get_slice(k).get_dtype()) for k in keys})
            except Exception as e:
                adapter["read_error"] = str(e)[:200]
    except Exception as e:
        adapter = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

    mean_w = statistics.mean(ps.samples) if ps.samples else None
    net_w = (mean_w - idle_w) if mean_w else None
    key = "fused_grouped_lora" if a.framework == "e4b" else "moe_bnb4bit_backend"
    kps = [k[key] for k in kcalls]
    efw = [k["experts_forward"] for k in kcalls]
    steady = step_ms[10:] if len(step_ms) > 10 else step_ms
    cell = {
        "framework": a.framework, "fam": a.fam, "model": a.model, "revision": a.revision, "model_type": x.get("model_type"), "n_layers": x.get("n_layers"),
        "snapshot_dir": x.get("snapshot_dir"), "arm": a.arm, "tag": a.tag,
        "status": "ok" if c1_ok else "c1_failed", "steps": a.steps, "seq": a.seq, "accum": a.accum, "autocast": bool(a.autocast),
        "r": a.r, "alpha": a.alpha, "lr": a.lr, "seed": a.seed, "offload": bool(a.offload),
        "optimizer": "torch.optim.AdamW(lr) torch defaults", "grad_ckpt": x["ckpt_mode"], "attn_4bit": bool(a.attn_4bit), "n_attn4": n_attn4, "attn4_probe": x.get("attn4_probe"),
        "structural_expected_n_attn4": x.get("structural_expected_n_attn4"), "detector_version": x.get("detector_version"),
        "tokens": {"path": os.path.basename(a.tokens), "sha256": tk["sha256"], "n_train": len(train), "eval_rows_used": len(ev), "tokenizer_agree": tokenizer_agree},
        "prereg": getattr(a, "prereg", None) or PREREG, "harness": HARNESS, "env": env, "load_s": round(load_s, 1),
        "verify": x.get("verify"), "census": census, "engagement_banners": banner_lines, "unsloth_bnb4bit_modules": bnb4,
        "trainable_params": n_trainable, "trainable_tensors": len(tr), "trainable_by_group": groups,
        "expect_trainable": a.expect_trainable, "trainable_mismatch": trainable_mismatch,
        "adapter_dtypes_before": dtypes_before, "adapter_dtypes_after": dtypes_after,
        "lora_cast_to_fp32": cast, "init_sha": init_sha, "n_patched": n_patched, "enable_reason": reason, "probes": x.get("probes"),
        "kernel_counter_key": key, "kernel_calls_per_step": kps, "kernel_calls_per_step_min": (min(kps) if kps else 0),
        "experts_forward_calls_per_step_min": (min(efw) if efw else 0), "kernel_calls_all": kcalls,
        "C1_tensors_hashed": len(h_before), "C1_bytes_hashed": bytes_before, "C1_empties_skipped": empties_before,
        "C1_control_detects_flipped_byte": True, "C1_experts_changed": len(changed), "C1_bit_exact": c1_ok, "C1_changed_sample": changed[:3],
        "loss_first": losses[0], "loss_last": losses[-1], "loss_mean_last20": round(statistics.mean(losses[-20:]), 5),
        "eval_loss_step0": round(ev0, 5), "eval_loss_final": ev1, "eval_curve": curve,
        "s_per_step": round(wall / a.steps, 4), "s_per_step_median_11plus": round(statistics.median(steady) / 1e3, 4), "step_ms": step_ms,
        "train_wall_s": round(train_wall, 2), "window_wall_s": round(wall, 2),
        "tokens_per_step": tokens_per_step, "tokens_total": sum(tokens_per_step), "tokens_per_s": round(sum(tokens_per_step) / train_wall, 1) if train_wall else None,
        "peak_vram_gb": peak, "idle_w": round(idle_w, 1), "mean_w": round(mean_w, 1) if mean_w else None, "power_samples": len(ps.samples),
        "sampler": bool(sampler), "joules_per_step": round(net_w * (train_wall / a.steps), 2) if net_w else None,
        "adapter": adapter, "losses": losses,
    }
    write_json(receipt_path(a), cell)
    print(("CELL OK " if c1_ok else "CELL C1_FAILED ") + json.dumps(
        {k: v for k, v in cell.items() if k not in ("losses", "step_ms", "tokens_per_step", "kernel_calls_all", "env", "census", "eval_curve", "kernel_calls_per_step")}), flush=True)
    if not c1_ok:
        sys.exit(4)
    return cell


# ----------------------------------------------------------------------------- T8: the CPU self-test (tiny synthetic model, mocked kernels)
class _FakeTok:
    """Byte-level tokenizer stand-in with AutoTokenizer's call shape."""
    def __call__(self, text, truncation=True, max_length=512):
        ids = [1 + (b % 62) for b in text.encode()][:max_length]
        return types.SimpleNamespace(input_ids=ids)


class Params4bit(nn.Parameter):
    """A uint8 parameter whose class NAME is what the census and hashers key on (bitsandbytes' Params4bit)."""
    def __new__(cls, data):
        return super().__new__(cls, data, requires_grad=False)


class _LoRALinear(nn.Module):
    def __init__(self, base, r):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) * 0.1)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t()) @ self.lora_B.t()


class _TinyExperts(nn.Module):
    """Frozen uint8 stacks (the bytes C1 hashes) + a trainable expert LoRA; the forward goes through the mocked kernel
    when `patched` (e4b fused) or through the mocked bnb4bit backend (Unsloth)."""
    def __init__(self, E, H, I, r, mode, param_cls=nn.Parameter):
        super().__init__()
        self.mode, self.E, self.H, self.I = mode, E, H, I
        g = torch.Generator().manual_seed(7)
        self.gate_up_proj = param_cls(torch.randint(0, 255, (E, 2 * I, H), generator=g, dtype=torch.uint8))
        self.down_proj = param_cls(torch.randint(0, 255, (E, H, I), generator=g, dtype=torch.uint8))
        if mode == "e4b":
            self.gate_up_absmax = nn.Parameter(torch.rand(E, 4, generator=g), requires_grad=False)
            self.down_absmax = nn.Parameter(torch.rand(E, 4, generator=g), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(E, r, H) * 0.1)
        self.lora_B = nn.Parameter(torch.zeros(E, H, r))
        self.patched = False

    def _dense(self, x):
        w1 = (self.gate_up_proj.float() / 255.0 - 0.5) * 0.05
        w2 = (self.down_proj.float() / 255.0 - 0.5) * 0.05
        h = torch.einsum("bth,eih->beti", x, w1)
        gate, up = h.chunk(2, dim=-1)
        y = torch.einsum("beti,ehi->beth", torch.nn.functional.silu(gate) * up, w2).mean(1)
        return y

    def forward(self, x):
        if self.mode == "e4b" and self.patched:
            import nf4_qlora
            y = nf4_qlora.fused_grouped_lora(self, x)
        elif self.mode == "unsloth":
            from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as M
            y = M.forward_moe_backend_bnb4bit(self, x)
        else:
            y = self._dense(x)
        lo = torch.einsum("bth,erh->betr", x, self.lora_A)
        return y + torch.einsum("betr,ehr->beth", lo, self.lora_B).mean(1)


class _Wrapper(nn.Module):
    """PEFT ParamWrapper stand-in: the module named `experts` wraps the real one twice (amendment 4's shape)."""
    def __init__(self, inner):
        super().__init__()
        self.base_layer = inner

    def forward(self, x):
        return self.base_layer(x)


class _Layer(nn.Module):
    def __init__(self, H, E, I, r, mode, param_cls, wrap):
        super().__init__()
        self.self_attn = nn.Module()
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, p, _LoRALinear(nn.Linear(H, H, bias=False), r))
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(H, E, bias=False)
        self.mlp.gate.weight.requires_grad_(False)
        ex = _TinyExperts(E, H, I, r, mode, param_cls)
        self.mlp.experts = _Wrapper(_Wrapper(ex)) if wrap else ex

    def forward(self, x):
        a = self.self_attn
        x = x + a.o_proj(a.v_proj(x) * torch.sigmoid(a.q_proj(x) * a.k_proj(x)).mean(-1, keepdim=True))
        return x + self.mlp.experts(x)


class _TinyLM(nn.Module):
    def __init__(self, mode, V=64, H=16, E=4, I=8, L=2, r=2, wrap=False):
        super().__init__()
        param_cls = Params4bit if mode == "unsloth" else (lambda t: nn.Parameter(t, requires_grad=False))
        self.config = types.SimpleNamespace(model_type=f"tiny_{mode}", num_hidden_layers=L, use_cache=False)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(V, H)
        self.model.embed_tokens.weight.requires_grad_(False)
        self.model.layers = nn.ModuleList([_Layer(H, E, I, r, mode, param_cls, wrap) for _ in range(L)])
        self.lm_head = nn.Linear(H, V, bias=False)
        self.lm_head.weight.requires_grad_(False)

    def gradient_checkpointing_enable(self, **k):
        pass

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = self.model.embed_tokens(input_ids)
        for l in self.model.layers:
            x = l(x)
        logits = self.lm_head(x)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1))
        return types.SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items() if "lora" in k}, os.path.join(d, "adapter_model.safetensors"))
        open(os.path.join(d, "adapter_config.json"), "w").write("{}")


def _install_fake_modules():
    nf4 = types.ModuleType("nf4_qlora")
    nf4.fused_grouped_lora = lambda mod, x: mod._dense(x)
    sys.modules["nf4_qlora"] = nf4
    zoo = types.ModuleType("unsloth_zoo")
    tp = types.ModuleType("unsloth_zoo.temporary_patches")
    mb = types.ModuleType("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")
    mb.forward_moe_backend_bnb4bit = lambda mod, x: mod._dense(x)
    mb._moe_uses_bnb4bit_expert_weights = lambda m: type(getattr(m, "gate_up_proj", None)).__name__ == "Params4bit"
    sys.modules["unsloth_zoo"], sys.modules["unsloth_zoo.temporary_patches"], sys.modules["unsloth_zoo.temporary_patches.moe_utils_bnb4bit"] = zoo, tp, mb
    zoo.temporary_patches = tp
    tp.moe_utils_bnb4bit = mb


def _selftest_load_e4b(a):
    m = _TinyLM("e4b")
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [], "probes": {}, "n_layers": 2, "model_type": "tiny_e4b",
         "verify": {"n_quantized": 2, "n_unquantized": 0}, "ckpt_mode": "hf:use_reentrant=False", "hashes": hashes_e4b,
         "fwd_kwargs": lambda t: {}, "tokenizer_obj": _FakeTok(), "snapshot_dir": None, "attn4_probe": attn4_bias_probe(m),
         "structural_expected_n_attn4": None, "detector_version": None}
    if a.attn_4bit:
        x["n_attn4"] = 4 * 2
        x["structural_expected_n_attn4"] = 4 * 2     # T10: simulated census here; the REAL detector is dry-run tested below
        x["detector_version"] = "selftest"
    if a.arm == "fused":
        for l in m.model.layers:
            l.mlp.experts.patched = True
        x["n_patched"], x["reason"] = 2, "[selftest] fused on 2 module(s)"
    elif a.arm == "attn_only":
        for n, p in m.named_parameters():
            if "lora" in n and "experts" in n:
                p.requires_grad_(False)
        x["probes"] = {"fused": {"n_patched": 0, "reason": "[selftest] 0 ExpertsLoRA"}, "batched": {"n_patched": 0, "reason": "[selftest] 0"}}
        refresh_stub(a, "fused_attn4", "fused", "refused", "enable_fast_train(dgrad=True) patched 0 modules on this box: [selftest]", {"probed_by": "attn_only", "n_patched": 0, "n_layers": 2})
    return m, x


def _selftest_load_unsloth(a):
    if a.model == "selftest/refuse":
        raise NotImplementedError("selftest: loader refuses this family")
    m = _TinyLM("unsloth", wrap=True)
    x = {"n_attn4": 0, "n_patched": 0, "reason": "", "banner_lines": [f"Unsloth: Detected MoE model. {BANNER}: ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']"],
         "probes": {}, "n_layers": 2, "model_type": "tiny_unsloth", "verify": {"n_quantized": None, "n_unquantized": None},
         "ckpt_mode": "unsloth" if a.grad_ckpt == "unsloth" else "hf:True (via get_peft_model)", "hashes": hashes_unsloth,
         "fwd_kwargs": lambda t: {"attention_mask": torch.ones_like(t)},
         "tokenizer_obj": _FakeTok(), "snapshot_dir": "/selftest/snapshots/deadbeef",
         "structural_expected_n_attn4": None, "detector_version": None}
    return m, x


def _selftest_detector(d, a):
    """T10 (#434) dry-runs: the REAL structural detector against synthetic stacks, driven through the same
    attn4_census_check the e4b arm uses. CPU-only; the library's package __init__ needs bitsandbytes but lora.py
    itself does not, so on a box without the installed package lora.py is loaded by path from the repo checkout."""
    try:
        from experts4bit_qlora.lora import detect_attention_projections as det
    except Exception:
        import importlib.util
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experts4bit_qlora", "lora.py"))
        spec = importlib.util.spec_from_file_location("e4b_lora_selftest", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod   # dataclass() resolves cls.__module__ through sys.modules
        spec.loader.exec_module(mod)
        det = mod.detect_attention_projections

    def stack(L, keqv=(), drop_k=()):
        m = nn.Module()
        blocks = []
        for i in range(L):
            b = nn.Module()
            b.q_proj = nn.Linear(8, 8, bias=False)
            if i not in drop_k:
                b.k_proj = nn.Linear(8, 8, bias=False)
            b.v_proj = None if i in keqv else nn.Linear(8, 8, bias=False)   # Gemma-4 k_eq_v: transformers sets v_proj = None
            b.o_proj = nn.Linear(8, 8, bias=False)
            blocks.append(b)
        m.layers = nn.ModuleList(blocks)
        return m

    def sim_quantize(model, skip=0):
        c = det(model, exact_linear=True)
        for j, (mod, name) in enumerate(c.candidates):
            if j >= skip:
                setattr(mod, name, nn.Module())   # stands in for Linear4bit: afterwards not exactly nn.Linear
        return len(c.candidates) - skip

    def arm_x(fam, tag):
        a.fam, a.framework, a.arm, a.tag = fam, "e4b", "reference", tag
        return {"n_attn4": 0, "n_layers": None, "model_type": "tiny_det", "attn4_probe": None,
                "structural_expected_n_attn4": None, "detector_version": "selftest"}

    # 1. tiny_keqv_30: 30 layers, 5 K-as-V -> census 115; converting exactly the candidates passes;
    #    a simulated 114-conversion fires void_attn4 naming 115.
    x = arm_x("tcdet", "keqv30")
    attn4_census_check(a, stack(30, keqv={0, 6, 12, 18, 24}), x, det, sim_quantize)
    assert x["structural_expected_n_attn4"] == 115 and x["n_attn4"] == 115, x
    x = arm_x("tcdet", "keqv30_drift")
    try:
        attn4_census_check(a, stack(30, keqv={0, 6, 12, 18, 24}), x, det, lambda mm: sim_quantize(mm, skip=1))
        raise AssertionError("a 114-conversion did not fire void_attn4")
    except SystemExit as e:
        assert e.code == 3, e.code
    r = json.load(open(os.path.join(d, "tcdet_e4b_keqv30_drift.json")))
    assert r["status"] == "void_attn4" and r["structural_expected_n_attn4"] == 115 and "115" in r["reason"], r

    # 2. tiny_plain_4: an ordinary 4-projection stack -> expected 4 x L; passes (the four-projection families untouched).
    x = arm_x("tcdet", "plain4")
    attn4_census_check(a, stack(4), x, det, sim_quantize)
    assert x["structural_expected_n_attn4"] == 16 == 4 * 4 and x["n_attn4"] == 16, x

    # 3. tiny_missing_k: q/o present but k_proj missing -> a REFUSED row (code 3, the layer named), never void_attn4, never a guess.
    x = arm_x("tcdet", "missing_k")
    try:
        attn4_census_check(a, stack(3, drop_k={1}), x, det, sim_quantize)
        raise AssertionError("a missing k_proj did not refuse")
    except SystemExit as e:
        assert e.code == 3, e.code
    r = json.load(open(os.path.join(d, "tcdet_e4b_missing_k.json")))
    assert r["status"] == "refused" and r["phase"] == "attn4" and r["attn4_refusing_modules"] == ["layers.1"], r
    assert "structural_expected_n_attn4" not in r, r   # the census never completed -- nothing guessed
    return {"tiny_keqv30": 115, "tiny_plain4": 16, "tiny_missing_k": r["attn4_refusing_modules"]}


def selftest(a):
    global DEV
    DEV = "cpu"
    _install_fake_modules()
    import tempfile
    d = tempfile.mkdtemp(prefix="tp3_selftest_")
    os.makedirs(os.path.join(d, "data"))
    rows = [{"instruction": f"Q{i}: describe patient {i}", "output": f"Patient {i} is stable; plan {i % 5} continues. " * 2} for i in range(40)]
    ds = {"train": rows[:32], "eval": rows[32:]}
    dp = os.path.join(d, "data", "ds_selftest.json")
    json.dump(ds, open(dp, "w"))
    a.fam, a.model, a.revision, a.data, a.data_sha = "tiny", "selftest/tiny", "0" * 40, dp, sha_bytes(open(dp, "rb").read())
    a.seq, a.steps, a.eval_every, a.eval_n, a.accum, a.autocast, a.lr, a.r, a.alpha = 64, 12, 4, 6, a.accum, a.autocast, 1e-3, 2, 4
    a.out, a.adapter_dir, a.tokens = d, os.path.join(d, "adapters"), os.path.join(d, "tokens_tiny.json")
    rec = prepare(a, tok=_FakeTok())
    a.tokens_sha = rec["sha256"]
    got = {}
    for fw, arm, tag, kw in (("e4b", "reference", "reference_attn4", {"attn_4bit": 1}), ("e4b", "fused", "fused_attn4", {"attn_4bit": 1}),
                             ("unsloth", "unsloth", "ckpt_unsloth", {"attn_4bit": 0}), ("unsloth", "unsloth", "ckpt_hf", {"attn_4bit": 0, "grad_ckpt": "hf"})):
        a.framework, a.arm, a.tag, a.attn_4bit, a.grad_ckpt = fw, arm, tag, kw["attn_4bit"], kw.get("grad_ckpt", "unsloth")
        a.expect_trainable = got.get("e4b_reference_attn4", {}).get("trainable_params") if fw == "unsloth" else None
        got[f"{fw}_{tag}"] = run_arm(a, _selftest_load_e4b if fw == "e4b" else _selftest_load_unsloth, sampler=False)
    # gpt-oss shape (its own family, its own tokens file): attn_only over bare experts (expert LoRA frozen), with the
    # fused_attn4 refusal stub written first by "the run script" (the tp1 citation), refreshed by the arm's own probe
    a.fam, a.tokens = "tinygo", os.path.join(d, "tokens_tinygo.json")
    rec_go = prepare(a, tok=_FakeTok())
    a.tokens_sha = rec_go["sha256"]
    a.framework, a.arm, a.tag, a.attn_4bit, a.expect_trainable = "e4b", "attn_only", "attn_only", 0, None
    stub(a, "refused", "SKIPPED as REFUSED: tp1 row cited (selftest)", {"cited": "tp1"}, fw="e4b", tag="fused_attn4", arm="fused")
    got["e4b_attn_only"] = run_arm(a, _selftest_load_e4b, sampler=False)
    a.fam, a.tokens, a.tokens_sha = "tiny", os.path.join(d, "tokens_tiny.json"), rec["sha256"]
    # a loader refusal -> refused stub, exit 3 (caught here)
    a.framework, a.arm, a.tag, a.model = "unsloth", "unsloth", "ckpt_unsloth_refuse", "selftest/refuse"
    try:
        run_arm(a, _selftest_load_unsloth, sampler=False)
        raise AssertionError("refusal did not exit")
    except SystemExit as e:
        assert e.code == 3, e.code
    a.model = "selftest/tiny"
    # a tokens-sha mismatch -> exit 13
    a.tokens_sha = "f" * 64
    a.tag = "ckpt_unsloth_badsha"
    try:
        run_arm(a, _selftest_load_unsloth, sampler=False)
        raise AssertionError("tokens mismatch did not exit")
    except SystemExit as e:
        assert e.code == 13, e.code
    a.tokens_sha = rec["sha256"]

    # ---- assertions on the receipts
    R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(d, "tiny*_*.json"))}
    need = ["s_per_step_median_11plus", "peak_vram_gb", "tokens_per_s", "joules_per_step", "losses", "eval_curve", "eval_loss_final",
            "kernel_calls_per_step_min", "n_patched", "n_attn4", "census", "unsloth_bnb4bit_modules", "C1_bit_exact", "C1_bytes_hashed",
            "C1_empties_skipped", "C1_control_detects_flipped_byte", "adapter", "env", "prereg", "init_sha", "trainable_params", "n_layers", "accum", "autocast",
            "structural_expected_n_attn4", "detector_version"]
    for k in ("tiny_e4b_reference_attn4", "tiny_e4b_fused_attn4", "tiny_unsloth_ckpt_unsloth", "tiny_unsloth_ckpt_hf", "tinygo_e4b_attn_only"):
        r = R[k]
        assert r["status"] == "ok", (k, r.get("status"), r.get("reason"))
        for f in need:
            assert f in r, (k, f)
        assert r["C1_bit_exact"] and r["C1_bytes_hashed"] > 0 and r["C1_empties_skipped"] == 0
        assert len(r["losses"]) == a.steps and r["eval_curve"][-1]["step"] == a.steps and len(r["eval_curve"]) == 1 + a.steps // a.eval_every
        assert r["adapter"].get("bytes", 0) > 0 and r["adapter"].get("dtypes"), r["adapter"]
        assert r["tokens"]["sha256"] == (rec_go if k.startswith("tinygo") else rec)["sha256"] and r["tokens_total"] == sum(r["tokens_per_step"])
        assert all(t == sum(len(rec["train"][(i * a.accum + j) % len(rec["train"])]) for j in range(a.accum)) for i, t in enumerate(r["tokens_per_step"]))
    e_ref, e_fu, u1, u2, ao = R["tiny_e4b_reference_attn4"], R["tiny_e4b_fused_attn4"], R["tiny_unsloth_ckpt_unsloth"], R["tiny_unsloth_ckpt_hf"], R["tinygo_e4b_attn_only"]
    assert e_ref["init_sha"] == e_fu["init_sha"], "e4b arms did not start bit-identical"
    assert e_ref["n_attn4"] == 8 and e_fu["n_attn4"] == 8 and e_fu["n_patched"] == 2 and e_ref["n_patched"] == 0
    assert e_ref["structural_expected_n_attn4"] == 8 and e_fu["structural_expected_n_attn4"] == 8 and e_ref["detector_version"] == "selftest"
    assert u1["structural_expected_n_attn4"] is None and u1["detector_version"] is None and ao["detector_version"] is None
    assert e_fu["kernel_calls_per_step_min"] == 2 * a.accum and e_ref["kernel_calls_per_step_min"] == 0, (e_fu["kernel_calls_per_step_min"], e_ref["kernel_calls_per_step_min"])
    assert e_fu["kernel_counter_key"] == "fused_grouped_lora"
    assert u1["kernel_counter_key"] == "moe_bnb4bit_backend" and u1["kernel_calls_per_step_min"] == 2 * a.accum and u1["experts_forward_calls_per_step_min"] == 2 * a.accum
    assert u1["census"]["Params4bit_expert_stacks"] == 4 and u1["unsloth_bnb4bit_modules"]["n_bnb4bit"] == 0 and u1["unsloth_bnb4bit_modules"]["n_bnb4bit_unwrapped"] == 2
    assert u1["unsloth_bnb4bit_modules"]["wrap_depth"] == 2 and u1["unsloth_bnb4bit_modules"]["inner_param_types"] == ["Params4bit"]
    assert any(BANNER in s for s in u1["engagement_banners"])
    assert u1["init_sha"] == u2["init_sha"] and u1["grad_ckpt"] == "unsloth" and u2["grad_ckpt"].startswith("hf")
    assert u1["trainable_params"] == e_ref["trainable_params"] and u1["expect_trainable"] == e_ref["trainable_params"] and u1["trainable_mismatch"] is None
    assert e_ref["trainable_by_group"]["experts"] > 0 and ao["trainable_by_group"]["experts"] == 0 and ao["trainable_params"] < e_ref["trainable_params"]
    assert ao["probes"]["fused"]["n_patched"] == 0
    st = R["tinygo_e4b_fused_attn4"]   # the run-script stub (tp1 citation) refreshed by the attn_only arm's own probe
    assert st["status"] == "refused" and st["cited"] == "tp1" and st["probed_by"] == "attn_only" and "patched 0 modules on this box" in st["probe_reason"], st
    assert R["tiny_e4b_fused_attn4"]["status"] == "ok" and "probe_reason" not in R["tiny_e4b_fused_attn4"], "a receipt that trained must never be touched by refresh_stub"
    assert "tinygo_e4b_reference_attn4" not in R, "no bias on the tiny attention -> no reference_attn4 refusal stub"
    rf = R["tiny_unsloth_ckpt_unsloth_refuse"]
    assert rf["status"] == "refused" and rf["phase"] == "load" and "NotImplementedError" in rf["reason"]
    bs = R["tiny_unsloth_ckpt_unsloth_badsha"]
    assert bs["status"] == "tokens_mismatch"
    assert R["tiny_e4b_fused_attn4"]["accum"] == a.accum and R["tiny_e4b_fused_attn4"]["autocast"] == bool(a.autocast)
    # T6: a trainable mismatch is recorded, not a stub
    a.framework, a.arm, a.tag, a.expect_trainable = "unsloth", "unsloth", "ckpt_unsloth_mismatch", e_ref["trainable_params"] + 1
    r = run_arm(a, _selftest_load_unsloth, sampler=False)
    assert r["status"] == "ok" and r["trainable_mismatch"]["expected"] == e_ref["trainable_params"] + 1 and r["trainable_mismatch"]["got"] == e_ref["trainable_params"]
    # --prereg wiring (#434 follow-up): a P41 run's receipts name its own pre-registration, never P40's by accident
    assert e_ref["prereg"] == PREREG and r["prereg"] == PREREG   # the default cites tp2's document (byte-for-byte behaviour)
    a.framework, a.arm, a.tag, a.prereg, a.expect_trainable = "unsloth", "unsloth", "ckpt_unsloth_prereg", "p41/P41-PREREG.md", None
    r = run_arm(a, _selftest_load_unsloth, sampler=False)
    assert r["prereg"] == "p41/P41-PREREG.md", r["prereg"]
    a.prereg = PREREG
    det = _selftest_detector(d, a)   # T10: the three dry-run tests against the REAL structural detector (#434)
    print(f"SELFTEST OK dir={d} receipts={sorted(R)} e4b ref/fused loss_last {e_ref['loss_last']}/{e_fu['loss_last']} unsloth {u1['loss_last']} "
          f"accum={a.accum} autocast={a.autocast} kcalls fused={e_fu['kernel_calls_per_step_min']} unsloth={u1['kernel_calls_per_step_min']} "
          f"detector_dryruns={det}")
    return d


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="T8: CPU, tiny synthetic model, both branches, mocked kernels; + T10 detector dry-runs (#434)")
    ap.add_argument("--framework", choices=["e4b", "unsloth"], default="e4b")
    ap.add_argument("--arm", choices=["reference", "fused", "attn_only", "unsloth"], default="fused")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fam", default="qwen3")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--accum", type=int, default=4, help="T5: micro-batches (batch 1 each) per optimizer step (P40: 4)")
    ap.add_argument("--autocast", type=int, default=1, help="T5: torch.autocast bf16 around forward+loss (P40: bf16 autocast)")
    ap.add_argument("--offload", type=int, default=0, help="e4b: expert offload to pinned host RAM (tp1: Mixtral only)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--data-sha", default=None)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--tokens-sha", default=None)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn-4bit", type=int, default=0, help="e4b: quantize_attention_projections_4bit before the attention LoRA (U4)")
    ap.add_argument("--grad-ckpt", choices=["unsloth", "hf"], default="unsloth", help="Unsloth: use_gradient_checkpointing mode (U1)")
    ap.add_argument("--unsloth-loader", choices=["FastLanguageModel", "FastModel"], default="FastLanguageModel", help="T4: P38's loader; FastModel is an amendment")
    ap.add_argument("--expect-trainable", type=int, default=None, help="T6: the family's e4b trainable count; a mismatch is recorded")
    ap.add_argument("--prereg", default=PREREG, help="the governing pre-registration path written into every receipt and stub; a P41 run MUST pass p41/P41-PREREG.md (default keeps tp2's byte-for-byte behaviour)")
    ap.add_argument("--no-sampler", type=int, default=0)
    ap.add_argument("--out", default="/root/tp2")
    ap.add_argument("--adapter-dir", default="/root/tp2/adapters")
    a = ap.parse_args()
    a.tag = a.tag or a.arm
    if a.selftest:
        return selftest(a)
    if a.prepare:
        return prepare(a)
    if not torch.cuda.is_available():
        stub(a, "harness_error", "torch.cuda.is_available() is False on a GPU lane", code=10)
    return run_arm(a, load_e4b if a.framework == "e4b" else load_unsloth, sampler=not a.no_sampler)


if __name__ == "__main__":
    main()
