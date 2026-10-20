"""DQ11 MLP-adapter BUILD Phase 1 on the RTX A2000 ($0, correctness only): the runner over the real L fixture
(``DQ11-MLP-BUILD-PREREG.md``; Amendments 1-3).

Each part runs in its own process (``--part``) and writes one JSON record into ``--out``:

- ``reads``: the per-module read (``dq11_mlp_phase1.read_layers``) at Amendment 1's ruled state, layers 0, 15 and 31.
- ``reported``: the same read after two AdamW updates of the L path. It is REPORTED and never ruled.
- ``step-today`` / ``step-off`` / ``step-fused``: one update of the L path, run three ways:
  - with the engine never imported;
  - with it imported and the knob off;
  - with the knob on.
- ``k8-unfused`` / ``k8-fused``: ``K8_UPDATES`` updates, scored on both texts at the start and the end.
- ``stage-check``: every model file against ``model_files.json`` (name, size, sha256), the sealed inputs against
  ``locked_inputs.json``, and the runtime. It runs first; every other part re-checks the model files' sizes.
- ``reduce``: reads the records and writes ``phase1.json`` with the outcome.

The workspace (``--directory``) has the DQ11 box layout:
- ``hf-cache/model`` at the registered revision;
- ``data/tokens.json`` and ``adapters/adapter_init.safetensors``, checked against ``locked_inputs.json``;
- ``wheels.json``, a copy of ``rehearsal/wheels.json``, installed by ``dq11_bootstrap.py`` with ``--cache``.

Every part refuses unless:
- the bus claim on the A2000 is confirmed (``DQ11_GPU_CLAIM_CONFIRMED=1``);
- the device is the A2000 at sm86 / cu128;
- PEFT, transformers, bitsandbytes, accelerate and torch are the lock's (Amendment 3).

It records each of the five's installed version, its locked wheel's sha256, and whether the installed files are that
wheel's.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import os
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCHEMA_A3 = "dq11-mlp-phase1/1"      # the -1 read's records (Amendments 1-3)
LAYERS = (0, 15, 31)
ADAPTERS = 96
REPORTED_UPDATES = 2
K8_UPDATES = 8          # Amendment 3: the short K8 run
K8_BAND = 0.05          # K8's absolute PPL band, unchanged
LOCKED = ("peft", "transformers", "bitsandbytes", "accelerate", "torch")
#: The -1 read's parts and record schema (Amendments 1-3); ``reduce`` still rebuilds that read from them.
PARTS_A3 = ("stage-check", "reads", "reported", "step-today", "step-off", "step-fused", "k8-unfused", "k8-fused")
STEPS_A3 = ("step-today", "step-off", "step-fused")
#: Amendment 4's read (dq11-mlp-phase1-a2000-2): three today steps per environment, the C parts only when the read
#: environment's null fails (the driver asks ``--part need-c``).
SCHEMA = "dq11-mlp-phase1/2"
TODAY_READ = ("step-today", "step-today-2", "step-today-3")
TODAY_C = ("step-today-c", "step-today-c2", "step-today-c3")
C_PARTS = TODAY_C + ("step-off-c",)
PARTS = (("stage-check", "reads", "reported") + TODAY_READ + ("step-off", "step-fused") + C_PARTS
         + ("k8-unfused", "k8-fused"))
STEPS = TODAY_READ + ("step-off", "step-fused") + C_PARTS
SPREAD_MARGIN = 1.0     # Amendment 4 (c) 3: off within 2x the null's largest spread
SPREAD_PAIRS_MAX_RATIO = 2.0


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _records(text) -> set:
    """``{(path, hash)}`` of a RECORD's hashed ``.py`` entries. RECORD is CSV: torch's wheel quotes every path, and pip
    writes the installed copy unquoted, so it is parsed as CSV, never split by hand."""
    out = set()
    for row in csv.reader(text.splitlines()):
        if len(row) == 3 and row[0].endswith(".py") and row[1]:
            out.add((row[0], row[1]))
    return out


def wheel_identity(name, row, cache) -> dict:
    """The installed distribution against its locked wheel: the version, the cached wheel's sha256 (which must be the
    lock's), and whether every hashed ``.py`` entry the wheel's RECORD lists is installed with that hash."""
    installed = md.version(name)
    wheel = Path(cache) / row["filename"]
    if not wheel.is_file():
        raise RuntimeError(f"PHASE1_REFUSAL: the locked wheel {row['filename']} is not in the install cache")
    sha = _sha(wheel)
    with zipfile.ZipFile(wheel) as archive:
        record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        locked = _records(archive.read(record).decode())
    present = _records(md.distribution(name).read_text("RECORD") or "")
    return {"installed": installed, "locked": row["version"], "lock_sha256": row["sha256"], "wheel_sha256": sha,
            "files_match_wheel": bool(locked) and locked <= present}


def verify_model(directory, *, hashes) -> dict:
    """The snapshot against ``model_files.json``: every registered file present at its size, and with ``hashes`` its
    sha256 too (the weights' are the hub's LFS hashes)."""
    manifest = json.loads((HERE / "model_files.json").read_text())
    root = Path(directory) / "hf-cache/model"
    bad = []
    for row in manifest["files"]:
        path = root / row["name"]
        if not path.is_file() or path.stat().st_size != row["size"] or (hashes and _sha(path) != row["sha256"]):
            bad.append(row["name"])
    if bad:
        raise RuntimeError(f"PHASE1_REFUSAL: the model snapshot differs from model_files.json: {bad}")
    return {"model": manifest["model"], "revision": manifest["revision"], "files": len(manifest["files"]),
            "hashed": hashes, "manifest_sha256": _sha(HERE / "model_files.json")}


def require_runtime(directory, cache) -> dict:
    """Refuse anything but the bus-claimed A2000 under the rehearsal lock; return the runtime record."""
    if os.environ.get("DQ11_GPU_CLAIM_CONFIRMED") != "1":
        raise RuntimeError("PHASE1_REFUSAL: the gpu:a2000 bus claim is not confirmed")
    lock_path = HERE / "rehearsal" / "wheels.json"
    if _sha(Path(directory) / "wheels.json") != _sha(lock_path):
        raise RuntimeError("PHASE1_REFUSAL: the workspace's wheels.json is not the A2000 rehearsal lock")
    rows = {r["name"].lower(): r for r in json.loads(lock_path.read_text())["packages"]}
    wheels = {name: wheel_identity(name, rows[name], cache) for name in LOCKED}
    bad = [n for n, w in wheels.items() if w["installed"] != w["locked"] or w["wheel_sha256"] != w["lock_sha256"]
           or not w["files_match_wheel"]]
    if bad:
        raise RuntimeError(f"PHASE1_REFUSAL: not the lock's install: {bad} ({ {n: wheels[n] for n in bad} })")
    import torch

    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != "NVIDIA RTX A2000 12GB" or torch.cuda.get_device_capability() != (8, 6)
            or torch.version.cuda != "12.8"):
        raise RuntimeError("PHASE1_REFUSAL: Phase 1 runs on the A2000 (sm86, cu128) only")
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    if free < 9 * (1 << 30):
        raise RuntimeError(f"PHASE1_REFUSAL: {free / (1 << 30):.1f} GiB free on the A2000, under 9")
    cap = min(9 * (1 << 30), int(total * 0.75))
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return {"wheels": wheels, "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
            "free_before_bytes": free, "process_cap_bytes": cap, "e4b_sha": os.environ.get("E4B_SHA")}


def build_l(directory):
    """The L fixture as ``dq11_arm.prepare`` builds it, with the canonical initializer installed: NF4 bases streamed with
    train_prefetch, seven FP32 r16/a32 targets, non-reentrant checkpointing, TF32 off, seed 3407.

    Two of ``prepare``'s checks are left out because :func:`require_runtime` covers them: its 5090-only device check (here
    the A2000, sm86 / cu128) and its ``device_count() == 1``. ``prepare``'s ``dq11_rehearsal.prepare_log`` branch is
    rehearsal-only (``DQ11_REHEARSAL=1``) and is not used."""
    import torch
    from dq11_common import install_initial
    from loggetta import Constraints, Workload, plan
    from loggetta.backends import dense, dense_train
    from loggetta.dense_policy import validate_plan
    from loggetta.hardware import probe

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(3407)
    snapshot = Path(directory) / "hf-cache/model"
    workload = Workload(seq_len=2048, micro_batch=1, grad_accum=1, steps=40,
                        optimizer="adamw", learning_rate=2e-4, lr_schedule="constant", warmup_steps=0)
    constraints = Constraints(fixed={"base": "nf4", "placement": "stream", "adapter_dtype": "fp32",
                                     "r": 16, "alpha": 32, "targets": list(dense.ROLES),
                                     "attn_impl": "sdpa", "loss_chunk": 0}, allow_development_executor=True)
    actual = plan(dense.describe(str(snapshot)), probe(), workload, constraints, backends=(dense,))
    if actual.status != "feasible":
        raise RuntimeError("PHASE1_REFUSAL: L admission refused on the A2000: " + actual.render())
    validate_plan(actual)
    prepared = dense_train.prepare(actual)              # not prepare_log: that wrapper is the tiny rehearsal's only
    model = prepared.model
    checkpoints = [m for m in model.modules() if getattr(m, "gradient_checkpointing", False)]
    if not checkpoints or any(getattr(getattr(m, "_gradient_checkpointing_func", None), "keywords", {})
                              .get("use_reentrant") is not False for m in checkpoints):
        raise RuntimeError("PHASE1_REFUSAL: non-reentrant checkpointing did not engage")
    model.train()
    initial = install_initial(model, Path(directory) / "adapters/adapter_init.safetensors")
    return model, prepared.trainable, workload, {"engaged": prepared.report, "initial_sha256": initial}


def decoder_mlps(model) -> dict:
    from experts4bit_qlora.engines.dense_offload import decoder_layers

    layers = [layer for _name, layer in decoder_layers(model)]
    if len(layers) != 32:
        raise RuntimeError(f"PHASE1_REFUSAL: {len(layers)} decoder layers, not Mistral-7B's 32")
    return {i: layers[i] for i in range(len(layers))}


def capture(model, ids, layers) -> tuple:
    """Each checked MLP's input ``x`` and the loss gradient ``gy`` at its output, from one forward and backward of the
    model as it stands (forward hooks on the MLP modules, never on their projections; a recompute is not recaptured)."""
    xs, gys, hooks = {}, {}, []
    for i, layer in layers.items():
        def pre(_m, args, i=i):
            xs.setdefault(i, args[0].detach().clone())

        def post(_m, _args, out, i=i):
            if i not in gys and out.requires_grad:
                out.register_hook(lambda g, i=i: gys.setdefault(i, g.detach().clone()))
        hooks += [layer.mlp.register_forward_pre_hook(pre), layer.mlp.register_forward_hook(post)]
    try:
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
    finally:
        for h in hooks:
            h.remove()
    model.zero_grad(set_to_none=True)
    if set(xs) != set(layers) or set(gys) != set(layers):
        raise RuntimeError("PHASE1_REFUSAL: the capture missed a checked layer")
    return xs, gys, float(loss.detach())


def _adapter_shas(model) -> dict:
    from dq11_common import adapter_slots, tensor_sha

    return {k: tensor_sha(p) for k, p in adapter_slots(model)[0].items()}


def _train(model, trainable, directory, workload, tokens, locked, updates) -> dict:
    from loggetta.backends import dense_train
    from loggetta.backends.experts4bit_train import train_loop
    from loggetta.measure import DriverMemorySampler

    data = SimpleNamespace(blocks=tokens["train"], info={"source": "DQ11 sealed tokens", "tokens": locked["tokens"]["train"]})
    with DriverMemorySampler() as sampler:
        out = train_loop(model, trainable, str(Path(directory) / "hf-cache/model"), replace(workload, steps=updates),
                         sampler, {}, seed=3407, warmup=0, prepared_data=data,
                         frozen_digest=dense_train.frozen_digest, frozen_kind="dense")
    if out.get("status") != "OK":
        raise RuntimeError("PHASE1_REFUSAL: the shipped loop failed its integrity check")
    return {"losses": out["correctness"]["losses"]}


def run_part(part, directory, cache, out=None) -> dict:
    from dq11_arm import read_inputs
    from dq11_proof_policy import process_environment

    record = {"schema": SCHEMA, "part": part, "science_eligible": False}
    record["environment"] = "C" if part in C_PARTS else "read"
    with process_environment("proof" if part in C_PARTS else "read"):
        record["runtime"] = require_runtime(directory, cache)
        record["model_files"] = verify_model(directory, hashes=part == "stage-check")
        if part == "stage-check":
            locked, _tokens = read_inputs(Path(directory))
            record["inputs"] = {"locked_inputs_sha256": _sha(Path(directory) / "locked_inputs.json"),
                                "assets": locked["assets"], "status": "PASS"}
            if _sha(Path(directory) / "locked_inputs.json") != _sha(HERE / "locked_inputs.json"):
                raise RuntimeError("PHASE1_REFUSAL: the workspace's locked_inputs.json is not the repository's")
            return record
        import torch

        if part == "step-fused" or part == "k8-fused":
            os.environ["E4B_DENSE_MLP_LORA_FUSE"] = "1"
        else:
            os.environ.pop("E4B_DENSE_MLP_LORA_FUSE", None)
        locked, tokens = read_inputs(Path(directory))
        model, trainable, workload, built = build_l(directory)
        record["built"] = {"initial_sha256": built["initial_sha256"]}
        layers = decoder_mlps(model)
        ids = torch.tensor([tokens["train"][0]], device="cuda", dtype=torch.long)
        if part in ("reads", "reported"):
            import dq11_mlp_phase1 as p1
            from experts4bit_qlora.engines import dense_lora_fuse as dlf

            if part == "reported":
                record["training"] = _train(model, trainable, directory, workload, tokens, locked, REPORTED_UPDATES)
                record["snapshot_sha256"] = _adapter_shas(model)
            checked = {i: layers[i] for i in LAYERS}
            xs, gys, record["capture_loss"] = capture(model, ids, checked)
            record["read"] = p1.read_layers_a4(
                {i: checked[i].mlp for i in LAYERS}, xs, gys, sigma=(False if part == "reported" else None),
                enable=lambda: dlf.enable_dense_mlp_lora_fuse(model), disable=lambda: dlf.disable_dense_mlp_lora_fuse(model),
                stats=dlf.DENSE_MLP_LORA_STATS, stage=lambda i: checked[i]._dense_offload.stage(),
                evict=lambda i: checked[i]._dense_offload.evict(), adapters=ADAPTERS)
            return record
        stats = None
        if part not in TODAY_READ + TODAY_C:                  # today's steps never import the engine
            from experts4bit_qlora.engines import dense_lora_fuse as dlf

            record["knob_requested"] = dlf.dense_mlp_lora_fuse_requested()
            if record["knob_requested"]:
                record["enabled_adapters"] = dlf.enable_dense_mlp_lora_fuse(model)
                record["refused"] = dict(dlf.DENSE_MLP_LORA_STATS["refused"])
            stats = dlf.DENSE_MLP_LORA_STATS
        if part in STEPS:
            import dq11_mlp_phase1 as p1
            from dq11_stream_witness import train_prefetch_snapshot

            before = train_prefetch_snapshot(model)
            if part in C_PARTS:                               # Amendment 4 (c) 2: DQ11's CONFIGURATION_C
                from dq11_proof_policy import observer_policy

                with observer_policy() as policy, p1.lora_forward_counter(model) as counts:
                    record["training"] = _train(model, trainable, directory, workload, tokens, locked, 1)
                record["policy"] = policy
            else:
                with p1.lora_forward_counter(model) as counts:
                    record["training"] = _train(model, trainable, directory, workload, tokens, locked, 1)
            record["prefetch_before"], record["prefetch_after"] = before, train_prefetch_snapshot(model)
            record["lora_forwards"] = dict(counts)
            record["adapters_sha256"] = _adapter_shas(model)
            if part in C_PARTS and out is not None:            # (c) 3's aggregate reads the B update itself
                from dq11_common import adapter_slots
                from safetensors.torch import save_file

                path = Path(out) / f"{part}-B.safetensors"
                save_file({k: p.detach().float().cpu().contiguous() for k, p in adapter_slots(model)[0].items()
                           if k.endswith(".B")}, str(path))
                record["b_file"] = {"name": path.name, "sha256": _sha(path)}
        else:
            from dq11_common import score

            record["quality_initial"] = score(model, tokens)
            record["training"] = _train(model, trainable, directory, workload, tokens, locked, K8_UPDATES)
            record["quality_final"] = score(model, tokens)
        if stats is not None:
            record["engine_stats"] = {k: stats.get(k) for k in ("fused", "adapters", "calls", "fallback")}
        record["engine_imported"] = "experts4bit_qlora.engines.dense_lora_fuse" in sys.modules
    return record


def _deltas(rec) -> dict:
    out = {}
    for dev, counts in rec["prefetch_after"]["devices"].items():
        before = rec["prefetch_before"]["devices"][dev]
        out[dev] = {k: (v - before[k]) if isinstance(v, int) and isinstance(before.get(k), int) else v
                    for k, v in counts.items()}
    return out


def reduce(records) -> dict:
    """The Phase 1 outcome over the part records. ``PHASE1_PASS`` needs every part's check; any failure is named."""
    fails = []
    stage = records["stage-check"]
    if not (stage["model_files"]["hashed"] and stage["inputs"]["status"] == "PASS"):
        fails.append("STAGE: the stage check did not hash and pass")
    reads = records["reads"]["read"]
    if not reads["outcome"].startswith("PHASE1_PASS"):
        fails.append(reads["outcome"])
    today, off, fused = (records[p] for p in STEPS_A3)
    if today["engine_imported"] or off["knob_requested"] or "enabled_adapters" in off:
        fails.append("KNOB_OFF: the off step engaged or today's step imported the engine")
    if (off["adapters_sha256"] != today["adapters_sha256"]
            or off["training"]["losses"] != today["training"]["losses"]):
        fails.append("KNOB_OFF: the off step is not today's, bit for bit")
    if fused.get("enabled_adapters") != ADAPTERS or fused.get("refused"):
        fails.append(f"ENGAGE_COUNT: {fused.get('enabled_adapters')}")
    if fused["engine_stats"]["fallback"] or not fused["engine_stats"]["calls"]:
        fails.append(f"VOID_FALLBACK: {fused['engine_stats']}")
    mlp = {k: v for k, v in fused["lora_forwards"].items() if k.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj")}
    attn_f = {k: v for k, v in fused["lora_forwards"].items() if k not in mlp}
    attn_t = {k: v for k, v in today["lora_forwards"].items()
              if k.rsplit(".", 1)[-1] not in ("gate_proj", "up_proj", "down_proj")}
    if mlp:
        fails.append(f"ENGAGE_PEFT_MLP_FORWARDS: {sum(mlp.values())}")
    if attn_f != attn_t:
        fails.append("ENGAGE_QKVO_CHANGED")
    peft_mlp = sum(v for k, v in today["lora_forwards"].items()
                   if k.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj"))
    if 3 * fused["engine_stats"]["calls"] != peft_mlp:            # forwards plus recomputes, call for call
        fails.append(f"RECOMPUTE_CHANGED: {fused['engine_stats']['calls']} fused calls vs {peft_mlp} PEFT MLP forwards")
    if _deltas(fused) != _deltas(today):
        fails.append("STREAMING_CHANGED")
    k8 = {}
    for point in ("quality_initial", "quality_final"):
        for text in ("alpaca-heldout", "wikitext-test"):
            d = records["k8-fused"][point][text]["ppl"] - records["k8-unfused"][point][text]["ppl"]
            k8[f"{point}:{text}"] = d
            if not abs(d) <= K8_BAND:
                fails.append(f"K8: {point} {text} {d:+.4f}")
    if records["k8-fused"]["engine_stats"]["fallback"]:
        fails.append("VOID_FALLBACK: k8-fused")
    outcome = reads["outcome"] if not fails else "PHASE1_FAIL: " + "; ".join(fails)
    return {"schema": SCHEMA_A3, "outcome": outcome, "reads_outcome": reads["outcome"], "k8_ppl_deltas": k8,
            "reported_outcome": records["reported"]["read"]["outcome"], "science_eligible": False}


def _same(records) -> bool:
    first = records[0]
    return all(r["adapters_sha256"] == first["adapters_sha256"] and r["training"]["losses"] == first["training"]["losses"]
               for r in records[1:])


def need_c(records) -> bool:
    """Amendment 4 (c): the C parts run only when the read environment's three today steps disagree."""
    return not _same([records[p] for p in TODAY_READ])


def b_distance(path_p, path_q) -> dict:
    """D(P, Q) over every B slot concatenated (Amendment 4 (c) 3): ||B_P - B_Q|| / ||B_Q|| and max |B_P - B_Q|."""
    import torch
    from safetensors.torch import load_file

    a, b = load_file(str(path_p)), load_file(str(path_q))
    if set(a) != set(b) or len(a) != 224:
        raise RuntimeError("PHASE1_REFUSAL: the saved B updates do not hold the same 224 slots")
    keys = sorted(a)
    va = torch.cat([a[k].flatten().double() for k in keys])
    vb = torch.cat([b[k].flatten().double() for k in keys])
    d = va - vb
    return {"rel_rms": float(d.norm() / vb.norm()), "max_abs": float(d.abs().max())}


def knob_off(records, out) -> dict:
    """Amendment 4 (c), in order: the read environment's null and bitwise check; else C's; else C's spread."""
    if _same([records[p] for p in TODAY_READ]):
        off, today = records["step-off"], records["step-today"]
        ok = off["adapters_sha256"] == today["adapters_sha256"] and off["training"]["losses"] == today["training"]["losses"]
        return {"how": "bitwise in the read environment (its null held)", "ok": ok}
    if any(p not in records for p in C_PARTS):
        return {"how": "the read environment's null failed and the C parts are missing", "ok": False}
    if _same([records[p] for p in TODAY_C]):
        off, today = records["step-off-c"], records["step-today-c"]
        ok = off["adapters_sha256"] == today["adapters_sha256"] and off["training"]["losses"] == today["training"]["losses"]
        return {"how": "bitwise under C (the read null failed, C's held)", "ok": ok}
    files = {p: Path(out) / records[p]["b_file"]["name"] for p in C_PARTS}
    for p, f in files.items():
        if _sha(f) != records[p]["b_file"]["sha256"]:
            raise RuntimeError(f"PHASE1_REFUSAL: {f.name} changed since its part wrote it")
    # (c) 3: each of the four processes is scored by its mean relative-RMS distance to the other three (max-abs is
    # recorded only, as in (b)); off fails only if its score exceeds (1 + margin) x the largest today score
    procs = TODAY_C + ("step-off-c",)
    dist = {(a, b): b_distance(files[a], files[b]) for i, a in enumerate(procs) for b in procs[i + 1:]}
    d = lambda a, b: dist[(a, b)] if (a, b) in dist else dist[(b, a)]  # noqa: E731
    score = {p: sum(d(p, q)["rel_rms"] for q in procs if q != p) / (len(procs) - 1) for p in procs}
    null_pairs = [d(a, b)["rel_rms"] for i, a in enumerate(TODAY_C) for b in TODAY_C[i + 1:]]
    spread_ok = max(null_pairs) <= SPREAD_PAIRS_MAX_RATIO * min(null_pairs)
    today_max = max(score[p] for p in TODAY_C)
    out_rec = {"how": "KNOB_OFF_BY_SPREAD under C (both nulls failed)",
               "distances": {f"{a}|{b}": v for (a, b), v in dist.items()}, "scores": score,
               "null_pairs_rel_rms": null_pairs}
    if not spread_ok:
        return dict(out_rec, ok=None, indeterminate="the null's own distances differ by more than 2x")
    return dict(out_rec, ok=score["step-off-c"] <= (1 + SPREAD_MARGIN) * today_max)


def reduce_a4(records, out) -> dict:
    """Amendment 4's outcome for dq11-mlp-phase1-a2000-2: PHASE1_PASS needs (b)'s per-module PASS, KNOB_OFF held, and
    every registered part; INDETERMINATE from (b)'s calibration or (c) 3 is not a FAIL; anything else is PHASE1_FAIL."""
    fails, undecided = [], []
    stage = records["stage-check"]
    if not (stage["model_files"]["hashed"] and stage["inputs"]["status"] == "PASS"):
        fails.append("STAGE: the stage check did not hash and pass")
    reads = records["reads"]["read"]["outcome"]
    if reads.startswith("INDETERMINATE"):
        undecided.append(reads)
    elif not reads.startswith("PHASE1_PASS"):
        fails.append(reads)
    knob = knob_off(records, out)
    if knob["ok"] is None:
        undecided.append("KNOB_OFF: " + knob["indeterminate"])
    elif not knob["ok"]:
        fails.append("KNOB_OFF: " + knob["how"])
    today, fused = records["step-today"], records["step-fused"]
    if fused.get("enabled_adapters") != ADAPTERS or fused.get("refused"):
        fails.append(f"ENGAGE_COUNT: {fused.get('enabled_adapters')}")
    if fused["engine_stats"]["fallback"] or not fused["engine_stats"]["calls"]:
        fails.append(f"VOID_FALLBACK: {fused['engine_stats']}")
    is_mlp = lambda k: k.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj")  # noqa: E731
    if sum(v for k, v in fused["lora_forwards"].items() if is_mlp(k)):
        fails.append("ENGAGE_PEFT_MLP_FORWARDS")
    if ({k: v for k, v in fused["lora_forwards"].items() if not is_mlp(k)}
            != {k: v for k, v in today["lora_forwards"].items() if not is_mlp(k)}):
        fails.append("ENGAGE_QKVO_CHANGED")
    if _deltas(fused) != _deltas(today):
        fails.append("STREAMING_CHANGED")
    if 3 * fused["engine_stats"]["calls"] != sum(v for k, v in today["lora_forwards"].items() if is_mlp(k)):
        fails.append("RECOMPUTE_CHANGED")
    k8 = {}
    for point in ("quality_initial", "quality_final"):
        for text in ("alpaca-heldout", "wikitext-test"):
            d = records["k8-fused"][point][text]["ppl"] - records["k8-unfused"][point][text]["ppl"]
            k8[f"{point}:{text}"] = d
            if not abs(d) <= K8_BAND:
                fails.append(f"K8: {point} {text} {d:+.4f}")
    if records["k8-fused"]["engine_stats"]["fallback"]:
        fails.append("VOID_FALLBACK: k8-fused")
    if fails:
        outcome = "PHASE1_FAIL: " + "; ".join(fails)
    elif undecided:
        outcome = "INDETERMINATE: " + "; ".join(undecided)
    else:
        outcome = reads + (", KNOB_OFF_BY_SPREAD" if knob["how"].startswith("KNOB_OFF_BY_SPREAD") else "")
    return {"schema": SCHEMA, "outcome": outcome, "reads_outcome": reads, "knob_off": knob, "k8_ppl_deltas": k8,
            "reported_outcome": records["reported"]["read"]["outcome"], "science_eligible": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=PARTS + ("reduce", "need-c"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="the wheel cache dq11_bootstrap.py installed from")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part == "need-c":
        records = {p: json.loads((args.out / f"{p}.json").read_text()) for p in TODAY_READ}
        print("yes" if need_c(records) else "no")
        return 0
    if args.part == "reduce":
        first = json.loads((args.out / "stage-check.json").read_text())
        if first["schema"] == SCHEMA_A3:                       # the -1 read (Amendments 1-3)
            records = {p: json.loads((args.out / f"{p}.json").read_text()) for p in PARTS_A3}
            result = reduce(records)
        else:
            records = {p: json.loads((args.out / f"{p}.json").read_text()) for p in PARTS
                       if (args.out / f"{p}.json").exists()}
            result = reduce_a4(records, args.out)
        (args.out / "phase1.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["outcome"])
        return 0 if result["outcome"].startswith("PHASE1_PASS") else 1
    record = run_part(args.part, args.directory, args.cache, args.out)
    (args.out / f"{args.part}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
