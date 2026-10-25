# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""A backend-adapter/2 manifest DERIVED for a backend-adapter/1 fixture, beside a byte copy of its weights (CPU, USD 0).

D0's two fixtures (P224 shake-3) were written as backend-adapter/1: no fingerprints. The merged-arena writer and the
paged server's refusal 13 read /2 only. This tool writes, into a NEW directory, a byte copy of the fixture's
``adapter.safetensors`` (so the adapter id is unchanged) and a /2 manifest whose added fields come from evidence:

- ``expert_absmax``, ``train_fused_qkv``, ``attn_quant``, ``versions``, ``lora_scale_applied`` and ``run_id`` are read
  from the training receipt's ``engaged`` block, as the trainer recorded them. ``lora_scale_applied`` must equal the
  recorded alpha / r ("equal by the reader rule"), and the training code is read to show it applies alpha / r: the
  training commit is the receipt's recorded experts4bit-qlora version, resolved to its release tag in an e4b git
  repository; ``lora.py`` is read from that commit (``git show``, its blob id recorded), and must set
  ``self.scaling = alpha / r`` in both LoRA classes and nowhere else otherwise, with no rslora or DoRA path. The
  recorded setup must carry no scaling option.
- ``fingerprints.experts`` is e4b's own ``quant_fingerprint`` over the STORED trained base (fixture-base:
  ``stacks.safetensors``, plus ``absmax_dq.safetensors`` for a double-quantized one, installed as stored and never
  recomputed, since the compression's statistics depend on the device). Before that, the trainer's own record is
  checked: Loggetta's ``_expert_digest`` over the stored stacks must equal the training receipt's
  ``frozen_expert_digest`` and the P224 fixture_base.json record. That record covers 2 of 16 stacks (the first and
  the last); the other 14 rest on the deterministic rebuild by the same call, setup and revision.
- ``fingerprints.attention`` is qfp-v1 over the checkpoint's bf16 attention at the adapter's revision (``attn_quant``
  bf16, ``attn_4bit`` false), each checkpoint file checked against its LFS object id.
- ``created_at`` is the derivation time given on the command line; the fingerprints' ``seconds`` are 0 (not
  measured). Two runs with the same inputs write byte-identical manifests.
- ``derived_from`` records every input's sha256 and states the limitation: the fingerprints were measured after
  training, over the verified stored base, not recorded by the trainer.

Every link that is missing or does not verify is a refusal (:class:`Refused`); nothing falls back to an unverified
base. The fixture's own directory and receipts are only read.

    python bench/d0s/d0s_v2.py --fixture nodq --v1 DIR --receipt RECEIPT.json --base FIXTURE_BASE_DIR \\
        --base-record fixture_base.json --checkpoint HF_SNAPSHOT --e4b-repo E4B_GIT_CHECKOUT \\
        --created-at 2026-10-24T00:00:00Z --out NEW_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

TOOL = "bench/d0s/d0s_v2.py"
FIXTURES = {"nodq": "fp32", "dq": "double-quantized"}
DIGEST_STACKS = ("stack[0]", "stack[-1]")
WHICH = ("gate_up", "down")
SUFFIXES = ("_q", "_s", "_off", "_code")
LIMITATION = ("fingerprints measured after training, over the verified stored base, not recorded by the trainer; 2 of "
              "16 stacks verified against the trainer digest, the other 14 resting on the deterministic rebuild")
#: the v1 setup keys that must equal the training receipt's
SETUP_KEYS = ("r", "alpha", "quant_type", "blocksize", "attn_4bit", "train_attention", "train_experts")
_SCALING_OPTION = re.compile(r"rslora|dora|scal", re.I)


class Refused(RuntimeError):
    """A link of the derivation is missing or does not verify; nothing is written."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_scaling_code(text: str) -> None:
    """The training commit's lora.py applies alpha / r: ``self.scaling = alpha / r`` once in ExpertsLoRA and once in
    LoRALinear, no other assignment of ``self.scaling``, and no rslora / DoRA / sqrt-of-r path."""
    if re.search(r"rs_?lora|\bdora\b|use_dora|sqrt\(\s*(self\.)?r\s*\)", text, re.I):
        raise Refused("the training lora.py carries an rslora / DoRA / sqrt(r) path: alpha / r is not the scale")
    assignments = re.findall(r"self\.scaling\s*=(?!=)\s*([^\n#]+)", text)
    if [a.strip() for a in assignments] != ["alpha / r", "alpha / r"]:
        raise Refused(f"the training lora.py assigns self.scaling as {assignments}, not alpha / r in exactly its two "
                      f"LoRA classes")
    for cls in ("ExpertsLoRA", "LoRALinear"):
        m = re.search(rf"^class {cls}\b.*?(?=^class |\Z)", text, re.S | re.M)
        if m is None or "self.scaling = alpha / r" not in m.group(0):
            raise Refused(f"the training lora.py's {cls} does not set self.scaling = alpha / r")


def training_lora_py(e4b_repo: str, receipt: dict) -> dict:
    """``{"version", "tag", "commit", "blob", "sha256", "text"}``: the training commit's ``experts4bit_qlora/lora.py``,
    read with git from ``e4b_repo`` at the release tag of the experts4bit-qlora version the training receipt records."""
    version = ((receipt.get("engaged") or {}).get("versions") or {}).get("experts4bit-qlora")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise Refused(f"the training receipt records no experts4bit-qlora release version ({version!r})")
    tag = f"v{version}"

    def git(*args):
        r = subprocess.run(["git", "-C", e4b_repo, *args], capture_output=True, timeout=60)
        if r.returncode:
            raise Refused(f"git {' '.join(args)} in {e4b_repo}: {r.stderr.decode(errors='replace').strip()[:200]}")
        return r.stdout

    commit = git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}").decode().strip()
    blob = git("rev-parse", "--verify", "--quiet", f"{commit}:experts4bit_qlora/lora.py").decode().strip()
    body = git("cat-file", "blob", blob)
    return {"version": version, "tag": tag, "commit": commit, "blob": blob,
            "sha256": hashlib.sha256(body).hexdigest(), "text": body.decode("utf-8")}


def check_receipt(v1, receipt: dict, fixture: str) -> dict:
    """The training receipt's engaged record, checked against the v1 manifest and the fixture's declared form."""
    engaged = receipt.get("engaged")
    if not isinstance(engaged, dict):
        raise Refused("the training receipt has no engaged block")
    setup = engaged.get("setup") or {}
    for key in SETUP_KEYS:
        if v1.setup.get(key) != setup.get(key):
            raise Refused(f"setup {key}: the v1 manifest has {v1.setup.get(key)!r}, the training receipt "
                          f"{setup.get(key)!r}")
    options = sorted(k for k in set(setup) | set(v1.setup) if _SCALING_OPTION.search(k))
    if options:
        raise Refused(f"the recorded setup carries a scaling option {options}: alpha / r cannot be assumed")
    scale = float(setup["alpha"]) / int(setup["r"])
    applied = engaged.get("lora_scale_applied")
    if not isinstance(applied, dict) or any(applied.get(k) != scale for k in ("attention", "experts")):
        raise Refused(f"lora_scale_applied {applied!r} is not the recorded alpha / r {scale!r} for both kinds")
    form = (engaged.get("expert_absmax") or {}).get("form")
    if form != FIXTURES[fixture]:
        raise Refused(f"the training receipt's expert_absmax form is {form!r}; fixture {fixture!r} is "
                      f"{FIXTURES[fixture]!r}")
    if v1.revision != engaged.get("commit") or v1.revision != (receipt.get("model") or {}).get("revision"):
        raise Refused(f"the adapter's revision {v1.revision} is not the training receipt's {engaged.get('commit')}")
    correctness = receipt.get("correctness") or {}
    digest = correctness.get("frozen_expert_digest")
    if correctness.get("frozen_expert_bytes_unchanged") is not True or not isinstance(digest, dict) \
            or sorted(digest) != sorted(DIGEST_STACKS):
        raise Refused("the training receipt records no unchanged frozen-expert digest of stack[0] and stack[-1]")
    for key in ("train_fused_qkv", "attn_quant", "versions"):
        if key not in engaged:
            raise Refused(f"the training receipt's engaged block lacks {key}")
    if not receipt.get("run_id"):
        raise Refused("the training receipt names no run_id")
    return engaged


def build_trained_base(base_dir: str, fixture: str, *, r: int, alpha: int, dims: dict):
    """The trained base as stored: one ExpertsLoRA-wrapped Experts4bit per layer holding fixture-base's codes and
    fp32 absmax, and for a double-quantized base its stored payload installed in place of the fp32 absmax."""
    import torch
    from safetensors import safe_open
    from torch import nn

    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.absmax_dq import install_compressed_absmax_
    from experts4bit_qlora.lora import ExpertsLoRA

    st = safe_open(os.path.join(base_dir, "stacks.safetensors"), "pt")
    keys = set(st.keys())
    n_layers = len({k.split(".", 1)[0] for k in keys})
    if keys != {f"{li}.{w}_{t}" for li in range(n_layers) for w in WHICH for t in ("proj", "absmax")}:
        raise Refused("stacks.safetensors is not one codes + fp32 absmax tensor per layer and projection")
    dq = None
    if fixture == "dq":
        path = os.path.join(base_dir, "absmax_dq.safetensors")
        if not os.path.isfile(path):
            raise Refused("a double-quantized fixture needs fixture-base's absmax_dq.safetensors")
        dq = safe_open(path, "pt")
        if set(dq.keys()) != {f"{li}.{w}_absmax{x}" for li in range(n_layers) for w in WHICH for x in SUFFIXES}:
            raise Refused("absmax_dq.safetensors is not one payload of four tensors per layer and projection")
    layers = nn.ModuleList()
    for li in range(n_layers):
        base = Experts4bit(dims["num_experts"], dims["hidden"], dims["intermediate"], quant_type="nf4", blocksize=64)
        with torch.no_grad():
            for w in WHICH:
                for t, attr in (("proj", f"{w}_proj"), ("absmax", f"{w}_absmax")):
                    src, dst = st.get_tensor(f"{li}.{w}_{t}"), getattr(base, attr)
                    if src.dtype != dst.dtype or src.numel() != dst.numel():
                        raise Refused(f"stored {li}.{w}_{t} is {src.dtype} x {src.numel()}, the stack's {dst.dtype} "
                                      f"x {dst.numel()}")
                    dst.copy_(src.reshape(dst.shape))
        if dq is not None:
            install_compressed_absmax_(base, {w: {x: dq.get_tensor(f"{li}.{w}_absmax{x}") for x in SUFFIXES}
                                              for w in WHICH})
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = ExpertsLoRA(base, r=r, alpha=alpha)
        layers.append(layer)
    return layers


def attach_attention(layers, checkpoint: str) -> dict:
    """The checkpoint's bf16 q/k/v/o projections, as nn.Linear under each layer's self_attn; ``{file: sha256}`` of
    every checkpoint file read, each equal to its LFS object id when the snapshot is an HF cache entry."""
    import torch
    from safetensors import safe_open
    from torch import nn

    index = os.path.join(checkpoint, "model.safetensors.index.json")
    weight_map = (_json(index)["weight_map"] if os.path.isfile(index)
                  else {k: "model.safetensors" for k in safe_open(os.path.join(checkpoint, "model.safetensors"),
                                                                    "pt").keys()})
    files, handles = {}, {}
    for li, layer in enumerate(layers):
        layer.self_attn = nn.Module()
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = f"model.layers.{li}.self_attn.{proj}.weight"
            if name not in weight_map:
                raise Refused(f"the checkpoint has no {name}")
            fname = weight_map[name]
            if fname not in handles:
                path = os.path.join(checkpoint, fname)
                files[fname] = sha256_file(path)
                lfs = os.path.basename(os.path.realpath(path))
                if os.path.islink(path) and re.fullmatch(r"[0-9a-f]{64}", lfs) and lfs != files[fname]:
                    raise Refused(f"{fname}: sha256 {files[fname][:16]} is not its LFS object id {lfs[:16]}")
                handles[fname] = safe_open(path, "pt")
            w = handles[fname].get_tensor(name)
            if w.dtype != torch.bfloat16:
                raise Refused(f"{name} is {w.dtype}; the attention base is bf16")
            lin = nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.bfloat16)
            lin.weight.data = w
            setattr(layer.self_attn, proj, lin)
    return files


def derive(*, fixture: str, v1_dir: str, receipt_path: str, base_dir: str, base_record: str, checkpoint: str,
           e4b_repo: str, created_at: str, out: str, expert_digest=None) -> dict:
    """Write the derived /2 adapter directory ``out``; return its manifest. ``expert_digest`` is Loggetta's
    ``_expert_digest`` (the trainer's function; a test may pass the same computation)."""
    from torch import nn

    from experts4bit_qlora import adapter_schema as S
    from experts4bit_qlora.adapter_dir import SCHEMA_V1, SCHEMA_V2, verify_adapter_dir

    if fixture not in FIXTURES:
        raise Refused(f"fixture must be one of {sorted(FIXTURES)}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created_at or ""):
        raise Refused("created_at must be a UTC time as YYYY-MM-DDTHH:MM:SSZ")
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the derived directory is always new")
    if expert_digest is None:
        from loggetta.backends.experts4bit_train import _expert_digest as expert_digest
    v1 = verify_adapter_dir(v1_dir)
    if v1.manifest.get("schema") != SCHEMA_V1:
        raise Refused(f"{v1_dir} is {v1.manifest.get('schema')}, not {SCHEMA_V1}")
    receipt = _json(receipt_path)
    engaged = check_receipt(v1, receipt, fixture)
    lora = training_lora_py(e4b_repo, receipt)
    check_scaling_code(lora["text"])
    record = _json(base_record)
    want = receipt["correctness"]["frozen_expert_digest"]
    if record.get(f"{fixture}_ok") is not True or record.get(f"got_{fixture}") != want \
            or (record.get("want") or {}).get(fixture) != want:
        raise Refused(f"fixture_base.json does not record fixture-base's {fixture} digest equal to the trainer's")
    cfg = _json(os.path.join(checkpoint, "config.json"))
    dims = {"num_experts": int(cfg["num_experts"]), "hidden": int(cfg["hidden_size"]),
            "intermediate": int(cfg.get("moe_intermediate_size") or cfg["intermediate_size"])}
    layers = build_trained_base(base_dir, fixture, r=int(v1.setup["r"]), alpha=v1.setup["alpha"], dims=dims)
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = layers
    got = expert_digest(model, layers=(0, -1))
    if got != want:
        raise Refused(f"the stored base's digest {got} is not the trainer's {want}: it is not the trained base")
    checkpoint_files = attach_attention(layers, checkpoint)
    experts = S.quant_fingerprint(model, "experts")
    if fixture == "dq" and not isinstance(experts.get("fp32_source"), dict):
        raise Refused("the double-quantized base's fingerprint carries no fp32_source")
    if fixture == "nodq" and "fp32_source" in experts:
        raise Refused("an fp32 base's fingerprint carries an fp32_source")
    attention = S.quant_fingerprint(model, "attention")
    for fp in (experts, attention):
        fp["seconds"] = 0.0
    man = json.loads(json.dumps(v1.manifest))
    base_files = {"stacks.safetensors": sha256_file(os.path.join(base_dir, "stacks.safetensors"))}
    if fixture == "dq":
        base_files["absmax_dq.safetensors"] = sha256_file(os.path.join(base_dir, "absmax_dq.safetensors"))
    man.update({
        "schema": SCHEMA_V2,
        "expert_absmax": engaged["expert_absmax"], "train_fused_qkv": engaged["train_fused_qkv"],
        "attn_quant": engaged["attn_quant"], "versions": engaged["versions"],
        "lora_scale_applied": engaged["lora_scale_applied"],
        "fingerprints": {"experts": experts, "attention": attention},
        "run_id": receipt["run_id"], "created_at": created_at,
        "derived_from": {
            "tool": TOOL, "tool_sha256": sha256_file(os.path.abspath(__file__)),
            "v1_manifest_sha256": v1.manifest_sha256, "adapter_id": v1.adapter_id,
            "training_receipt_sha256": sha256_file(receipt_path), "fixture_base_record_sha256": sha256_file(base_record),
            "fixture_base_files": base_files, "checkpoint_files": checkpoint_files,
            "lora_py": {k: lora[k] for k in ("version", "tag", "commit", "blob", "sha256")},
            "digest_verified_stacks": list(DIGEST_STACKS), "fingerprint_seconds": "not measured (0)",
            "scale": "equal by the reader rule; the training code applies alpha/r",
            "limitation": LIMITATION},
    })
    tmp = out + ".tmp"
    if os.path.lexists(tmp):
        shutil.rmtree(tmp)
    os.mkdir(tmp)
    try:
        # every file the v1 manifest lists (the weights, and the tokenizer files a Loggetta adapter carries), each a
        # byte copy checked against the sha256 and size the manifest records; the weights' sha256 is the adapter id
        for name, info in v1.manifest["files"].items():
            dst = os.path.join(tmp, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(v1.path, name), dst, follow_symlinks=True)
            if os.path.islink(dst) or sha256_file(dst) != info["sha256"] or os.path.getsize(dst) != info["bytes"]:
                raise Refused(f"the copied {name} is not a byte copy of the fixture's")
        if sha256_file(os.path.join(tmp, "adapter.safetensors")) != v1.adapter_id:
            raise Refused("the copied adapter.safetensors is not the adapter id's bytes")
        with open(os.path.join(tmp, "adapter_manifest.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(man, indent=1, sort_keys=True) + "\n")
        v2 = verify_adapter_dir(tmp)
        if v2.manifest.get("schema") != SCHEMA_V2 or v2.adapter_id != v1.adapter_id:
            raise Refused("the derived directory does not verify as the same adapter under backend-adapter/2")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    os.rename(tmp, out)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fixture", required=True, choices=sorted(FIXTURES))
    for name in ("v1", "receipt", "base", "base-record", "checkpoint", "e4b-repo", "created-at", "out"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args(argv)
    try:
        man = derive(fixture=a.fixture, v1_dir=a.v1, receipt_path=a.receipt, base_dir=a.base, base_record=a.base_record,
                     checkpoint=a.checkpoint, e4b_repo=a.e4b_repo, created_at=a.created_at, out=a.out)
    except Refused as e:
        print(f"D0S_V2 REFUSED: {e}", file=sys.stderr)
        return 2
    except ValueError as e:                       # the verifier's refusal 7 (AdapterRefused is a ValueError)
        print(f"D0S_V2 REFUSED (verifier): {e}", file=sys.stderr)
        return 2
    fps = man["fingerprints"]
    print(json.dumps({"out": a.out, "adapter_id": man["derived_from"]["adapter_id"],
                      "experts_qfp_v1": fps["experts"]["sha256"],
                      "experts_fp32_source": (fps["experts"].get("fp32_source") or {}).get("sha256"),
                      "attention_qfp_v1": fps["attention"]["sha256"],
                      "manifest_sha256": sha256_file(os.path.join(a.out, "adapter_manifest.json"))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
