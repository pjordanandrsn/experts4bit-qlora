"""Opt-in A2000 correctness rehearsal; never a registered DQ11 scientific draw.

The original arm's proof, census, observer, scoring and training-loop code runs
unchanged. Only the hardware, cu128 lock and local random model/inputs differ.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
MODEL_CONFIG = {"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 32,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
                "vocab_size": 32000, "max_position_embeddings": 4096, "sliding_window": 4096,
                "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
                "tie_word_embeddings": False}
SCHEMA = "dq11-rehearsal/1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def mode(directory):
    """Overrides require both an explicit mode and a separately sealed workspace."""
    flags = {key: value for key, value in os.environ.items() if key.startswith("DQ11_REHEARSAL") and value}
    if not flags:
        return False
    if os.environ.get("DQ11_REHEARSAL") != "1" or os.environ.get("DQ11_REHEARSAL_TINY_MODEL") != "1":
        raise ValueError("tiny model/lock override refused in science mode")
    marker = json.loads((Path(directory) / "REHEARSAL.json").read_text())
    wheels = Path(directory) / "wheels.json"
    if (marker.get("schema") != SCHEMA or marker.get("science_eligible") is not False
            or marker.get("model_config") != MODEL_CONFIG or marker["wheel_lock_sha256"] != digest(wheels)):
        raise ValueError("rehearsal marker/lock changed")
    lock = json.loads(wheels.read_text())
    versions = {row["name"]: row["version"] for row in lock["packages"]}
    if (lock.get("schema") != "dq11-rehearsal-wheels/1" or lock.get("science_eligible") is not False
            or versions.get("torch") != "2.11.0+cu128" or versions.get("triton") != "3.6.0"):
        raise ValueError("unregistered rehearsal runtime")
    return True


def require_hardware(directory):
    if not mode(directory):
        raise ValueError("A2000 override requires explicit rehearsal mode")
    import torch

    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name() != "NVIDIA RTX A2000 12GB" or torch.version.cuda != "12.8"):
        raise ValueError("rehearsal requires the real single A2000 and CUDA 12.8")
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    if not 11 * (1 << 30) <= total <= 13 * (1 << 30) or free < 6 * (1 << 30):
        raise ValueError("A2000 identity/free-memory rehearsal guard refused")
    # Bound only this process. Never fill the device or disturb household users.
    torch.cuda.set_per_process_memory_fraction(min(0.25, 3 * (1 << 30) / total))


def prepare_log(directory, plan, prepare):
    """Scope the real offloader's tiny streaming threshold to real Log prepare.

    Science keeps its 1 MiB default. Tiny projections must stream too, rather
    than leaving every layer resident and failing Log's engagement assertion.
    The real function is delegated to; no model, proof or loop is substituted.
    """
    if not mode(directory):
        raise ValueError("tiny streaming adapter refused in science mode")
    from experts4bit_qlora.engines import dense_offload

    original = dense_offload.enable_dense_offload
    invocations = []

    @functools.wraps(original)
    def tiny_stream(*args, **kwargs):
        if "min_bytes" in kwargs and kwargs["min_bytes"] != 0:
            raise ValueError("conflicting tiny rehearsal streaming threshold")
        handles = original(*args, **dict(kwargs, min_bytes=0))
        streamed_bytes = sum(handle.bytes for handle in handles)
        if not handles or streamed_bytes <= 0:
            raise ValueError("tiny rehearsal offloader returned no streamed bytes")
        invocations.append({"min_bytes": 0, "handles": len(handles), "streamed_bytes": streamed_bytes})
        return handles

    # Pinned Loggetta imports this defining module's attribute inside prepare.
    dense_offload.enable_dense_offload = tiny_stream
    try:
        prepared = prepare(plan)
        if not invocations:
            raise ValueError("tiny rehearsal streaming adapter was not invoked")
        prepared.report["rehearsal_tiny_stream"] = invocations
        # A later scoring/proof refusal must not discard the preparation witness.
        write_json(Path(directory) / "receipts" / f"tiny-stream-prepare-{os.getpid()}.json",
                   {"schema": "dq11-rehearsal-stream/1", "science_eligible": False,
                    "source": os.environ.get("E4B_SHA"), "nonce": os.environ.get("TC1_RUN_NONCE"),
                    "invocations": invocations})
        return prepared
    finally:
        dense_offload.enable_dense_offload = original



from dq11_stream_witness import ROUTE_COUNTERS as ROUTE_COUNTERS  # noqa: E402


def train_prefetch_snapshot(directory, model):
    if not mode(directory):
        raise ValueError("CUDA route snapshot refused in science mode")
    from dq11_stream_witness import train_prefetch_snapshot as snapshot

    return snapshot(model)


def validate_train_prefetch_witness(witness):
    from dq11_stream_witness import validate_train_prefetch_witness as validate

    validate(witness, rehearsal=True)


def require_train_prefetch(directory, model, before, updates):
    if not mode(directory):
        raise ValueError("CUDA route witness refused in science mode")
    from dq11_stream_witness import require_train_prefetch as require

    return require(directory, model, before, updates, rehearsal=True)


def build_inputs(directory, canonical_tokens):
    """Local random Mistral and synthetic mapped tokens; no pretrained weights."""
    mode(directory)
    import inspect
    from types import SimpleNamespace

    from peft.tuners.lora.layer import LoraLayer
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    import torch
    from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

    directory = Path(directory)
    target = directory / "tiny-model"
    target.mkdir(exist_ok=False)
    torch.manual_seed(3407)
    model = MistralForCausalLM(MistralConfig(**MODEL_CONFIG)).to(dtype=torch.bfloat16)
    model.save_pretrained(target)
    del model
    vocabulary = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocabulary.update({f"word{i}": i for i in range(4, MODEL_CONFIG["vocab_size"])})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<s>",
                           eos_token="</s>", unk_token="<unk>", model_max_length=2048).save_pretrained(target)
    raw = json.loads(Path(canonical_tokens).read_text())
    blocks = {key: [[int(token) for token in block] for block in rows] for key, rows in raw.items()}
    if (set(blocks) != {"train", "alpaca-heldout", "wikitext-test"}
            or any(len(rows) != (40 if key == "train" else 8) for key, rows in blocks.items())
            or any(len(block) != 2048 for rows in blocks.values() for block in rows)
            or any(not 0 <= token < MODEL_CONFIG["vocab_size"]
                   for rows in blocks.values() for block in rows for token in block)):
        raise ValueError("rehearsal requires the original ordered block lengths")
    tokens = directory / "data/tokens.json"
    tokens.write_text(json.dumps(blocks, separators=(",", ":")) + "\n")
    torch.manual_seed(3407)
    tensors = {}
    projections = (("self_attn", "q_proj", 128, 128), ("self_attn", "k_proj", 128, 64),
                   ("self_attn", "v_proj", 128, 64), ("self_attn", "o_proj", 128, 128),
                   ("mlp", "gate_proj", 128, 256), ("mlp", "up_proj", 128, 256),
                   ("mlp", "down_proj", 256, 128))
    for layer in range(32):
        for group, name, dim_in, dim_out in projections:
            a, b = torch.empty((16, dim_in)), torch.empty((dim_out, 16))
            holder = SimpleNamespace(lora_A={"default": SimpleNamespace(weight=a)},
                                     lora_B={"default": SimpleNamespace(weight=b)},
                                     lora_bias={"default": False}, lora_embedding_A={}, lora_embedding_B={})
            LoraLayer.reset_lora_parameters(holder, "default", True)
            key = f"layers.{layer}.{group}.{name}"
            tensors[key + ".A"], tensors[key + ".B"] = a, b
    adapter = directory / "adapters/adapter_init.safetensors"
    save_file(tensors, str(adapter))
    locked = {"schema": "dq11-rehearsal-inputs/1", "science_eligible": False,
              "model": "local random tiny Mistral", "revision": "rehearsal-only",
              "mapping": "original ordered token IDs retained; synthetic word tokenizer/random model, never scientific quality",
              "canonical_tokens_sha256": digest(canonical_tokens), "model_config": MODEL_CONFIG,
              "initializer_source_sha256": digest(inspect.getsourcefile(LoraLayer.reset_lora_parameters)),
              "assets": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in (adapter, tokens)},
              "tokens": {key: {"blocks": len(rows), "targets": len(rows) * 2047,
                                "sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()}
                         for key, rows in blocks.items()}}
    write_json(directory / "locked_inputs.json", locked)
    (directory / "inputs.sha256").write_text("".join(digest(p) + "  " + str(p.relative_to(directory)) + "\n"
                                                      for p in (adapter, tokens)))
    write_json(directory / "model_files.json", {"schema": "dq11-rehearsal-model/1", "science_eligible": False,
               "model": str(target), "revision": "rehearsal-only", "files": [
                   {"name": p.name, "size": p.stat().st_size, "sha256": digest(p)} for p in sorted(target.iterdir())]})


def correctness(directory):
    """Apply the actual proof/read validators; issue no speed recommendation."""
    mode(directory)
    from dq11_reduce import initial_gate, validate_proofs, validate_read, ORDER

    proofs = [json.loads((directory / f"receipts/proof-{arm}.json").read_text()) for arm in ("L", "U", "U0")]
    validate_proofs(proofs, rehearsal=True)
    if not initial_gate(proofs):
        raise ValueError("rehearsal initial quality mismatch; no training")
    reads = []
    for rep, arm in ORDER:
        path = directory / f"receipts/read-{rep}-{arm}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if row.get("schema") != "dq11-rehearsal-arm/1" or row.get("science_eligible") is not False:
            raise ValueError("unmarked rehearsal reading")
        validate_read(row, next(p for p in proofs if p["arm"] == arm), rehearsal=True)
        if arm == "L":
            validate_train_prefetch_witness(row.get("rehearsal_train_prefetch"))
        reads.append({"arm": arm, "repetition": rep, "sha256": digest(path)})
    return {"schema": "dq11-rehearsal-correctness/1", "science_eligible": False,
            "recommendation": None, "proofs": [{"arm": p["arm"], "observer_same_arm_bitwise": True} for p in proofs],
            "reads": reads, "complete": len(reads) == 6,
            "limitations": ["cu128 and sm_86 cannot cover cu130 wheel loading or sm_120 kernels",
                            "local random tiny model and synthetic word tokenizer cannot license DQ11 quality or speed"]}


def timed_phase(directory, env, name, requested, *command):
    """Durable local rehearsal phase duration and actual time-left; never science timing."""
    deadline = int(env["TC1_DEADLINE_EPOCH"])
    started, monotonic = time.time(), time.monotonic()
    left = deadline - started - 300
    path = directory / "receipts/phase-timings.json"
    record = json.loads(path.read_text()) if path.exists() else {
        "schema": "dq11-rehearsal-phase-times/1", "science_eligible": False,
        "nonce": env["TC1_RUN_NONCE"], "deadline_epoch": deadline, "reserve_seconds": 300, "phases": []}
    row = {"phase": name, "started_epoch": started, "requested_cap_seconds": requested,
           "time_left_before_reserve_seconds": left, "effective_cap_seconds": min(requested, left) if left > 0 else None,
           "status": "RUNNING"}
    record["phases"].append(row)
    write_json(path, record)
    try:
        if left <= 0:
            raise TimeoutError("rehearsal deadline reserve")
        with (directory / f"logs/{name}.log").open("w") as log:
            subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=row["effective_cap_seconds"], check=True)
        row["status"] = "OK"
    except (subprocess.SubprocessError, TimeoutError, OSError) as error:
        row.update(status="REFUSED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        row.update(elapsed_seconds=time.monotonic() - monotonic, finished_epoch=time.time())
        row["time_left_after_reserve_seconds"] = deadline - row["finished_epoch"] - 300
        write_json(path, record)


def orchestrate():
    if os.environ.get("DQ11_REHEARSAL") != "1" or os.environ.get("DQ11_REHEARSAL_TINY_MODEL") != "1":
        raise ValueError("explicit correctness-only tiny-model opt-in required")
    sha = os.environ.get("E4B_SHA", "")
    nonce = os.environ.get("TC1_RUN_NONCE", "")
    if not re.fullmatch("[0-9a-f]{40}", sha) or not re.fullmatch("[a-zA-Z0-9_-]{1,128}", nonce):
        raise ValueError("reviewed source SHA and unique rehearsal nonce required")
    directory = Path(os.environ["DQ11_W"]).resolve()
    if directory.exists():
        raise ValueError("rehearsal requires a new owned workspace")
    canonical = Path(os.environ["DQ11_REHEARSAL_TOKENS"]).resolve()
    reference = json.loads((HERE / "locked_inputs.json").read_text())["assets"]["tokens.json"]
    if digest(canonical) != reference["sha256"] or canonical.stat().st_size != reference["bytes"]:
        raise ValueError("canonical token source changed before rehearsal input build")
    directory.mkdir(parents=True)
    for name in ("logs", "receipts", "adapters", "data", "hf-cache", "tmp"):
        (directory / name).mkdir()
    source_paths = (HERE, HERE.parent / "dq1", HERE.parent / "dq3")
    for line in (HERE / "science.sha256").read_text().splitlines():
        h, name = line.split()
        candidates = [p / name for p in source_paths if (p / name).is_file()]
        if len(candidates) != 1 or digest(candidates[0]) != h:
            raise ValueError("reviewed source closure changed: " + name)
        shutil.copy2(candidates[0], directory / name)
    shutil.copy2(HERE / "science.sha256", directory / "science-reference.sha256")
    for name in ("wheels.json", "requirements.lock"):
        shutil.copy2(HERE / "rehearsal" / name, directory / name)
    shutil.copy2(canonical, directory / "canonical-tokens.json")
    marker = {"schema": SCHEMA, "science_eligible": False, "source": sha, "nonce": nonce,
              "model_config": MODEL_CONFIG, "wheel_lock_sha256": digest(directory / "wheels.json"),
              "science_reference_sha256": digest(HERE / "science.sha256")}
    write_json(directory / "REHEARSAL.json", marker)
    env = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", TMPDIR=str(directory / "tmp"))
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env.pop(key, None)
    def phase(name, cap, *command):
        timed_phase(directory, env, name, cap, *command)

    rc = 11
    try:
        phase("git-prerequisite", 240, "bash", "dq11_require_git.sh")
        venv = directory / "venv-dq11"
        phase("create-venv", 120, "python3.11", "-m", "venv", str(venv))
        env["PATH"] = str(venv / "bin") + os.pathsep + env["PATH"]
        phase("bootstrap", 1800, "python", "dq11_bootstrap.py", "--cache", str(directory.parent / "wheel-cache"))
        phase("build-tiny-inputs", 120, "python", "dq11_rehearsal.py", "build", "--directory", str(directory),
              "--canonical-tokens", str(directory / "canonical-tokens.json"))
        # Alias filenames let the original arm consume the separate rehearsal
        # closure. This workspace's science.sha256 is explicitly NOT the science seal.
        payloads = [p for p in directory.iterdir() if p.is_file() and p.name not in {"science.sha256", "canonical-tokens.json"}]
        (directory / "science.sha256").write_text("".join(digest(p) + "  " + p.name + "\n" for p in sorted(payloads)))
        phase("prepare", 1800, "python", "dq11_prepare.py")
        for arm in ("L", "U", "U0"):
            phase("spread-" + arm, 900, "python", "dq11_arm.py", "--kind", "spread", "--arm", arm,
                  "--repetition", "0", "--out", f"receipts/spread-{arm}.json")
        for arm in ("L", "U", "U0"):
            phase("proof-" + arm, 900, "python", "dq11_arm.py", "--kind", "proof", "--arm", arm,
                  "--repetition", "0", "--out", f"receipts/proof-{arm}.json")
        phase("initial-correctness", 120, "python", "dq11_rehearsal.py", "check", "--directory", str(directory))
        for rep, arm in ((1, "L"), (1, "U"), (1, "U0"), (2, "U0"), (2, "U"), (2, "L")):
            phase(f"read-{rep}-{arm}", 900, "python", "dq11_arm.py", "--kind", "read", "--arm", arm,
                  "--repetition", str(rep), "--out", f"receipts/read-{rep}-{arm}.json")
        phase("final-correctness", 120, "python", "dq11_rehearsal.py", "check", "--directory", str(directory))
        rc = 0
    except (subprocess.SubprocessError, TimeoutError, OSError, ValueError) as error:
        write_json(directory / "receipts/refusal.json", {"schema": SCHEMA, "science_eligible": False,
                   "status": "REFUSED", "error": str(error), "nonce": nonce})
        raise
    finally:
        (directory / f"TC1_EXIT_CODE.{nonce}").write_text(str(rc) + "\n")
        (directory / f"TP_DONE.{nonce}").touch()
        if rc == 0:
            (directory / f"TC1_SUCCESS.{nonce}").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "build", "check"))
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--canonical-tokens", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        orchestrate()
    elif args.action == "build":
        build_inputs(args.directory, args.canonical_tokens)
    else:
        report = correctness(args.directory)
        write_json(args.directory / "receipts/correctness.json", report)
        print(json.dumps(report, indent=2))
