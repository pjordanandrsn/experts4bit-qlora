"""Exact fixture-only environment for fresh RA component processes."""
from __future__ import annotations

from pathlib import Path

PREFIXES = ("E4B_", "GNF4_", "NF4_", "TC1_", "P109_", "P115", "SC2", "TRAIN_")
SERVING = {"E4B_PAGED_MODEL", "E4B_PAGED_REVISION", "E4B_PAGED_ARENA", "E4B_PAGED_CALIB",
           "E4B_PAGED_PLACEMENT", "E4B_PAGED_DEVICE", "E4B_PAGED_TORCH_THREADS"}
CAPACITY = {"E4B_PAGED_MAX_TOKENS_PER_SEQ", "E4B_PAGED_CHUNK_TOKENS", "E4B_PAGED_TRACE", "E4B_PAGED_STEP_TRACE"}
BASE = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "LD_LIBRARY_PATH",
        "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_HUB_DISABLE_XET", "CUDA_VISIBLE_DEVICES"}


def allowed(component):
    if component in ("training", "training_profile"):
        return set()
    if component == "decode":
        return SERVING
    if component == "quality":
        return SERVING | {"E4B_PAGED_GRAPHS"}
    if component == "capacity":
        return SERVING | CAPACITY
    raise ValueError("unknown RA component")


def validate(component, fixture):
    if not set(fixture) <= allowed(component):
        raise ValueError("unregistered fixture environment key")
    fixed = {"E4B_PAGED_PLACEMENT": "all-vram", "E4B_PAGED_DEVICE": "cuda",
             "E4B_PAGED_GRAPHS": "0", "E4B_PAGED_MAX_TOKENS_PER_SEQ": "2048", "E4B_PAGED_CHUNK_TOKENS": "512"}
    if any(k in fixture and fixture[k] != v for k, v in fixed.items()):
        raise ValueError("fixture environment value changed")
    if component == "quality" and fixture.get("E4B_PAGED_GRAPHS") != "0":
        raise ValueError("quality requires named eager fixture")
    if component == "capacity" and any(fixture.get(k) != fixed[k] for k in
                                       ("E4B_PAGED_MAX_TOKENS_PER_SEQ", "E4B_PAGED_CHUNK_TOKENS")):
        raise ValueError("capacity requires named context/chunk fixture")
    for key in ("E4B_PAGED_ARENA", "E4B_PAGED_CALIB", "E4B_PAGED_TRACE", "E4B_PAGED_STEP_TRACE"):
        if key in fixture and not Path(fixture[key]).is_absolute():
            raise ValueError("fixture paths must be absolute")
    return dict(fixture)


def clean(base, *, component, fixture, venv, cache, threads, allocator):
    """Return a fresh environment and removed key names, never secret values."""
    validate(component, fixture)
    if not venv.is_absolute() or not cache.is_absolute() or type(threads) is not int or threads < 1:
        raise ValueError("invalid process paths/thread count")
    env = {k: v for k, v in base.items() if k in BASE}
    env.update(PATH=str(venv / "bin") + ":" + base.get("PATH", "/usr/bin:/bin"), VIRTUAL_ENV=str(venv),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               PYTORCH_CUDA_ALLOC_CONF=allocator, TRITON_CACHE_DIR=str(cache / "triton"),
               TORCH_EXTENSIONS_DIR=str(cache / "extensions"))
    env.update(fixture)
    removed = sorted(k for k in base if k.startswith(PREFIXES) or k in
                     ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "LD_PRELOAD"))
    return env, removed


def check_current(component, env):
    feature = {k: v for k, v in env.items() if k.startswith(PREFIXES)}
    validate(component, feature)
    return feature
