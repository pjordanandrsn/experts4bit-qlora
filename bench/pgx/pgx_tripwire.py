"""PGX tripwire: every check the shakedown box makes that needs no GPU, runnable at $0 against the launch commit.

    python bench/pgx/pgx_tripwire.py --kit DIR                       # CPU: the repo's own modules (the kit test runs this)
    python bench/pgx/pgx_tripwire.py --kit DIR --box E4B_SHA TORCH   # the box: plus the installed pins and CUDA

``DIR`` holds the kit's data files laid out as the box stages them (``fit_profile.jsonl``, ``bytes_per_expert.json``,
``prompts/``), so the CPU run and the box run read the same bytes. It checks, in order:

1. the scored prompt sets: each set's ``prompts_sha`` (``pgx_drive.prompts_sha``) equals the manifest's and the
   registered one, with the registered sizes and 1,024-token windows;
2. the fit profile: its sha256 is the registered one, and ``serve_paged.placement_profile`` accepts it for 48 x 128;
3. the K table from ``bytes_per_expert.json`` at 8 GB (the shakedown's budget) and E-hyb's GiB knob;
4. ``serve_paged``'s configuration as the E-hyb lifetime sets it (``PagedServeConfig.from_env`` reads the solver
   placement, both budgets, the profile and the step trace), that decode graphs resolve OFF at ``solver`` and ON at
   ``all-vram`` on a cc 8.9 CUDA device, and that ``build_engine`` hands ``enable_hybrid_tier`` the manifest in the
   position the placement-proof hook reads (``bench/pgx/hook_e4b``);
5. ``--box`` only: the installed e4b is the launch commit, grouped-nf4-gemm is the pinned release, torch is the image's,
   and CUDA is usable; then ``serve_paged``'s web stack (fastapi, uvicorn), imported by name at its pins, since the
   server imports both lazily and nothing else would before the first E-hyb start.

Prints ``PGX_TRIPWIRE OK`` and exits 0, or names the first failed check and exits 9 (the runner's install code).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

PROFILE_SHA = "0a46edd68005b46db6bb2174542ec948df2843071efce2ecb36099fda04c626a"
SETS = {"warm_c1": ("d45648005719bb7ff6979e814f0c9cde5824e85d12b60f5d1cd0e107b838f554", 8),
        "warm_c4": ("37589f7c124653b62f321a6641d76ab5162840d25e618321f574025704b11b12", 16),
        "shift": ("c4477b9ad77d5c02a2a30eade3ea1021e04e21299f9c43ce4bed31717403a295", 8)}
K_AT_8GB = {"S4": 67, "S8": 35, "V8": 35, "E-hyb": 62}
EHYB_VRAM_GIB_8GB = 7.451
GNF4_VERSION = "0.51.0"
# serve_paged's web stack, which it imports LAZILY (fastapi in create_app, uvicorn in main): nothing else on the box
# imports them before the first E-hyb server start, so a broken [serve] install would pass every other check
FASTAPI_VERSION = "0.141.1"
UVICORN_VERSION = "0.54.0"


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_prompts(kit: str) -> None:
    import pgx_drive
    with open(os.path.join(kit, "prompts", "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    for name, (want, n) in SETS.items():
        with open(os.path.join(kit, "prompts", f"{name}.json"), encoding="utf-8") as f:
            s = json.load(f)
        got = pgx_drive.prompts_sha(s)
        assert got == want == man["sets"][name]["prompts_sha"], f"{name}: prompts_sha {got[:12]} is not the registered {want[:12]}"
        assert len(s) == n and all(len(p["token_ids"]) == 1024 for p in s), f"{name}: not {n} windows of 1,024 tokens"


def check_profile(kit: str) -> None:
    path = os.path.join(kit, "fit_profile.jsonl")
    assert _sha(path) == PROFILE_SHA, f"fit_profile.jsonl sha256 {_sha(path)[:12]} is not the registered {PROFILE_SHA[:12]}"
    from experts4bit_qlora import serve_paged as sp
    assert sp.placement_profile(path, 48, 128) == path, "serve_paged does not accept the fit profile for 48 x 128"


def check_k(kit: str) -> None:
    import pgx_bytes
    import pgx_place_preview
    with open(os.path.join(kit, "bytes_per_expert.json"), encoding="utf-8") as f:
        kt = pgx_bytes.k_table(json.load(f))
    got = {arm: kt[arm]["8"] for arm in K_AT_8GB}
    assert got == K_AT_8GB, f"the K table at 8 GB is {got}, registered {K_AT_8GB}"
    assert pgx_place_preview.vram_gib(8) == EHYB_VRAM_GIB_8GB, pgx_place_preview.vram_gib(8)


def check_serve_paged(kit: str) -> None:
    from experts4bit_qlora import serve_paged as sp
    env = {"E4B_PAGED_PLACEMENT": "solver", "E4B_PAGED_VRAM_GB": str(EHYB_VRAM_GIB_8GB), "E4B_PAGED_DRAM_GB": "16",
           "E4B_PAGED_PROFILE": os.path.join(kit, "fit_profile.jsonl"), "E4B_PAGED_STEP_TRACE": "/tmp/pgx-step.jsonl"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        cfg = sp.PagedServeConfig.from_env()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    got = (cfg.placement, cfg.vram_gb, cfg.dram_gb, cfg.profile, cfg.step_trace_path)
    want = ("solver", EHYB_VRAM_GIB_8GB, 16.0, env["E4B_PAGED_PROFILE"], env["E4B_PAGED_STEP_TRACE"])
    assert got == want, f"PagedServeConfig.from_env read {got}, not {want}"
    assert sp._graphs_env("", "cuda", "solver", (8, 9)) is False, "decode graphs would be on at solver"
    assert sp._graphs_env("", "cuda", "all-vram", (8, 9)) is True, "decode graphs would be off at all-vram on cc 8.9"
    src = inspect.getsource(sp.build_engine)
    assert "from .engines.hybrid import enable_hybrid_tier" in src, "build_engine no longer imports enable_hybrid_tier by name"
    assert "enable_hybrid_tier(model, cfg.arena, man," in src, "build_engine no longer hands the manifest third"
    from experts4bit_qlora.engines import hybrid
    params = list(inspect.signature(hybrid.enable_hybrid_tier).parameters)[:3]
    assert params == ["model", "arena_path", "manifest"], params


def check_box(e4b_sha: str, torch_want: str) -> None:
    d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
    got = d.get("vcs_info", {}).get("commit_id")
    assert got == e4b_sha, f"installed e4b is {got}, not the launch commit {e4b_sha}"
    assert md.version("grouped-nf4-gemm") == GNF4_VERSION, f"grouped-nf4-gemm {md.version('grouped-nf4-gemm')} is not {GNF4_VERSION}"
    import torch
    assert torch.__version__ == torch_want, f"torch {torch.__version__} is not the image's {torch_want}"
    assert torch.cuda.is_available(), "CUDA is not usable"


def check_serve_stack() -> None:
    """Import serve_paged's web stack by name and check the pins (the g3-gpu-check-1 lesson: a lazy import is untested)."""
    import fastapi
    import uvicorn
    assert fastapi.__version__ == FASTAPI_VERSION, f"fastapi {fastapi.__version__} is not {FASTAPI_VERSION}"
    assert uvicorn.__version__ == UVICORN_VERSION, f"uvicorn {uvicorn.__version__} is not {UVICORN_VERSION}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--box", nargs=2, metavar=("E4B_SHA", "TORCH"))
    a = ap.parse_args(argv)
    steps = [("prompt sets", lambda: check_prompts(a.kit)), ("fit profile", lambda: check_profile(a.kit)),
             ("K table", lambda: check_k(a.kit)), ("serve_paged config", lambda: check_serve_paged(a.kit))]
    if a.box:
        steps.append(("installed pins", lambda: check_box(*a.box)))
        steps.append(("serve stack", check_serve_stack))
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- every failure names its step
            print(f"PGX_TRIPWIRE FAIL {name}: {type(exc).__name__}: {exc}")
            return 9
        print(f"PGX_TRIPWIRE ok {name}")
    print("PGX_TRIPWIRE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
