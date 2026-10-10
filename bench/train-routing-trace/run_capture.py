"""Capture a TRAINING routing trace by running the TC1 harness (bench/tc1/tc1_arm.py) unchanged, with routers hooked.

The harness is not edited. This wrapper:
  * wraps `tc1_arm.load_e4b` so the loaded model gets a RoutingRecorder on every `layers.<i>.mlp.gate`;
  * wraps `torch.Tensor.backward` so router calls inside backward (the checkpoint recompute) are labelled `recompute`;
  * registers a global optimizer step post-hook so each optimizer step closes a step in the trace;
then runs `tc1_arm.main()` with the harness's own arguments, and writes the trace when the harness returns.

Counts and expert ids only: the wrapper records no timings and the trace carries none.

    python bench/train-routing-trace/run_capture.py --trace-out /path/trace.npz -- <tc1_arm.py arguments ...>

The plan this implements (model, recipe, sizes, replay) is in this directory's README.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                       # not a checkout (an installed copy): recorded as unknown
        return None


def _sha16(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def environment(model) -> dict:
    """What the trace needs in order to be regenerable: library versions, commits, and the model's identity by hash."""
    import transformers
    cfg = getattr(model, "config", None)
    path = getattr(cfg, "_name_or_path", "") or ""
    env = {"torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0],
           "cuda": torch.version.cuda, "e4b_commit": _git(REPO), "name_or_path": path,
           "config_sha16": _sha16(os.path.join(path, "config.json")),
           "index_sha16": _sha16(os.path.join(path, "model.safetensors.index.json"))}
    try:
        import nf4_grouped
        env["gnf4_file"] = getattr(nf4_grouped, "__file__", None)
        from importlib.metadata import version
        env["grouped_nf4_gemm"] = version("grouped-nf4-gemm")
    except Exception:                                       # the kernel package absent: recorded as unknown
        env["grouped_nf4_gemm"] = None
    if cfg is not None:
        env["shape"] = {k: getattr(cfg, k, None) for k in
                        ("num_hidden_layers", "num_experts", "num_experts_per_tok", "hidden_size",
                         "moe_intermediate_size")}
    return env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-out", required=True, help="the .npz to write (a .meta.json lands beside it)")
    ap.add_argument("harness_args", nargs=argparse.REMAINDER, help="-- then the tc1_arm.py arguments")
    a = ap.parse_args(argv)
    harness_args = a.harness_args[1:] if a.harness_args[:1] == ["--"] else a.harness_args

    recorder_mod = _load("routing_recorder", HERE / "routing_recorder.py")
    tc1 = _load("tc1_arm", REPO / "bench" / "tc1" / "tc1_arm.py")
    rec = recorder_mod.RoutingRecorder()
    state = {"model": None}

    orig_load = tc1.load_e4b

    def load_e4b_hooked(*args, **kw):
        out = orig_load(*args, **kw)
        model = out[0] if isinstance(out, tuple) else out
        n = rec.attach(model)
        rec.n_experts = getattr(getattr(model, "config", None), "num_experts", None)
        state["model"] = model
        print(f"ROUTING-TRACE attached {n} router hooks", flush=True)
        return out

    tc1.load_e4b = load_e4b_hooked

    orig_backward = torch.Tensor.backward

    def backward_labelled(self, *args, **kw):
        with rec.backward_phase():
            return orig_backward(self, *args, **kw)

    torch.Tensor.backward = backward_labelled
    step_hook = torch.optim.optimizer.register_optimizer_step_post_hook(lambda *_: rec.end_step())

    rc = 0
    try:
        sys.argv = [str(REPO / "bench" / "tc1" / "tc1_arm.py"), *harness_args]
        rc = tc1.main() or 0
    finally:
        step_hook.remove()
        torch.Tensor.backward = orig_backward
        if state["model"] is not None:
            summary = rec.save(a.trace_out, {"environment": environment(state["model"]),
                                             "harness_args": harness_args, "plan": "README.md"})
            print(f"ROUTING-TRACE wrote {a.trace_out}: {summary['records']} records over {summary['steps']} steps; "
                  f"fwd/recompute differ {len(summary['fwd_recompute_mismatches']['differ'])}, "
                  f"forward-only {len(summary['fwd_recompute_mismatches']['no_recompute'])}; "
                  f"incomplete passes {len(summary['missing_layers'])}", flush=True)
        else:
            print("ROUTING-TRACE no model was loaded: nothing written", flush=True)
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
