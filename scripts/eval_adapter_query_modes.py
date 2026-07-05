"""Evaluate trained adapters under different ExpertsNbit query modes (the matrix columns).

For each requested query mode this loads the base model ONCE (streaming loader, that mode's
storage/offload), records the base/no-adapter held-out eval loss (a fresh ``ExpertsLoRA`` is
zero-delta at init, so the freshly loaded model IS the no-adapter baseline), then applies each
adapter from ``--adapter-root`` in turn and records its eval loss. Storage-mode mismatch between
an adapter's train-mode provenance (sidecar metadata) and the query mode is *intentional* here:
it is recorded on every row and surfaced as a warning, never blocked and never hidden. Cross-mode
query is an empirical path — same-mode query is the cleanest contract.

Legs fail independently (``--fail-fast`` to stop instead); rows append to ``--out`` as JSONL and
a rerun with ``--resume`` skips pairs that already passed.

Example:
    python scripts/eval_adapter_query_modes.py \\
      --model allenai/OLMoE-1B-7B-0924 \\
      --adapter-root runs/olmoe_mode_adapters \\
      --query-modes nf4,fp4,int8,bf16,fp16 \\
      --out runs/olmoe_mode_adapters/query_matrix.jsonl
"""

import argparse
import os
import subprocess
import sys
import uuid
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mode_matrix_common import (  # noqa: E402
    append_jsonl,
    compute_mismatch,
    parse_mode_list,
    read_jsonl,
    read_metadata,
    utc_now,
    validate_row,
)


def _repo_commit():
    try:
        r = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _discover_adapters(adapter_root):
    """Every subdirectory holding an adapter_best.pt; sidecar metadata may be absent (warned)."""
    out = []
    for name in sorted(os.listdir(adapter_root)):
        d = os.path.join(adapter_root, name)
        p = os.path.join(d, "adapter_best.pt")
        if os.path.isdir(d) and os.path.exists(p):
            out.append((name, d, p))
    if not out:
        raise SystemExit(f"no adapters (adapter_best.pt) found under {adapter_root}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--adapter-root", required=True)
    ap.add_argument("--query-modes", required=True, help="csv of mode labels, e.g. nf4,fp4,int8,bf16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=10000, help="eval split = train[n:n+64], as in training")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N legs (smoke runs)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    modes = parse_mode_list(args.query_modes)
    adapters = _discover_adapters(args.adapter_root)
    print(f"planned query grid: {len(adapters)} adapters x {len(modes)} query modes on {args.model}")
    for name, _, _ in adapters:
        print(f"  adapter: {name}")
    for label, storage, offload in modes:
        print(f"  query:   {label:<14} storage={storage} offload={offload}")
    if args.dry_run:
        return 0

    # Heavy imports only past --dry-run. The trainer module's helpers (dataset encode, eval_loss)
    # read SEQ from env at import time — set it first so the eval split tokenizes exactly as in
    # training.
    os.environ["SEQ"] = str(args.seq)
    import torch  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    from experts4bit_qlora import train as trainmod  # noqa: E402
    from experts4bit_qlora.loader import load_moe_4bit_streaming  # noqa: E402
    from experts4bit_qlora.lora import add_attention_lora  # noqa: E402

    done = set()
    if args.resume and os.path.exists(args.out):
        done = {(r["train_mode_label"], r["query_mode_label"]) for r in read_jsonl(args.out) if r["status"] == "pass"}

    commit = _repo_commit()
    run_id = uuid.uuid4().hex[:12]
    env_common = {
        "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "commit": commit,
    }
    try:
        import bitsandbytes

        env_common["bitsandbytes_version"] = bitsandbytes.__version__
    except Exception:
        env_common["bitsandbytes_version"] = None

    tok = AutoTokenizer.from_pretrained(args.model)
    eval_data = trainmod.encode_alpaca(tok, f"train[{args.n_train}:{args.n_train + 64}]")

    legs = 0
    failures = 0
    for q_label, q_storage, q_offload in modes:
        pending = [(n, d, p) for n, d, p in adapters if (n, q_label) not in done]
        if not pending:
            print(f"[skip] query {q_label}: all pairs already passed (--resume)")
            continue
        print(f"[load] base in query mode {q_label} ...")
        try:
            model, _ = load_moe_4bit_streaming(
                args.model, "cuda", torch.bfloat16, args.r, args.alpha,
                offload=q_offload, quant_type=q_storage,
            )
            if not q_offload:
                model.to("cuda")
            add_attention_lora(model, args.r, args.alpha, torch.bfloat16)
            model.eval()
            model.config.use_cache = False
            base_eval = trainmod.eval_loss(model, eval_data)
            print(f"  base (no adapter) eval loss under {q_label}: {base_eval:.4f}")
        except Exception as e:
            for name, d, p in pending:
                append_jsonl(args.out, validate_row(_row(
                    run_id, args, name, d, p, q_label, q_storage, q_offload, env_common,
                    status="fail", reason=f"query-mode load failed: {type(e).__name__}: {e}",
                )))
            failures += len(pending)
            if args.fail_fast:
                return 1
            continue

        for name, adir, apath in pending:
            if args.limit and legs >= args.limit:
                print(f"[limit] stopping after {legs} legs")
                return 1 if failures else 0
            legs += 1
            meta = read_metadata(adir)
            mm = compute_mismatch(meta, q_storage, q_offload)
            for w in mm["warnings"]:
                warnings.warn(w, stacklevel=1)
            row = _row(run_id, args, name, adir, apath, q_label, q_storage, q_offload, env_common,
                       meta=meta, mismatch=mm, base_eval=base_eval)
            try:
                sd = torch.load(apath, map_location="cuda", weights_only=True)
                missing_model = [k for k in sd if k not in dict(model.named_parameters())]
                if missing_model:
                    raise RuntimeError(
                        f"{len(missing_model)} adapter tensors have no home in the model "
                        f"(first: {missing_model[0]}) — check --r/--alpha against the training run"
                    )
                model.load_state_dict(sd, strict=False)
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    loss = trainmod.eval_loss(model, eval_data)
                row.update(
                    status="pass",
                    eval_loss_with_adapter=loss,
                    delta_vs_base_query_mode=loss - base_eval,
                    peak_gpu_query_gb=torch.cuda.max_memory_allocated() / 1e9,
                )
                print(f"  [pass] {name} under {q_label}: eval {loss:.4f} (base {base_eval:.4f}, "
                      f"mismatch={mm['storage_mode_mismatch']})")
            except Exception as e:
                row.update(status="fail", skip_or_fail_reason=f"{type(e).__name__}: {e}")
                failures += 1
                print(f"  [fail] {name} under {q_label}: {e}")
                if args.fail_fast:
                    append_jsonl(args.out, validate_row(row))
                    return 1
            append_jsonl(args.out, validate_row(row))
        del model
        torch.cuda.empty_cache()
    print(f"done: {legs} legs, {failures} failures -> {args.out}")
    return 1 if failures else 0


def _row(run_id, args, name, adir, apath, q_label, q_storage, q_offload, env_common,
         meta=None, mismatch=None, base_eval=None, status="fail", reason=None):
    meta = meta or {}
    mismatch = mismatch or {"storage_mode_mismatch": None, "offload_mismatch": None}
    return {
        "run_id": run_id,
        "base_model": args.model,
        "train_mode_label": name,
        "train_storage_mode": meta.get("train_storage_mode"),
        "train_offload": meta.get("train_offload"),
        "query_mode_label": q_label,
        "query_storage_mode": q_storage,
        "query_offload": q_offload,
        "storage_mode_mismatch": mismatch["storage_mode_mismatch"],
        "offload_mismatch": mismatch["offload_mismatch"],
        "adapter_path": apath,
        "adapter_metadata_path": os.path.join(adir, "expertsnbit_adapter_metadata.json"),
        "eval_loss_base_query_mode_no_adapter": base_eval,
        "eval_loss_with_adapter": None,
        "delta_vs_base_query_mode": None,
        "delta_vs_same_mode_adapter": None,  # derived by the summarizer from the full matrix
        "peak_gpu_query_gb": None,
        "decode_tok_s": None,  # not measured in this pass
        "train_peak_gpu_gb": meta.get("peak_train_gpu_gb"),
        "train_s_per_step": meta.get("seconds_per_step"),
        "train_eval_before": meta.get("train_eval_before"),
        "train_eval_after": meta.get("train_eval_after"),
        "status": status,
        "skip_or_fail_reason": reason,
        "timestamp": utc_now(),
        **env_common,
    }


if __name__ == "__main__":
    sys.exit(main())
