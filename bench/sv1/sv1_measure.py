"""bench/sv1/sv1_measure.py -- lane SV1, one arm (bench/sv1/SV1-PREREG.md). Builds serve_paged's engine in-process with
exactly the environment ServeSetup.to_env() gives, serves 16 seeded requests to idle, and writes ``sv1-arm/1``: the
estimate's items beside the allocator / reserved / driver peaks, and the runner's own graph and prefill-graph status.
Nothing here creates, destroys or approves compute."""
import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import traceback


class DriverPeak:
    """Peak driver-reported device memory of this process, sampled every 0.25 s."""

    def __init__(self):
        self.peak, self.samples, self._stop = 0, 0, threading.Event()
        self._smi = shutil.which("nvidia-smi")
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        pid = str(os.getpid())
        while not self._stop.is_set():
            try:
                out = subprocess.run([self._smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                for line in out.splitlines():
                    p, _, mib = line.partition(",")
                    if p.strip() == pid and mib.strip().isdigit():
                        self.peak, self.samples = max(self.peak, int(mib) << 20), self.samples + 1
            except Exception:  # noqa: BLE001 - a missed sample is a missed sample
                pass
            self._stop.wait(0.25)

    def __enter__(self):
        if self._smi:
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--graphs", type=int, choices=(0, 1), required=True)
    ap.add_argument("--prefill-graph", choices=("0", "1", "auto"), required=True)
    ap.add_argument("--max-seqs", type=int, default=16)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import importlib.metadata as md

    import torch

    from experts4bit_qlora.arch.topology import describe_moe
    from experts4bit_qlora.serve_recipe import ServeSetup, estimate_serve_footprint

    setup = ServeSetup(max_seqs=a.max_seqs, max_tokens_per_seq=a.context, chunk_tokens=512, graphs=bool(a.graphs),
                       prefill_graph=a.prefill_graph, placement="all-vram", hot_rows=64)
    fp = estimate_serve_footprint(describe_moe(a.model, revision=a.revision), setup)
    items = [{"name": i.name, "where": i.where, "bytes": i.bytes, "basis": i.basis} for i in fp.items]
    env = {**setup.to_env(), "E4B_PAGED_MODEL": a.model, "E4B_PAGED_REVISION": a.revision, "E4B_PAGED_ARENA": a.arena,
           "E4B_PAGED_CALIB": a.calib, "E4B_PAGED_TORCH_THREADS": "8"}
    os.environ.update(env)
    rec = {"schema": "sv1-arm/1", "args": vars(a), "setup": setup.to_dict(), "env": env,
           "estimate": {"items": items, "unmodelled": list(fp.unmodelled), "refusals": list(fp.refusals),
                        "device_total": sum(i["bytes"] for i in items if i["where"] == "device")},
           "versions": {d: md.version(d) for d in ("experts4bit-qlora", "grouped-nf4-gemm", "torch", "transformers")},
           "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())}
    m = rec["measured"] = {}
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    try:
        from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

        cfg = PagedServeConfig.from_env()
        with DriverPeak() as smi:
            t0 = time.time()
            parts = build_engine(cfg)
            torch.cuda.synchronize()
            m.update(load_seconds=round(time.time() - t0, 1), load_device_peak_bytes=torch.cuda.max_memory_allocated(),
                     load_reserved_peak_bytes=torch.cuda.max_memory_reserved())
            runner = parts.runner
            rec["graph_status"] = {str(k): v for k, v in (getattr(runner, "graph_status", None) or {}).items()}
            rec["prefill_graph_at_load"] = runner.prefill_graph_stats() if hasattr(runner, "prefill_graph_stats") else None
            g = torch.Generator().manual_seed(0)
            vocab = int(getattr(parts.tokenizer, "vocab_size", 0) or 32000)
            for _ in range(a.max_seqs):
                parts.scheduler.add_request(torch.randint(10, vocab - 10, (a.prompt_tokens,), generator=g).tolist(),
                                            max_new_tokens=a.new_tokens)
            t1 = time.time()
            steps = parts.scheduler.run_until_idle()
            torch.cuda.synchronize()
            dt = time.time() - t1
        done = len(parts.scheduler.done)
        m.update(device_peak_bytes=torch.cuda.max_memory_allocated(), device_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                 driver_process_peak_bytes=smi.peak or None, driver_samples=smi.samples, scheduler_steps=steps,
                 serve_seconds=round(dt, 2), requests_done=done, tokens_out=done * a.new_tokens,
                 tokens_per_s=round(done * a.new_tokens / dt, 1) if dt else None)
        rec["prefill_graph"] = runner.prefill_graph_stats() if hasattr(runner, "prefill_graph_stats") else None
        rec["status"] = "OK" if done == a.max_seqs else "INCOMPLETE"
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="OOM", error=str(e)[:600])
        m.update(device_peak_bytes=torch.cuda.max_memory_allocated(), device_reserved_peak_bytes=torch.cuda.max_memory_reserved())
    except Exception as e:  # noqa: BLE001 - a refusal is a reading, recorded with its reason
        rec.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:600], traceback=traceback.format_exc()[-2000:])
        m.update(device_peak_bytes=torch.cuda.max_memory_allocated(), device_reserved_peak_bytes=torch.cuda.max_memory_reserved())
    if m.get("device_peak_bytes"):
        m["estimate_minus_peak_bytes"] = rec["estimate"]["device_total"] - m["device_peak_bytes"]
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"{rec['status']} est {rec['estimate']['device_total'] / 2**30:.3f} GiB peak "
          f"{(m.get('device_peak_bytes') or 0) / 2**30:.3f} reserved {(m.get('device_reserved_peak_bytes') or 0) / 2**30:.3f} "
          f"driver {(m.get('driver_process_peak_bytes') or 0) / 2**30:.3f} prefill_graph {rec.get('prefill_graph')}")
    raise SystemExit(0 if rec["status"] == "OK" else 1)


if __name__ == "__main__":
    main()
