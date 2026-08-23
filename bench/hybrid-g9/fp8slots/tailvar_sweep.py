"""Tail-rate variance sweep (PREREG-tailvar.md): one serving run per
disjoint corpus window, identical config, uniform placement -- the
touch counters are placement-independent (the co-routing cycle's law),
so no profile and no ladder are needed. Each run is a fresh subprocess.
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
STEP_DECOMP = HERE.parent / "step_decomp.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--window-span", type=int, default=28000)
    ap.add_argument("--window-stride", type=int, default=29000)
    ap.add_argument("--vram-gb", type=float, default=10.0)
    ap.add_argument("--dram-gb", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--series", action="store_true",
                    help="also dump each window's per-step touched-expert "
                         "series (gzipped)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    pathlib.Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    tags = []
    for w in range(a.windows):
        off = w * a.window_stride
        tag = f"w{w}"
        amort = pathlib.Path(a.out_dir) / f"{tag}.amort.json"
        cmd = [sys.executable, str(STEP_DECOMP),
               "--model", a.model, "--arena", a.arena, "--calib", a.calib,
               "--vram-gb", str(a.vram_gb), "--dram-gb", str(a.dram_gb),
               "--batch", str(a.batch), "--prompt-len", str(a.prompt_len),
               "--gen-tokens", str(a.gen_tokens), "--chunk", str(a.chunk),
               "--threads", str(a.threads),
               "--cpu-us-fixed", "55", "--cpu-us-per-row", "2",
               "--prompt-offset", str(off),
               "--prompt-span", str(a.window_span),
               "--out", str(pathlib.Path(a.out_dir) / f"{tag}.json"),
               "--amort-out", str(amort)]
        if a.series:
            cmd += ["--series-out",
                    str(pathlib.Path(a.out_dir) / f"{tag}.series.json.gz")]
        print(f"RUN {tag} offset={off} span={a.window_span}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"window {tag} failed rc={r.returncode}")
        tags.append(tag)
    idx = pathlib.Path(a.out_dir) / "windows.json"
    idx.write_text(json.dumps({"windows": tags, "stride": a.window_stride,
                               "span": a.window_span}, indent=1))
    print("TAILVAR_DONE", idx, flush=True)


if __name__ == "__main__":
    main()
