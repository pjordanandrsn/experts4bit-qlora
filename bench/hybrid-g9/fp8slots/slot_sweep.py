"""Slot-value sweep orchestrator (PREREG-slotvalue.md).

Runs step_decomp.py once per (pass, vram_gb) point as a SUBPROCESS —
fresh process per point so allocator state, triton caches, and torch
pools cannot couple adjacent points. Pass 1 walks the ladder descending,
pass 2 ascending: a monotonic drift in the box shows up as a systematic
pass1/pass2 split, which the A/A gate in slot_verdict.py scores.

Profile-pass mode (--profile-pass) runs one point at the top budget with
--profile-out to capture the model's own routing; the sweep then feeds
that profile to every placement.
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
STEP_DECOMP = HERE.parent / "step_decomp.py"


def run_point(a, vram_gb, tag, extra):
    out = pathlib.Path(a.out_dir) / f"{tag}.json"
    amort = pathlib.Path(a.out_dir) / f"{tag}.amort.json"
    cmd = [sys.executable, str(STEP_DECOMP),
           "--model", a.model, "--arena", a.arena, "--calib", a.calib,
           "--vram-gb", str(vram_gb), "--dram-gb", str(a.dram_gb),
           "--batch", str(a.batch), "--prompt-len", str(a.prompt_len),
           "--gen-tokens", str(a.gen_tokens), "--chunk", str(a.chunk),
           "--threads", str(a.threads),
           "--cpu-us-fixed", "55", "--cpu-us-per-row", "2",
           "--out", str(out), "--amort-out", str(amort)] + extra
    print("RUN", tag, "vram_gb=", vram_gb, flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"point {tag} failed rc={r.returncode}")
    return json.loads(amort.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--ladder", default="10.0,12.0,13.0,13.5,14.0")
    ap.add_argument("--dram-gb", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--profile-pass", action="store_true",
                    help="run one top-budget point emitting the routing "
                         "profile, then exit")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    pathlib.Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    ladder = [float(x) for x in a.ladder.split(",")]

    if a.profile_pass:
        prof = pathlib.Path(a.out_dir) / "expert_profile.jsonl"
        run_point(a, ladder[-1], "profilepass",
                  ["--profile-out", str(prof)])
        print("PROFILE_PASS_DONE", prof, flush=True)
        return

    if not a.profile:
        raise SystemExit("the sweep needs --profile (run --profile-pass first)")
    points = {}
    order1 = sorted(ladder, reverse=True)
    order2 = sorted(ladder)
    for pno, order in ((1, order1), (2, order2)):
        for v in order:
            tag = f"p{pno}_v{v:g}"
            points[tag] = run_point(a, v, tag, ["--profile", a.profile])
            am = points[tag]
            nv = sum(pl["uniq_nvme"] for pl in am["per_layer"])
            if nv:
                raise SystemExit(
                    f"{tag}: NVMe tier touched ({nv} uniques) — the cold "
                    "path would contaminate the DRAM bucket; raise --dram-gb")
    sweep = pathlib.Path(a.out_dir) / "sweep.json"
    sweep.write_text(json.dumps(
        {"ladder": ladder, "profile": a.profile, "points": points}, indent=1))
    print("SWEEP_DONE", sweep, flush=True)


if __name__ == "__main__":
    main()
