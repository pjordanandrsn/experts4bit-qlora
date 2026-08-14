#!/bin/bash
# The PREREG's run, on the pod. Published wheels + this branch's overlay.
#
# Two markers drive teardown, and the split is the point (teardown-on-fatal must
# know what the pod is HOLDING):
#   DONE  -> cheap to recreate; the puller may tear down.
#   HOLD  -> the pod holds a 160 GB download and/or a 147 GB arena that cost real
#            wall-clock. The puller SKIPS teardown on HOLD and the wall-clock
#            backstop stays the bill cap.
# Written the moment the download starts paying for itself, not at the end.
#
# No stage is `cmd | tail || fail`. That construct is why attempt 1 ran both arms
# against an arena that did not exist.
set -uo pipefail
W=/root/v4
OUT=/root/v4out
mkdir -p "$OUT"
cd "$W" || exit 90

say() { echo "[$(date -u +%FT%TZ)] $*"; }
hold() { touch "$OUT/HOLD"; say "HOLD written — pod holds expensive state, teardown deferred to the cap"; }
done_() { touch "$OUT/DONE"; say "DONE written ($1)"; }
finish() {  # finish <status> <note>   — always exit 0; outcomes live in files
  ${PY:-python3} - "$1" "$2" <<'PYEOF' || true
import json, os, sys
r = {"status": sys.argv[1], "note": sys.argv[2]}
out = "/root/v4out"
for f in sorted(os.listdir(out)):
    if f.endswith(".json") and f != "result.json":
        try:
            r[f[:-5]] = json.load(open(os.path.join(out, f)))
        except Exception as e:
            r[f[:-5]] = f"unparsable: {e}"
json.dump(r, open(os.path.join(out, "result.json"), "w"), indent=1, default=str)
print("RESULT", json.dumps({k: r[k] for k in ("status", "note")}))
PYEOF
  if [ "$1" = "COMPLETE" ]; then
    rm -f "$OUT/HOLD"                 # nothing left worth paying to keep
    done_ "$1"
  elif [ -f "$OUT/HOLD" ]; then
    say "HOLD retained: $1 happened AFTER the download/bake, so the pod is kept for"
    say "  inspection and the wall-clock backstop stays the bill cap."
  else
    done_ "$1"                        # failed cheap; nothing to preserve
  fi
  say "finished: $1 — $2"
  exit 0
}

# ---------------------------------------------------------------- interpreter
PY=""
for c in python /venv/bin/python /opt/conda/bin/python python3 /usr/local/bin/python3; do
  command -v "$c" > /dev/null 2>&1 || [ -x "$c" ] || continue
  if "$c" -c "import torch" > /dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo '{"status":"ENV_BROKEN"}' > "$OUT/result.json"; done_ ENV_BROKEN; exit 0; }
export PY
say "interpreter: $PY"

# ------------------------------------------------------------------- install
say "=== install ==="
"$PY" -m pip install -q --no-cache-dir "experts4bit-qlora[fast]" grouped-nf4-gemm \
  bitsandbytes transformers accelerate safetensors huggingface_hub hf_transfer > pip.log 2>&1
PIP_RC=$?
tail -3 pip.log
[ $PIP_RC -eq 0 ] || finish PIP_FAILED "pip exit $PIP_RC"

"$PY" - <<'PYEOF'
import json, sys, torch
env = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "vram_gib": round(torch.cuda.mem_get_info()[1] / 2**30, 1) if torch.cuda.is_available() else 0}
try:
    import triton; env["triton"] = triton.__version__
except Exception as e:
    env["triton"] = f"MISSING: {e}"
import importlib.metadata as m
for p in ("experts4bit-qlora", "grouped-nf4-gemm", "transformers"):
    try: env[p] = m.version(p)
    except Exception as e: env[p] = f"MISSING: {e}"
json.dump(env, open("/root/v4out/env.json", "w"), indent=1)
print(json.dumps(env))
sys.exit(0 if env["cuda"] else 7)
PYEOF
[ $? -eq 0 ] || finish ENV_BROKEN "no CUDA"

say "=== overlay ==="
rm -rf ov && mkdir ov && tar xzf e4b_overlay.tgz -C ov
[ $? -eq 0 ] || finish OVERLAY_FAILED "untar"
OUT="$OUT" PY="$PY" bash ov/apply_overlay.sh > overlay.log 2>&1
[ $? -eq 0 ] || { cat overlay.log; finish OVERLAY_FAILED "apply_overlay"; }

say "=== G2/G4 pre-gates in the IMPORTED module ==="
"$PY" v4_run.py gates > gates.log 2>&1
[ $? -eq 0 ] || { tail -20 gates.log; finish G2_FAILED "overlay not in the imported module"; }
tail -3 gates.log

# --------------------------------------------------------------- local disk
# `/workspace` is MooseFS over the network; only the container overlay is local
# NVMe. The arena read path is the thing under test, so putting it on a network
# mount would measure the network.
say "=== disk ==="
findmnt -n -o SOURCE,FSTYPE --target /root | tee "$OUT/mount.txt"
if findmnt -n -o SOURCE --target /root | grep -q '^mfs'; then
  finish WRONG_DISK "/root is a network mount; the arena must live on local NVMe"
fi
df -h /root | tail -1

# ---------------------------------------------------------------- download
say "=== download DeepSeek-V4-Flash (~160 GB) to LOCAL disk ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
T0=$(date +%s)
"$PY" - <<'PYEOF' > dl.log 2>&1
from huggingface_hub import snapshot_download
p = snapshot_download("deepseek-ai/DeepSeek-V4-Flash", local_dir="/root/ckpt",
                      max_workers=16, ignore_patterns=["assets/*", "*.png"])
print("snapshot:", p)
PYEOF
DL_RC=$?
tail -4 dl.log
[ $DL_RC -eq 0 ] || finish DOWNLOAD_FAILED "snapshot_download exit $DL_RC"
say "download took $(( $(date +%s) - T0 ))s; $(du -sh /root/ckpt | cut -f1) on disk"
hold                      # from here the pod holds something expensive

# -------------------------------------------------------------------- bake
# RELOCATION bake, not a re-quantization: the arena carries the checkpoint's own
# MXFP4 bytes. name_template verified against the real index before renting —
# V4 is `layers.{layer}.ffn.experts.{expert}.{kind}`, NOT the `model.layers…mlp`
# form the test fixture uses.
say "=== bake the MXFP4 arena (~147 GB) ==="
T0=$(date +%s)
"$PY" - <<'PYEOF' > bake.log 2>&1
import json, time
from nvme_arena import bake_expert_tensors, load_index
from mxfp4_residency import V4_RESIDENCY_KINDS
t0 = time.time()
man = bake_expert_tensors("/root/ckpt", "/root/v4.arena",
                          name_template="layers.{layer}.ffn.experts.{expert}.{kind}",
                          kinds=V4_RESIDENCY_KINDS, align=4096, workers=8)
idx = load_index("/root/v4.arena")
rec = {"seconds": round(time.time() - t0, 1),
       "n_layers": idx["n_layers"], "n_experts_per_layer": idx["n_experts_per_layer"],
       "row_stride": idx["row_stride"],
       "segments": [s["suffix"] for s in idx["segments"]],
       "manifest_keys": sorted(man)[:12] if isinstance(man, dict) else str(type(man))}
json.dump(rec, open("/root/v4out/bake.json", "w"), indent=1)
print(json.dumps(rec, indent=1))
PYEOF
BAKE_RC=$?
tail -20 bake.log
[ $BAKE_RC -eq 0 ] || finish BAKE_FAILED "bake exit $BAKE_RC (HOLD is set; pod kept)"
say "bake took $(( $(date +%s) - T0 ))s; arena $(du -sh /root/v4.arena 2>/dev/null | cut -f1)"

# -------------------------------------------------------------------- arms
# ARENA first. It is the claim; STOCK is its control, and a control that eats the
# clock before the claim is measured is the wrong order.
say "=== ARM: ARENA ==="
"$PY" v4_run.py arena > arena.log 2>&1
ARENA_RC=$?
tail -30 arena.log

say "=== ARM: STOCK (run, not asserted) ==="
"$PY" v4_run.py stock > stock.log 2>&1
tail -12 stock.log

if [ $ARENA_RC -eq 0 ]; then finish COMPLETE "both arms ran"; fi
finish ARENA_FAILED "arena arm exit $ARENA_RC (see arm_arena.json / stage_arena_crash.json)"
