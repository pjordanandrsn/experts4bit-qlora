#!/bin/bash
# pod-ab-run.sh — ON-POD payload: gpt-oss-20b hybrid-vs-llama same-box A/B.
#   ours arm : e4b NF4 hot/cold hybrid (feat/hot-residency-gptoss), HOT_K in {0,4,8}
#   llama arm: llama.cpp CUDA build, MXFP4 GGUF, --n-cpu-moe in {0,12,24}
# House metric both sides: decode tok/s. Evidence lands in /root/ab-out/;
# the mini watcher tears the pod down on "AB-DONE" / "AB-FATAL" / deadline.
# Every cell is timeout-guarded and failure-tolerant: one bad cell never kills
# the run; missing cells show up as absent JSONs in the summary.
set -u
mkdir -p /root/ab-out
echo "== AB start $(date -u +%FT%TZ) =="
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- receipts: what box did we actually get ---
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee /root/ab-out/gpu.txt
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)' | tee /root/ab-out/cpu.txt
free -g | tee /root/ab-out/ram.txt

# --- two-pythons trap: pick the interpreter that owns torch (mx7 lesson) ---
PY=""
for p in python3.12 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null && "$p" -c 'import torch' 2>/dev/null; then PY="$p"; break; fi
done
[ -n "$PY" ] || { echo "NO-TORCH-PYTHON"; echo "AB-FATAL"; exit 0; }
echo "PY=$PY torch=$($PY -c 'import torch; print(torch.__version__)')"

# --- gates: broken host / dead network cost cents, not hours ---
$PY -c "import torch; assert torch.cuda.is_available(); x = torch.zeros(1024, device='cuda') + 1; torch.cuda.synchronize(); print('CUDA-OK', torch.cuda.get_device_name(0))" \
  || { echo "CUDA-GATE-FAIL"; echo "AB-FATAL"; exit 0; }
(curl -sS -m 15 -o /dev/null https://pypi.org && curl -sS -m 15 -o /dev/null https://github.com) \
  || { echo "NET-GATE-FAIL"; echo "AB-FATAL"; exit 0; }

# --- ours arm setup ---
cd /root
git clone --depth 1 -b feat/hot-residency-gptoss https://github.com/pjordanandrsn/experts4bit-qlora.git e4b \
  || { echo "CLONE-FAIL"; echo "AB-FATAL"; exit 0; }
cd /root/e4b
git rev-parse HEAD | tee /root/ab-out/e4b_sha.txt
$PY -m pip install -q ".[train,fast]" 2>&1 | tail -2   # [fast]: hot_residency runs on nf4_grouped
$PY -c "import torch, transformers, bitsandbytes as b; print('versions torch', torch.__version__, 'tf', transformers.__version__, 'bnb', b.__version__)" \
  | tee /root/ab-out/versions.txt || { echo "DEPS-FAIL"; echo "AB-FATAL"; exit 0; }

# --- ours cells: fresh process per K (clean VRAM), K=0 = all-cold streamed ---
for K in 0 4 8; do
  echo "== OURS HOT_K=$K $(date -u +%TZ) =="
  MODEL=openai/gpt-oss-20b HOT_K=$K BENCH_TOKENS=128 OUT=/root/ab-out/ours_k$K.json \
    timeout 3600 "$PY" bench/bench_gptoss_hybrid.py 2>&1 | tail -15 \
    || echo "OURS K=$K FAILED"
done

# --- llama arm: build ---
cd /root
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq cmake build-essential >/dev/null 2>&1 || true
export PATH=/usr/local/cuda/bin:$PATH
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git 2>&1 | tail -1
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
  > /root/ab-out/llama_cmake.log 2>&1
cmake --build llama.cpp/build -j "$(nproc)" --target llama-bench \
  > /root/ab-out/llama_build.log 2>&1 || echo "LLAMA-BUILD-FAIL"
B=/root/llama.cpp/build/bin/llama-bench

# --- llama arm: model (pick the mxfp4 gguf from the official repo) ---
"$PY" - <<'PY' > /root/ab-out/gguf_path.txt 2>/root/ab-out/gguf_dl.err || echo "GGUF-DL-FAIL"
import re
from huggingface_hub import HfApi, hf_hub_download
repo = "ggml-org/gpt-oss-20b-GGUF"
files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf")]
assert files, "no gguf files in repo"
pick = sorted(files, key=lambda f: ("mxfp4" not in f.lower(), f))[0]
stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", pick).replace(".gguf", "")
group = sorted(f for f in files if f.startswith(stem))
paths = [hf_hub_download(repo, f) for f in group]
print(paths[0])
PY
GGUF=$(tail -1 /root/ab-out/gguf_path.txt)

# --- llama cells ---
if [ -x "$B" ] && [ -f "$GGUF" ]; then
  if "$B" --help 2>&1 | grep -q "n-cpu-moe"; then NCFLAG="--n-cpu-moe"; else NCFLAG=""; fi
  for NC in 0 12 24; do
    echo "== LLAMA n-cpu-moe=$NC $(date -u +%TZ) =="
    if [ "$NC" = "0" ]; then
      timeout 1800 "$B" -m "$GGUF" -ngl 99 -p 64 -n 128 -r 3 -o json \
        > /root/ab-out/llama_ncmoe$NC.json 2>/root/ab-out/llama_ncmoe$NC.err || echo "LLAMA NC=$NC FAILED"
    elif [ -n "$NCFLAG" ]; then
      timeout 1800 "$B" -m "$GGUF" -ngl 99 $NCFLAG "$NC" -p 64 -n 128 -r 3 -o json \
        > /root/ab-out/llama_ncmoe$NC.json 2>/root/ab-out/llama_ncmoe$NC.err || echo "LLAMA NC=$NC FAILED"
    else
      # fallback: regex tensor override sends ALL expert tensors to CPU (NC=24 analogue only)
      [ "$NC" = "24" ] && timeout 1800 "$B" -m "$GGUF" -ngl 99 -ot "blk\..*\.ffn_.*_exps\.=CPU" -p 64 -n 128 -r 3 -o json \
        > /root/ab-out/llama_ncmoe$NC.json 2>/root/ab-out/llama_ncmoe$NC.err || echo "LLAMA NC=$NC SKIPPED (no n-cpu-moe flag)"
    fi
    grep -aE '"avg_ts"|error' /root/ab-out/llama_ncmoe$NC.json 2>/dev/null | head -4
  done
else
  echo "LLAMA-ARM-SKIPPED (build or gguf missing)"
fi

# --- summary (verdict numbers must survive into the email body) ---
"$PY" - <<'PY' | tee /root/ab-out/SUMMARY.txt
import json, glob, os
rows = []
for f in sorted(glob.glob("/root/ab-out/ours_k*.json")):
    d = json.load(open(f))
    rows.append(("ours hot_k=%d" % d["hot_k"], d["decode_toks"], d.get("peak_gb"), d.get("coherent")))
for f in sorted(glob.glob("/root/ab-out/llama_ncmoe*.json")):
    try:
        d = json.load(open(f))
        nc = f.rsplit("ncmoe", 1)[1].split(".")[0]
        tg = [r for r in d if r.get("n_gen", 0) > 0]
        if tg:
            rows.append(("llama n-cpu-moe=%s" % nc, round(tg[0]["avg_ts"], 2), None, None))
    except Exception as e:
        rows.append((os.path.basename(f), "parse-error: %s" % e, None, None))
print("cell                        decode_tok/s   peak_gb  coherent")
for r in rows:
    print("%-27s %-14s %-8s %s" % r)
json.dump([list(r) for r in rows], open("/root/ab-out/SUMMARY.json", "w"), indent=1)
PY

echo "AB-DONE — evidence complete $(date -u +%FT%TZ)"
