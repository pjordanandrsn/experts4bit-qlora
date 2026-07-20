#!/bin/bash
# informed-pod-run.sh — ON-POD payload: (a) routing-informed hot sets vs naive
# (gpt-oss K=4/8 + Gemma-4 K=8, calibration = same greedy workload = oracle
# upper bound), (b) weak-CPU regime cells (taskset 0-3 both arms: ours K=0
# streams over PCIe, llama --n-cpu-moe computes on 4 cores). Same-box naive
# baselines included so no cross-box comparisons are needed.
# Evidence contract: /root/ab-out + /root/ab-run.log + AB-DONE / AB-FATAL.
set -u -o pipefail
mkdir -p /root/ab-out
echo "== INFORMED+WEAKCPU start $(date -u +%FT%TZ) =="
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee /root/ab-out/gpu.txt
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)' | tee /root/ab-out/cpu.txt
free -g | tee /root/ab-out/ram.txt

PY=""
for p in python3.12 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null && "$p" -c 'import torch' 2>/dev/null; then PY="$p"; break; fi
done
[ -n "$PY" ] || { echo "NO-TORCH-PYTHON"; echo "AB-FATAL"; exit 0; }
echo "PY=$PY torch=$($PY -c 'import torch; print(torch.__version__)')"
$PY -c "import torch; assert torch.cuda.is_available(); x = torch.zeros(1024, device='cuda') + 1; torch.cuda.synchronize(); print('CUDA-OK', torch.cuda.get_device_name(0))" \
  || { echo "CUDA-GATE-FAIL"; echo "AB-FATAL"; exit 0; }
(curl -sS -m 15 -o /dev/null https://pypi.org && curl -sS -m 15 -o /dev/null https://github.com) \
  || { echo "NET-GATE-FAIL"; echo "AB-FATAL"; exit 0; }

cd /root
git clone --depth 1 -b feat/hot-residency-gptoss https://github.com/pjordanandrsn/experts4bit-qlora.git e4b \
  || { echo "CLONE-FAIL"; echo "AB-FATAL"; exit 0; }
cd /root/e4b
git rev-parse HEAD | tee /root/ab-out/e4b_sha.txt
$PY -m pip install -q ".[train,fast]" 2>&1 | tail -2 \
  || { echo "PIP-INSTALL-FAIL"; echo "AB-FATAL"; exit 0; }
$PY -m pip install -q sentencepiece protobuf 2>&1 | tail -1
$PY -c "import transformers, bitsandbytes as b, nf4_grouped; print('versions tf', transformers.__version__, 'bnb', b.__version__, '| nf4_grouped OK')" \
  | tee /root/ab-out/versions.txt || { echo "DEPS-FAIL"; echo "AB-FATAL"; exit 0; }

run_ours() { # $1 model-id  $2 hot_k  $3 hot_mode  $4 out-tag  $5 extra-prefix (e.g. "taskset -c 0-3")
  echo "== OURS $4 $(date -u +%FT%TZ) =="
  MODEL="$1" HOT_K="$2" HOT_MODE="$3" BENCH_TOKENS=128 OUT="/root/ab-out/$4.json" \
    timeout 4500 $5 "$PY" bench/bench_gptoss_hybrid.py 2>&1 | tail -8 || echo "OURS $4 FAILED"
}

# --- gpt-oss: informed vs naive + same-box K=0 baseline + weak-CPU K=0 ---
run_ours openai/gpt-oss-20b 4 informed ours_gptoss_k4_informed ""
run_ours openai/gpt-oss-20b 8 informed ours_gptoss_k8_informed ""
run_ours openai/gpt-oss-20b 4 naive    ours_gptoss_k4_naive ""
run_ours openai/gpt-oss-20b 0 naive    ours_gptoss_k0 ""
run_ours openai/gpt-oss-20b 0 naive    ours_gptoss_k0_t4 "taskset -c 0-3"

# --- llama arm: build + MXFP4 gguf, full-cores and 4-core CPU-MoE ---
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
"$PY" - <<'PYS' > /root/ab-out/gguf_path.txt 2>/root/ab-out/gguf_dl.err || echo "GGUF-DL-FAIL"
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
PYS
GGUF=$(tail -1 /root/ab-out/gguf_path.txt)
if [ -x "$B" ] && [ -f "$GGUF" ]; then
  echo "== LLAMA ncmoe24 full-cores $(date -u +%TZ) =="
  timeout 1800 "$B" -m "$GGUF" -ngl 99 --n-cpu-moe 24 -p 64 -n 128 -r 3 -o json \
    > /root/ab-out/llama_ncmoe24.json 2>/root/ab-out/llama_ncmoe24.err || echo "LLAMA full FAILED"
  echo "== LLAMA ncmoe24 taskset-0-3 t4 $(date -u +%TZ) =="
  timeout 1800 taskset -c 0-3 "$B" -m "$GGUF" -ngl 99 --n-cpu-moe 24 -t 4 -p 64 -n 128 -r 3 -o json \
    > /root/ab-out/llama_ncmoe24_t4.json 2>/root/ab-out/llama_ncmoe24_t4.err || echo "LLAMA t4 FAILED"
else
  echo "LLAMA-ARM-SKIPPED (build or gguf missing)"
fi

# --- gemma informed K=8 (gated; wait for the operator-scp'd token) ---
for i in $(seq 1 60); do
  [ -s /root/hf/token ] && { echo "hf token present"; break; }
  [ "$i" = 60 ] && echo "TOKEN-WAIT-TIMEOUT — skipping gemma cell"
  sleep 10
done
if [ -s /root/hf/token ]; then
  cd /root/e4b
  # same-box pair: naive first, then informed (the published Gemma comparison)
  run_ours google/gemma-4-26B-A4B 8 naive    gemma_k8_naive ""
  run_ours google/gemma-4-26B-A4B 8 informed gemma_k8_informed ""
fi

"$PY" - <<'PYS' | tee /root/ab-out/SUMMARY.txt
import json, glob
print("cell                          decode_tok/s  peak_gb  coverage  coherent")
rows = []
for f in sorted(glob.glob("/root/ab-out/ours_*.json")) + sorted(glob.glob("/root/ab-out/gemma_*.json")):
    d = json.load(open(f))
    name = f.split("/")[-1].replace(".json", "")
    rows.append([name, d["decode_toks"], d.get("peak_gb"), d.get("cal_coverage"), d.get("coherent")])
    print("%-29s %-13s %-8s %-9s %s" % tuple(rows[-1]))
for f in sorted(glob.glob("/root/ab-out/llama_*.json")):
    try:
        d = json.load(open(f))
        tg = [r for r in d if r.get("n_gen", 0) > 0]
        if tg:
            name = f.split("/")[-1].replace(".json", "")
            rows.append([name, round(tg[0]["avg_ts"], 2), None, None, None])
            print("%-29s %-13s" % (name, rows[-1][1]))
    except Exception as e:
        rows.append([f, "parse-error: %s" % e, None, None, None])
json.dump(rows, open("/root/ab-out/SUMMARY.json", "w"), indent=1)
PYS

echo "AB-DONE — evidence complete $(date -u +%FT%TZ)"
