#!/usr/bin/env bash
# bench/sc1/llamacpp/llamacpp_box.sh -- shell functions for the SC1 box script (SOURCE this file; do not run it).
#
#   llamacpp_build        <dir> <commit>                 clone at the exact commit, CUDA sm_120 build, harness, --version
#   llamacpp_fetch        <repo> <file> <dest> [<rev>]   one GGUF via huggingface_hub, sha256 + fetch receipt
#   llamacpp_server_start <gguf> <np> <port> <log> [<ctx>]   llama-server with the registered flags; waits for /health;
#                                                         REFUSES unless the log says every layer is on the GPU
#   llamacpp_server_stop                                 stop the server started by llamacpp_server_start
#   llamacpp_server_dump  <port> <dest-prefix>           /props, /health, /metrics, /slots to files
#
# Environment: LLAMACPP_PY (python with huggingface_hub; default python3), LLAMACPP_JOBS (build -j; default nproc),
# CUDAARCHS (default 120a-real -- the RTX 5090; the prebuilt cuda-12.4 binaries carry no sm_120 SASS).
#
# Context size (-c is the TOTAL across slots; libllama gives each slot -c/np padded UP to a multiple of 256,
# src/llama-context.cpp: n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256); the server refuses a request whose prompt + n_predict
# would not fit the slot and stops a row early at the slot boundary -- which the arm driver reports as VOID):
#   slope arms   : each slot must hold 512 prompt + 128 generated = 640 tokens  ->  minimum -c = np * 640
#                  (np=16: 10240 total, 768 per slot after padding; np=1: 640 -> 768 per slot)
#   TTFT arm     : one slot must hold 4096 prompt + 8 generated = 4104 tokens   ->  minimum -c = 4104 (4352 after padding)
#   DEFAULT here : -c = np * 1024 (np=16 -> 16384 = 1024 per slot; np=1 -> 1024) for the slope arms; the TTFT caller
#                  passes <ctx> = 8192 explicitly. f16 KV for Qwen3-30B-A3B is 48 layers x 2 x 4 heads x 128 x 2 B
#                  = 96 KiB/token, so 16384 tokens = 1.5 GiB -- comfortably inside the 32 GB card beside an 18.6 GB GGUF.

LLAMACPP_PIN_COMMIT=${LLAMACPP_PIN_COMMIT:-552f18f912a32ea86edf82e2b76431cb7131538d}
LLAMACPP_PIN_TAG=${LLAMACPP_PIN_TAG:-b11327}
LLAMACPP_REPO_URL=${LLAMACPP_REPO_URL:-https://github.com/ggml-org/llama.cpp}
LLAMACPP_SRC_DIR=${LLAMACPP_SRC_DIR:-}      # set by llamacpp_build
LLAMACPP_BIN=${LLAMACPP_BIN:-}              # set by llamacpp_build (= <dir>/build/bin)
LLAMACPP_SERVER_PID=
LLAMACPP_SERVER_PORT=
LLAMACPP_SERVER_LOG=
_SC1_LLAMACPP_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

_llamacpp_log() { printf '[llamacpp_box %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

llamacpp_build() {
    local dir=$1 commit=${2:-$LLAMACPP_PIN_COMMIT}
    local jobs=${LLAMACPP_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
    local arch=${CUDAARCHS:-120a-real}
    if [ ! -d "$dir/.git" ]; then
        # clone by the TAG for a shallow fetch, then VERIFY the sha -- a tag can be moved, the sha cannot
        git clone --quiet --depth 1 --branch "$LLAMACPP_PIN_TAG" "$LLAMACPP_REPO_URL" "$dir" || return 1
    fi
    if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
        _llamacpp_log "tag $LLAMACPP_PIN_TAG is not $commit; fetching the commit directly"
        git -C "$dir" fetch --quiet --depth 1 origin "$commit" || return 1
        git -C "$dir" checkout --quiet --detach FETCH_HEAD || return 1
    fi
    local head; head=$(git -C "$dir" rev-parse HEAD)
    if [ "$head" != "$commit" ]; then
        _llamacpp_log "REFUSE: $dir is at $head, the pin is $commit"; return 1
    fi
    _llamacpp_log "llama.cpp at $head (tag $(git -C "$dir" describe --tags --exact-match 2>/dev/null || echo none))"
    cmake -S "$dir" -B "$dir/build" -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="$arch" -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$dir/build-configure.log" 2>&1 || { tail -30 "$dir/build-configure.log" >&2; return 1; }
    cmake --build "$dir/build" --config Release -j"$jobs" --target llama llama-server llama-batched-bench llama-bench \
        > "$dir/build-compile.log" 2>&1 || { tail -40 "$dir/build-compile.log" >&2; return 1; }
    bash "$_SC1_LLAMACPP_HERE/build_harness.sh" "$dir" "$dir/build" > "$dir/build-harness.log" 2>&1 || { tail -20 "$dir/build-harness.log" >&2; return 1; }
    LLAMACPP_SRC_DIR=$dir
    LLAMACPP_BIN=$dir/build/bin
    export LLAMACPP_SRC_DIR LLAMACPP_BIN
    export LD_LIBRARY_PATH="$LLAMACPP_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$LLAMACPP_BIN/llama-server" --version > "$dir/build/llama-server.version.txt" 2>&1 || true
    local short=${commit:0:8}
    if ! grep -q "$short" "$dir/build/llama-server.version.txt"; then
        _llamacpp_log "REFUSE: llama-server --version does not name commit $short:"; cat "$dir/build/llama-server.version.txt" >&2; return 1
    fi
    {
        echo "commit: $head"
        echo "cmake: -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=$arch -DLLAMA_CURL=OFF"
        echo "targets: llama llama-server llama-batched-bench llama-bench nll_teacher_forced"
        echo "nvcc: $(nvcc --version 2>/dev/null | tail -1)"
        echo "version:"; cat "$dir/build/llama-server.version.txt"
    } > "$dir/build/BUILD-RECEIPT.txt"
    cat "$dir/build/llama-server.version.txt" >&2
}

llamacpp_fetch() {
    local repo=$1 file=$2 dest=$3 rev=${4:-main}
    mkdir -p "$dest"
    local py=${LLAMACPP_PY:-python3}
    HF_HUB_DISABLE_XET=1 "$py" - "$repo" "$file" "$dest" "$rev" <<'PY' || return 1
import hashlib, json, os, sys, time
from huggingface_hub import HfApi, hf_hub_download
repo, file, dest, rev = sys.argv[1:5]
t0 = time.time()
info = HfApi().model_info(repo, revision=rev, files_metadata=False)
path = hf_hub_download(repo_id=repo, filename=file, revision=info.sha, local_dir=dest)
h = hashlib.sha256()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
rec = {"repo": repo, "file": file, "revision_requested": rev, "revision_sha": info.sha, "path": path,
       "size_bytes": os.path.getsize(path), "sha256": h.hexdigest(), "fetch_s": round(time.time() - t0, 1)}
json.dump(rec, open(path + ".fetch.json", "w"), indent=1)
open(path + ".sha256", "w").write(f"{rec['sha256']}  {os.path.basename(path)}\n")
print(json.dumps(rec))
PY
}

llamacpp_server_start() {
    local gguf=$1 np=$2 port=$3 log=$4 ctx=${5:-}
    [ -n "$ctx" ] || ctx=$(( np * 1024 ))
    local need=$(( np * 640 ))
    if [ "$ctx" -lt "$need" ]; then _llamacpp_log "REFUSE: -c $ctx < np*640 = $need (512 prompt + 128 generated per slot)"; return 1; fi
    [ -n "$LLAMACPP_BIN" ] || { _llamacpp_log "LLAMACPP_BIN unset (run llamacpp_build first)"; return 1; }
    local flags=(-m "$gguf" -ngl 99 -fa on -ctk f16 -ctv f16 -np "$np" --cont-batching -c "$ctx" -b 2048 -ub 512
                 --temp 0 --metrics --host 127.0.0.1 --port "$port" -lm mmap --no-context-shift --no-webui)
    _llamacpp_log "llama-server ${flags[*]}"
    printf 'FLAGS: %s\n' "${flags[*]}" > "$log"
    env | grep -E '^(GGML_|LLAMA_ARG_|CUDA_VISIBLE)' | sed 's/^/ENV: /' >> "$log"
    nohup "$LLAMACPP_BIN/llama-server" "${flags[@]}" >> "$log" 2>&1 &
    LLAMACPP_SERVER_PID=$!
    LLAMACPP_SERVER_PORT=$port
    LLAMACPP_SERVER_LOG=$log
    local deadline=$(( $(date +%s) + ${LLAMACPP_START_TIMEOUT:-900} ))
    while :; do
        if ! kill -0 "$LLAMACPP_SERVER_PID" 2>/dev/null; then _llamacpp_log "server died during startup; log tail:"; tail -30 "$log" >&2; return 1; fi
        if curl -fsS "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok"'; then break; fi
        if [ "$(date +%s)" -ge "$deadline" ]; then _llamacpp_log "server not healthy after timeout"; llamacpp_server_stop; return 1; fi
        sleep 2
    done
    # every layer on the GPU (49/49 for Qwen3-30B-A3B: 48 blocks + output), flash attention enabled, the slot count
    local line n of
    line=$(grep -E 'offloaded [0-9]+/[0-9]+ layers to GPU' "$log" | tail -1)
    n=$(printf '%s' "$line" | sed -E 's/.*offloaded ([0-9]+)\/([0-9]+) layers.*/\1/')
    of=$(printf '%s' "$line" | sed -E 's/.*offloaded ([0-9]+)\/([0-9]+) layers.*/\2/')
    if [ -z "$line" ] || [ "$n" != "$of" ]; then
        _llamacpp_log "REFUSE: not every layer is on the GPU ('${line:-no offload line}')"; llamacpp_server_stop; return 1
    fi
    if ! grep -qE 'flash_attn\s*=\s*enabled' "$log"; then
        _llamacpp_log "REFUSE: the context log does not say flash_attn = enabled"; llamacpp_server_stop; return 1
    fi
    if ! grep -qE "n_slots = $np\b" "$log"; then
        _llamacpp_log "REFUSE: the server did not initialise n_slots = $np"; llamacpp_server_stop; return 1
    fi
    _llamacpp_log "healthy on :$port, $line, pid $LLAMACPP_SERVER_PID"
    llamacpp_server_dump "$port" "$log.start"
}

llamacpp_server_dump() {
    local port=$1 prefix=$2
    curl -fsS "http://127.0.0.1:$port/props"   -o "$prefix.props.json"   2>/dev/null || true
    curl -fsS "http://127.0.0.1:$port/health"  -o "$prefix.health.json"  2>/dev/null || true
    curl -fsS "http://127.0.0.1:$port/metrics" -o "$prefix.metrics.txt"  2>/dev/null || true
    curl -fsS "http://127.0.0.1:$port/slots"   -o "$prefix.slots.json"   2>/dev/null || true
}

llamacpp_server_stop() {
    if [ -n "$LLAMACPP_SERVER_PID" ] && kill -0 "$LLAMACPP_SERVER_PID" 2>/dev/null; then
        kill "$LLAMACPP_SERVER_PID" 2>/dev/null
        local i=0
        while kill -0 "$LLAMACPP_SERVER_PID" 2>/dev/null && [ $i -lt 30 ]; do sleep 1; i=$((i + 1)); done
        kill -9 "$LLAMACPP_SERVER_PID" 2>/dev/null || true
        wait "$LLAMACPP_SERVER_PID" 2>/dev/null || true
        _llamacpp_log "stopped pid $LLAMACPP_SERVER_PID"
    fi
    LLAMACPP_SERVER_PID=
}
