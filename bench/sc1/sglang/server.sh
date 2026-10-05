#!/bin/bash
# bench/sc1/sglang/server.sh -- SOURCE me. The SGLang server of lane SC1 (bench/sc1/SC1-PREREG.md, experts4bit-qlora#846).
#
#   sglang_server_start <model_dir_or_id> <revision> <port> <log> <mode>     mode: matched | native | kvfp8 | ttft_matched | quality
#   sglang_server_stop
#
# Every flag spelling below was read from the v0.5.20 field declarations (python/sglang/srt/arg_groups/fields/*.py; the CLI
# name is the field name with underscores -> dashes): disable_radix_cache (memory.py:65), max_running_requests
# (schedule.py:38), chunked_prefill_size (schedule.py:59, -1 = no chunking), schedule_policy (schedule.py:89, choice fcfs),
# cuda_graph_bs_decode (exec_.py:488, explicit capture list), cuda_graph_backend_prefill (exec_.py:475, choice disabled),
# moe_runner_backend (exec_.py:683, default auto), dtype (model.py:159, choice float16 since A9), context_length (model.py:109),
# kv_cache_dtype (model.py:196, choice fp8_e4m3), random_seed (device.py:43), revision (model.py:121), host/port
# (serving.py:72-73). There is no --cuda-graph-max-bs / --cuda-graph-bs any more (per-phase flags only).
#
# Modes (the draft's SGLang block, "Environments" + "Validity"):
#   matched       --disable-radix-cache --max-running-requests 16 --chunked-prefill-size -1 --schedule-policy fcfs
#                 --cuda-graph-bs-decode 1 16 --moe-runner-backend auto --dtype float16 --context-length 2048 --random-seed 0
#                 (2048 holds the 512-token rows + 128 new: tokenizer_manager.py:1260-1305 refuses input >= context_len and
#                 input + max_new_tokens > context_len)
#   kvfp8         matched + --kv-cache-dtype fp8_e4m3 (FlashInfer takes the kv dtype straight: flashinfer_backend.py:356-367)
#   ttft_matched  matched + --cuda-graph-backend-prefill disabled (prefill graphs are ON by default in 0.5.20:
#                 model_executor/cuda_graph_config.py:112-120 `breakable`) and --context-length 4608 (4096-token prompt + 8 new)
#   native        SGLang's defaults + --disable-radix-cache ONLY (kept ON for every timed arm): chunked prefill 4096 and decode
#                 graph max_bs 48 from the 20-35 GB memory tier (arg_groups/memory_hook.py:100-116), capture list
#                 [1,2,4,8,12,16,24,32,40,48] (cuda_graph_hook.py:577-608; 16 is native), prefill graphs `breakable`,
#                 overlap scheduler ON, max_running_requests derived (mem_cache/kv_cache_configurator.py resolve_max_num_reqs)
#   quality       the NLL scorer's server: radix cache ON (default), --max-running-requests 1, --chunked-prefill-size -1
#                 (the 2561-token prefill-shaped request is one extend), --context-length 4096, --dtype float16 --random-seed 0
#   A8 (receipt sc1c-prove-13): matched / kvfp8 / ttft_matched / quality also pass --mem-fraction-static 0.75. With
#                 --chunked-prefill-size -1, v0.5.20's own rule (arg_groups/memory_hook.py handle_gpu_memory_settings) reserves
#                 512 + 1.5 MB x max(max_prefill_tokens = 16384, 2048) + 128 + 2 x decode max_bs MB of activations -- 25.2 GB of a
#                 32,607 MiB 5090, i.e. mem_fraction_static 0.226, below the 0.514 the 15.7 GB GPTQ checkpoint alone needs ("Loaded
#                 weights leave no GPU memory for the KV cache"). 0.75 gives a ~7.3 GiB KV pool (~79,800 16-bit tokens: each server's
#                 registered capacity, 16 x 2048 and the TTFT server's 16 x 4608) and keeps 7.7 GiB outside the static pool, >= 5x
#                 the peak activation + CUDA-graph memory vLLM 0.30 measured for the same checkpoint, card and 8192-token batch
#                 (<= 0.86 + 0.50 GiB, sc1a-5090-1). The fraction sizes the static pool only; no kernel or scheduling flag changes.
#                 native keeps SGLang's own resolution (chunked prefill on: ~0.79 on this card).
#   A9 (receipt sc1c-prove-14): every pinned mode runs --dtype float16 (was bfloat16). The lane's GPTQ checkpoint declares
#                 torch_dtype float16 and stores float16 scales, and v0.5.20's GPTQ Marlin MoE asserts the activations are in the
#                 scales' dtype (layers/moe/fused_moe_triton/fused_marlin_moe.py: "moe_wna16_marlin_gemm assumes
#                 hidden_states.dtype == w1_scale.dtype"). vLLM's arms pass no dtype and `auto` resolves float16 on the same
#                 checkpoint (sc1a-5090-1), so float16 is also the matched activation dtype. native passes none (auto = float16).
#
# Readiness + engagement (refused with a non-zero return otherwise):
#   * GET /health -> 200 (http_server.py:669-735; 503 while ServerStatus.Starting, i.e. until the startup warmup request has
#     run and set Up: http_server.py:2210-2260, 2423), then GET /server_info (http_server.py:818-850; /get_server_info is the
#     deprecated alias at 808) dumped to <log>.server_info.json = server_args.resolved_dict() + launch_command +
#     scheduler_info + version.
#   * attention_backend resolved to "flashinfer" in that dump (arg_groups/model_override_base.py:295-351: on an MHA model that
#     is neither Hopper nor SM100, flashinfer when available; sm_120 lands there).
#   * the GPTQ -> Marlin upgrade banner in the log: "The model is convertible to gptq_marlin during runtime. Using gptq_marlin
#     kernel." (layers/quantization/gptq/gptq.py:414-428 `override_quantization_method`); the MoE method is then
#     GPTQMarlinMoEMethod -> MoeRunner(MoeRunnerBackend.MARLIN) (gptq.py:440-447; hardware_backend/gpu/quantization/
#     gptq_kernels.py:306-311, which ASSERTS --moe-runner-backend auto).
#   * the Marlin MoE JIT: the loader logs NOTHING at INFO on a build or a cache hit (kernels/jit/utils/compile/loader.py:48-182
#     has only two warnings; cache.py logs at DEBUG), so the proof is ON DISK -- the startup warmup (8 generated tokens,
#     http_server.py:2258) is the first MoE call and compiles
#     $SGLANG_JIT_CACHE_DIR/<sm120f|sm120a>/sgl_kernel_jit_moe_wna16_marlin_<dtype>_<ep>_<bias>/build-*/deps-*/
#     {<module>.so, build.ninja, sgl_deps.json} (spec.py:20,55-57 module_name = "sgl_kernel_jit_" + "_".join(args);
#     cache.py:16-23 layout, :210-221 target tag, :512-527 the staging dir incl. build.ninja is renamed into the leaf).
#     build.ninja carries the target: -gencode=arch=compute_120f,code=sm_120f (toolchain.py:113-124). A failed build raises
#     "Failed to build JIT module ..." (ninja.py:167-171) -> the warmup fails -> the server never becomes healthy.
set -uo pipefail
SGLANG_WORK=${SGLANG_WORK:-/root/sc1}
SGLANG_VENV=${SGLANG_VENV:-$SGLANG_WORK/venv-sglang}
SGLANG_ENV_FILE=${SGLANG_ENV_FILE:-$SGLANG_WORK/sglang.env}
SGLANG_START_TIMEOUT=${SGLANG_START_TIMEOUT:-1800}   # model load + Marlin JIT (nvcc, minutes) + graph capture + warmup
SC1_SGLANG_MEM_FRACTION_STATIC=0.75   # A8: pinned on every chunking-off mode (see the header); deliberately not env-overridable
SGLANG_SERVER_PID=""; SGLANG_SERVER_PGID=""; SGLANG_SERVER_PORT=""; SGLANG_SERVER_LOG=""
_sgl_say(){ echo "[$(date -u +%FT%TZ)] sglang-server: $*"; }

sglang_server_flags(){  # <mode> <port> -> echoes the flag list (one per line; the common block first)
  local MODE=$1 PORT=$2
  printf '%s\n' --host 127.0.0.1 --port "$PORT" --random-seed 0 --moe-runner-backend auto --log-level info
  case "$MODE" in
    matched)      printf '%s\n' --disable-radix-cache --max-running-requests 16 --chunked-prefill-size -1 --schedule-policy fcfs \
                                --cuda-graph-bs-decode 1 16 --dtype float16 --context-length 2048 --mem-fraction-static "$SC1_SGLANG_MEM_FRACTION_STATIC" ;;
    kvfp8)        printf '%s\n' --disable-radix-cache --max-running-requests 16 --chunked-prefill-size -1 --schedule-policy fcfs \
                                --cuda-graph-bs-decode 1 16 --dtype float16 --context-length 2048 --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$SC1_SGLANG_MEM_FRACTION_STATIC" ;;
    ttft_matched) printf '%s\n' --disable-radix-cache --max-running-requests 16 --chunked-prefill-size -1 --schedule-policy fcfs \
                                --cuda-graph-bs-decode 1 16 --dtype float16 --context-length 4608 --cuda-graph-backend-prefill disabled --mem-fraction-static "$SC1_SGLANG_MEM_FRACTION_STATIC" ;;
    native)       printf '%s\n' --disable-radix-cache ;;
    gptoss)       printf '%s\n' --disable-radix-cache ;;   # SC2g (bench/sc2): gpt-oss-20b's own MXFP4 path at SGLang's defaults
    quality)      printf '%s\n' --max-running-requests 1 --chunked-prefill-size -1 --dtype float16 --context-length 4096 --mem-fraction-static "$SC1_SGLANG_MEM_FRACTION_STATIC" ;;
    *) return 1 ;;
  esac
}

sglang_server_start(){
  local MODEL=$1 REV=$2 PORT=$3 LOG=$4 MODE=$5
  [ -f "$SGLANG_ENV_FILE" ] && . "$SGLANG_ENV_FILE"
  export SGLANG_CACHE_DIR=${SGLANG_CACHE_DIR:-$SGLANG_WORK/sglang-cache} SGLANG_JIT_CACHE_DIR=${SGLANG_JIT_CACHE_DIR:-$SGLANG_WORK/sglang-cache/jit}
  export PATH=$SGLANG_VENV/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
  [ -x "$SGLANG_VENV/bin/python" ] || { _sgl_say "refusing: no venv at $SGLANG_VENV"; return 78; }
  local -a FLAGS; mapfile -t FLAGS < <(sglang_server_flags "$MODE" "$PORT") || { _sgl_say "refusing: unknown mode $MODE"; return 78; }
  [ ${#FLAGS[@]} -gt 0 ] || { _sgl_say "refusing: unknown mode $MODE"; return 78; }
  # shellcheck disable=SC2206
  local -a EXTRA=(${SGLANG_EXTRA_ARGS:-})
  local -a CMD=("$SGLANG_VENV/bin/python" -m sglang.launch_server --model-path "$MODEL" --revision "$REV" "${FLAGS[@]}" "${EXTRA[@]}")
  { echo "# sc1 sglang server mode=$MODE at=$(date -u +%FT%TZ)"; echo "# cmd: ${CMD[*]}"; echo "# SGLANG_CACHE_DIR=$SGLANG_CACHE_DIR SGLANG_JIT_CACHE_DIR=$SGLANG_JIT_CACHE_DIR CUDA_HOME=${CUDA_HOME:-}"; } > "$LOG"
  _sgl_say "start mode=$MODE port=$PORT log=$LOG"; _sgl_say "cmd: ${CMD[*]}"
  local t0; t0=$(date +%s)
  # SC1b: an optional launch prefix (`nsys launch --session-new=...`), empty unless set, so SC1's command is unchanged
  local -a PREFIX=(${SC1_LAUNCH_PREFIX:-})
  setsid ${PREFIX[@]+"${PREFIX[@]}"} "${CMD[@]}" >> "$LOG" 2>&1 < /dev/null &
  SGLANG_SERVER_PID=$!; SGLANG_SERVER_PORT=$PORT; SGLANG_SERVER_LOG=$LOG
  SGLANG_SERVER_PGID=$(ps -o pgid= -p "$SGLANG_SERVER_PID" 2>/dev/null | tr -d ' ')
  local code="" last=0
  while :; do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
    [ "$code" = 200 ] && break
    if ! kill -0 "$SGLANG_SERVER_PID" 2>/dev/null; then _sgl_say "server process exited before /health (mode=$MODE)"; tail -25 "$LOG" | cut -c1-300; return 45; fi
    local now; now=$(date +%s)
    if [ $((now - t0)) -ge "$SGLANG_START_TIMEOUT" ]; then _sgl_say "startup timeout ${SGLANG_START_TIMEOUT}s (last /health=$code)"; tail -25 "$LOG" | cut -c1-300; sglang_server_stop; return 44; fi
    if [ $((now - last)) -ge 30 ]; then last=$now; _sgl_say "waiting ($((now - t0))s, /health=$code): $(grep -av '^\s*$' "$LOG" | tail -1 | cut -c1-160)"; fi
    sleep 2
  done
  local startup_s=$(( $(date +%s) - t0 )); _sgl_say "healthy after ${startup_s}s"
  curl -fsS -m 30 "http://127.0.0.1:$PORT/server_info" -o "$LOG.server_info.json" 2>/dev/null \
    || curl -fsS -m 30 "http://127.0.0.1:$PORT/get_server_info" -o "$LOG.server_info.json" 2>/dev/null \
    || { _sgl_say "no /server_info"; sglang_server_stop; return 45; }
  # ---- engagement: resolved args, the gptq_marlin banner, no JIT build failure, the Marlin MoE leaf on disk
  SC1_MODE=$MODE SC1_MFS=$SC1_SGLANG_MEM_FRACTION_STATIC SC1_LOG=$LOG SC1_STARTUP_S=$startup_s SC1_JIT=$SGLANG_JIT_CACHE_DIR SC1_CMD="${CMD[*]}" "$SGLANG_VENV/bin/python" - <<'PYT'
import glob, json, os, re, sys
mode, log, jit = os.environ["SC1_MODE"], os.environ["SC1_LOG"], os.environ["SC1_JIT"]
info = json.load(open(log + ".server_info.json"))
text = open(log, errors="replace").read()
eng = {"mode": mode, "startup_s": int(os.environ["SC1_STARTUP_S"]), "cmd": os.environ["SC1_CMD"], "errors": []}
def need(cond, msg):
    if not cond: eng["errors"].append(msg)
eng["version"] = info.get("version"); need(info.get("version") == "0.5.20", f"server version {info.get('version')} != 0.5.20")
want_attn = "triton" if mode == "gptoss" else "flashinfer"   # SC2g: SGLang forces its triton kernels for gpt-oss's sinks + window
eng["attention_backend"] = info.get("attention_backend"); need(info.get("attention_backend") == want_attn, f"attention_backend resolved to {info.get('attention_backend')!r}, not {want_attn}")
for k in ("prefill_attention_backend", "decode_attention_backend", "kv_cache_dtype", "disable_radix_cache", "max_running_requests",
          "chunked_prefill_size", "schedule_policy", "context_length", "dtype", "quantization", "moe_runner_backend",
          "speculative_algorithm", "disable_overlap_schedule", "cuda_graph_config", "page_size", "mem_fraction_static",
          "max_total_num_tokens", "max_prefill_tokens", "random_seed", "model_path", "revision"):
    eng[k] = info.get(k)
radix_off = mode in ("matched", "kvfp8", "ttft_matched", "native", "gptoss")
need(bool(info.get("disable_radix_cache")) == radix_off, f"disable_radix_cache={info.get('disable_radix_cache')} but mode {mode} expects {radix_off}")
if mode == "kvfp8": need(info.get("kv_cache_dtype") == "fp8_e4m3", f"kv_cache_dtype={info.get('kv_cache_dtype')}")
if mode in ("matched", "kvfp8", "ttft_matched"): need(info.get("max_running_requests") == 16, f"max_running_requests={info.get('max_running_requests')}")
if mode == "quality": need(info.get("max_running_requests") == 1, f"max_running_requests={info.get('max_running_requests')}")
if mode in ("matched", "kvfp8", "ttft_matched", "quality"):  # A8: the pinned static pool, and the capacity it has to hold
    mfs = float(os.environ["SC1_MFS"])
    need(abs(float(info.get("mem_fraction_static") or 0) - mfs) < 1e-9, f"mem_fraction_static={info.get('mem_fraction_static')} != {mfs} (A8)")
    cap = int(info.get("context_length") or 0) * (16 if mode in ("matched", "kvfp8") else 1)
    need(int(info.get("max_total_num_tokens") or 0) >= cap, f"max_total_num_tokens={info.get('max_total_num_tokens')} < {cap}, the registered capacity (A8)")
    need(info.get("dtype") == "float16", f"dtype={info.get('dtype')!r}, not float16: the GPTQ Marlin MoE needs the activations in the scales' dtype (A9)")
cg = json.dumps(info.get("cuda_graph_config"), default=str)
eng["cuda_graph_config_json"] = cg[:2000]
if mode == "ttft_matched": need('"disabled"' in cg or "DISABLED" in cg.upper(), "prefill cuda graphs not disabled in the resolved cuda_graph_config")
banner = "The model is convertible to gptq_marlin during runtime. Using gptq_marlin kernel."
eng["gptq_marlin_banner"] = banner in text
if mode != "gptoss":   # the GPTQ checkpoint's engagement; gpt-oss's MXFP4 path records moe_runner_backend / quantization instead
    need(banner in text, "gptq_marlin upgrade banner missing from the log (gptq.py:424-428)")
bad = [l for l in text.splitlines() if "Failed to build JIT module" in l]
eng["jit_build_failures"] = bad[:3]; need(not bad, "JIT build failure in the log")
leaves = sorted(glob.glob(os.path.join(jit, "*", "*moe_wna16_marlin*", "build-*", "deps-*", "*.so")))
eng["marlin_moe_jit_leaves"] = leaves
if mode != "gptoss":
    need(bool(leaves), f"no moe_wna16_marlin JIT leaf under {jit} (the startup warmup should have compiled it)")
if leaves:
    leaf = os.path.dirname(leaves[-1]); eng["jit_target_tag"] = leaf.split(os.sep)[-4]
    try:
        bn = open(os.path.join(leaf, "build.ninja")).read()
        m = re.search(r"-gencode=arch=compute_(\w+),code=sm_(\w+)", bn); eng["gencode"] = m.group(0) if m else None
        need(m is not None and m.group(2) in ("120f", "120a"), f"build.ninja target {m.group(0) if m else None} is not sm_120f/sm_120a")
    except OSError as e: eng["gencode"] = None; eng["errors"].append(f"cannot read build.ninja in {leaf}: {e}")
eng["fired_up"] = "The server is fired up and ready to roll!" in text
sa = re.search(r"server_args=\{.*", text); eng["server_args_log_line"] = (sa.group(0)[:3000] if sa else None)
json.dump(eng, open(log + ".engagement.json", "w"), indent=1)
print("SGLANG_ENGAGEMENT " + json.dumps({k: eng.get(k) for k in ("mode", "version", "attention_backend", "kv_cache_dtype", "disable_radix_cache", "max_running_requests", "gptq_marlin_banner", "jit_target_tag", "gencode", "startup_s")}))
for e in eng["errors"]: print("SGLANG_ENGAGEMENT REFUSED: " + e)
sys.exit(43 if eng["errors"] else 0)
PYT
  local rc=$?
  if [ $rc -ne 0 ]; then _sgl_say "engagement refused (rc=$rc) -- stopping"; sglang_server_stop; return 43; fi
  _sgl_say "ready: $LOG.server_info.json $LOG.engagement.json"
  return 0
}

sglang_server_stop(){
  local pid=${SGLANG_SERVER_PID:-} pgid=${SGLANG_SERVER_PGID:-} port=${SGLANG_SERVER_PORT:-}
  [ -n "$pid" ] || return 0
  _sgl_say "stop pid=$pid pgid=$pgid"
  [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
  local i; for i in $(seq 1 "${SGLANG_STOP_TERM_S:-30}"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then _sgl_say "SIGKILL"; [ -n "$pgid" ] && kill -KILL -- "-$pgid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; fi
  # the scheduler / detokenizer children are spawned processes; sweep anything still bound to this launch's port
  [ -n "$port" ] && pkill -KILL -f "sglang.launch_server.*--port $port( |$)" 2>/dev/null
  # A10: never an unbounded `wait`. A process stuck in the GPU driver sits in uninterruptible sleep and ignores SIGKILL;
  # `wait` on it never returns (sc1c-5090-1 stalled with the server's memory resident and the GPU idle). Poll, reap
  # only a process that has exited, and report a stuck one through SGLANG_STOP_STUCK so the lane skips later starts.
  SGLANG_STOP_STUCK=""
  for i in $(seq 1 "${SGLANG_STOP_WAIT_S:-60}"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    SGLANG_STOP_STUCK=$pid
    _sgl_say "STOP STUCK: pid $pid still present ${SGLANG_STOP_WAIT_S:-60} s after SIGKILL (uninterruptible?) -- not waiting on it"
  else
    wait "$pid" 2>/dev/null
  fi
  for i in $(seq 1 30); do curl -s -o /dev/null -m 2 "http://127.0.0.1:$port/health" 2>/dev/null || break; sleep 1; done
  SGLANG_SERVER_PID=""; SGLANG_SERVER_PGID=""; SGLANG_SERVER_PORT=""
  return 0
}
