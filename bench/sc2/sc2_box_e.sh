# shellcheck shell=bash
# bench/sc2/sc2_box_e.sh -- lane SC2 (#846), box E: request-level serving (bench/sc2/SC2-PREREG.md). Sourced by
# sc1_run.sh when SC1_BOX=E, after SC1's common setup (pre-flight, the e4b venv and tripwires, the installs). Every helper
# used here (say, line, phase, can_run, arm_alarm, rec, have, unsupported, sampler_*, host_snapshot, fetch, fetch_common,
# bake_qwen3, quiesce, sglang_server_start / _stop, llamacpp_server_start / _stop) is SC1's. Staged flat in $W.
#
# Each engine is served behind its OpenAI-compatible /v1/completions, one at a time, and driven by ONE driver
# (sc2_driver.py) with ONE request plan: identical token-id prompts (sc2_prompts.py), identical max_tokens, greedy,
# ignore_eos, streaming. Per engine: a warm pass, then SC2_DRAWS draws of {Q1 serial, Q2 Poisson at each rate}.
# Engine order: e4b int4, vLLM, SGLang, llama.cpp, then the e4b NF4 default (a labelled row), so the deadline drops it first.

SC2_RATES="1 2 4 8"; SC2_DRAWS=2; SC2_N=120; SC2_SERIAL_N=24; SC2_LO=64; SC2_HI=256; SC2_MAXLEN=2048
PORT_E4B=8778; PORT_VLLM=8000; PORT_SGL=30000; PORT_LL=8080
mkdir -p "$W/sc2"

gpu_free(){ local i; for i in $(seq 1 "${1:-120}"); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ] && return 0; sleep 1; done
  line "SC2 GPU still held after ${1:-120}s: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; return 1; }
fetch_q4km(){ have llamacpp || return 1; mkdir -p "$W/gguf"; say "fetch $GGUF_REPO/$GGUF_Q4KM @ $GGUF_REV"
  LLAMACPP_PY=$PY perl -e "alarm $(arm_alarm 3600); exec @ARGV" bash -c ". $W/llamacpp/llamacpp_box.sh && llamacpp_fetch $GGUF_REPO $GGUF_Q4KM $W/gguf $GGUF_REV" \
      > logs/fetch_gguf_q4km.log 2>&1 || { tail -2 logs/fetch_gguf_q4km.log; line "FETCH gguf $GGUF_Q4KM FAILED"; OK[llamacpp]=0; return 11; }
  line "FETCH gguf $GGUF_Q4KM $(cut -c1-16 "$W/gguf/$GGUF_Q4KM.sha256" 2>/dev/null)"; }
# the driver's client (aiohttp) AND serve_paged's web stack (e4b's `serve` extra: fastapi + uvicorn) into the e4b venv.
# A1: SC1 drove e4b's scheduler in-process and never needed the web stack; sc2-prove-1's e4b server died on
# `No module named 'uvicorn'`. Exact pins; the import check covers all three.
install_sc2_client(){ say "install the SC2 driver's client (aiohttp) and serve_paged's web stack (fastapi, uvicorn) into the e4b venv"
  "$PY" -m pip install -q --no-input "aiohttp==3.14.3" "fastapi==0.141.1" "uvicorn==0.54.0" > logs/pip_aiohttp.log 2>&1 \
    && "$PY" -c "import aiohttp, fastapi, uvicorn" 2>/dev/null \
    && { OK[sc2client]=1; "$PY" -c 'import aiohttp, fastapi, uvicorn; print("aiohttp", aiohttp.__version__, "fastapi", fastapi.__version__, "uvicorn", uvicorn.__version__)' | tee -a versions.txt summary.txt; } \
    || unsupported sc2client "aiohttp / fastapi / uvicorn install failed" logs/pip_aiohttp.log; }
sc2_prompts(){ say "SC2 prompt pool"; perl -e "alarm 1200; exec @ARGV" "$PY" $W/sc2_prompts.py --model "$MID" --revision "$REV" --out $W/sc2/prompts.json \
    > logs/sc2_prompts.log 2>&1 || { tail -3 logs/sc2_prompts.log; return 19; }
  grep -a "^SC2_PROMPTS" logs/sc2_prompts.log | tee -a summary.txt; }
# wait_http URL CAP PID LOG -- poll URL until HTTP 200, the process exits, or CAP seconds pass
wait_http(){ local url=$1 cap=$2 pid=$3 log=$4 t0 code; t0=$(date +%s)
  while :; do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$url" 2>/dev/null || true); [ "$code" = 200 ] && return 0
    kill -0 "$pid" 2>/dev/null || { line "SC2 server exited before $url answered: $(tail -2 "$log" | tr '\n' ' ' | cut -c1-240)"; return 45; }
    [ $(( $(date +%s) - t0 )) -ge "$cap" ] && { line "SC2 server startup timeout ${cap}s at $url"; return 44; }
    sleep 2
  done; }
# wait_e4b_ready URL CAP PID LOG -- serve_paged answers /health with HTTP 200 while it is still building the engine
# ({"status":"loading"}; generation gets 503 until then), so HTTP 200 is not readiness. Poll until status is ready
# (or busy), fail fast on error. A1: wait_http alone would have driven e4b mid-load.
wait_e4b_ready(){ local url=$1 cap=$2 pid=$3 log=$4 t0 st; t0=$(date +%s)
  while :; do
    st=$(curl -s -m 5 "$url" 2>/dev/null | grep -a -o -E '"status": ?"[a-z]+"' | head -1 | grep -a -o -E '[a-z]+"$' | tr -d '"')
    case "$st" in ready|busy) return 0;; error) line "SC2 e4b engine error: $(curl -s -m 5 "$url" | cut -c1-300)"; return 45;; esac
    kill -0 "$pid" 2>/dev/null || { line "SC2 server exited before $url reported ready: $(tail -2 "$log" | tr '\n' ' ' | cut -c1-240)"; return 45; }
    [ $(( $(date +%s) - t0 )) -ge "$cap" ] && { line "SC2 e4b not ready after ${cap}s (status=${st:-none}) at $url"; return 44; }
    sleep 3
  done; }
stop_pid(){ local pid=$1 i; [ -n "$pid" ] || return 0; kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done; kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; }

# ---- the servers (each returns 0 healthy; sets SRV_PID for e4b / vLLM)
e4b_server_start(){ local MODE=$1 MODEL=${2:-$MID} R=${3:-$REV} ARENA=${4:-$QA} LOG=$W/logs/sc2_server_e4b_$1.log LEV=""
  case "$MODE" in int4) LEV="$SPEEDENV E4B_PAGED_FUSE_QKV=1";; nf4) LEV="";; granite) LEV="$GR_ENV";; esac
  # shellcheck disable=SC2086  # assignment lists by design
  setsid env PYTHONPATH= $ROUTEENV $LEV E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$R E4B_PAGED_ARENA=$ARENA E4B_PAGED_CALIB=$W/calib.json \
      E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2_MAXLEN E4B_PAGED_TRACE=$W/sc2/trace_e4b_$MODE.jsonl E4B_HOST=127.0.0.1 E4B_PORT=$PORT_E4B \
      "$PY" -m experts4bit_qlora.serve_paged > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_e4b_ready "http://127.0.0.1:$PORT_E4B/health" 2400 $SRV_PID "$LOG" || return $?
  curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_e4b_$MODE.json || return 45
  line "SC2 e4b ($MODE) healthy: $(cut -c1-300 $W/sc2/health_e4b_$MODE.json)"; }
vllm_server_start(){ local LOG=$W/logs/sc2_server_vllm.log
  setsid env VLLM_LOGGING_LEVEL=INFO "$W/venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port $PORT_VLLM \
      --model "$GPTQ_MID" --revision "$GPTQ_REV" --tokenizer-revision "$GPTQ_REV" --served-model-name sc2 --max-num-seqs 16 \
      --max-model-len $SC2_MAXLEN --no-enable-prefix-caching --seed 0 --gpu-memory-utilization 0.90 > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_http "http://127.0.0.1:$PORT_VLLM/v1/models" 1800 $SRV_PID "$LOG" || return $?
  curl -fsS -m 30 "http://127.0.0.1:$PORT_VLLM/v1/models" -o $W/sc2/models_vllm.json || true
  line "SC2 vllm healthy: $(grep -a -o -i -E 'quantization=[a-z_0-9]+|gptq_marlin|Marlin[A-Za-z]*MoE[A-Za-z]*' "$LOG" | sort | uniq -c | tr '\n' ' ' | cut -c1-240)"; }
sgl_server_start(){ have sglang || return 1
  SGLANG_EXTRA_ARGS="--max-running-requests 16 --dtype float16 --context-length $SC2_MAXLEN --mem-fraction-static $SC1_SGLANG_MEM_FRACTION_STATIC --served-model-name sc2" \
    sglang_server_start "$GPTQ_MID" "$GPTQ_REV" $PORT_SGL "$W/logs/sc2_server_sglang.log" native > logs/sc2_sglang_start.log 2>&1 || return $?
  cp "$W/logs/sc2_server_sglang.log.server_info.json" $W/sc2/server_info_sglang.json 2>/dev/null || true
  cp "$W/logs/sc2_server_sglang.log.engagement.json" $W/sc2/engagement_sglang.json 2>/dev/null || true
  line "SC2 sglang healthy: $(grep -a SGLANG_ENGAGEMENT logs/sc2_sglang_start.log | tail -1 | cut -c1-300)"; }
ll_server_start(){ have llamacpp && [ -s "$W/gguf/$GGUF_Q4KM" ] || return 1
  llamacpp_server_start "$W/gguf/$GGUF_Q4KM" 16 $PORT_LL "$W/logs/sc2_server_llamacpp.log" > logs/sc2_llamacpp_start.log 2>&1 || return $?
  line "SC2 llamacpp healthy: $(tail -1 logs/sc2_llamacpp_start.log | cut -c1-240)"; }

# drive TAG PORT MODEL PROFILE MODE RATE N SEED -- one driver run into sc2/TAG.json
drive(){ local TAG=$1 PORT=$2 MODEL=$3 PROFILE=$4 MODE=$5 RATE=$6 N=$7 SEED=$8 AL rc
  AL=$(arm_alarm 1800)
  perl -e "alarm $AL; exec @ARGV" "$PY" $W/sc2_driver.py run --base "http://127.0.0.1:$PORT" --model "$MODEL" --prompts $W/sc2/prompts.json \
      --mode "$MODE" --rate "$RATE" --n "$N" --seed "$SEED" --max-tokens-lo $SC2_LO --max-tokens-hi $SC2_HI --profile "$PROFILE" \
      --out $W/sc2/$TAG.json > logs/sc2_$TAG.log 2>&1
  rc=$?; line "SC2 $TAG rc=$rc $(grep -a '^SC2_RUN' logs/sc2_$TAG.log | tail -1 | cut -c1-320)"; return $rc; }
# engine NAME PORT MODEL PROFILE -- warm, then every draw of Q1 serial and the Q2 rates
engine_runs(){ local E=$1 PORT=$2 MODEL=$3 PROFILE=$4 D r
  sampler_start sc2_$E
  drive ${E}_warm $PORT "$MODEL" $PROFILE serial 0 4 999
  for D in $(seq 1 $SC2_DRAWS); do
    can_run 300 "$E draw $D serial" && drive ${E}_serial_d$D $PORT "$MODEL" $PROFILE serial 0 $SC2_SERIAL_N $D
    for r in $SC2_RATES; do
      can_run $(( SC2_N / r + 240 )) "$E draw $D rate $r" && drive ${E}_r${r}_d$D $PORT "$MODEL" $PROFILE poisson $r $SC2_N $(( D * 100 + r ))
    done
  done
  sampler_stop sc2_$E; }

# ---- the real lane
box_e(){
  phase 0 "fetches (bf16 for the bake, GPTQ, the Q4_K_M GGUF), bake, the SC2 prompt pool; vLLM / SGLang / llama.cpp / aiohttp installed above"
  fetch_common || finish 11; fetch_q4km; bake_qwen3 || finish 12; sc2_prompts || finish 19
  phase SG0 "SGLang first JIT before any timing"; sgl_server_start && sglang_server_stop > /dev/null 2>&1; gpu_free 180
  quiesce arms
  phase E4B "e4b serve_paged, SC1's int4 levers"
  if can_run 1200 e4b_int4 && e4b_server_start int4; then engine_runs e4b_int4 $PORT_E4B "$MID" e4b; fi; stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  phase VLLM "vLLM 0.30.0 OpenAI server (GPTQ Marlin)"
  if have vllm && can_run 1200 vllm && vllm_server_start; then engine_runs vllm $PORT_VLLM sc2 vllm; fi; stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  phase SGL "SGLang 0.5.20 server (GPTQ Marlin)"
  if can_run 1200 sglang && sgl_server_start; then engine_runs sglang $PORT_SGL sc2 sglang; fi; sglang_server_stop > /dev/null 2>&1; gpu_free 180
  phase LL "llama.cpp server (Q4_K_M)"
  if can_run 1200 llamacpp && ll_server_start; then engine_runs llamacpp $PORT_LL sc2 llamacpp; fi; llamacpp_server_stop > /dev/null 2>&1; gpu_free 120
  phase NF4 "e4b serve_paged, the shipped NF4 default (a labelled row)"
  if can_run 1200 e4b_nf4 && e4b_server_start nf4; then engine_runs e4b_nf4 $PORT_E4B "$MID" e4b; fi; stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  phase RD "the reading"
  "$PY" $W/sc2_reduce.py --dir $W/sc2 --out $W/sc2/verdict.json 2>&1 | tail -30 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): every server, the real comparator checkpoints, e4b on Granite; each must answer the driver
# with every request VALID on a serial smoke and a short Poisson run
prove_e(){ local ok=0
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  "$PY" $W/sc2_driver.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  "$PY" $W/sc2_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  # the prompt pool from the Granite tokenizer (the comparators' ids then exceed nothing: Qwen3's vocabulary is larger)
  perl -e "alarm 900; exec @ARGV" "$PY" $W/sc2_prompts.py --model "$GR" --revision "$GR_REV" --out $W/sc2/prompts.json > logs/sc2_prompts.log 2>&1 \
    || { tail -3 logs/sc2_prompts.log; say "PROVE: prompt pool failed -- NOT PROVED"; rec 23; return; }
  smoke(){ local E=$1 PORT=$2 MODEL=$3 PROFILE=$4
    drive ${E}_prove_serial $PORT "$MODEL" $PROFILE serial 0 6 1 && drive ${E}_prove_poisson $PORT "$MODEL" $PROFILE poisson 4 16 2 \
      || { say "PROVE: $E smoke failed -- NOT PROVED"; ok=1; }; }
  if e4b_server_start granite "$GR" "$GR_REV" "$W/work_granite/nf4.arena"; then smoke e4b_granite $PORT_E4B "$GR" e4b; else say "PROVE: e4b server failed"; ok=1; fi
  stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  if fetch gptq "$GPTQ_MID" "$GPTQ_REV" 1800; then
    if vllm_server_start; then smoke vllm $PORT_VLLM sc2 vllm; else say "PROVE: vLLM server failed"; ok=1; fi
    stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
    if sgl_server_start; then smoke sglang $PORT_SGL sc2 sglang; else say "PROVE: SGLang server failed"; ok=1; fi
    sglang_server_stop > /dev/null 2>&1; gpu_free 180
  else say "PROVE: GPTQ fetch failed -- NOT PROVED"; ok=1; fi
  if fetch_q4km && ll_server_start; then smoke llamacpp $PORT_LL sc2 llamacpp; else say "PROVE: llama.cpp server failed"; ok=1; fi
  llamacpp_server_stop > /dev/null 2>&1; gpu_free 120
  [ $ok = 0 ] || rec 23; }
