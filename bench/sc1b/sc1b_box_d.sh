# shellcheck shell=bash
# bench/sc1b/sc1b_box_d.sh -- lane SC1b (#846), box D: the census box (bench/sc1b/SC1b-PREREG.md). Sourced by sc1_run.sh
# when SC1_BOX=D, after SC1's common setup (pre-flight, e4b venv + tripwires, installs); every helper used here (say, line,
# phase, can_run, arm_alarm, rec, have, unsupported, sched_env, sampler_*, host_snapshot, the SC1 arm recipes, vllm_arm,
# sglang_up/down, sglang_server_start/stop, llamacpp_server_start/stop, fetch, prompts, bake_qwen3, fetch_common) is SC1's.
# Staged flat in $W beside sc1_run.sh.
#
# Per (engine, B): pass 1 = SC1's own unprofiled arm (one draw: this box's speed); pass 2 = an `nsys profile` graph-mode
# capture; pass 3 = an `nsys profile` node-mode capture. e4b / vLLM / SGLang bracket 64 steady decode steps with their own
# cudaProfilerStart/Stop; llama.cpp (no hook) is captured whole and the reducer selects the registered positions.
# Every capture ends with the app exiting on its own or by SIGTERM to the APP (never to nsys), then nsys exiting (the
# report is written), then the GPU found empty.

NSYS_VER=2025.6.1; NSYS=""
NSYS_TRACE="-t cuda-sw,nvtx --sample=none --cpuctxsw=none"
SC1B_SKIP=32; SC1B_STEPS=64; SC1B_TOKENS=160; VOCAB=151936; LL_POSITIONS=33:64
mkdir -p "$W/census"

install_nsys(){ say "install nsight-systems-cli-$NSYS_VER (NVIDIA devtools apt repo; the image's own nsys is Nsight Compute's, never used)"
  { apt-get update -qq && apt-get install -y -qq --no-install-recommends gnupg2 wget ca-certificates \
    && wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | gpg --dearmor > /usr/share/keyrings/nvidia-devtools-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu$(. /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /" > /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-get update -qq && apt-get install -y -qq --no-install-recommends "nsight-systems-cli-$NSYS_VER"; } > logs/install_nsys.log 2>&1
  NSYS=$(dpkg -L "nsight-systems-cli-$NSYS_VER" 2>/dev/null | grep -E '/nsys$' | head -1)
  if [ -n "$NSYS" ] && [ -x "$NSYS" ]; then
    OK[nsys]=1; echo "nsys $("$NSYS" --version 2>&1 | tail -1) at $NSYS" | tee -a versions.txt summary.txt
    "$NSYS" status -e >> logs/install_nsys.log 2>&1; { echo "nsys status -e:"; tail -12 logs/install_nsys.log; } >> forensics.txt
  else unsupported nsys "install failed: $(tail -1 logs/install_nsys.log | cut -c1-160)" logs/install_nsys.log; fi; }

# ---- capture plumbing -------------------------------------------------------------------------------------------------
nsys_export(){ local rep=$1; [ -s "$rep.nsys-rep" ] || { say "no report $rep.nsys-rep"; return 1; }
  "$NSYS" export --type sqlite --force-overwrite true -o "$rep.sqlite" "$rep.nsys-rep" > "logs/export_$(basename "$rep").log" 2>&1; }
# gpu_free [S] -- wait (bounded) until no process holds the GPU; a holder after the bound is named and the step fails
gpu_free(){ local i; for i in $(seq 1 "${1:-120}"); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ] && return 0; sleep 1; done
  line "SC1B GPU still held after ${1:-120}s: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; return 1; }
# end_capture NSYS_PID APP_PATTERN -- SIGTERM the traced APP (never nsys), wait for nsys to write the report and exit. Only
# DESCENDANTS of nsys are candidates (the app is nsys's child): a bare `pkill -f` also matches nsys itself (its command line
# carries the app's) and any unrelated process whose command line happens to hold the pattern, such as a calling shell
_descendants(){ local c; for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; _descendants "$c"; done; }
end_capture(){ local npid=$1 pat=$2 i p kids
  kids=" $(_descendants "$npid" | tr '\n' ' ') "
  for p in $(pgrep -f "$pat" 2>/dev/null); do
    case "$kids" in *" $p "*) ;; *) continue;; esac
    case "$(cat "/proc/$p/comm" 2>/dev/null)" in nsys*|setsid|perl|"") continue;; esac
    kill -TERM "$p" 2>/dev/null
  done
  for i in $(seq 1 300); do kill -0 "$npid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$npid" 2>/dev/null; then line "SC1B nsys pid $npid still running 300 s after the app's SIGTERM"; return 1; fi; }
clock_of(){ "$PY" -c "
import csv, statistics, sys
v = [float(r[3]) for r in csv.reader(open(sys.argv[1])) if len(r) > 3 and r[3].strip().replace('.', '', 1).isdigit()]
print(round(statistics.median(v)) if v else '')" "$W/samples/$1.csv" 2>/dev/null; }

# e4b_census B MODE [STACK MODEL REV ARENA TAG FUSE] -- nsys profile around sc1b_e4b_census.py with the sched arm's exact
# env (SC1's int4_sched: SPEEDENV, fuse_qkv 1); the proof's Granite capture passes GR_ENV and fuse_qkv 0, as SC1's smokes do
e4b_census(){ local B=$1 MODE=$2 STACK=${3:-$SPEEDENV} M=${4:-$MID} R=${5:-$REV} ARENA=${6:-$QA} TAG=${7:-e4b} FUSE=${8:-1}
  local S=census_${TAG}_b${B}_$MODE AL; AL=$(arm_alarm 1800); say "census $S (alarm=$AL)"; sampler_start $S
  { echo "SC1B census=$S at=$(date -u +%FT%TZ) $(host_snapshot)"; echo "ENV: $ROUTEENV $STACK $(sched_env "$B" 2048 "$FUSE" "$M" "$R" "$ARENA")"; } > logs/run_$S.log
  env PYTHONPATH= $ROUTEENV $STACK $(sched_env "$B" 2048 "$FUSE" "$M" "$R" "$ARENA") E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA SC1_ARM=$S SC1_BATCH=$B \
      SC1_PROMPTS=$W/prompts_b$B.json SC1_OUT=$W/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID \
    perl -e "alarm $AL; exec @ARGV" "$NSYS" profile $NSYS_TRACE --cuda-graph-trace=$MODE --capture-range=cudaProfilerApi \
      --capture-range-end=stop --force-overwrite true -o "$W/census/$S" "$PY" "$W/sc1b_e4b_census.py" --skip $SC1B_SKIP --steps $SC1B_STEPS >> logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  pkill -TERM -f "sc1b_e4b_census.py" 2>/dev/null; gpu_free 120 || rc=${rc/#0/32}      # perl's alarm kills nsys, not the app
  [ "$rc" = 0 ] && { nsys_export "$W/census/$S" || rc=30; }
  line "$S B=$B mode=$MODE rc=$rc $(grep -a 'SC1B_E4B' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }

# vllm_census B MODE -- nsys profile around sc1b_vllm_census.py (SC1's graph_r1 kwargs + profiler_config cuda); spawn workers
vllm_census(){ local B=$1 MODE=$2 S=census_vllm_b$1_$2 AL; AL=$(arm_alarm 1800); say "census $S (alarm=$AL)"; sampler_start $S
  { echo "SC1B census=$S at=$(date -u +%FT%TZ) $(host_snapshot)"; } > logs/run_$S.log
  env SC1_ARM=graph_r1 SC1_BATCH=$B SC1_PROMPTS=$W/prompts_b$B.json SC1_MODEL=$GPTQ_MID SC1_REV=$GPTQ_REV SC1_OUT=$W/$S.json \
      SC1_LOG=$W/logs/run_$S.engine.log VLLM_LOGGING_LEVEL=INFO VLLM_WORKER_MULTIPROC_METHOD=spawn \
    perl -e "alarm $AL; exec @ARGV" "$NSYS" profile $NSYS_TRACE --cuda-graph-trace=$MODE --capture-range=cudaProfilerApi \
      --capture-range-end=stop --force-overwrite true -o "$W/census/$S" "$W/venv-vllm/bin/python" "$W/sc1b_vllm_census.py" \
      --skip $SC1B_SKIP --steps $SC1B_STEPS --tokens $SC1B_TOKENS >> logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  pkill -TERM -f "sc1b_vllm_census.py" 2>/dev/null; gpu_free 180 || rc=${rc/#0/32}
  [ "$rc" = 0 ] && { nsys_export "$W/census/$S" || rc=30; }
  line "$S B=$B mode=$MODE rc=$rc $(grep -a 'SC1B_VLLM' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }

# sglang_census B MODE -- a matched server under `nsys profile -c cudaProfilerApi --capture-range-end=stop` (one capture per
# launch); the client posts SC1's batch and /start_profile on the clock; then the server is SIGTERMed and nsys writes the report
sglang_census(){ local B=$1 MODE=$2 S=census_sglang_b$1_$2 AL; AL=$(arm_alarm 900)
  have sglang && [ -z "$SGL_DEAD" ] || { line "$S SKIPPED (no sglang)"; return 0; }
  say "census $S (alarm=$AL)"; line "sglang census server $S starting"
  SC1_LAUNCH_PREFIX="$NSYS profile $NSYS_TRACE --cuda-graph-trace=$MODE --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true -o $W/census/$S" \
    sglang_server_start "$GPTQ_MID" "$GPTQ_REV" 30000 "$W/logs/sglang_server_$S.log" matched > logs/sglang_start_$S.log 2>&1 \
    || { line "$S FAILED server start: $(tail -1 logs/sglang_start_$S.log | cut -c1-160)"; sglang_server_stop > /dev/null 2>&1; gpu_free 180; return 31; }
  local npid=$SGLANG_SERVER_PID; sampler_start $S
  perl -e "alarm $AL; exec @ARGV" "$PY" "$W/sc1b_serve_census.py" --engine sglang --batch "$B" --prompts "$W/prompts_b$B.json" --port 30000 \
      --pass1 "$W/sglang_gptq_matched_b${B}_r1.json" --skip $SC1B_SKIP --steps $SC1B_STEPS --tokens $SC1B_TOKENS --out "$W/$S.json" > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  end_capture "$npid" "sglang.launch_server.*--port 30000" || rc=${rc/#0/33}
  sglang_server_stop > logs/sglang_stop_$S.log 2>&1; gpu_free 180 || rc=${rc/#0/32}
  [ "$rc" = 0 ] && { nsys_export "$W/census/$S" || rc=30; }
  line "$S B=$B mode=$MODE rc=$rc $(grep -a 'SC1B_SERVE' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }

# llamacpp_census B MODE -- llama-server -np B under plain `nsys profile`: the whole 160-token run is captured; SIGTERM to
# llama-server ends it and nsys writes the report; the reducer selects positions LL_POSITIONS by counting logits copies
llamacpp_census(){ local B=$1 MODE=$2 S=census_llamacpp_b$1_$2 AL; AL=$(arm_alarm 1200)
  have llamacpp && [ -s "$W/gguf/$GGUF_Q4KM" ] || { line "$S SKIPPED no llama.cpp build or GGUF"; return 0; }
  say "census $S (alarm=$AL)"
  SC1_LAUNCH_PREFIX="$NSYS profile $NSYS_TRACE --cuda-graph-trace=$MODE --force-overwrite true -o $W/census/$S" \
    llamacpp_server_start "$W/gguf/$GGUF_Q4KM" "$B" 8080 "$W/logs/llamacpp_server_$S.log" > logs/llamacpp_start_$S.log 2>&1 \
    || { line "$S FAILED server start: $(tail -1 logs/llamacpp_start_$S.log | cut -c1-160)"; llamacpp_server_stop > /dev/null 2>&1; gpu_free 120; return 31; }
  local npid=$LLAMACPP_SERVER_PID; sampler_start $S
  perl -e "alarm $AL; exec @ARGV" "$PY" "$W/sc1b_serve_census.py" --engine llamacpp --batch "$B" --prompts "$W/prompts_b$B.json" --port 8080 \
      --tokens $SC1B_TOKENS --out "$W/$S.json" > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  end_capture "$npid" "llama-server.*--port 8080" || rc=${rc/#0/33}
  llamacpp_server_stop > /dev/null 2>&1; gpu_free 120 || rc=${rc/#0/32}
  [ "$rc" = 0 ] && { nsys_export "$W/census/$S" || rc=30; }
  line "$S B=$B mode=$MODE rc=$rc $(grep -a 'SC1B_SERVE' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }

# ---- the reading ------------------------------------------------------------------------------------------------------
# sc1b_arm ENGINE B PASS1_STEM [DELIM] -- census arm record from the two exports, pass 1's median ms/step, the three clocks
sc1b_arm(){ local E=$1 B=$2 P1=$3 DELIM=${4:-graph} U POS="" BYTES=""
  U=$("$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); print(r.get('decode_ms_per_step_median') or r.get('decode_ms_per_step') or '')" "$W/$P1.json" 2>/dev/null)
  local G=$W/census/census_${E}_b${B}_graph.sqlite N=$W/census/census_${E}_b${B}_node.sqlite
  [ -s "$G" ] && [ -s "$N" ] || { line "SC1B arm $E B=$B UNREAD (missing export)"; return 0; }
  [ "$DELIM" = d2h ] && { POS="--positions $LL_POSITIONS"; BYTES="--d2h-bytes $((B * VOCAB * 4))"; }
  local CLK="pass1=$(clock_of "$P1"),graph=$(clock_of "census_${E}_b${B}_graph"),node=$(clock_of "census_${E}_b${B}_node")"
  "$PY" "$W/sc1b_census.py" arm --graph "$G" --node "$N" --engine "$E" --batch "$B" --classes "$W/kernel_classes.json" \
      --delim "$DELIM" $BYTES $POS ${U:+--unprofiled-ms $U} --clocks "$CLK" --out "$W/sc1b_arm_${E}_b$B.json" 2>&1 | tail -1 | tee -a summary.txt; }
sc1b_gap(){ local C=$1 B=$2 R=${3:-}
  [ -s "$W/sc1b_arm_e4b_b$B.json" ] && [ -s "$W/sc1b_arm_${C}_b$B.json" ] || { line "SC1B gap e4b-$C B=$B UNREAD"; return 0; }
  "$PY" "$W/sc1b_census.py" gap --e4b "$W/sc1b_arm_e4b_b$B.json" --comp "$W/sc1b_arm_${C}_b$B.json" ${R:+--sc1-ratio $R} \
      --out "$W/sc1b_gap_e4b_${C}_b$B.json" 2>&1 | tail -1 | tee -a summary.txt; }

# fetch_q4km -- the Q4_K_M GGUF only (SC1 box B's fetch_ggufs also pulls IQ4_XS, which SC1b does not read)
fetch_q4km(){ have llamacpp || return 1; mkdir -p "$W/gguf"; say "fetch $GGUF_REPO/$GGUF_Q4KM @ $GGUF_REV"
  LLAMACPP_PY=$PY perl -e "alarm $(arm_alarm 3600); exec @ARGV" bash -c ". $W/llamacpp/llamacpp_box.sh && llamacpp_fetch $GGUF_REPO $GGUF_Q4KM $W/gguf $GGUF_REV" \
      > logs/fetch_gguf_q4km.log 2>&1 || { tail -2 logs/fetch_gguf_q4km.log; line "FETCH gguf $GGUF_Q4KM FAILED"; OK[llamacpp]=0; return 11; }
  line "FETCH gguf $GGUF_Q4KM $(cut -c1-16 "$W/gguf/$GGUF_Q4KM.sha256" 2>/dev/null)"; }

# ---- the real lane (box D): run order e4b -> llama.cpp -> vLLM -> SGLang, B=1 before B=16, graph before node; the deadline
# drops from the end (SGLang B=16 node first)
box_d(){ SCHED_NAME=int4; SCHED_STACK="$SPEEDENV"
  phase 0 "fetches (bf16 for the bake, GPTQ, the Q4_K_M GGUF), bake, prompts; vLLM / SGLang / llama.cpp / nsys installed above"
  fetch_common || finish 11; fetch_q4km; bake_qwen3 || finish 12; prompts || finish 19
  phase SG0 "SGLang first JIT (matched) before any timing"; sglang_up matched; sglang_down
  quiesce arms
  for B in 1 16; do
    phase "E4B$B" "e4b B=$B: SC1's unprofiled int4_sched arm, then graph + node captures"
    can_run 900 e4bsched_int4_sched_b${B}_r1 && { arm_int4_sched r1 "$B"; rec $?; }
    for M in graph node; do can_run 900 census_e4b_b${B}_$M && { e4b_census "$B" $M; rec $?; }; done
    phase "LL$B" "llama.cpp B=$B: SC1's unprofiled q4km arm, then graph + node captures (whole run; positions $LL_POSITIONS)"
    can_run 900 llamacpp_q4km_b${B}_r1 && { arm_llamacpp_q4km r1 "$B"; rec $?; }
    for M in graph node; do can_run 900 census_llamacpp_b${B}_$M && { llamacpp_census "$B" $M; rec $?; }; done
  done
  for B in 1 16; do
    phase "VL$B" "vLLM B=$B (spawn workers in every pass)"
    can_run 900 vllm_gptq_graph_b${B}_r1 && { vllm_arm gptq_graph_b${B}_r1 graph_r1 "$B" "$GPTQ_MID" "$GPTQ_REV" VLLM_WORKER_MULTIPROC_METHOD=spawn; rec $?; }
    for M in graph node; do can_run 900 census_vllm_b${B}_$M && { vllm_census "$B" $M; rec $?; }; done
  done
  phase SGL "SGLang: SC1's unprofiled matched arms, then one nsys-launched server per (B, mode)"
  sglang_up matched; for B in 1 16; do can_run 900 sglang_gptq_matched_b${B}_r1 && { arm_sglang_matched r1 "$B"; rec $?; }; done; sglang_down
  for B in 1 16; do for M in graph node; do can_run 900 census_sglang_b${B}_$M && { sglang_census "$B" $M; rec $?; }; done; done
  phase RD "the reading: census arms, then the registered gaps (SC1's own reducer is not run on box D)"
  for B in 1 16; do
    sc1b_arm e4b "$B" "e4bsched_int4_sched_b${B}_r1"
    sc1b_arm llamacpp "$B" "llamacpp_q4km_b${B}_r1" d2h
    sc1b_arm vllm "$B" "vllm_gptq_graph_b${B}_r1"
    sc1b_arm sglang "$B" "sglang_gptq_matched_b${B}_r1"
  done
  sc1b_gap llamacpp 1 1.484; sc1b_gap llamacpp 16 0.624                  # SC1 box B's measured ratios
  sc1b_gap vllm 1 1.203; sc1b_gap vllm 16 1.172                          # SC1 box B
  sc1b_gap sglang 1 1.268; sc1b_gap sglang 16 1.191; }                   # SC1 box C

# ---- the proof (SC1_PROVE=1): every proof capture must REDUCE (status ok, kept steps >= the floor, map and segments clean),
# not merely exit 0 (round 2 M11)
prove_red(){ local tag=$1; shift
  "$PY" - "$@" <<'PYR' > "logs/prove_red_$tag.log" 2>&1
import json, sys
sys.path.insert(0, "/root/sc1")
import sc1b_census as C
kind, path, engine, extra = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
ex = C.load(path)
if kind == "graph":
    r = C.graph_mode(ex, extra.get("delim", "graph"), extra.get("d2h_bytes"), positions=tuple(extra["positions"]) if extra.get("positions") else None)
else:
    r = C.node_mode(ex, C.load_classes("/root/sc1/kernel_classes.json", engine))
bad = [d["text"][:160] for d in ex.get("diagnostics", []) if any(k in d["text"].lower() for k in C.DIAG_BAD)]
ok = r["status"] == "ok" and not bad
if kind == "node" and ok:
    ok = r["residual_fraction"] <= C.G_MAP and not (r.get("segments") or {}).get("replays_broken")
print("SC1B_PROVE_RED " + json.dumps({"kind": kind, "engine": engine, "ok": ok, "status": r["status"], "why": r.get("why"),
      "steps_kept": r.get("steps_kept"), "residual": r.get("residual_fraction"), "segments": r.get("segments"),
      "top_residual": r.get("top_residual_kernels"), "diagnostic_errors": bad[:3]}))
sys.exit(0 if ok else 1)
PYR
  local rc=$?; line "PROVE $tag $(grep -a SC1B_PROVE_RED "logs/prove_red_$tag.log" | cut -c1-300)"; return $rc; }
prove_d(){ local ok=0 r
  have nsys || { say "PROVE: nsys did not install -- NOT PROVED"; rec 23; return; }
  # 2. the toy, graph and node, through the reducer's loader (a spawned child, the graph captured before the range, >= 90 %
  #    memory, one eager kernel and one H2D copy inside the range)
  for M in graph node; do
    perl -e "alarm 600; exec @ARGV" "$NSYS" profile $NSYS_TRACE --cuda-graph-trace=$M --capture-range=cudaProfilerApi --capture-range-end=stop \
        --force-overwrite true -o "$W/census/toy_$M" "$PY" "$W/sc1b_toy.py" run > logs/prove_toy_$M.log 2>&1 && nsys_export "$W/census/toy_$M" || ok=1
  done
  "$PY" "$W/sc1b_toy.py" check "$W/census/toy_graph.sqlite" "$W/census/toy_node.sqlite" > logs/prove_toy_check.log 2>&1 || ok=1
  line "PROVE toy $(grep -a SC1B_TOY_CHECK logs/prove_toy_check.log | cut -c1-300) $(grep -a 'SC1B_TOY ' logs/prove_toy_node.log | cut -c1-120)"
  [ $ok = 0 ] || { say "PROVE: the toy failed -- the instrument does not record what the census reads -- NOT PROVED"; rec 23; return; }
  # 3. e4b B=16 graph capture on Granite (the staggered-admission bracket, B1), with the Granite arena the common proof baked
  e4b_census 16 graph "$GR_ENV" "$GR" "$GR_REV" "$W/work_granite/nf4.arena" granite 0; r=$?
  [ $r = 0 ] && prove_red e4b_granite_b16 graph "$W/census/census_granite_b16_graph.sqlite" e4b '{}' || { say "PROVE: e4b B=16 census rc=$r -- NOT PROVED"; rec 23; }
  # 4. vLLM B=1 node mode on the lane's checkpoint (spawned EngineCore, in-process start_profile, the class map on real kernels)
  if fetch gptq "$GPTQ_MID" "$GPTQ_REV" 1800 && prompts; then           # prompts() fetches the bf16 repo's tokenizer itself
    vllm_census 1 node; r=$?
    [ $r = 0 ] && prove_red vllm_b1_node node "$W/census/census_vllm_b1_node.sqlite" vllm '{}' || { say "PROVE: vLLM B=1 node census rc=$r -- NOT PROVED"; rec 23; }
    # 5. SGLang B=1 node mode (the server under nsys profile, /start_profile's text reply, the JIT on this box)
    printf '{"decode_ms_per_step": 3.76, "walls_short_s": [0.2142]}' > "$W/sglang_gptq_matched_b1_r1.json"     # SC1 box C's B=1 timing
    sglang_census 1 node; r=$?
    [ $r = 0 ] && prove_red sglang_b1_node node "$W/census/census_sglang_b1_node.sqlite" sglang '{}' || { say "PROVE: SGLang B=1 node census rc=$r -- NOT PROVED"; rec 23; }
    rm -f "$W/sglang_gptq_matched_b1_r1.json"
  else say "PROVE: GPTQ fetch or prompts failed -- NOT PROVED"; rec 23; fi
  # 6. llama.cpp B=16 graph mode, whole run, -lv 5 (graph-replay evidence); the reducer must find >= the floor at the positions
  if fetch_q4km && [ -s "$W/gguf/$GGUF_Q4KM" ]; then
    LLAMACPP_EXTRA_FLAGS="-lv 5" llamacpp_census 16 graph; r=$?
    line "PROVE llama.cpp warmups=$(grep -ac 'CUDA graph warmup complete' "$W/logs/llamacpp_server_census_llamacpp_b16_graph.log" 2>/dev/null)"
    [ $r = 0 ] && prove_red llamacpp_b16_graph graph "$W/census/census_llamacpp_b16_graph.sqlite" llamacpp \
        "{\"delim\": \"d2h\", \"d2h_bytes\": $((16 * VOCAB * 4)), \"positions\": [33, 64]}" || { say "PROVE: llama.cpp B=16 census rc=$r -- NOT PROVED"; rec 23; }
  else say "PROVE: Q4_K_M fetch failed -- NOT PROVED"; rec 23; fi; }
