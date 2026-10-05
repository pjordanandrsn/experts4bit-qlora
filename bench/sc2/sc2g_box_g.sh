# shellcheck shell=bash
# bench/sc2/sc2g_box_g.sh -- lane SC2g (#846), box G: request-level serving of gpt-oss-20b (bench/sc2/SC2g-PREREG.md).
# Sourced by sc1_run.sh when SC1_BOX=G, AFTER sc2_box_e.sh (its helpers: gpu_free, wait_http, wait_e4b_ready, stop_pid,
# drive, install_sc2_client, sc2_prompts' tool, PORT_*, SC2_* constants) and SC1's common setup.
#
# The same driver and plan as SC2, on openai/gpt-oss-20b, each engine on its own path to the checkpoint's MXFP4 experts.
# The ARITHMETIC differs per engine and is a label on every row (bench/sc2/SC2g-PREREG.md):
#   e4b       native MXFP4 decode (gemv_mxfp4 at T == 1, K21 at <= 256 rows); NF4 prefill (E4B_INT4_KEEP_NF4=1), sinks
#             layers on the explicit-mask prefill path;
#   vLLM      Marlin W4A16 (--moe-backend marlin), --attention-backend TRITON_ATTN pinned;
#   SGLang    its default MXFP4 runner on sm_120, triton attention (forced for gpt-oss), recorded from server_info;
#   llama.cpp the published ggml-org MXFP4 GGUF (attention Q8_0), default MMQ: W4A8 decode, W4A4 prefill.
# Like box F, box G exports NEITHER of SC1's prefill route pins and every e4b server starts under `env -u` on both.

SC2G_MID=openai/gpt-oss-20b; SC2G_REV=6cee5e81ee83917806bbde320786a8fb61efebee
SC2G_GGUF_REPO=ggml-org/gpt-oss-20b-GGUF; SC2G_GGUF_REV=ef9b12f2ff56c69cf32153a02784e7a3c88bf524; SC2G_GGUF=gpt-oss-20b-MXFP4.gguf
SC2G_LAYERS=24
# GNF4_TRITON_PREBIND=1 is gnf4 v0.41.0's default (kernel/_triton_shim.py:269), pinned so the receipt shows it: bit-identical, a
# launch-overhead lever that wraps _gemm_nf4_grouped, whose reach into the NF4 prefill path is not verified
SC2G_E4B_ENV="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 GNF4_TRITON_PREBIND=1 $FOLDS"
SC2G_ENGINES="e4b_gptoss vllm sglang llamacpp"

# the checkpoint without its duplicate `original/` and `metal/` copies (bo3's patterns; the xet backend can stall)
fetch_gptoss(){ say "fetch $SC2G_MID @ $SC2G_REV (no original/, metal/)"
  HF_HUB_DISABLE_XET=1 perl -e "alarm $(arm_alarm 3600); exec @ARGV" "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$SC2G_MID', revision='$SC2G_REV', ignore_patterns=['original/*', 'metal/*', 'consolidated*'], max_workers=4))" > logs/fetch_gptoss.log 2>&1 \
    || { tail -2 logs/fetch_gptoss.log; say "DL FAIL (gptoss)"; line "FETCH gptoss FAILED"; return 11; }
  tail -1 logs/fetch_gptoss.log > fetch_gptoss.path; line "FETCH gptoss $SC2G_MID@$SC2G_REV $(cat fetch_gptoss.path)"; }
fetch_gptoss_gguf(){ have llamacpp || return 1; mkdir -p "$W/gguf"; say "fetch $SC2G_GGUF_REPO/$SC2G_GGUF @ $SC2G_GGUF_REV"
  LLAMACPP_PY=$PY perl -e "alarm $(arm_alarm 3600); exec @ARGV" bash -c ". $W/llamacpp/llamacpp_box.sh && llamacpp_fetch $SC2G_GGUF_REPO $SC2G_GGUF $W/gguf $SC2G_GGUF_REV" \
      > logs/fetch_gguf_gptoss.log 2>&1 || { tail -2 logs/fetch_gguf_gptoss.log; line "FETCH gguf $SC2G_GGUF FAILED"; OK[llamacpp]=0; return 11; }
  line "FETCH gguf $SC2G_GGUF $(cut -c1-16 "$W/gguf/$SC2G_GGUF.sha256" 2>/dev/null)"; }
bake_gptoss(){ bake gptoss "$SC2G_MID" 5400 || return 12; GA_GPTOSS=$W/work_gptoss/nf4.arena
  "$PY" -c "import json,sys; r=json.load(open('$W/work_gptoss/bake.json')); sys.exit(0 if r.get('status') == 'OK' else 1)" \
    || { line "BAKE gptoss: bake.json status is not OK"; return 12; }; }
gptoss_prompts(){ say "SC2g prompt pool (gpt-oss tokenizer)"
  perl -e "alarm 1200; exec @ARGV" "$PY" $W/sc2_prompts.py --model "$SC2G_MID" --revision "$SC2G_REV" --out $W/sc2/prompts.json > logs/sc2_prompts.log 2>&1 \
    || { tail -3 logs/sc2_prompts.log; return 19; }
  grep -a "^SC2_PROMPTS" logs/sc2_prompts.log | tee -a summary.txt; }

# g_e4b_check HEALTH.json -- the e4b server's own record: the native MXFP4 store on every layer, routes as box F's, and
# the prefill graph reported (whatever `auto` decided is recorded, not gated)
g_e4b_check(){ "$PY" - "$1" $SC2G_LAYERS <<'PYG'
import json, sys
h, n = json.load(open(sys.argv[1])), int(sys.argv[2])
lv, r = h.get("levers") or {}, h.get("prefill_routes") or {}
bad = []
if lv.get("int4_store_kinds") != ["mxfp4"]:
    bad.append(f"int4_store_kinds {lv.get('int4_store_kinds')!r}, not ['mxfp4']")
if lv.get("exp_int4_layers_enabled") != n:
    bad.append(f"exp_int4_layers_enabled {lv.get('exp_int4_layers_enabled')!r}, not {n}")
want = {"int4_prefill": "k19", "int4_prefill_above_256_rows": "k19", "prefill_attn": "flash", "device_grouping": True,
        "int4_prefill_env": None, "prefill_attn_env": None}   # box F's assertion: main's defaults, neither pin present
off = {k: r.get(k, "<missing>") for k, v in want.items() if r.get(k, "<missing>") != v}
if off:
    bad.append(f"prefill_routes {off!r}")
pg = h.get("prefill_graph") or {}
print("SC2G_E4B " + json.dumps({"ok": not bad, "bad": bad, "store": lv.get("int4_store_kinds"), "routes": r,
                                "prefill_graph": {k: pg.get(k) for k in ("status", "requested", "T", "why", "pool_mib", "free_after_mib")}}),
      flush=True)
sys.exit(1 if bad else 0)
PYG
}
g_e4b_start(){ local LOG=$W/logs/sc2g_server_e4b.log
  # shellcheck disable=SC2086  # assignment lists by design
  setsid env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN PYTHONPATH= $ROUTEENV $SC2G_E4B_ENV \
      E4B_PAGED_MODEL=$SC2G_MID E4B_PAGED_REVISION=$SC2G_REV E4B_PAGED_ARENA=$GA_GPTOSS E4B_PAGED_CALIB=$W/calib.json \
      E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2_MAXLEN E4B_PAGED_TRACE=$W/sc2/trace_e4b_gptoss.jsonl E4B_HOST=127.0.0.1 E4B_PORT=$PORT_E4B \
      "$PY" -m experts4bit_qlora.serve_paged > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_e4b_ready "http://127.0.0.1:$PORT_E4B/health" 2400 $SRV_PID "$LOG" || return $?
  curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_e4b_gptoss.json || return 45
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits > $W/sc2/vram_e4b_gptoss.txt 2>/dev/null
  g_e4b_check $W/sc2/health_e4b_gptoss.json | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { line "SC2G STOP: e4b not on its registered path"; return 47; }
  line "SC2G e4b healthy"; }
g_vllm_start(){ local LOG=$W/logs/sc2g_server_vllm.log
  setsid env VLLM_LOGGING_LEVEL=INFO "$W/venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port $PORT_VLLM \
      --model "$SC2G_MID" --revision "$SC2G_REV" --tokenizer-revision "$SC2G_REV" --served-model-name sc2 --max-num-seqs 16 \
      --max-model-len $SC2_MAXLEN --no-enable-prefix-caching --seed 0 --gpu-memory-utilization 0.90 \
      --moe-backend marlin --attention-backend TRITON_ATTN > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_http "http://127.0.0.1:$PORT_VLLM/v1/models" 1800 $SRV_PID "$LOG" || return $?
  curl -fsS -m 30 "http://127.0.0.1:$PORT_VLLM/v1/models" -o $W/sc2/models_vllm.json || true
  line "SC2G vllm healthy: $(grep -a -o -i -E 'marlin[A-Za-z_ ]*|TRITON_ATTN|mxfp4[A-Za-z_]*' "$LOG" | sort | uniq -c | tr '\n' ' ' | cut -c1-240)"; }
g_sgl_start(){ have sglang || return 1
  SGLANG_EXTRA_ARGS="--max-running-requests 16 --context-length $SC2_MAXLEN --mem-fraction-static $SC1_SGLANG_MEM_FRACTION_STATIC --served-model-name sc2" \
    sglang_server_start "$SC2G_MID" "$SC2G_REV" $PORT_SGL "$W/logs/sc2g_server_sglang.log" gptoss > logs/sc2g_sglang_start.log 2>&1 || return $?
  cp "$W/logs/sc2g_server_sglang.log.server_info.json" $W/sc2/server_info_sglang.json 2>/dev/null || true
  cp "$W/logs/sc2g_server_sglang.log.engagement.json" $W/sc2/engagement_sglang.json 2>/dev/null || true
  line "SC2G sglang healthy: $(grep -a SGLANG_ENGAGEMENT logs/sc2g_sglang_start.log | tail -1 | cut -c1-300)"; }
g_ll_start(){ have llamacpp && [ -s "$W/gguf/$SC2G_GGUF" ] || return 1
  llamacpp_server_start "$W/gguf/$SC2G_GGUF" 16 $PORT_LL "$W/logs/sc2g_server_llamacpp.log" > logs/sc2g_llamacpp_start.log 2>&1 || return $?
  line "SC2G llamacpp healthy: $(tail -1 logs/sc2g_llamacpp_start.log | cut -c1-240)"; }

# g_engine_runs NAME PORT MODEL PROFILE -- SC2's plan, but BOTH draws repeat ONE realisation: the serial seed and each
# rate's seed are the same in draw 1 and draw 2. SC2's read named its per-draw seeds as the reason knee rows read
# UNSTABLE (the two draws realised different rates); here a draw disagreement is the engine's, not the dice's.
g_engine_runs(){ local E=$1 PORT=$2 MODEL=$3 PROFILE=$4 D r
  sampler_start sc2g_$E
  drive ${E}_warm $PORT "$MODEL" $PROFILE serial 0 4 999
  for D in $(seq 1 $SC2_DRAWS); do
    can_run 300 "$E draw $D serial" && drive ${E}_serial_d$D $PORT "$MODEL" $PROFILE serial 0 $SC2_SERIAL_N 1
    for r in $SC2_RATES; do
      can_run $(( SC2_N / r + 240 )) "$E draw $D rate $r" && drive ${E}_r${r}_d$D $PORT "$MODEL" $PROFILE poisson $r $SC2_N $(( 100 + r ))
    done
  done
  sampler_stop sc2g_$E; }

# ---- the real lane
box_g(){
  phase 0 "fetches (gpt-oss-20b, its published MXFP4 GGUF), the NF4 bake, the gpt-oss prompt pool"
  fetch_gptoss || finish 11; fetch_gptoss_gguf; bake_gptoss || finish 12; gptoss_prompts || finish 19
  phase SG0 "SGLang first JIT before any timing"; g_sgl_start && sglang_server_stop > /dev/null 2>&1; gpu_free 180
  quiesce arms
  phase E4B "e4b serve_paged on gpt-oss: native MXFP4 decode, NF4 prefill"
  if can_run 1200 e4b_gptoss && g_e4b_start; then g_engine_runs e4b_gptoss $PORT_E4B "$SC2G_MID" e4b; fi; stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  phase VLLM "vLLM 0.30.0 (Marlin W4A16, TRITON_ATTN)"
  if have vllm && can_run 1200 vllm && g_vllm_start; then g_engine_runs vllm $PORT_VLLM sc2 vllm; fi; stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  phase SGL "SGLang 0.5.20 (gpt-oss defaults)"
  if can_run 1200 sglang && g_sgl_start; then g_engine_runs sglang $PORT_SGL sc2 sglang; fi; sglang_server_stop > /dev/null 2>&1; gpu_free 180
  phase LL "llama.cpp server (the published MXFP4 GGUF)"
  if can_run 1200 llamacpp && g_ll_start; then g_engine_runs llamacpp $PORT_LL sc2 llamacpp; fi; llamacpp_server_stop > /dev/null 2>&1; gpu_free 120
  phase RD "the reading"
  "$PY" $W/sc2g_reduce.py --dir $W/sc2 --out $W/sc2/verdict_sc2g.json 2>&1 | tail -30 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): every server on gpt-oss-20b ITSELF (13 GB; e4b's first gpt-oss run through serve_paged),
# each answering a serial and a Poisson smoke with every request VALID
prove_g(){ local ok=0
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  for s in sc2_driver.py sc2_reduce.py sc2g_reduce.py; do
    "$PY" $W/$s --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  done
  fetch_gptoss || { say "PROVE: gpt-oss fetch failed -- NOT PROVED"; rec 23; return; }
  bake_gptoss || { say "PROVE: gpt-oss bake failed -- NOT PROVED"; rec 23; return; }
  gptoss_prompts || { say "PROVE: prompt pool failed -- NOT PROVED"; rec 23; return; }
  gsmoke(){ local E=$1 PORT=$2 MODEL=$3 PROFILE=$4
    drive ${E}_prove_serial $PORT "$MODEL" $PROFILE serial 0 6 1 && drive ${E}_prove_poisson $PORT "$MODEL" $PROFILE poisson 4 16 2 \
      || { say "PROVE: $E smoke failed -- NOT PROVED"; ok=1; }; }
  if g_e4b_start; then gsmoke e4b_gptoss $PORT_E4B "$SC2G_MID" e4b; curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_e4b_gptoss_end.json; else say "PROVE: e4b server failed"; ok=1; fi
  stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  if g_vllm_start; then gsmoke vllm $PORT_VLLM sc2 vllm; else say "PROVE: vLLM server failed"; ok=1; fi
  stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  if g_sgl_start; then gsmoke sglang $PORT_SGL sc2 sglang; else say "PROVE: SGLang server failed"; ok=1; fi
  sglang_server_stop > /dev/null 2>&1; gpu_free 180
  if fetch_gptoss_gguf && g_ll_start; then gsmoke llamacpp $PORT_LL sc2 llamacpp; else say "PROVE: llama.cpp server failed"; ok=1; fi
  llamacpp_server_stop > /dev/null 2>&1; gpu_free 120
  [ $ok = 0 ] || rec 23; }
