# shellcheck shell=bash
# bench/sc2/sc1g_box_i.sh -- lane SC1g (#846), box I: the quality of each engine's gpt-oss-20b arithmetic on identical tokens
# (bench/sc2/SC1g-PREREG.md, amendment A1). Sourced by sc1_run.sh when SC1_BOX=I, AFTER sc2_box_e.sh (gpu_free, stop_pid) and
# sc2g_box_g.sh (fetch_gptoss, fetch_gptoss_gguf, bake_gptoss and the SC2G_* pins: the same checkpoint, GGUF and arena).
# Top level holds constants and functions only: sc1_run.sh sources this before it defines FOLDS / ROUTEENV (SC2g A1).
#
# Teacher-forced NLL in nats, SC1's two shapes, every engine scoring the SAME ids (each row's text_sha). A1's texts:
#   conv1, conv2  GRADED: the first two ultrachat_200k test_sft conversations (by index, pinned revision) whose single
#                 rendering in gpt-oss's chat template reaches 2,561 tokens -- in distribution by format and register
#   wikitext      DESCRIPTIVE: the chat-framed wikitext the proof showed is out of distribution (ppl ~560)
# Arms: e4b_serve (served = MXFP4 GEMV W4A8; prefill = kept-NF4 host M-tile at chunk 64 and 128, the floor) and the e4b_nf4
# control; vllm Marlin W4A16; sglang flashinfer_mxfp4 and Marlin; llama.cpp's published GGUF at default MMQ and MMQ_PREC=q8.
# Diagnostics (descriptive, conv1/conv2): e4b served at GNF4_PDL=0, with the folds off, and --ppl-oracle eager --ppl-chunk 1
# (T == 1 expert routes under transformers' attention and a bf16 cache) for MXFP4 and NF4; and the kernel check
# (sc1g_gemv_check.py) on activations the served arms capture.
# The e4b arms and the windows run through sc1g_k8.py: P39's step_decomp and SC1's window rule UNMODIFIED (byte-pinned by other
# lanes), the chat date pinned, the window read from its file, a route record written at exit (a missing record FAILS the row).

SC1G_CHAT_SUFFIX='<|channel|>final<|message|>'
# gpt-oss's template writes "Current date: <today>"; pinned (sc1g_k8.py's SC1G_CHAT_DATE) so the windows and their shas are the
# same on every box and every day -- the proof's and the reading's windows are byte-identical, and no run VOIDs itself at midnight
SC1G_CHAT_DATE=2026-10-05
SC1G_UC_REPO=HuggingFaceH4/ultrachat_200k; SC1G_UC_REV=8049631c405ae6576f93f445c6b8166f76f5505a
SC1G_UC_FILE=data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet
SC1G_E4B_SERVE="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 GNF4_TRITON_PREBIND=1"
SC1G_E4B_NF4="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 GNF4_TRITON_PREBIND=1"
SC1G_NOFOLD="E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0"
# SC1's K8ARGS minus --ppl-source (the window comes from its file)
SC1G_K8ARGS="--placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv"
SC1G_SRCS="conv1 conv2"       # graded
SC1G_CTRL="wikitext"          # descriptive

i_ultrachat(){ say "fetch $SC1G_UC_REPO/$SC1G_UC_FILE @ $SC1G_UC_REV"
  HF_HUB_DISABLE_XET=1 perl -e "alarm 1200; exec @ARGV" "$PY" -c "from huggingface_hub import hf_hub_download as d; print(d('$SC1G_UC_REPO', '$SC1G_UC_FILE', repo_type='dataset', revision='$SC1G_UC_REV'))" \
      > logs/fetch_ultrachat.log 2>&1 || { tail -2 logs/fetch_ultrachat.log; line "FETCH ultrachat FAILED"; return 11; }
  tail -1 logs/fetch_ultrachat.log > fetch_ultrachat.path; line "FETCH ultrachat $SC1G_UC_REPO@$SC1G_UC_REV $(cat fetch_ultrachat.path)"; }

# the windows (sc1g_k8.py windows): the chat-framed wikitext control and the two graded conversations
i_windows(){ say "SC1g windows (gpt-oss tokenizer, chat template, date $SC1G_CHAT_DATE)"; mkdir -p $W/sc1g
  i_ultrachat || return 19
  SC1G_CHAT_DATE=$SC1G_CHAT_DATE perl -e "alarm 1200; exec @ARGV" "$PY" $W/sc1g_k8.py windows --model "$SC2G_MID" --rev "$SC2G_REV" --out $W/sc1g \
      --suffix "$SC1G_CHAT_SUFFIX" --ultrachat "$(cat fetch_ultrachat.path)" --n-conv 2 > logs/sc1g_windows.log 2>&1 || { tail -3 logs/sc1g_windows.log; return 19; }
  grep -a "^WINDOW" logs/sc1g_windows.log | tee -a summary.txt; }

# step_decomp loads --model by id with no revision; refuse the e4b arms if the hub's main is not the pin (it is today)
i_pin_ok(){ "$PY" -c "from huggingface_hub import HfApi; import sys; s = HfApi().model_info('$SC2G_MID').sha; print('SC1G_MAIN', s); sys.exit(0 if s == '$SC2G_REV' else 1)" \
    2>&1 | tail -1 | tee -a summary.txt; return "${PIPESTATUS[0]}"; }

# i_e4b NAME STACK SRC [EXTRA ARGS...] -- step_decomp (through sc1g_k8.py) at SC1's K8 args on the window file
# sc1g/k8_window_<SRC>.json; receipt sc1g/<NAME>.json, route record sc1g/routes/<NAME>.<pid>.json. The arm's STACK comes
# after SC1's FOLDS, so an arm can override a fold (the folds-off diagnostic). The prefill route pins are absent (box I is
# in sc1_run.sh's F/G branch).
i_e4b(){ local NAME=$1 STACK=$2 SRC=$3; shift 3; local AL; AL=$(arm_alarm 2400); mkdir -p $W/sc1g/routes; rm -f $W/sc1g/routes/$NAME.*
  [ -n "${SC1G_PIN_BAD:-}" ] && { stub $W/sc1g/$NAME.json e4b "$NAME" 1 refused "the hub's main is not $SC2G_REV" logs/sc1g_windows.log; line "$NAME REFUSED (pin)"; return 0; }
  say "arm $NAME ($SRC $* alarm=$AL)"; sampler_start $NAME
  { echo "SC1g arm=$NAME box=$BOX at=$(date -u +%FT%TZ) $(host_snapshot)"; echo "ENV: $ROUTEENV $FOLDS $STACK"; } > logs/run_$NAME.log
  env $ROUTEENV $FOLDS $STACK SC1G_CHAT_DATE=$SC1G_CHAT_DATE SC1G_WINDOW_FILE=$W/sc1g/k8_window_$SRC.json SC1G_ROUTE_OUT=$W/sc1g/routes/$NAME \
      perl -e "alarm $AL; exec @ARGV" "$PY" $W/sc1g_k8.py k8 -- --model "$SC2G_MID" --arena "$GA_GPTOSS" --calib $W/calib.json $SC1G_K8ARGS "$@" \
      --out $W/sc1g/$NAME.json >> logs/run_$NAME.log 2>&1
  local rc=$?; sampler_stop $NAME
  [ -s $W/sc1g/$NAME.json ] || stub $W/sc1g/$NAME.json e4b "$NAME" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$NAME.log
  line "$NAME src=$SRC rc=$rc $(grep -aE 'K8_PPL|INT4EXP|REFUSED|Error' logs/run_$NAME.log | tail -1 | cut -c1-200)"; return $rc; }

# the e4b rows of one window: the served shape (capturing GEMV activations), the floor pair, the NF4 control
i_e4b_rows(){ local SRC=$1
  can_run 900 e4b_serve_served_$SRC && i_e4b e4b_serve_served_$SRC "$SC1G_E4B_SERVE SC1G_CAPTURE_OUT=$W/sc1g/capture_$SRC.pt" $SRC
  can_run 600 e4b_serve_prefill64_$SRC && i_e4b e4b_serve_prefill64_$SRC "$SC1G_E4B_SERVE" $SRC --ppl-oracle eager --ppl-chunk 64
  can_run 600 e4b_serve_prefill128_$SRC && i_e4b e4b_serve_prefill128_$SRC "$SC1G_E4B_SERVE" $SRC --ppl-oracle eager --ppl-chunk 128
  can_run 900 e4b_nf4_served_$SRC && i_e4b e4b_nf4_served_$SRC "$SC1G_E4B_NF4" $SRC
  can_run 600 e4b_nf4_prefill128_$SRC && i_e4b e4b_nf4_prefill128_$SRC "$SC1G_E4B_NF4" $SRC --ppl-oracle eager --ppl-chunk 128; }
# the diagnostics of one graded window (descriptive): what each moving would mean is registered in A1
i_e4b_diag(){ local SRC=$1
  can_run 900 e4b_serve_pdl0_$SRC && i_e4b e4b_serve_pdl0_$SRC "$SC1G_E4B_SERVE GNF4_PDL=0" $SRC
  can_run 900 e4b_serve_nofold_$SRC && i_e4b e4b_serve_nofold_$SRC "$SC1G_E4B_SERVE $SC1G_NOFOLD" $SRC
  can_run 1200 e4b_serve_chunk1_$SRC && i_e4b e4b_serve_chunk1_$SRC "$SC1G_E4B_SERVE" $SRC --ppl-oracle eager --ppl-chunk 1
  can_run 1200 e4b_nf4_chunk1_$SRC && i_e4b e4b_nf4_chunk1_$SRC "$SC1G_E4B_NF4" $SRC --ppl-oracle eager --ppl-chunk 1; }

# the kernel check on THIS card: every capture the served arms wrote, synthetic rows, and the mutation (must disagree)
i_gemv(){ local MD; MD=$(cat fetch_gptoss.path); local c t
  for c in $W/sc1g/capture_*.pt; do [ -s "$c" ] || continue; t=$(basename "$c" .pt)
    perl -e "alarm 900; exec @ARGV" "$PY" $W/sc1g_gemv_check.py --model-dir "$MD" --acts "$c" --out $W/sc1g/gemv_${t}_5090.json > logs/gemv_$t.log 2>&1
    grep -a "^SC1G_GEMV_CHECK" logs/gemv_$t.log | tail -1 | sed "s/^/$t /" | tee -a summary.txt; done
  perl -e "alarm 900; exec @ARGV" "$PY" $W/sc1g_gemv_check.py --model-dir "$MD" --synthetic --layers 0,12 --out $W/sc1g/gemv_synthetic_5090.json > logs/gemv_synthetic.log 2>&1
  grep -a "^SC1G_GEMV_CHECK" logs/gemv_synthetic.log | tail -1 | sed "s/^/synthetic /" | tee -a summary.txt
  perl -e "alarm 900; exec @ARGV" "$PY" $W/sc1g_gemv_check.py --model-dir "$MD" --synthetic --layers 0 --mutate --out $W/sc1g/gemv_mutate_5090.json > logs/gemv_mutate.log 2>&1
  grep -a "^SC1G_GEMV_CHECK" logs/gemv_mutate.log | tail -1 | sed "s/^/mutate /" | tee -a summary.txt; }

# i_vllm MODE SRC -- SC1's vLLM scorer on gpt-oss: Marlin W4A16, TRITON_ATTN (FA2 has no sinks); receipt sc1g/nll_vllm_<mode>_<src>.json
i_vllm(){ local MODE=$1 SRC=$2 S=nll_vllm_$1_$2
  have vllm || { stub $W/sc1g/$S.json vllm "nll_$MODE" 1 unsupported "vllm not installed"; line "$S SKIPPED unsupported"; return 0; }
  local AL; AL=$(arm_alarm 2400); say "arm $S (nll $MODE $SRC alarm=$AL)"; sampler_start $S
  env SC1_WINDOW=$W/sc1g/k8_window_$SRC.json SC1_MODE=$MODE SC1_MODEL=$SC2G_MID SC1_REV=$SC2G_REV SC1_ATTN_BACKEND=TRITON_ATTN SC1_MOE_BACKEND=marlin \
      SC1_OUT=$W/sc1g/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID SC1_LOG=$W/logs/run_$S.engine.log VLLM_LOGGING_LEVEL=INFO \
    perl -e "alarm $AL; exec @ARGV" $W/venv-vllm/bin/python $W/vllm/sc1_vllm_nll.py > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/sc1g/$S.json ] || stub $W/sc1g/$S.json vllm "nll_$MODE" 1 "$(status_of_rc $rc load_fault)" "rc=$rc" logs/run_$S.log
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|VOID|Error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }

# i_sgl_up VARIANT (native|marlin) / i_sgl VARIANT MODE SRC -- SC1's SGLang scorer against a gptoss_q / gptoss_qm server;
# receipt sc1g/nll_sglang_<variant>_<mode>_<src>.json
i_sgl_up(){ local V=$1 M; M=$([ "$1" = marlin ] && echo gptoss_qm || echo gptoss_q); have sglang || return 1
  SGL_LOG=$W/logs/sglang_server_$M.log; line "sglang server mode=$M starting"
  sglang_server_start "$SC2G_MID" "$SC2G_REV" 30000 "$SGL_LOG" "$M" > logs/sglang_start_$M.log 2>&1; local rc=$?
  if [ $rc -ne 0 ]; then line "sglang server mode=$M FAILED rc=$rc $(tail -1 logs/sglang_start_$M.log | cut -c1-160)"; SGL_MODE=""; return $rc; fi
  SGL_MODE=$M; line "sglang server mode=$M up $(grep -a 'SGLANG_ENGAGEMENT {' logs/sglang_start_$M.log | tail -1 | cut -c1-200)"; }
i_sgl(){ local V=$1 MODE=$2 SRC=$3 S=nll_sglang_$1_$2_$3
  [ -n "$SGL_MODE" ] || { stub $W/sc1g/$S.json sglang "nll_$MODE" 1 "$(have sglang && echo harness_error || echo unsupported)" "no sglang server" $W/logs/install_sglang.log; line "$S SKIPPED no server"; return 0; }
  local AL; AL=$(arm_alarm 2400); say "arm $S (nll $MODE $SRC alarm=$AL)"; sampler_start $S
  perl -e "alarm $AL; exec @ARGV" $W/venv-sglang/bin/python $W/sglang/sc1_sglang_nll.py --window $W/sc1g/k8_window_$SRC.json --mode $MODE --port 30000 \
      --server-info $SGL_LOG.server_info.json --engagement $SGL_LOG.engagement.json --out $W/sc1g/$S.json > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/sc1g/$S.json ] || stub $W/sc1g/$S.json sglang "nll_$MODE" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|VOID|Error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }

# i_ll VARIANT (default|q8) MODE (prefill|decode) SRC -- SC1's libllama harness on the published GGUF; q8 sets GGML_CUDA_MMQ_PREC=q8
# (read at run time, ggml-cuda/mmq.cu:96 at 552f18f). The harness does not record the env, so the box writes it into the receipt.
i_ll(){ local V=$1 MODE=$2 SRC=$3 S; S=nll_llamacpp$([ "$1" = q8 ] && echo _q8)_$2_$3; local G=$W/gguf/$SC2G_GGUF
  have llamacpp && [ -s "$G" ] || { stub $W/sc1g/$S.json llamacpp "nll_$MODE" 1 unsupported "no llama.cpp build or GGUF"; line "$S SKIPPED"; return 0; }
  local AL; AL=$(arm_alarm 2400); say "arm $S (nll $MODE $SRC alarm=$AL)"; sampler_start $S
  env $([ "$V" = q8 ] && echo GGML_CUDA_MMQ_PREC=q8) perl -e "alarm $AL; exec @ARGV" $LLAMACPP_BIN/nll_teacher_forced --model "$G" --tokens $W/sc1g/k8_window_$SRC.json \
      --out $W/sc1g/$S.json --prompt-len 512 --steps 2048 --mode $MODE --n-gpu-layers 99 --flash-attn on --type-kv f16 > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  if [ -s $W/sc1g/$S.json ]; then "$PY" - "$W/sc1g/$S.json" "$V" <<'PYL'
import json, sys
p, v = sys.argv[1], sys.argv[2]
r = json.load(open(p)); r["mmq_prec_env"] = "q8" if v == "q8" else None
json.dump(r, open(p, "w"), indent=1)
PYL
  else stub $W/sc1g/$S.json llamacpp "nll_$MODE" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log; fi
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|REFUSE|error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }

# every arm, graded windows first then the control, in the registered order; the GPU is freed between engines. The deadline
# drops from the end: llama.cpp, then SGLang, then the control's rows go first.
i_arms(){ local SRC ALL="$SC1G_SRCS $SC1G_CTRL"
  phase E4B "e4b: serve env served + floor pair + NF4 control on every window; the diagnostics on the graded windows"
  i_pin_ok || { SC1G_PIN_BAD=1; line "SC1G_PIN_BAD: the e4b arms are refused"; }
  for SRC in $SC1G_SRCS; do i_e4b_rows $SRC; i_e4b_diag $SRC; done
  for SRC in $SC1G_CTRL; do i_e4b_rows $SRC; done; gpu_free 120
  phase GEMV "the kernel check on this card (captured activations, synthetic, mutation)"; i_gemv; gpu_free 60
  phase VLLM "vLLM: Marlin W4A16, TRITON_ATTN"
  for SRC in $ALL; do can_run 900 vllm_$SRC && { i_vllm prefill $SRC; i_vllm served $SRC; }; done; gpu_free 120
  phase SGL "SGLang: flashinfer_mxfp4 (native), then Marlin"
  for V in native marlin; do
    if can_run 1500 sglang_$V && i_sgl_up $V; then for SRC in $ALL; do i_sgl $V prefill $SRC; i_sgl $V served $SRC; done; fi
    sglang_server_stop > /dev/null 2>&1; SGL_MODE=""; gpu_free 180
  done
  phase LL "llama.cpp: the published GGUF, default MMQ and MMQ_PREC=q8"
  for V in default q8; do for SRC in $ALL; do can_run 600 ll_${V}_$SRC && { i_ll $V prefill $SRC; i_ll $V decode $SRC; }; done; done; gpu_free 60; }

# ---- the real lane
box_i(){
  phase 0 "fetches (gpt-oss-20b, its published MXFP4 GGUF, ultrachat_200k test_sft), the NF4 bake, the windows"
  fetch_gptoss || finish 11; fetch_gptoss_gguf; bake_gptoss || finish 12; i_windows || finish 19
  quiesce arms
  i_arms
  phase RD "the reading"
  "$PY" $W/sc1g_reduce.py --dir $W/sc1g --out $W/sc1g/verdict_sc1g.json 2>&1 | tail -40 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): A1's new paths on conv1 -- the window from its file, the capture, eager chunk 1, the kernel check
# with its mutation -- plus one scoring per comparator engine, every route gate read
prove_i(){ local ok=0
  "$PY" $W/sc1g_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  fetch_gptoss || { say "PROVE: gpt-oss fetch failed -- NOT PROVED"; rec 23; return; }
  bake_gptoss || { say "PROVE: gpt-oss bake failed -- NOT PROVED"; rec 23; return; }
  i_windows || { say "PROVE: windows failed -- NOT PROVED"; rec 23; return; }
  fetch_gptoss_gguf || { say "PROVE: GGUF fetch failed"; ok=1; }
  i_pin_ok || { SC1G_PIN_BAD=1; say "PROVE: the hub's main is not the pin"; ok=1; }
  i_e4b e4b_serve_served_conv1 "$SC1G_E4B_SERVE SC1G_CAPTURE_OUT=$W/sc1g/capture_conv1.pt" conv1
  i_e4b e4b_serve_prefill64_conv1 "$SC1G_E4B_SERVE" conv1 --ppl-oracle eager --ppl-chunk 64
  i_e4b e4b_serve_chunk1_conv1 "$SC1G_E4B_SERVE" conv1 --ppl-oracle eager --ppl-chunk 1; gpu_free 120
  i_gemv; gpu_free 60
  i_vllm prefill conv1; gpu_free 120
  if i_sgl_up native; then i_sgl native prefill conv1; fi; sglang_server_stop > /dev/null 2>&1; SGL_MODE=""; gpu_free 180
  i_ll q8 prefill conv1
  "$PY" $W/sc1g_reduce.py --dir $W/sc1g --out $W/sc1g/verdict_sc1g_prove.json --prove 2>&1 | tail -20 | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: an arm did not score VALID, a route gate failed, or the kernel check failed -- NOT PROVED"; ok=1; }
  [ $ok = 0 ] || rec 23; }
