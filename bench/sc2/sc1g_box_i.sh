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
# A3: the MXFP4 store with the NF4 stacks EMPTIED, so rows above 256 run mxfp4_grouped_v1 on bf16 activations (a true MXFP4-weights
# prefill); under SC1G_E4B_SERVE (KEEP_NF4=1) those rows take the kept NF4 stacks, which is why box J's MXFP4 prefill read as NF4
SC1G_E4B_MXPRE="E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 GNF4_TRITON_PREBIND=1"
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
      --suffix "$SC1G_CHAT_SUFFIX" --ultrachat "$(cat fetch_ultrachat.path)" --n-conv ${SC1G_NCONV:-2} > logs/sc1g_windows.log 2>&1 || { tail -3 logs/sc1g_windows.log; return 19; }
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
      SC1_OUT=$W/sc1g/$S.json SC1_INSTANCE_ID=$SC1_INSTANCE_ID SC1_LOG=$W/logs/run_$S.engine.log VLLM_LOGGING_LEVEL=INFO ${SC1G_NAMED_ENV:-} \
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
      --server-info $SGL_LOG.server_info.json --engagement $SGL_LOG.engagement.json --out $W/sc1g/$S.json ${SC1G_NAMED_ARGS:-} > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  [ -s $W/sc1g/$S.json ] || stub $W/sc1g/$S.json sglang "nll_$MODE" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|VOID|Error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }

# i_ll VARIANT (default|q8) MODE (prefill|decode) SRC -- SC1's libllama harness on the published GGUF; q8 sets GGML_CUDA_MMQ_PREC=q8
# (read at run time, ggml-cuda/mmq.cu:96 at 552f18f). The harness does not record the env, so the box writes it into the receipt.
i_ll(){ local V=$1 MODE=$2 SRC=$3 S; S=nll_llamacpp$([ "$1" = q8 ] && echo _q8)_$2_$3; local G=$W/gguf/$SC2G_GGUF
  have llamacpp && [ -s "$G" ] || { stub $W/sc1g/$S.json llamacpp "nll_$MODE" 1 unsupported "no llama.cpp build or GGUF"; line "$S SKIPPED"; return 0; }
  local AL; AL=$(arm_alarm 2400); say "arm $S (nll $MODE $SRC alarm=$AL)"; sampler_start $S
  env $([ "$V" = q8 ] && echo GGML_CUDA_MMQ_PREC=q8) perl -e "alarm $AL; exec @ARGV" $LLAMACPP_BIN/nll_teacher_forced --model "$G" --tokens $W/sc1g/k8_window_$SRC.json \
      --out $W/sc1g/$S.json --prompt-len 512 --steps 2048 --mode $MODE --n-gpu-layers 99 --flash-attn on --type-kv f16 ${SC1G_NAMED_LL:-} > logs/run_$S.log 2>&1
  local rc=$?; sampler_stop $S
  if [ -s $W/sc1g/$S.json ]; then "$PY" - "$W/sc1g/$S.json" "$V" <<'PYL'
import json, sys
p, v = sys.argv[1], sys.argv[2]
r = json.load(open(p)); r["mmq_prec_env"] = "q8" if v == "q8" else None
json.dump(r, open(p, "w"), indent=1)
PYL
  else stub $W/sc1g/$S.json llamacpp "nll_$MODE" 1 "$(status_of_rc $rc)" "rc=$rc" logs/run_$S.log; fi
  line "$S mode=$MODE src=$SRC rc=$rc $(grep -aE 'mean_nll|ppl|REFUSE|error' logs/run_$S.log | tail -1 | cut -c1-200)"; return $rc; }

# ---- A4 (bench/sc2/SC1g-PREREG.md, amendment A4): the fidelity instrument's ENGINE side. Box R's artifacts (ref_<src>.npz: the
# bf16-dequant reference's top-64 named ids + fp64 log-probs per scored position) and their registered shas (ref_shas.json) are
# staged into $SC1G_REF_DIR by the post-R registration; absent, or not hashing to the registered sha, every KL arm is REFUSED
# (no artifact, no KL). Each served arm then also writes named_<stem>.npz: the engine's log-probs on those 64 ids at each
# position and on the target (sc1g_kl.py reads KL65 from them; the reader proves alignment against the arm's own mean NLL).
SC1G_REF_DIR=$W/sc1g_ref
SC1G_A4_SRCS="conv1 conv2 conv3 conv4 wikitext"   # conv1-conv4 graded, wikitext the descriptive control (last per engine)
i_ref(){ local SRC=$1 F=$SC1G_REF_DIR/ref_$1.npz WANT GOT
  [ -s "$F" ] && [ -s "$SC1G_REF_DIR/ref_shas.json" ] || return 1
  WANT=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$SC1G_REF_DIR/ref_shas.json" "$SRC" 2>/dev/null) || return 1
  GOT=$(sha256sum "$F" | cut -d' ' -f1); [ "$GOT" = "$WANT" ] || { line "SC1G_REF_BAD $SRC sha $GOT != registered $WANT" >&2; return 1; }
  echo "$WANT"; }
# the artifacts (and box R's verdict + calibration) ride in the receipt beside the rows they grade
i_ref_stage(){ mkdir -p $W/sc1g/ref; cp -p $SC1G_REF_DIR/ref_*.npz $SC1G_REF_DIR/ref_*.npz.json $SC1G_REF_DIR/ref_shas.json \
    $SC1G_REF_DIR/r_verdict.json $SC1G_REF_DIR/r_calib.json $W/sc1g/ref/ 2>/dev/null
  line "SC1G_REF staged $(ls $W/sc1g/ref/ref_*.npz 2>/dev/null | wc -l | tr -d ' ') artifacts"; }
i_noref(){ stub $W/sc1g/$1.json "$2" "named" 1 refused "no registered reference artifact for $3 (A4: no artifact, no KL)"; line "$1 REFUSED (no reference)"; }
i_e4b_named(){ local NAME=$1 STACK=$2 SRC=$3 SHA; SHA=$(i_ref $SRC) || { i_noref $NAME e4b $SRC; return 0; }
  i_e4b $NAME "$STACK SC1G_REF_FILE=$SC1G_REF_DIR/ref_$SRC.npz SC1G_REF_SHA=$SHA SC1G_NAMED_OUT=$W/sc1g/named_$NAME.npz" $SRC; }
i_vllm_named(){ local SRC=$1 SHA; SHA=$(i_ref $SRC) || { i_noref nll_vllm_served_$SRC vllm $SRC; return 0; }
  SC1G_NAMED_ENV="SC1_NAMED_REF=$SC1G_REF_DIR/ref_$SRC.npz SC1_NAMED_REF_SHA=$SHA SC1_NAMED_OUT=$W/sc1g/named_nll_vllm_served_$SRC.npz" i_vllm served $SRC; }
i_sgl_named(){ local V=$1 SRC=$2 SHA; SHA=$(i_ref $SRC) || { i_noref nll_sglang_${V}_served_$SRC sglang $SRC; return 0; }
  SC1G_NAMED_ARGS="--named-ref $SC1G_REF_DIR/ref_$SRC.npz --named-ref-sha $SHA --named-out $W/sc1g/named_nll_sglang_${V}_served_$SRC.npz" i_sgl $V served $SRC; }
# llama.cpp's harness reads raw int32 ids and writes raw float64 [steps x 65]; converted on either side
i_ll_named(){ local V=$1 SRC=$2 SHA S; S=nll_llamacpp$([ "$V" = q8 ] && echo _q8)_decode_$SRC
  SHA=$(i_ref $SRC) || { i_noref $S llamacpp $SRC; return 0; }
  "$PY" -c "import numpy as np, sys; np.load(sys.argv[1])['ids'].astype('<i4').tofile(sys.argv[2])" $SC1G_REF_DIR/ref_$SRC.npz $W/sc1g/named_ids_$SRC.bin
  SC1G_NAMED_LL="--named $W/sc1g/named_ids_$SRC.bin --named-out $W/sc1g/named_$S.bin --named-k 64" i_ll $V decode $SRC
  [ -s $W/sc1g/named_$S.bin ] && "$PY" -c "
import numpy as np, sys
a = np.fromfile(sys.argv[1], '<f8').reshape(-1, 65)
np.savez(sys.argv[2], eng_lp=a[:, :64], eng_target_lp=a[:, 64])" $W/sc1g/named_$S.bin $W/sc1g/named_$S.npz; }
# A4's arm list: the named KL rows first (by engine, so each server starts once), then descriptive prefill rows. The deadline
# drops from the end.
i_arms_a4(){ local SRC V
  phase A4E4B "A4: e4b served with named-token capture -- MXFP4 (GEMV) then NF4, conv1-conv4 then the control"
  i_pin_ok || { SC1G_PIN_BAD=1; line "SC1G_PIN_BAD: the e4b arms are refused"; }
  for SRC in $SC1G_A4_SRCS; do
    can_run 900 a4_e4b_$SRC && { i_e4b_named e4b_serve_served_$SRC "$SC1G_E4B_SERVE" $SRC; i_e4b_named e4b_nf4_served_$SRC "$SC1G_E4B_NF4" $SRC; }; done
  gpu_free 120
  phase A4VLLM "A4: vLLM served (Marlin W4A16, TRITON_ATTN), logprob_token_ids = the named ids + the target"
  for SRC in $SC1G_A4_SRCS; do can_run 900 a4_vllm_$SRC && i_vllm_named $SRC; done; gpu_free 120
  phase A4SGL "A4: SGLang served (flashinfer_mxfp4, then Marlin), token_ids_logprob = the target + the named ids"
  for V in native marlin; do
    if can_run 1500 a4_sglang_$V && i_sgl_up $V; then for SRC in $SC1G_A4_SRCS; do can_run 600 a4_sgl_${V}_$SRC && i_sgl_named $V $SRC; done; fi
    sglang_server_stop > /dev/null 2>&1; SGL_MODE=""; gpu_free 180
  done
  phase A4LL "A4: llama.cpp decode (the published GGUF; default MMQ, then MMQ_PREC=q8), the harness's --named"
  for V in default q8; do for SRC in $SC1G_A4_SRCS; do can_run 600 a4_ll_${V}_$SRC && i_ll_named $V $SRC; done; done; gpu_free 60
  phase A4DESC "A4: descriptive prefill-shaped NLL rows (no KL; the deadline drops these first)"
  for SRC in $SC1G_A4_SRCS; do
    can_run 600 d_e4b_$SRC && { i_e4b e4b_serve_prefill128_$SRC "$SC1G_E4B_SERVE" $SRC --ppl-oracle eager --ppl-chunk 128
                                i_e4b e4b_nf4_prefill128_$SRC "$SC1G_E4B_NF4" $SRC --ppl-oracle eager --ppl-chunk 128; }; done
  gpu_free 60
  for SRC in $SC1G_A4_SRCS; do can_run 600 d_ll_$SRC && i_ll default prefill $SRC; done; gpu_free 60; }

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
  phase 0 "fetches (gpt-oss-20b, its published MXFP4 GGUF, ultrachat_200k test_sft), the NF4 bake, the windows (A4: four conversations)"
  fetch_gptoss || finish 11; fetch_gptoss_gguf; bake_gptoss || finish 12; SC1G_NCONV=4 i_windows || finish 19; i_ref_stage
  quiesce arms
  i_arms_a4
  phase RD "the reading"
  "$PY" $W/sc1g_reduce.py --dir $W/sc1g --out $W/sc1g/verdict_sc1g.json 2>&1 | tail -40 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1), A4: every engine path's named-token KL row on conv1 -- e4b served (MXFP4 GEMV), vLLM served,
# SGLang native served, llama.cpp decode at MMQ_PREC=q8 -- each VALID, its named record complete and aligned with its own NLL,
# against box R's registered artifact. (A1's proof paths -- the capture, eager chunk 1, the kernel check -- ran on box J.)
prove_i(){ local ok=0
  "$PY" $W/sc1g_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  "$PY" $W/sc1g_kl.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  fetch_gptoss || { say "PROVE: gpt-oss fetch failed -- NOT PROVED"; rec 23; return; }
  bake_gptoss || { say "PROVE: gpt-oss bake failed -- NOT PROVED"; rec 23; return; }
  SC1G_NCONV=4 i_windows || { say "PROVE: windows failed -- NOT PROVED"; rec 23; return; }
  i_ref_stage
  i_ref conv1 > /dev/null || { say "PROVE: no registered reference artifact for conv1 (box R's post-run registration) -- NOT PROVED"; rec 23; return; }
  fetch_gptoss_gguf || { say "PROVE: GGUF fetch failed"; ok=1; }
  i_pin_ok || { SC1G_PIN_BAD=1; say "PROVE: the hub's main is not the pin"; ok=1; }
  i_e4b_named e4b_serve_served_conv1 "$SC1G_E4B_SERVE" conv1; gpu_free 120
  i_vllm_named conv1; gpu_free 120
  if i_sgl_up native; then i_sgl_named native conv1; fi; sglang_server_stop > /dev/null 2>&1; SGL_MODE=""; gpu_free 180
  i_ll_named q8 conv1
  "$PY" $W/sc1g_reduce.py --dir $W/sc1g --out $W/sc1g/verdict_sc1g_prove_a4.json --prove-a4 2>&1 | tail -20 | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: a named KL row was not VALID (arm, record, alignment or box R) -- NOT PROVED"; ok=1; }
  [ $ok = 0 ] || rec 23; }

# ---- box J (SC1_BOX=J), the e4b-only diagnostic box -- no comparator installs, guard <= 1 h (so no proof). A2 ran it as
# sc1g-diag-1 (e4b 3e133cf7): the paged fp8-KV path carries no gap; the MXFP4 T == 1 route costs +0.17-0.22 nats (J5). A3 (this
# arm list) splits that cost: the weights (MXFP4 prefill at KEEP_NF4=0 vs NF4 prefill), the decode-row route on bf16 activations
# (E4B_MXFP4_GEMV=0 vs NF4 served), the int8 activations (MXFP4 served vs GEMV=0); folds-off and PDL=0; a determinism repeat.
# Priority order, so the deadline drops the least important (sc1g-diag-1 reached ~15 arms in its 1 h): conv1's K1/K2/K5 rows,
# the attention check, then the three rows K2 and K5 rest on for conv2, conv3 and conv4 (the next two test_sft conversations by
# the same rule; conv1/conv2 unchanged) -- box J saw a 0.051-nat NF4 path-to-path spread on conv2, so one window cannot carry a
# ~0.1 effect -- then the controls (kvg4, folds-off, PDL=0) and conv2's K1 pair. Every comparison is within this box except K4's
# repeat of box J.
i_j(){ local TAG=$1; shift; can_run 600 "$TAG" && i_e4b "$@"; }
box_j(){
  phase 0 "fetches (gpt-oss-20b, ultrachat_200k test_sft), the NF4 bake, the windows (no GGUF, no comparators)"
  fetch_gptoss || finish 11; bake_gptoss || finish 12; SC1G_NCONV=4 i_windows || finish 19
  quiesce arms
  phase DJ "e4b diagnostics (A3): where the MXFP4 route's cost lives"
  i_pin_ok || { SC1G_PIN_BAD=1; line "SC1G_PIN_BAD: the e4b arms are refused"; }
  i_j k1 e4b_serve_v1_conv1 "$SC1G_E4B_SERVE E4B_MXFP4_GEMV=0" conv1
  i_j k2 e4b_mxpre_prefill128_conv1 "$SC1G_E4B_MXPRE" conv1 --ppl-oracle eager --ppl-chunk 128
  i_j k3 e4b_nf4_prefill128_conv1 "$SC1G_E4B_NF4" conv1 --ppl-oracle eager --ppl-chunk 128
  i_j k4 e4b_serve_served_conv1 "$SC1G_E4B_SERVE" conv1
  i_j k5 e4b_nf4_served_conv1 "$SC1G_E4B_NF4" conv1
  # e4b#1175's check (the maintainer, 2026-10-05): the fp8 paged decode attention at k_groups 4 / 8 / 16 vs a dequantize-then-attend
  # reference on gpt-oss's geometry -- sm_89+ only, so it rides this 5090; refuse-until-validated stands until kg16 agrees
  gpu_free 60
  if can_run 600 attn_check; then phase ATTN "fp8 paged decode attention at k_groups 4 / 8 / 16 (e4b#1175)"
    perl -e "alarm 600; exec @ARGV" "$PY" $W/sc1g_attn_check.py --out $W/sc1g/attn_check_5090.json > logs/attn_check.log 2>&1
    grep -a "^SC1G_ATTN_CHECK" logs/attn_check.log | tail -1 | tee -a summary.txt | grep -q . ||
      line "SC1G_ATTN_CHECK ERROR $(tail -1 logs/attn_check.log | cut -c1-200)"; fi
  local c n=5; for c in conv2 conv3 conv4; do
    i_j k$((n += 1)) e4b_serve_served_$c "$SC1G_E4B_SERVE" $c
    i_j k$((n += 1)) e4b_nf4_served_$c "$SC1G_E4B_NF4" $c
    i_j k$((n += 1)) e4b_serve_v1_$c "$SC1G_E4B_SERVE E4B_MXFP4_GEMV=0" $c; done
  i_j k15 e4b_serve_kvg4_conv1 "$SC1G_E4B_SERVE" conv1 --kv-groups 4
  i_j k16 e4b_serve_nofold_conv1 "$SC1G_E4B_SERVE $SC1G_NOFOLD" conv1
  i_j k17 e4b_serve_pdl0_conv1 "$SC1G_E4B_SERVE GNF4_PDL=0" conv1
  i_j k18 e4b_mxpre_prefill128_conv2 "$SC1G_E4B_MXPRE" conv2 --ppl-oracle eager --ppl-chunk 128
  i_j k19 e4b_nf4_prefill128_conv2 "$SC1G_E4B_NF4" conv2 --ppl-oracle eager --ppl-chunk 128
  gpu_free 60
  phase RD "the diagnostic reading (comparator rows absent by design: G1-G5 read UNREAD here; K1-K5 are read)"
  "$PY" $W/sc1g_reduce.py --dir $W/sc1g --out $W/sc1g/verdict_sc1g_diag.json 2>&1 | tail -40 | tee -a summary.txt; }
