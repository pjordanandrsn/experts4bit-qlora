# shellcheck shell=bash
# bench/sc2/sc2d_box_k.sh -- lane SC2d (#846), box K: the E4B_PAGED_BULK_KV default's engagement reads
# (bench/sc2/SC2d-PREREG.md). Sourced by sc1_run.sh when SC1_BOX=K, AFTER sc2_box_e.sh (drive, gpu_free,
# wait_e4b_ready, stop_pid, install_sc2_client) and sc2g_box_g.sh (fetch_gptoss, bake_gptoss, SC2G_MID / SC2G_REV,
# SC2G_E4B_ENV) and SC1's common setup (fetch, arm_alarm, can_run, phase, quiesce, line, rec, ROUTEENV, FOLDS).
#
# Two models the bulk path has not been read on, each served by `python -m experts4bit_qlora.serve_paged` at its
# registered serving config with only the knob changed:
#   gptoss  openai/gpt-oss-20b: attention sinks on every layer, sliding layers alternating with full ones; SC2g's e4b
#           path (native MXFP4 decode, NF4 prefill).
#   qw36    Qwen/Qwen3.6-35B-A3B: a hybrid; its Gated DeltaNet layers keep no K/V, so the pool holds its attention
#           layers only (the pool-layer subset); the server's defaults on P98's NF4 arena (P98, P105, P106 served it).
# Per model two servers, OFF (E4B_PAGED_BULK_KV=0) then ON (=1). OFF: 4 warm requests, the serial plan, the serial plan
# again (the determinism control). ON: 4 warm, the serial plan. Then each server's own /health. No time is read.
# gpt-oss runs end to end before Qwen3.6 is fetched, so a Qwen3.6 fetch or bake failure still leaves gpt-oss's read.

SC2D_KNOB=E4B_PAGED_BULK_KV; SC2D_SERIAL_N=16
QW36_MID=Qwen/Qwen3.6-35B-A3B; QW36_REV=995ad96eacd98c81ed38be0c5b274b04031597b0   # P98's pin
QA36=""

# k_tripwire -- the installed e4b counts a bulk flush written per layer (`flush_bulk_fallback`, e4b#1174): without
# it the ENGAGED gate would read <missing> on every server and report a false NOT_ENGAGED. Refuse (rc 9) first.
k_tripwire(){ "$PY" -c "import inspect, experts4bit_qlora.engines.paged_runner as p, experts4bit_qlora.engines.fp8_paged_kv as k; assert 'flush_bulk_fallback' in inspect.getsource(p) and inspect.signature(k.Fp8PagedKV.append_prompt).return_annotation in (bool, 'bool'), 'e4b predates #1174'" \n    2>&1 | tail -1 | sed 's/^/SC2D_TRIPWIRE /' | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "SC2d: the installed e4b has no flush_bulk_fallback (#1174)"; finish 9; }
  line "SC2D_TRIPWIRE ok: flush_bulk_fallback (e4b#1174) present"; }

# k_prompts M MODEL REV -- the SC2 prompt pool in that model's tokenizer (512-token prompts), kept per model
k_prompts(){ local M=$1 MODEL=$2 R=$3; say "SC2d prompt pool ($M)"
  perl -e "alarm 1200; exec @ARGV" "$PY" $W/sc2_prompts.py --model "$MODEL" --revision "$R" --out $W/sc2/prompts_$M.json \
    > logs/sc2_prompts_$M.log 2>&1 || { tail -3 logs/sc2_prompts_$M.log; return 19; }
  grep -a "^SC2_PROMPTS" logs/sc2_prompts_$M.log | sed "s/^/$M /" | tee -a summary.txt; }

# k_arch M FETCHNAME -- the checkpoint's own facts, from the fetched snapshot's config.json and weight index: layer
# types (a hybrid's attention layers), and which layers carry an attention-sinks tensor
k_arch(){ "$PY" - "$1" "$(cat fetch_$2.path 2>/dev/null)" $W/sc2/arch_$1.json <<'PYA' | tee -a summary.txt; return "${PIPESTATUS[0]}"
import glob, json, os, re, sys
m, snap, out = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(os.path.join(snap, "config.json")))
tc = cfg.get("text_config") or cfg
types = list(tc.get("layer_types") or [])
n = tc.get("num_hidden_layers") or len(types)
names = []
for p in glob.glob(os.path.join(snap, "*.safetensors.index.json")):
    names += list(json.load(open(p))["weight_map"])
if not names:          # a single-file checkpoint: read the safetensors header
    from safetensors import safe_open
    for p in glob.glob(os.path.join(snap, "*.safetensors")):
        with safe_open(p, "pt") as f:
            names += list(f.keys())
sinks = {int(x.group(1)) for x in (re.search(r"\.layers\.(\d+)\.self_attn\.sinks$", k) for k in names) if x}
rec = {"model": m, "n_layers": n, "layer_types": sorted(set(types)),
       "attention_layers": [i for i, t in enumerate(types) if t in ("full_attention", "sliding_attention", "attention")]
       if types else list(range(n)),
       "linear_layers": sum(1 for t in types if t == "linear_attention"), "sinks_layers": len(sinks)}
json.dump(rec, open(out, "w"), indent=1)
print("SC2D_ARCH " + json.dumps(dict(rec, attention_layers=len(rec["attention_layers"]))), flush=True)
PYA
}

# bake_qw36 -- P98's arena bake (p98_bake.py: k8_bake's steps with the revision pinned)
bake_qw36(){ say "bake the Qwen3.6 NF4 arena (p98_bake.py)"; mkdir -p $W/work_qw36
  perl -e "alarm $(arm_alarm 3600); exec @ARGV" "$PY" $W/p98_bake.py --model "$QW36_MID" --revision "$QW36_REV" --work $W/work_qw36 \
      > logs/bake_qw36.log 2>&1 || { tail -3 logs/bake_qw36.log; line "BAKE qw36 FAILED"; return 12; }
  "$PY" -c "import json,sys; r=json.load(open('$W/work_qw36/bake.json')); print('BAKE qw36', {k: r.get(k) for k in ('status','layers','experts','snapshot_gib','bake_s')}); sys.exit(0 if r.get('status') == 'OK' else 1)" \
      | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || return 12
  [ -e $W/work_qw36/nf4.arena ] || { line "BAKE qw36: no arena"; return 12; }
  QA36=$W/work_qw36/nf4.arena; }

# k_server_start TAG ARM(off|on) MODEL REV ARENA LEV -- one e4b server at the model's registered config plus the knob
k_server_start(){ local TAG=$1 ARM=$2 MODEL=$3 R=$4 ARENA=$5 LEV=$6 LOG=$W/logs/sc2d_server_$1.log K=0
  [ "$ARM" = on ] && K=1
  # shellcheck disable=SC2086  # assignment lists by design
  setsid env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN -u E4B_PAGED_PREFILL_GRAPH -u E4B_PAGED_GRAPHS PYTHONPATH= \
      $ROUTEENV $LEV $SC2D_KNOB=$K E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$R E4B_PAGED_ARENA=$ARENA \
      E4B_PAGED_CALIB=$W/calib.json E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2_MAXLEN E4B_HOST=127.0.0.1 E4B_PORT=$PORT_E4B \
      "$PY" -m experts4bit_qlora.serve_paged > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_e4b_ready "http://127.0.0.1:$PORT_E4B/health" 2400 $SRV_PID "$LOG" || return $?
  curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_${TAG}_start.json || return 45
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits > $W/sc2/vram_${TAG}.txt 2>/dev/null   # record only
  line "SC2D $TAG healthy ($SC2D_KNOB=$K) vram_used_mib=$(cut -d, -f1 $W/sc2/vram_${TAG}.txt 2>/dev/null | tr -d ' ')"; }

# k_model M MODEL REV ARENA LEV -- the model's two servers, OFF then ON, on its own prompt pool
k_model(){ local M=$1 MODEL=$2 R=$3 ARENA=$4 LEV=$5 a
  cp $W/sc2/prompts_$M.json $W/sc2/prompts.json || { line "SC2D $M: no prompt pool"; return 19; }
  for a in off on; do
    if can_run 1500 "${M}_$a" && k_server_start "${M}_$a" "$a" "$MODEL" "$R" "$ARENA" "$LEV"; then
      drive ${M}_${a}_warm $PORT_E4B "$MODEL" e4b serial 0 4 999
      drive ${M}_${a}_serial $PORT_E4B "$MODEL" e4b serial 0 $SC2D_SERIAL_N 1
      [ "$a" = off ] && drive ${M}_off_serial_repeat $PORT_E4B "$MODEL" e4b serial 0 $SC2D_SERIAL_N 1
      curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o $W/sc2/health_${M}_${a}_end.json || line "SC2D ${M}_$a: no end /health"
    else line "SC2D ${M}_$a: the server did not start, or no time was left"; tail -5 $W/logs/sc2d_server_${M}_$a.log 2>/dev/null | cut -c1-300 >> summary.txt; fi
    stop_pid "$SRV_PID"; SRV_PID=""; gpu_free 180
  done; }

# ---- the real lane
box_k(){
  k_tripwire
  phase 0 "gpt-oss-20b: fetch, bake, its prompt pool, the checkpoint's own facts"
  fetch_gptoss || finish 11; bake_gptoss || finish 12; k_prompts gptoss "$SC2G_MID" "$SC2G_REV" || finish 19; k_arch gptoss gptoss || finish 19
  quiesce arms
  phase K1 "gpt-oss-20b: OFF then ON"; k_model gptoss "$SC2G_MID" "$SC2G_REV" "$GA_GPTOSS" "$SC2G_E4B_ENV $FOLDS"
  phase K2 "Qwen3.6-35B-A3B: fetch, bake, its prompt pool, the checkpoint's own facts; OFF then ON"
  if fetch qw36 "$QW36_MID" "$QW36_REV" 3600 && bake_qw36 && k_prompts qw36 "$QW36_MID" "$QW36_REV" && k_arch qw36 qw36; then
    k_model qw36 "$QW36_MID" "$QW36_REV" "$QA36" ""
  else line "SC2D qw36: fetch, bake, prompts or arch failed -- its read is UNREAD"; rec 12; fi
  phase RD "the reading"
  "$PY" $W/sc2d_reduce.py --dir $W/sc2 --out $W/sc2/verdict_sc2d.json 2>&1 | tail -20 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): box K's whole flow on the proof model -- its prompt pool, its arch record, both servers
# with every request, both knobs' engagement as the reducer reads it, determinism and identity -- plus the self-tests
prove_k(){
  k_tripwire
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  for s in sc2_driver.py sc2_identity.py sc2d_reduce.py; do
    "$PY" $W/$s --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { rec 23; return; }
  done
  "$PY" -c "import ast; ast.parse(open('$W/p98_bake.py').read())" || { say "PROVE: p98_bake.py does not parse -- NOT PROVED"; rec 23; return; }
  k_prompts granite "$GR" "$GR_REV" || { say "PROVE: prompt pool failed -- NOT PROVED"; rec 23; return; }
  k_arch granite granite || { say "PROVE: the arch record failed -- NOT PROVED"; rec 23; return; }
  k_model granite "$GR" "$GR_REV" "$W/work_granite/nf4.arena" "$GR_ENV"
  "$PY" $W/sc2d_reduce.py --dir $W/sc2 --proof granite | tee -a summary.txt
  [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: box K's flow on Granite is not ENGAGED_IDENTICAL -- NOT PROVED"; rec 23; }; }
