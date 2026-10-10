# shellcheck shell=bash
# bench/sc5/sc5_box_m.sh -- lane SC5 (#1478 item 4), SC1 box M: same-box serving head-to-head on one RTX 5090
# (bench/sc5/SC5-PREREG.md). Sourced by sc1_run.sh when SC1_BOX=M, after sc2_box_e.sh, whose server helpers it reuses
# (wait_http, wait_e4b_ready, stop_pid, gpu_free, install_sc2_client, sc2_prompts), and SC1's common setup (say, line,
# phase, can_run, arm_alarm, rec, have, unsupported, quiesce, fetch, fetch_common, bake_qwen3, pipx, install_vllm,
# install_sglang, sglang_server_start / _stop). Staged flat in $W.
#
# What box M does differently from SC1's other boxes:
#   - e4b and grouped-nf4-gemm are the RELEASE WHEELS of e4b-wheels.lock, each verified by sha256 before pip sees it
#     (m_install_e4b, called by sc1_run.sh in place of the common git-SHA install and its tripwire);
#   - vLLM and SGLang install through SC1's installers in their lock mode: the whole closure by sha256;
#   - e4b serves as "e4b int4 (documented serving configuration)": SC1's SPEEDENV + E4B_PAGED_FUSE_QKV=1, every other
#     knob at its 0.52.0 default. e4b's OUT-OF-BOX serving is NF4 and is NOT measured here.
#   - the closed-loop driver (sc5_driver.py), ABBA blocks with cold starts, two memory settings, the capacity readouts,
#     a 1 Hz GPU-memory sampler per block, and the quality passes against one bf16 reference verified by sha256.
#
# SC1_SC5_PHASE (forwarded by sc1_drive.sh; read by sc1_run.sh's dispatch): read (default) -- the registered reading;
# ref -- compute the bf16 reference once (a >= 98 GB RAM host).
# SC1_PROVE=1 runs prove_m instead.

SC5_CS="1 16 64"; SC5_MAXTOK=256; SC5_DRAWS=2; SC5_MEMS="default matched"   # registered constants, not knobs
SC5_MATCHED_SEQS=64; SC5_MATCHED_TPS=1024; SC5_MATCHED_KV=65536; SC5_VLLM_BLOCK=16
SC5_VLLM_VERSION=0.31.0; SC5_SGLANG_VERSION=0.5.21; SC5_SGLANG_TAG_COMMIT=e00930c5489053f26d86b179cee0d087f846acbb
SC5_REF_SHA256=""   # the registered reference's sha256: set by the amendment that commits bench/sc5/ref/ (empty: quality UNREAD)
SC5_D=$W/sc5; mkdir -p "$SC5_D/blocks" "$SC5_D/quality"
sc5_n(){ case "$1" in 1) echo 48;; 16) echo 160;; 64) echo 320;; *) return 1;; esac; }
# the model name each server answers to: serve_paged serves its model id and refuses any other ("unknown model 'sc5'",
# sc5-prove-4); vLLM and SGLang serve --served-model-name sc5
sc5_model(){ case "$1" in e4b) echo "$MID";; *) echo sc5;; esac; }

# ---- installs ------------------------------------------------------------------------------------------------------
# m_install_e4b: venv-e4b (--system-site-packages, as SC1's), torch 2.8.0 cu128 (as boxes C-L), then the two release
# wheels by sha256 with e4b's [train] extra (transformers lives there: loc-a2000-4) and SC1's pins; a tripwire of its own.
m_install_e4b(){ local name ver fn sha url
  say "install e4b + gnf4 RELEASE WHEELS by sha256 (e4b-wheels.lock) into venv-e4b (--system-site-packages)"
  "$BASEPY" -m venv --system-site-packages $W/venv-e4b > logs/venv_e4b.log 2>&1 || { tail -3 logs/venv_e4b.log; say "VENV FAIL (e4b)"; return 9; }
  PY=$W/venv-e4b/bin/python
  pipx logs/pip_torch.log 1800 "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 || { tail -3 logs/pip_torch.log; say "PIP FAIL (torch cu128)"; return 9; }
  mkdir -p $W/wheels
  while read -r name ver fn sha url; do
    case "$name" in \#*|"") continue;; esac
    perl -e 'alarm 900; exec @ARGV' curl -fsSL --retry 2 -o "$W/wheels/$fn" "$url" || { say "WHEEL DOWNLOAD FAIL: $fn"; return 9; }
    [ "$(sha256sum "$W/wheels/$fn" | cut -d' ' -f1)" = "$sha" ] || { say "WHEEL SHA256 MISMATCH: $fn (want $sha)"; return 9; }
    echo "wheel $name $ver $fn sha256 $sha" | tee -a versions.txt
  done < $W/e4b-wheels.lock
  pipx logs/pip_e4b.log 1800 --prefer-binary $W/wheels/grouped_nf4_gemm-*.whl "$(ls $W/wheels/experts4bit_qlora-*.whl)[train]" \
    "transformers==5.16.1" "bitsandbytes==0.50.1" sentencepiece tiktoken "huggingface_hub>=0.23" || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b wheels)"; return 9; }
  LOCK=$W/e4b-wheels.lock "$PY" - <<'PYT' || { say "TRIPWIRE FAIL (e4b wheels)"; return 9; }
import importlib.metadata as md, json, os
want = {r.split()[0]: r.split()[1] for r in open(os.environ["LOCK"]) if r.strip() and not r.startswith("#")}
for name, ver in want.items():
    got = md.version(name)
    assert got == ver, f"{name} {got} installed, the lock pins {ver}"
    du = json.loads(md.distribution(name).read_text("direct_url.json") or "{}")
    assert du.get("url", "").endswith(".whl") and "vcs_info" not in du, f"{name} did not install from the verified wheel: {du}"
import torch, transformers, experts4bit_qlora as e
assert torch.cuda.is_available(), "torch sees no CUDA device in venv-e4b"
from experts4bit_qlora.serve_paged import build_engine, PagedServeConfig  # noqa: F401  (the server and the quality pass)
open("/root/sc1/versions.txt", "a").write(f"e4b {e.__version__} (wheel)\ngnf4 {md.version('grouped-nf4-gemm')} (wheel)\n"
                                          f"torch(e4b) {torch.__version__} cuda {torch.version.cuda}\ntransformers(e4b) {transformers.__version__}\n")
print("tripwire OK (e4b wheels):", {n: md.version(n) for n in want}, "torch", torch.__version__)
PYT
}
m_install_competitors(){
  export SC1_VLLM_VERSION=$SC5_VLLM_VERSION SC1_VLLM_WHEEL=lock SC1_VLLM_LOCK=$W/vllm.lock.txt
  export SGLANG_PIN=$SC5_SGLANG_VERSION SGLANG_TAG_COMMIT=$SC5_SGLANG_TAG_COMMIT SGLANG_LOCK=$W/sglang.lock.txt
  install_vllm; install_sglang; install_sc2_client; }

# ---- the servers, per memory setting (each returns 0 healthy; sets SRV_PID for e4b / vLLM) ----------------------------
m_e4b_start(){ local MEM=$1 LOG=$2 MEMENV=""
  [ "$MEM" = matched ] && MEMENV="E4B_PAGED_MAX_SEQS=$SC5_MATCHED_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC5_MATCHED_TPS"
  # shellcheck disable=SC2086  # assignment lists by design
  setsid env PYTHONPATH= $SPEEDENV E4B_PAGED_FUSE_QKV=1 $MEMENV E4B_PAGED_MODEL=$MID E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$QA \
      E4B_PAGED_CALIB=$W/calib.json E4B_HOST=127.0.0.1 E4B_PORT=$PORT_E4B "$PY" -m experts4bit_qlora.serve_paged > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_e4b_ready "http://127.0.0.1:$PORT_E4B/health" 2400 $SRV_PID "$LOG"; }
m_vllm_start(){ local MEM=$1 LOG=$2 MEMARGS=""
  [ "$MEM" = matched ] && MEMARGS="--block-size $SC5_VLLM_BLOCK --num-gpu-blocks-override $(( SC5_MATCHED_KV / SC5_VLLM_BLOCK ))"
  # shellcheck disable=SC2086
  setsid env VLLM_LOGGING_LEVEL=INFO "$W/venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port $PORT_VLLM \
      --model "$GPTQ_MID" --revision "$GPTQ_REV" --tokenizer-revision "$GPTQ_REV" --served-model-name sc5 --max-num-seqs 64 \
      --no-enable-prefix-caching --seed 0 $MEMARGS > "$LOG" 2>&1 < /dev/null &
  SRV_PID=$!; wait_http "http://127.0.0.1:$PORT_VLLM/v1/models" 1800 $SRV_PID "$LOG"; }
m_sgl_start(){ local MEM=$1 LOG=$2 MEMARGS=""
  [ "$MEM" = matched ] && MEMARGS="--max-total-tokens $SC5_MATCHED_KV"
  have sglang || return 1
  SGLANG_EXTRA_ARGS="--max-running-requests 64 --dtype float16 --served-model-name sc5 $MEMARGS" \
    sglang_server_start "$GPTQ_MID" "$GPTQ_REV" $PORT_SGL "$LOG" native > "$LOG.start" 2>&1; }
m_stop(){ case "$1" in sglang) sglang_server_stop > /dev/null 2>&1;; *) stop_pid "$SRV_PID"; SRV_PID="";; esac; gpu_free 180; }

# m_capacity FW BD -> "kv_tokens kv_rounding" from the framework's own report (the health / metrics / server info)
m_capacity(){ local FW=$1 BD=$2
  case "$FW" in
    e4b) curl -fsS -m 30 "http://127.0.0.1:$PORT_E4B/health" -o "$BD/health.json" || return 1
         "$PY" - "$BD/health.json" <<'PYC'
import json, sys
def find(o):
    if isinstance(o, dict):
        if "max_seqs" in o and "max_tokens_per_seq" in o:
            return o
        for v in o.values():
            r = find(v)
            if r is not None:
                return r
    return None
e = find(json.load(open(sys.argv[1])))
print(int(e["max_seqs"]) * int(e["max_tokens_per_seq"]), 0)
PYC
         ;;
    vllm) curl -fsS -m 30 "http://127.0.0.1:$PORT_VLLM/metrics" -o "$BD/metrics.txt" || return 1
          "$PY" - "$BD/metrics.txt" <<'PYC'
import re, sys
t = open(sys.argv[1]).read()
m = re.search(r'^vllm:cache_config_info\{([^}]*)\}', t, re.M)
kv = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
print(int(kv["num_gpu_blocks"]) * int(kv["block_size"]), int(kv["block_size"]))
PYC
          ;;
    sglang) cp "${SGLANG_SERVER_LOG:-$BD/server.log}.server_info.json" "$BD/server_info.json" 2>/dev/null \
              || curl -fsS -m 30 "http://127.0.0.1:$PORT_SGL/get_server_info" -o "$BD/server_info.json" || return 1
            "$PY" - "$BD/server_info.json" <<'PYC'
import json, sys
s = json.load(open(sys.argv[1]))
# SGLang 0.5.21's /get_server_info reports the KV pool at the top level (max_total_num_tokens) and per scheduler as
# internal_states[i].memory_usage.token_capacity; internal_states[0] has no max_total_num_tokens (sc5-prove-5: "?").
st = s["internal_states"] if isinstance(s.get("internal_states"), list) and s["internal_states"] else [{}]
cap = (st[0].get("memory_usage") or {}).get("token_capacity")
kv = s.get("max_total_num_tokens", st[0].get("max_total_num_tokens", cap))
if kv is None or (cap is not None and int(cap) != int(kv)):
    sys.exit(f"sglang capacity: max_total_num_tokens={kv}, token_capacity={cap}")
print(int(kv), int(s.get("page_size") or st[0].get("page_size") or 1))
PYC
            ;;
  esac; }

# m_cell FW PORT BD C -- one closed-loop cell into BD/cC.json; the plan is identical across blocks (seed = C)
m_cell(){ local FW=$1 PORT=$2 BD=$3 C=$4 N AL rc
  N=$(sc5_n "$C"); AL=$(arm_alarm 2400)
  perl -e "alarm $AL; exec @ARGV" "$PY" $W/sc5_driver.py run --base "http://127.0.0.1:$PORT" --model "$(sc5_model "$FW")" --prompts $W/sc2/prompts.json \
      --concurrency "$C" --n "$N" --seed "$C" --max-tokens $SC5_MAXTOK --profile "$FW" --out "$BD/c$C.json" > "$BD/c$C.log" 2>&1
  rc=$?; line "SC5 $(basename "$BD") C=$C rc=$rc $(grep -a '^SC5_RUN' "$BD/c$C.log" | tail -1 | cut -c1-300)"; return $rc; }

# m_block NN DRAW MEM FW K -- one cold-started block: quiescence gate, start, capacity, a warm-up, the three cells, stop
m_block(){ local NN=$1 D=$2 MEM=$3 FW=$4 K=$5 BD PORT ready=False kv="" rnd=0 smp c vok
  M_KV="" M_RND=0 M_READY=0   # the block's capacity readout and rounding, and whether its server came up (prove_m, box_m)
  BD=$SC5_D/blocks/$(printf %02d "$NN")_d${D}_${MEM}_${FW}_b${K}; mkdir -p "$BD"
  case "$FW" in e4b) PORT=$PORT_E4B; vok=True;; vllm) PORT=$PORT_VLLM; vok=$( [ "${OK[vllm]:-0}" = 1 ] && echo True || echo False );;
                sglang) PORT=$PORT_SGL; vok=$( [ "${OK[sglang]:-0}" = 1 ] && echo True || echo False );; esac
  phase "SC5_$NN" "draw $D, $MEM memory, $FW block $K"
  # SC1's quiesce records quiesced=true|false in quiesce_<tag>.json and always returns 0, so the gate reads the record:
  # a block whose card or host never quiesced is VOID (SC5-PREREG.md "Order and noise" / VOID), never run.
  if gpu_free 180 && quiesce "sc5_$NN" && grep -q '"quiesced": true' "quiesce_sc5_$NN.json" && can_run 1500 "sc5 block $NN"; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$BD/mem.csv" 2>/dev/null & smp=$!
    if "m_${FW/sglang/sgl}_start" "$MEM" "$BD/server.log"; then
      ready=True; M_READY=1; read -r kv rnd < <(m_capacity "$FW" "$BD") || kv=""
      line "SC5 $(basename "$BD") ready: kv_tokens=${kv:-?} rounding=$rnd"; M_KV=${kv:-}; M_RND=${rnd:-0}
      perl -e "alarm 900; exec @ARGV" "$PY" $W/sc5_driver.py run --base "http://127.0.0.1:$PORT" --model "$(sc5_model "$FW")" --prompts $W/sc2/prompts.json \
          --concurrency 1 --n 4 --seed 999 --max-tokens $SC5_MAXTOK --profile "$FW" --out "$BD/warm.json" > "$BD/warm.log" 2>&1
      for c in $SC5_CS; do m_cell "$FW" "$PORT" "$BD" "$c"; done
    fi
    m_stop "$FW"; kill "$smp" 2>/dev/null; wait "$smp" 2>/dev/null
  else
    line "SC5 $(basename "$BD"): the quiescence gate or the deadline refused the block (VOID)"
  fi
  "$PY" - "$BD" "$FW" "$K" "$D" "$MEM" "$ready" "$vok" "${kv:-}" "${rnd:-0}" <<'PYB'
import json, sys
bd, fw, k, d, mem, ready, vok, kv, rnd = sys.argv[1:10]
peak = None
try:
    vals = [float(x) for x in open(f"{bd}/mem.csv").read().split() if x.strip()]
    peak = max(vals) if vals else None
except OSError:
    pass
json.dump({"framework": fw, "block": int(k), "draw": d, "memory": mem, "ready": ready == "True", "versions_ok": vok == "True",
           "kv_tokens": int(kv) if kv else None, "kv_rounding": int(rnd), "peak_mem_mib": peak}, open(f"{bd}/block.json", "w"))
PYB
}

# m_draw_order DRAW -> the ABBA order (draw 1: e4b vLLM SGLang SGLang vLLM e4b; draw 2 the reverse)
m_draw_order(){ case "$1" in 1) echo "e4b:1 vllm:1 sglang:1 sglang:2 vllm:2 e4b:2";; *) echo "sglang:1 vllm:1 e4b:1 e4b:2 vllm:2 sglang:2";; esac; }

# m_matched_ok -- the last block's capacity readout is the registered matched capacity, within its rounding
m_matched_ok(){ local off; [ -n "$M_KV" ] || return 1; off=$(( M_KV - SC5_MATCHED_KV )); [ "${off#-}" -le "${M_RND:-0}" ]; }

# ---- quality: the windows and the reference verified by sha256, then one pass per framework at the default setting --
m_quality(){ local q=$SC5_D/quality
  cp $W/sc5_windows_w64.json $q/windows.json
  [ -n "$SC5_REF_SHA256" ] && [ -s $W/sc5_ref.json ] || { line "SC5 quality: no registered reference staged -- quality UNREAD"; return 0; }
  sha256sum $W/sc5_ref.json | cut -d' ' -f1 > $q/ref_sha256.txt
  phase SC5_Q "quality passes (windows sha256 $(python3 -c "import json; print(json.load(open('$q/windows.json'))['windows_sha256'])"))"
  if gpu_free 180 && m_vllm_start default $q/server_vllm.log; then
    "$PY" $W/sc5_quality.py score --framework vllm --base "http://127.0.0.1:$PORT_VLLM" --model sc5 --windows $q/windows.json --out $q/q_vllm.json | tee -a summary.txt
  fi; m_stop vllm
  if m_sgl_start default $q/server_sglang.log; then
    "$PY" $W/sc5_quality.py score --framework sglang --base "http://127.0.0.1:$PORT_SGL" --windows $q/windows.json --out $q/q_sglang.json | tee -a summary.txt
  fi; m_stop sglang
  # shellcheck disable=SC2086
  env PYTHONPATH=$W $SPEEDENV E4B_PAGED_FUSE_QKV=1 E4B_PAGED_MODEL=$MID E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$QA E4B_PAGED_CALIB=$W/calib.json \
    perl -e "alarm $(arm_alarm 3600); exec @ARGV" "$PY" $W/sc5_e4b_quality.py run --windows $q/windows.json \
      --prefill-out $q/q_e4b.json --decode-out $q/q_e4b_decode.json 2>&1 | tail -3 | tee -a summary.txt
  gpu_free 180
  for row in vllm sglang e4b e4b_decode; do
    [ -s $q/q_$row.json ] && "$PY" $W/sc5_quality.py compare --ref $W/sc5_ref.json --ref-sha256 "$SC5_REF_SHA256" --got $q/q_$row.json --out $q/cmp_$row.json
  done
  [ -s $W/sc5_ref_chunked.json ] && "$PY" $W/sc5_quality.py compare --ref $W/sc5_ref.json --ref-sha256 "$SC5_REF_SHA256" --got $W/sc5_ref_chunked.json --out $q/cmp_floor.json
  return 0; }

# ---- the reading -------------------------------------------------------------------------------------------------------
box_m(){ local d m nn=0 fk k miss
  phase 0 "fetches (bf16 for the bake, the GPTQ checkpoint), bake, SC2's prompt pool; installs above"
  fetch_common || finish 11; fetch gptq "$GPTQ_MID" "$GPTQ_REV" 1800 || finish 11; bake_qwen3 || finish 12; sc2_prompts || finish 19
  phase SG0 "SGLang's first JIT before any timing"; m_sgl_start default "$W/logs/sc5_sglang_jit.log" && m_stop sglang
  for d in $(seq 1 "$SC5_DRAWS"); do
    for m in $SC5_MEMS; do k=0; miss=""
      for fk in $(m_draw_order "$d"); do
        nn=$((nn + 1)); k=$((k + 1)); m_block "$nn" "$d" "$m" "${fk%%:*}" "${fk##*:}"
        # Cost guard (SC5-PREREG.md "Box log"): draw 1's first three matched blocks are one per framework. When a server
        # that came up read a capacity off the registered one, the reading stops; the reducer VOIDs those blocks anyway.
        if [ "$d:$m" = 1:matched ] && [ "$k" -le 3 ]; then
          [ "$M_READY" = 1 ] && ! m_matched_ok && miss="$miss ${fk%%:*}=${M_KV:-unread}"
          [ "$k" = 3 ] && [ -n "$miss" ] && { line "SC5_MATCHED_STOP draw 1 matched capacity off $SC5_MATCHED_KV:$miss -- the reading stops (cost guard)"; finish 35; }
        fi
      done
    done
  done
  can_run 4800 "quality passes" && m_quality
  phase RD "the record and the reading"
  "$PY" $W/sc5_record.py --dir $SC5_D --ref-sha256 "${SC5_REF_SHA256:-unset}" --out $SC5_D/sc5.json | tee -a summary.txt
  "$PY" $W/sc5_reduce.py --record $SC5_D/sc5.json --out $SC5_D/verdict.json 2>&1 | tail -30 | tee -a summary.txt; }

# ---- the reference, once (SC5_PHASE=ref; a >= 98 GB RAM host): full forward and the chunked ordering floor ----------
box_m_ref(){ local q=$SC5_D/quality
  phase REF "the bf16 reference over the committed windows (sc5_ref.py; device_map auto over the card and host RAM)"
  fetch_common || finish 11
  cp $W/sc5_windows_w64.json $q/windows.json
  perl -e "alarm $(arm_alarm 14400); exec @ARGV" "$PY" $W/sc5_ref.py --windows $q/windows.json --model "$MID" --revision "$REV" --out $q/ref.json \
    > logs/sc5_ref.log 2>&1 || { tail -3 logs/sc5_ref.log; finish 34; }
  perl -e "alarm $(arm_alarm 14400); exec @ARGV" "$PY" $W/sc5_ref.py --windows $q/windows.json --model "$MID" --revision "$REV" --out $q/ref_chunked.json \
    --chunked 256 > logs/sc5_ref_chunked.log 2>&1 || { tail -3 logs/sc5_ref_chunked.log; finish 34; }
  cat $q/ref.json.sha256 $q/ref_chunked.json.sha256 | tee -a summary.txt; }

# ---- the proof (SC1_PROVE=1): every install from its lock, every server at the default setting, one block each at C = 1
# and 16, the capacity readouts, and every quality scorer on 8 windows (complete positions, no reference compare).
# A failed capacity readout fails the proof: sc5-prove-5 printed "kv_tokens=?" for SGLang and still proved, and at the
# matched setting the reducer VOIDs a block whose kv_tokens is missing.
prove_m(){ local ok=0 fw nn=0 q=$SC5_D/quality
  have sc2client || { say "PROVE: aiohttp / fastapi / uvicorn did not install -- NOT PROVED"; rec 23; return; }
  for t in sc5_driver sc5_quality sc5_e4b_quality sc5_reduce sc5_record; do
    "$PY" $W/$t.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "PROVE: $t self-test failed"; ok=1; }
  done
  # sc1_run.sh's proof block runs before it defines fetch_common / bake_qwen3 (sc5-prove-3: rc 11, "fetch_common:
  # command not found"), so the proof calls the earlier-defined fetch / bake, as every other box's proof does.
  fetch qwen3 "$MID" "$REV" 4800 || finish 11; fetch gptq "$GPTQ_MID" "$GPTQ_REV" 1800 || finish 11
  bake qwen3 "$MID" 5400 || finish 12; QA=$W/work_qwen3/nf4.arena; sc2_prompts || finish 19
  SC5_CS="1 16"
  for fw in e4b vllm sglang; do nn=$((nn + 1)); m_block "$nn" 1 default "$fw" 1
    [ -n "$M_KV" ] || { say "PROVE: $fw capacity readout failed (kv_tokens unread)"; ok=1; }
    for c in 1 16; do [ -s "$SC5_D/blocks/$(printf %02d $nn)_d1_default_${fw}_b1/c$c.json" ] || { say "PROVE: $fw C=$c missing"; ok=1; }; done
  done
  PYTHONPATH=$W "$PY" -c "import json, sc5_ref; w = json.load(open('$W/sc5_windows_w64.json')); w['windows'] = w['windows'][:8]; w['windows_sha256'] = sc5_ref.windows_sha256(w['windows']); json.dump(w, open('$q/windows8.json', 'w'))" || ok=1
  if m_vllm_start default $q/server_vllm.log; then "$PY" $W/sc5_quality.py score --framework vllm --base "http://127.0.0.1:$PORT_VLLM" --model sc5 --windows $q/windows8.json --out $q/q_vllm.json || ok=1; else ok=1; fi; m_stop vllm
  if m_sgl_start default $q/server_sglang.log; then "$PY" $W/sc5_quality.py score --framework sglang --base "http://127.0.0.1:$PORT_SGL" --windows $q/windows8.json --out $q/q_sglang.json || ok=1; else ok=1; fi; m_stop sglang
  # shellcheck disable=SC2086
  env PYTHONPATH=$W $SPEEDENV E4B_PAGED_FUSE_QKV=1 E4B_PAGED_MODEL=$MID E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$QA E4B_PAGED_CALIB=$W/calib.json \
    "$PY" $W/sc5_e4b_quality.py run --windows $q/windows8.json --prefill-out $q/q_e4b.json --decode-out $q/q_e4b_decode.json || ok=1
  [ $ok = 0 ] && echo "SC5_PROVE ok" | tee -a summary.txt || rec 23; }
