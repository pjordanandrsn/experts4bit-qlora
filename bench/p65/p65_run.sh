#!/bin/bash
# bench/p65/p65_run.sh -- lane P65, BOX side (P65-PREREG.md). Started detached by p65_drive.sh with the run's nonce;
# writes P65_RUN_NONCE first, then P65_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P65_SUCCESS.<nonce> only
# when every registered census is present, complete and passed its selfcheck.
#
# One RTX 5090. Per family (Granite, OLMoE, Mixtral, in that order): fetch the pinned revision, bake the NF4 arena
# (P39's k8_bake.py), run p65_census.py (P44-a's census rows + activation entropy, wikitext and c4val1, two halves
# each), free the family. The reducer runs here only as a smoke (p65_table.md); the read of record is
# `p65_reduce.py` over the FETCHED receipts. Nothing here changes a kernel or a default.
set -uo pipefail
W=/root/p65; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] p65: $*"; }
NONCE=${P65_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P65_RUN_NONCE.tmp && mv $W/P65_RUN_NONCE.tmp $W/P65_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P65_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P65_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P65_RUN_ID P65_DEADLINE_EPOCH P65_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA; do case "${!v}" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac; [ ${#E4B_SHA} -eq 40 ] && [ ${#GNF4_SHA} -eq 40 ] || { say "refusing: $v is not a 40-char sha"; finish 78; }; done
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
FAMILIES=${P65_FAMILIES:-granite,olmoe,mixtral}
NSEQ=${P65_NSEQ:-64}                                  # windows of 512 per text; each half = 32 = P44-a's census size
MIXTRAL_LAYERS=${P65_MIXTRAL_LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}   # P44-a's measured half, registered
MIN_MBPS=${P65_MIN_MBPS:-50}; MIN_DISK_GB=${P65_MIN_DISK_GB:-200}; MIN_RAM_GB=${P65_MIN_RAM_GB:-64}
# seconds a family needs, fetch + bake + census, from the rehearsal (P65-PREREG "Box and cost"); a family that cannot
# fit before the deadline is SKIPPED (host-limited), never started and cut
NEED_granite=${P65_NEED_GRANITE_S:-1500}; NEED_olmoe=${P65_NEED_OLMOE_S:-1800}; NEED_mixtral=${P65_NEED_MIXTRAL_S:-4200}
: > summary.txt; echo "$P65_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte (names as the box sees them)
for f in p65_run.sh p65_census.py expert_entropy.py p65_reduce.py expert_residuals.py serve_stack.py p44_reduce.py k8_bake.py calib.json staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p65/staged.sha256"; finish 9; }
# ---- card class, host floors, egress: refusal rows, not stalls
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac
DISK_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9); RAM_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "disk_avail_gb=$DISK_GB ram_avail_gb=$RAM_GB" >> forensics.txt
[ "${DISK_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${DISK_GB} GB free < ${MIN_DISK_GB} (Mixtral bf16 + its NF4 snapshot + arena)"; echo "refused: disk $DISK_GB GB" > REFUSAL; finish 13; }
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB} GB RAM available < ${MIN_RAM_GB} (Mixtral's Hessians + one layer's weights)"; echo "refused: host RAM $RAM_GB GB" > REFUSAL; finish 13; }
# the Hessian budget scales with the host (bigger chunks = fewer forward passes); recorded
BUDGET=$(( RAM_GB * 2 / 5 )); [ "$BUDGET" -lt 16 ] && BUDGET=16; [ "$BUDGET" -gt 64 ] && BUDGET=64
export E4B_INT4_HESSIAN_BUDGET_GB=$BUDGET; echo "E4B_INT4_HESSIAN_BUDGET_GB=$BUDGET" >> forensics.txt
say "egress pre-flight: HF CDN, 50 MB range, 20 s cap (floor ${MIN_MBPS} MB/s)"
BPS=$(curl -sSL --max-time 20 -r 0-52428800 -o /dev/null -w '%{speed_download}' https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors 2>/dev/null || echo 0)
MBPS=$(python3 -c "print(round(float('${BPS:-0}')/1e6,1))"); say "HF CDN ${MBPS} MB/s"; echo "hf_cdn_mbps=$MBPS" >> forensics.txt
if python3 -c "import sys; sys.exit(0 if float('${MBPS:-0}') < float('$MIN_MBPS') else 1)"; then say "REFUSED: egress ${MBPS} MB/s < ${MIN_MBPS} -- host-limited, not a result"; echo "refused: egress $MBPS MB/s" > REFUSAL; finish 14; fi
# ---- install: e4b pinned + P37's toolchain pins, then gnf4 at the pin (last, so the kernel pin wins)
say "install e4b @$E4B_SHA (image python; P37 pins) + gnf4 @$GNF4_SHA"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import inspect, os, importlib.metadata as md
from experts4bit_qlora.engines.int4_experts import calibrate_expert_hessians, _ExpertHessianSink
from experts4bit_qlora.engines.expert_profile import hot_sets_from_profile  # noqa: F401
from gptq_pack import HessianAccumulator  # noqa: F401
from int4_pack_ref import pack_int4_b32  # noqa: F401
assert "activation_means" in inspect.signature(calibrate_expert_hessians).parameters, "installed e4b predates P65's tap"
assert "means" in inspect.signature(_ExpertHessianSink).parameters
import experts4bit_qlora as e, torch, triton, transformers
open("/root/p65/versions.txt", "w").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK: e4b", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
# ---- helpers (P44-a's)
left(){ echo $(( P65_DEADLINE_EPOCH - $(date +%s) )); }
can_run(){ local need=$1; [ $(( $(date +%s) + need + 600 )) -le "$P65_DEADLINE_EPOCH" ] || { say "STOP: $2 needs ${need}s, only $(left)s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local l=$(( P65_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$l" -lt 900 ] && l=900; [ "$l" -gt 14400 ] && l=14400; echo "$l"; }
fetch(){ local MID=$1 REV=$2; say "fetch $MID @ $REV"
  perl -e "alarm $(arm_alarm); exec @ARGV" python - "$MID" "$REV" <<'PYF' > logs/fetch_${MID//\//--}.log 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json", "*.tiktoken", "*.jinja"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}"); print("STAGED", os.path.basename(os.path.realpath(p)))
PYF
  local rc=$?; [ $rc -ne 0 ] && { tail -3 logs/fetch_${MID//\//--}.log; say "FETCH FAIL rc=$rc"; return 11; }
  local GOT; GOT=$(grep -a "^STAGED " logs/fetch_${MID//\//--}.log | tail -1 | awk '{print $2}')
  [ "$GOT" = "$REV" ] || { say "PIN MISMATCH staged=$GOT != pin=$REV -- not coerced"; return 12; }
  local RDIR=/root/.cache/huggingface/hub/models--${MID//\//--}; mkdir -p "$RDIR/refs" && printf '%s' "$REV" > "$RDIR/refs/main"; say "PIN OK $GOT"; }
bake(){ local MID=$1 TAG=$2; [ -e "$W/work_$TAG/nf4.arena" ] && return 0; say "bake $TAG"; mkdir -p $W/work_$TAG
  K8_MODEL="$MID" K8_WORK="$W/work_$TAG" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/k8_bake.py > logs/bake_$TAG.log 2>&1 || { tail -3 logs/bake_$TAG.log; say "BAKE FAIL $TAG"; return 12; }
  [ -e "$W/work_$TAG/nf4.arena" ] || { say "BAKE FAIL $TAG (no arena)"; return 12; }; }
free_family(){ rm -rf $W/work_$1; rm -rf /root/.cache/huggingface/hub/models--${2//\//--}; say "freed $1 ($(df -h /root | tail -1 | awk '{print $4}') free)"; }
census(){ local FAM=$1 NEED=$2 LAYERS=${3:-}; read MID REV <<<"$(python -c "import sys; sys.path.insert(0, '$W'); from serve_stack import MODELS; print(*MODELS['$FAM'])")"
  can_run "$NEED" "${FAM}_census" || return 40
  local t0; t0=$(date +%s)
  fetch "$MID" "$REV" && bake "$MID" $FAM || { echo "CENSUS $FAM fetch/bake FAILED" >> summary.txt; return 11; }
  say "census $FAM (nseq $NSEQ per text${LAYERS:+, layers $LAYERS})"
  E4B_MODEL_ID=$MID perl -e "alarm $(arm_alarm); exec @ARGV" python $W/p65_census.py --family $FAM --arena $W/work_$FAM/nf4.arena --calib $W/calib.json \
      --nseq "$NSEQ" ${LAYERS:+--layers "$LAYERS"} --out $W/census_$FAM.json > logs/census_$FAM.log 2>&1; local rc=$?
  tail -3 logs/census_$FAM.log | sed "s/^/    /"; { echo -n "census $FAM rc=$rc wall=$(( $(date +%s) - t0 ))s "; tail -1 logs/census_$FAM.log | cut -c1-200; } >> summary.txt
  free_family $FAM "$MID"; return $rc; }
rc_any=0
FAMS=",$FAMILIES,"
[[ "$FAMS" == *,granite,* ]] && { census granite "$NEED_granite" || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }; }
[[ "$FAMS" == *,olmoe,* ]] && { census olmoe "$NEED_olmoe" || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }; }
[[ "$FAMS" == *,mixtral,* ]] && { census mixtral "$NEED_mixtral" "$MIXTRAL_LAYERS" || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }; }
# ---- every registered census present, complete (every requested layer, both texts, both halves) and self-checked
for FAM in ${FAMILIES//,/ }; do
  python - "$W/census_$FAM.json" <<'PYC' >> summary.txt 2>&1 || { echo "ROW census_$FAM INCOMPLETE" >> summary.txt; [ "$rc_any" = 0 ] && rc_any=41; }
import json, sys
c = json.load(open(sys.argv[1]))
assert (c.get("selfcheck") or {}).get("ok"), "selfcheck did not pass"
assert c["layers_censused"] == c["layers_requested"], (c["layers_censused"], c["layers_requested"])
cells = {(r["text"], r["half"]) for r in c["rows"]}
assert cells == {(t, h) for t in ("wikitext", "c4val1") for h in (0, 1, "full")}, cells
print(f"ROW census_{c['family']} complete: {len(c['layers_censused'])} layers, {len(c['rows'])} rows, selfcheck max rel diff {c['selfcheck']['max_rel_diff']:.2e}")
PYC
done
python $W/p65_reduce.py $W --md $W/p65_table.md --json $W/p65_verdicts.json > logs/reduce.log 2>&1 && cat $W/p65_table.md >> summary.txt || say "reducer smoke failed (the fetched receipts are re-reduced off the box)"
say "----- summary -----"; cat summary.txt
finish "$rc_any"
