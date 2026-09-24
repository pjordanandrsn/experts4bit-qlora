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
# Mixtral: the first 16 layers of the plan's own enumeration order -- the order P44-a's census walked (checkpoint-index
# key order, lexicographic), so the SAME half P44-a measured; the census records the list (layers_requested)
MIXTRAL_FIRST=${P65_MIXTRAL_FIRST_LAYERS:-16}
MIN_MBPS=${P65_MIN_MBPS:-80}; MIN_DISK_GB=${P65_MIN_DISK_GB:-200}; MIN_RAM_GB=${P65_MIN_RAM_GB:-64}
# seconds a family needs, fetch + bake + census, from the rehearsal (P65-PREREG "Box and cost"); a family that cannot
# fit before the deadline is SKIPPED (host-limited), never started and cut
NEED_granite=${P65_NEED_GRANITE_S:-1500}; NEED_olmoe=${P65_NEED_OLMOE_S:-2100}; NEED_mixtral=${P65_NEED_MIXTRAL_S:-4500}
: > summary.txt; echo "$P65_INSTANCE_ID" > INSTANCE_ID
# ---- staged pieces, byte-for-byte (names as the box sees them)
for f in p65_run.sh p65_census.py expert_entropy.py p65_reduce.py expert_residuals.py serve_stack.py p44_reduce.py k8_bake.py calib.json staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p65/staged.sha256"; finish 9; }
# ---- card class, host floors, egress: refusal rows, not stalls
# floor(): a host floor the READING needs. The reading refuses on it (rc 13 / 14, host-limited). A PROOF (P65_PROVE=1)
# only records it: the proof's box is not the reading's box, so a reading-only floor that refuses a proof buys nothing
# and burns the proof budget (Amendment 1: p65-prove-1 refused at 0.152 s vs 0.15, p65-prove-3 at 97.8 vs 100 MB/s).
# Card class (15), a dud (10) and disk (13) still refuse a proof: those would break the proof itself.
floor(){ local rc=$1 msg=$2
  if [ "${P65_PROVE:-0}" = "1" ]; then say "PROOF: would refuse the READING -- $msg (recorded, not enforced in a proof)"
    echo "floor_would_refuse_reading rc=$rc $msg" | tee -a forensics.txt >> summary.txt; return 0; fi
  say "REFUSED: $msg -- host-limited, not a result"; echo "refused: $msg" > REFUSAL; finish "$rc"; }
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac
DISK_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9); RAM_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "disk_avail_gb=$DISK_GB ram_avail_gb=$RAM_GB" >> forensics.txt
[ "${DISK_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${DISK_GB} GB free < ${MIN_DISK_GB} (Mixtral bf16 + its NF4 snapshot + arena)"; echo "refused: disk $DISK_GB GB" > REFUSAL; finish 13; }
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || floor 13 "host RAM ${RAM_GB} GB available < ${MIN_RAM_GB} (Mixtral's Hessians + one layer's weights)"
# host-side Hessian pre-flight: every calibration batch lands a K x K fp32 gram on the host and scales + adds it into
# the running mean (gptq_pack.HessianAccumulator, hessian_device="cpu"). At Mixtral's down shape (14336^2, 822 MB) that
# is ~4,100 updates for the registered census, and it is HOST-bound: 0.33 s per scale+add on the NAS Xeon W-1250, 0.027 s
# on an Apple M1 Max (bench/p65/rehearsal-a2000/accbench_*.json). A host above the floor would push Mixtral past the guard.
MAX_ACC_S=${P65_MAX_ACC_S:-0.15}
ACC_S=$(python - <<'PYA' 2>/dev/null || echo 99
import time, torch
H = torch.zeros(14336, 14336); g = torch.randn(14336, 14336); ts = []
for i in range(3):
    t0 = time.perf_counter(); H *= 0.9; H.add_(g, alpha=0.001); ts.append(time.perf_counter() - t0)
print(f"{sorted(ts)[1]:.3f}")
PYA
)
echo "host_scale_add_14336_s=$ACC_S (floor $MAX_ACC_S)" >> forensics.txt; say "host Hessian scale+add at 14336^2: ${ACC_S} s (floor ${MAX_ACC_S})"
if python3 -c "import sys; sys.exit(0 if float('${ACC_S:-99}') > float('$MAX_ACC_S') else 1)"; then floor 13 "host memory bandwidth (scale+add ${ACC_S} s > ${MAX_ACC_S})"; fi
# the Hessian budget scales with the host (bigger chunks = fewer forward passes); recorded
BUDGET=$(( RAM_GB * 2 / 5 )); [ "$BUDGET" -lt 16 ] && BUDGET=16; [ "$BUDGET" -gt 64 ] && BUDGET=64
export E4B_INT4_HESSIAN_BUDGET_GB=$BUDGET; echo "E4B_INT4_HESSIAN_BUDGET_GB=$BUDGET" >> forensics.txt
# Amendment 2: measure the fetch's OWN path. snapshot_download pulls with max_workers=4, so the probe is four parallel
# 50 MB ranges of one HF CDN file and the rate is their total bytes over the wall time of all four (a single-stream
# probe read 36.1 MB/s on p65-prove-4's box, whose 4-worker Granite fetch then ran at ~83.5 MB/s).
say "egress pre-flight: HF CDN, 4 parallel 50 MB ranges, 30 s cap (floor ${MIN_MBPS} MB/s aggregate; Mixtral is 93 GB)"
MBPS=$(python3 - <<'PYEG' 2>/dev/null || echo 0
# four parallel 50 MB byte ranges of one HF CDN file, in Python (the image need not ship curl); the rate is total bytes
# over the wall time of all four -- the way snapshot_download's four workers share the path
import time, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL = "https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors"
CH, N, CAP = 52428800, 4, 30.0
def get(i):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={i * CH}-{(i + 1) * CH - 1}"})
    got, t0 = 0, time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=CAP) as r:
            while time.perf_counter() - t0 < CAP:
                b = r.read(1 << 20)
                if not b:
                    break
                got += len(b)
    except Exception:
        pass
    return got
t0 = time.perf_counter()
with ThreadPoolExecutor(N) as ex:
    total = sum(ex.map(get, range(N)))
print(round(total / 1e6 / max(1e-3, time.perf_counter() - t0), 1))
PYEG
)
say "HF CDN ${MBPS} MB/s (4 parallel ranges)"; echo "hf_cdn_mbps_4x=$MBPS" >> forensics.txt
if python3 -c "import sys; sys.exit(0 if float('${MBPS:-0}') < float('$MIN_MBPS') else 1)"; then floor 14 "egress ${MBPS} MB/s < ${MIN_MBPS}"; fi
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
# ---- the PROVING mode (P65-PREREG "Box and cost"): the compute rule puts a proving rental (<= 10 min, <= $0.15) in front
# of any guard over 1 h. P65_PROVE=1 runs everything above -- card class, every host floor, egress, install, tripwire --
# then ONE registered family end to end on a cut-down census: Granite, its first plan-order layer, nseq 8 (fetch at the
# pin, bake, build, selfcheck, both texts, both halves, full rows). Its census is written as prove_census_granite.json,
# which the reducer never reads (it reads census_<family>.json), and MODE says which run this was. Not a reading.
PFX=""; MARGIN=600; ALARM_FLOOR=900; PROVE_FIRST=""
if [ "${P65_PROVE:-0}" = "1" ]; then
  FAMILIES=granite; NSEQ=8; PFX=prove_; MARGIN=60; ALARM_FLOOR=120; PROVE_FIRST=1; NEED_granite=${P65_NEED_PROVE_S:-240}
  echo prove > MODE; say "PROVING run (not a reading): granite, first 1 layer, nseq 8"; echo "mode=PROVE granite first-layers 1 nseq 8" | tee -a summary.txt
else
  echo reading > MODE
fi
# ---- helpers (P44-a's)
left(){ echo $(( P65_DEADLINE_EPOCH - $(date +%s) )); }
can_run(){ local need=$1; [ $(( $(date +%s) + need + MARGIN )) -le "$P65_DEADLINE_EPOCH" ] || { say "STOP: $2 needs ${need}s, only $(left)s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local l=$(( P65_DEADLINE_EPOCH - $(date +%s) - MARGIN )); [ "$l" -lt "$ALARM_FLOOR" ] && l=$ALARM_FLOOR; [ "$l" -gt 14400 ] && l=14400; echo "$l"; }
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
census(){ local FAM=$1 NEED=$2 FIRST=${3:-}; read MID REV <<<"$(python -c "import sys; sys.path.insert(0, '$W'); from serve_stack import MODELS; print(*MODELS['$FAM'])")"
  can_run "$NEED" "${FAM}_census" || return 40
  local t0; t0=$(date +%s)
  fetch "$MID" "$REV" && bake "$MID" $FAM || { echo "CENSUS $FAM fetch/bake FAILED" >> summary.txt; return 11; }
  say "census $FAM (nseq $NSEQ per text${FIRST:+, first $FIRST layers of the plan order})"
  E4B_MODEL_ID=$MID perl -e "alarm $(arm_alarm); exec @ARGV" python $W/p65_census.py --family $FAM --arena $W/work_$FAM/nf4.arena --calib $W/calib.json \
      --nseq "$NSEQ" ${FIRST:+--first-layers "$FIRST"} --out $W/${PFX}census_$FAM.json > logs/${PFX}census_$FAM.log 2>&1; local rc=$?
  tail -3 logs/${PFX}census_$FAM.log | sed "s/^/    /"; { echo -n "${PFX}census $FAM rc=$rc wall=$(( $(date +%s) - t0 ))s "; tail -1 logs/${PFX}census_$FAM.log | cut -c1-200; } >> summary.txt
  free_family $FAM "$MID"; return $rc; }
rc_any=0
FAMS=",$FAMILIES,"
[[ "$FAMS" == *,granite,* ]] && { census granite "$NEED_granite" "$PROVE_FIRST" || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }; }
[[ "$FAMS" == *,olmoe,* ]] && { census olmoe "$NEED_olmoe" || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }; }
[[ "$FAMS" == *,mixtral,* ]] && { census mixtral "$NEED_mixtral" "$MIXTRAL_FIRST" || { r=$?; [ "$rc_any" = 0 ] && rc_any=$r; }; }
# ---- every registered census present, complete (every requested layer, both texts, both halves) and self-checked
for FAM in ${FAMILIES//,/ }; do
  python - "$W/${PFX}census_$FAM.json" <<'PYC' >> summary.txt 2>&1 || { echo "ROW ${PFX}census_$FAM INCOMPLETE" >> summary.txt; [ "$rc_any" = 0 ] && rc_any=41; }
import json, sys
c = json.load(open(sys.argv[1]))
assert (c.get("selfcheck") or {}).get("ok"), "selfcheck did not pass"
assert c["layers_censused"] == c["layers_requested"], (c["layers_censused"], c["layers_requested"])
cells = {(r["text"], r["half"]) for r in c["rows"]}
assert cells == {(t, h) for t in ("wikitext", "c4val1") for h in (0, 1, "full")}, cells
print(f"ROW census_{c['family']} complete: {len(c['layers_censused'])} layers, {len(c['rows'])} rows, selfcheck max rel diff {c['selfcheck']['max_rel_diff']:.2e}")
PYC
done
if [ -z "$PFX" ]; then
  python $W/p65_reduce.py $W --md $W/p65_table.md --json $W/p65_verdicts.json > logs/reduce.log 2>&1 && cat $W/p65_table.md >> summary.txt || say "reducer smoke failed (the fetched receipts are re-reduced off the box)"
fi
say "----- summary -----"; cat summary.txt
finish "$rc_any"
