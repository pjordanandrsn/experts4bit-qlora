#!/bin/bash
# bench/p124/p124_run.sh -- lane P124, BOX side (bench/p124/PREREG-p124.md; e4b#846). Derived from
# bench/p122/p122_run.sh by named substitutions. Started detached by p124_drive.sh with the run's nonce; P124_RUN_NONCE
# first, then P124_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P124_SUCCESS.<nonce> only when the reducer ran
# (or, under P124_PROVE=1, when the proof passed).
#
# Does E4B_ATTN_INT4_WIDE=1 (the attention projections of a 17-64-row decode step on grouped-nf4-gemm's int4 small-M
# GEMM, #522 + e4b #1410, instead of a cached bf16 copy on cuBLAS) make SC2e's 64- and 32-row decode steps faster without
# costing teacher-forced quality? On ONE RTX 5090: SC2e's int4 server built by build_engine (eager, one slot, the route
# enabled; NF4 arena baked on the box); p124_box.py profiles the eager twins at 64 and 32 rows with the route off and on,
# times four captured-graph arms OFF, ON, ON, OFF at each depth (and their GPU busy fraction and peak memory), and runs
# P117's teacher-forced passes with the route off (the reference and the floor) and on (the subjects), two mutants and a
# captured function check; p124_reduce.py applies the registered rule.
#   premise  on THIS card, before anything is fetched: tests/test_decode_graph_buckets.py, 7 passed; grouped-nf4-gemm's
#            K16 tests compiled for this card (TRITON_INTERPRET=0), 25 passed; tests/test_int4_attn_wide.py, 11 passed
#            (its two CUDA tests: graphs sharing the route's workspace, replayed in either order); none skipped
#   order    install + tripwire; reducer self-test; premise; fetch; NF4 arena bake; the box; reduce
#
# Knobs (recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P124_GPU_CLASS P124_MIN_DISK_GB P124_MIN_RAM_GB P124_REHEARSAL P124_WINDOWS.
# P124_PROVE=1 is the PROVING RUN: everything above, end to end, on ibm-granite/granite-3.1-3b-a800m-instruct (its NF4
# expert store with int4 RTN attention, so the route under test exists) with 40 rows (still one bucket-64 piece) and 32
# teacher-forced positions.
set -uo pipefail
W=/root/p124; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p124: $*"; }
NONCE=${P124_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P124_RUN_NONCE.tmp && mv $W/P124_RUN_NONCE.tmp $W/P124_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P124_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P124_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P124_RUN_ID P124_DEADLINE_EPOCH P124_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=5cff3bdf9d8eaf63505d676168de211e7ddcc7a5   # grouped-nf4-gemm #522 (gemm_int4_b32_smallm block_m=), its merge commit; a registered constant
PROVE=${P124_PROVE:-0}
if [ "$PROVE" = 1 ]; then
  MODEL=ibm-granite/granite-3.1-3b-a800m-instruct; REV=a02780686e08a03fe0d2679a293b5c74a90efa89   # P94's pin (SC1's proof model)
  WINDOWS_DEF=40; CONT_DEF=32
  NEED_FETCH=300; NEED_BAKE=300; NEED_BOX=1200                                                  # the proof's own time-left checks (P109's lesson)
else
  MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39                            # SC1's pin, SC2e's model
  WINDOWS_DEF=64; CONT_DEF=128
  NEED_FETCH=2400; NEED_BAKE=1800; NEED_BOX=3000
fi
GPU_CLASS=${P124_GPU_CLASS:-5090}; MIN_DISK_GB=${P124_MIN_DISK_GB:-150}; MIN_RAM_GB=${P124_MIN_RAM_GB:-60}
REHEARSAL=${P124_REHEARSAL:-0}; WINDOWS=${P124_WINDOWS:-$WINDOWS_DEF}; CONT=${P124_CONT:-$CONT_DEF}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
# every serving lever and engine knob starts unset; the engine is built with E4B_ATTN_INT4_WIDE=1 and the box sets the
# route per arm; the reading then sets SC2e's int4 stack (below)
unset E4B_SERVE_EXP_INT4 E4B_SERVE_EXP_INT4_CALIB E4B_SERVE_ATTN_INT4_CALIB E4B_SERVE_ATTN_INT4 E4B_FUSE_T1_GLUE E4B_FUSE_T1_GLUE_R2 \
      E4B_FUSE_ROUTER_EPI E4B_INT4_KEEP_NF4 E4B_INT4_GROUPED_SMALLM E4B_INT4_LEAN_GLUE E4B_INT4_DECODE_A16 E4B_ROUTER_EPI_CAST \
      E4B_FUSED_KV_APPEND E4B_MXFP4_GEMV E4B_MXFP4_GROUPED_SMALLM E4B_NF4_GROUPED_SMALLM E4B_NF4_T1_DEVICE_GROUPING E4B_CALIB_SOURCE \
      E4B_CALIB_NSEQ E4B_MODEL_ID E4B_INT4_PREFILL E4B_PAGED_PREFILL_ATTN USE_HUB_KERNELS \
      E4B_PAGED_GRAPHS E4B_PAGED_BUCKETS E4B_PAGED_MAX_SEQS E4B_PAGED_MAX_TOKENS_PER_SEQ E4B_PAGED_CHUNK_TOKENS \
      E4B_PAGED_MAX_PREFILL_TOKENS E4B_PAGED_PLACEMENT E4B_PAGED_KV_GROUPS E4B_PAGED_FUSE_QKV E4B_PAGED_TORCH_THREADS \
      E4B_PAGED_PREFILL_GRAPH E4B_PAGED_BULK_KV E4B_INT4_WIDE_TILES E4B_ATTN_INT4_WIDE E4B_ATTN_INT4_SMALLM
# SC2e's served stack (bench/sc1/sc1_run.sh's SPEEDENV, FOLDS and ROUTEENV, byte for byte, + fused q/k/v); the proof's
# Granite runs its NF4 expert store with int4 RTN attention only, so the route under test exists there
FOLDS="E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
SPEEDENV="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 $FOLDS"
ROUTEENV="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"
if [ "$PROVE" = 1 ]; then LEVERS="$ROUTEENV E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0"; else LEVERS="$ROUTEENV $SPEEDENV E4B_PAGED_FUSE_QKV=1"; fi
# the route under test is enabled at build (its capability check runs on this card); the box sets it per arm
LEVERS="$LEVERS E4B_ATTN_INT4_WIDE=1"
# the engine is built eager with ONE small slot and no prefill graph: the box's passes build their own pools
ENGINE_KNOBS="E4B_PAGED_GRAPHS=0 E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=768 E4B_PAGED_PREFILL_GRAPH=0"
: > summary.txt; echo "$P124_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB prove=$PROVE windows=$WINDOWS cont=$CONT" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != 150 ] || [ "$MIN_RAM_GB" != 60 ] \
   || [ "$WINDOWS" != "$WINDOWS_DEF" ] || [ "$CONT" != "$CONT_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in p124_box.py p124_reduce.py p119_box.py p117_box.py p108_box.py p97_box.py k8_bake.py calib.json test_decode_graph_buckets.py \
         test_int4_smallm_interp.py test_int4_attn.py test_int4_attn_wide.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p124/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: the card class, the disk, the host RAM
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 61 GB checkpoint, its NF4 snapshot and arena)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P124_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P124_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P124_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.17.0; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, inspect, importlib.metadata as md
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
assert md.version("grouped-nf4-gemm") == "0.43.0", md.version("grouped-nf4-gemm")
import int4_smallm
assert "block_m" in inspect.signature(int4_smallm.gemm_int4_b32_smallm).parameters, "grouped-nf4-gemm #522: block_m="
assert int4_smallm.SMALLM_ROWS_MAX == 64 and int4_smallm.smallm_block_m(16) == 16 and int4_smallm.smallm_block_m(64) == 64
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import serve_paged
from experts4bit_qlora.engines import hot_residency
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
from experts4bit_qlora.serve_recipe import default_buckets
assert default_buckets(64) == (1, 2, 4, 8, 16, 32, 64), "E4B_PAGED_BUCKETS=auto (#1319): the buckets under test"
assert "E4B_PAGED_GRAPHS" in inspect.getsource(serve_paged.PagedServeConfig.from_env)
from experts4bit_qlora.engines import int4_attn
assert int4_attn.Int4Linear.SMALLM_ROWS_MAX == 16 and int4_attn.Int4Linear.WIDE_ROWS_MAX == 64, "the routes by row count"
assert int4_attn._wide_supported(int4_smallm.gemm_int4_b32_smallm), "E4B_ATTN_INT4_WIDE (#1410) sees block_m="
assert callable(int4_attn.resolve_wide) and callable(int4_attn.wide_workspace_bytes)
from experts4bit_qlora.engines.step_trace import StepTrace  # noqa: F401  (the busy fraction's dec_prep / dec_issue events)
import int4_b32  # noqa: F401  (K19 and the tile builders: the expert routes)
assert "DEVICE_GROUPING[0] = True" in inspect.getsource(serve_paged._batched_graph_grouping)
assert hasattr(PagedModelRunner, "enable_decode_graphs") and isinstance(hot_residency.DEVICE_GROUPING, list)
assert "capture" in inspect.signature(PagedModelRunner.enable_decode_graphs).parameters, "the padded eager twins (P109)"
assert hasattr(PagedModelRunner, "disable_decode_graphs"), "the arms release their graphs"
import sys; sys.path[:0] = ["/root/p124"]
import p119_box  # noqa: F401  (P119's decode bracket, pool and grouping, at its registered bytes)
import p117_box  # noqa: F401  (P117's windows() and paged_pass(), at its registered bytes)
assert callable(p117_box.paged_pass) and "mutant" in inspect.signature(p117_box.paged_pass).parameters
import p108_box  # noqa: F401  (P108's Attention patch, _score, _kl, _release at their registered bytes)
assert hasattr(p108_box, "Attention") and hasattr(p108_box, "_release")
import fp8_paged_attn, fp8_kv, nvme_arena  # noqa: F401  (the decode kernel, the KV append, the arena bake)
assert hasattr(fp8_kv, "fp8_kv_append_bt1"), "the buckets' fused append"
import experts4bit_qlora as e, torch, triton
open("/root/p124/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p124_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: the bucketed decode graphs, the K16 kernel up to 64 rows
# compiled here, and the route's e4b tests (two Int4Linear of one width captured in ONE graph, sharing the workspace)
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_decode_graph_buckets.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log)
(cd $W && PYTHONPATH='' TRITON_INTERPRET=0 perl -e 'alarm 1200; exec @ARGV' python -m pytest test_int4_smallm_interp.py -q -rs -p no:cacheprovider) > logs/premise_k16.log 2>&1
rc2=$?; LASTK=$(tail -1 logs/premise_k16.log)
(cd $W && PYTHONPATH='' perl -e 'alarm 900; exec @ARGV' python -m pytest test_int4_attn_wide.py -q -rs -p no:cacheprovider) > logs/premise_wide.log 2>&1
rc3=$?; LASTW=$(tail -1 logs/premise_wide.log)
{ echo -n "premise run rc=$rc: "; echo "$LASTL"; echo -n "premise k16 rc=$rc2: "; echo "$LASTK"; echo -n "premise wide rc=$rc3: "; echo "$LASTW"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "7 passed" && ! echo "$LASTL" | grep -q skipped \
   && [ "$rc2" = 0 ] && echo "$LASTK" | grep -q "25 passed" && ! echo "$LASTK" | grep -q skipped \
   && [ "$rc3" = 0 ] && echo "$LASTW" | grep -q "11 passed" && ! echo "$LASTW" | grep -q skipped; then
  echo "premise ok" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc k16 rc=$rc2 wide rc=$rc3" > REFUSAL; finish 25
fi
# ---- the checkpoint and its NF4 arena (P39's k8_bake.py, as SC1 and P119 bake it)
can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL @ $REV"
perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena"; mkdir -p $W/work
K8_MODEL="$SNAP" K8_WORK="$W/work" perl -e "alarm $(step_alarm 5400); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 \
  || { tail -3 logs/bake.log; say "BAKE FAIL"; finish 12; }
[ -e $W/work/nf4.arena ] || { say "BAKE FAIL: no arena"; finish 12; }
grep -a "BAKE" logs/bake.log | tail -1 | tee -a summary.txt
# ---- the box: SC2e's int4 server built eager with one slot, then every pass on its model
ENGINE_ENV="E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$W/work/nf4.arena E4B_PAGED_CALIB=$W/calib.json $ENGINE_KNOBS $LEVERS"
can_run $NEED_BOX box || finish 40
AL=$(step_alarm 4500); say "the box (alarm=$AL)"
# shellcheck disable=SC2086  # ENGINE_ENV is an assignment list by design
env PYTHONPATH= $ENGINE_ENV E4B_SHA=$E4B_SHA GNF4_SHA=$GNF4_SHA \
  perl -e "alarm $AL; exec @ARGV" python $W/p124_box.py --out $W/box.json --rows $WINDOWS --cont $CONT > logs/box.log 2>&1
rc=$?
{ echo -n "box rc=$rc "; grep -aE "^P124_BOX" logs/box.log | tail -1 | cut -c1-1500; echo; } | tee -a summary.txt
grep -aE "^P124_(ARM|QUALITY)" logs/box.log | tee -a summary.txt
[ "$rc" = 0 ] || { tail -8 logs/box.log | cut -c1-300 | tee -a summary.txt; say "BOX FAILED"; finish 26; }
say "reduce"; python $W/p124_reduce.py --dir $W --out $W/verdict.json --e4b-sha $E4B_SHA 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
if [ "$PROVE" = 1 ]; then
  V=$(python -c "import json; print(json.load(open('$W/verdict.json'))['verdict'])")
  case "$V" in VOID|NO_READING) say "PROVE: the reducer read $V on the proof -- not proved"; finish 27;; esac
  echo "PROVE -- the proving run: the whole box on $MODEL; reducer verdict $V (not a reading)" | tee -a summary.txt
  : > PROVED
fi
finish 0
