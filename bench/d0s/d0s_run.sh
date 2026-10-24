#!/bin/bash
# bench/d0s/d0s_run.sh -- lane D0s, BOX side (bench/d0s/PREREG-d0s.md). Started detached by d0s_drive.sh with the run's
# nonce: D0S_RUN_NONCE first, then D0S_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit, and D0S_SUCCESS.<nonce> only
# when every step ran (D0S_SHAKE=1: SHAKE_OK).
#   order   refusals (card, disk, RAM); install e4b @E4B_SHA + grouped-nf4-gemm @GNF4_SHA; a clean checkout of E4B_SHA
#           whose kit files equal staged.sha256; tripwire; the premise (tests/test_merged_arena_gpu.py on THIS card,
#           both cases PASSED, none skipped); then
#   shake   (D0S_SHAKE=1) d0s_box.py prepare --mode shake + run: the whole flow on the GPU serve smoke's tiny random
#           Qwen3-MoE, every writer cell WRITTEN and every arm rc 0 -> SHAKE_OK. Never data.
#   read    the checkpoint at its pin; d0s_box.py prepare --mode read (the base bake on the card, the attention-only
#           adapters, the writer's four merges) + run (every arm in its own process). The reducer runs on the controller.
# Knobs (recorded; a value off its registered default marks the run a REHEARSAL, NOT a reading):
# D0S_GPU_CLASS D0S_MIN_DISK_GB D0S_MIN_RAM_GB.
set -uo pipefail
W=/root/d0s; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] d0s: $*"; }
NONCE=${D0S_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/D0S_RUN_NONCE.tmp && mv $W/D0S_RUN_NONCE.tmp $W/D0S_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > D0S_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > D0S_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in D0S_RUN_ID D0S_DEADLINE_EPOCH D0S_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
case "$E4B_SHA" in *[!0-9a-f]*|"") say "refusing: E4B_SHA is not hex"; finish 78;; esac
[ ${#E4B_SHA} -eq 40 ] || { say "refusing: E4B_SHA is not a 40-char sha"; finish 78; }
GNF4_SHA=a83f6bb3a3a69aef8981fb81b6361bf62b06f6e7   # grouped-nf4-gemm v0.49.0, e4b CI's pin; a registered constant
MODEL=allenai/OLMoE-1B-7B-0924-Instruct; REV=7f1c97f440f06ce36705e4f2b843edb5925f4498   # D0's checkpoint
SHAKE=${D0S_SHAKE:-0}
GPU_CLASS=${D0S_GPU_CLASS:-4090}; MIN_DISK_GB=${D0S_MIN_DISK_GB:-60}; MIN_RAM_GB=${D0S_MIN_RAM_GB:-30}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false HF_HUB_CACHE=$W/hf
# every serving lever and engine knob starts unset: the arms set only what the registration names
for v in $(compgen -e | grep '^E4B_' | grep -v '^E4B_SHA$'); do unset "$v"; done
: > summary.txt; echo "$D0S_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV shake=$SHAKE gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB" | tee -a summary.txt
if [ "$GPU_CLASS" != 4090 ] || [ "$MIN_DISK_GB" != 60 ] || [ "$MIN_RAM_GB" != 30 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
[ -s $W/staged.sha256 ] || { say "STAGE MISSING: staged.sha256"; finish 9; }
if [ "$SHAKE" != 1 ]; then
  for fx in nodq dq; do [ -s $W/adapters/fixture-$fx-v2/adapter_manifest.json ] || { say "STAGE MISSING: adapter fixture-$fx-v2"; finish 9; }; done
fi
# ---- refusals before anything is installed or fetched
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
step_alarm(){ local cap=$1 left=$(( D0S_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 300 ] && left=300; echo "$left"; }
TORCH_PIN=$(python -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==$TORCH_PIN" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (transformers 5.17.0, bitsandbytes 0.50.2; torch held at $TORCH_PIN)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest fastapi httpx || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
# ---- a clean checkout of the launch commit: the kit runs from it, and its kit files must equal staged.sha256
rm -rf $W/e4bsrc && mkdir -p $W/e4bsrc && cd $W/e4bsrc && git init -q && git remote add origin https://github.com/pjordanandrsn/experts4bit-qlora.git \
  && perl -e 'alarm 600; exec @ARGV' git fetch -q --depth 1 origin "$E4B_SHA" && git checkout -q FETCH_HEAD || { cd $W; say "CHECKOUT FAIL"; finish 9; }
[ "$(git rev-parse HEAD)" = "$E4B_SHA" ] && [ -z "$(git status --porcelain)" ] || { cd $W; say "CHECKOUT is not a clean $E4B_SHA"; finish 9; }
sha256sum -c $W/staged.sha256 > $W/logs/staged.log 2>&1 || { cd $W; tail -3 $W/logs/staged.log; say "KIT FILES DIFFER FROM staged.sha256"; finish 9; }
cd $W
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md, hashlib
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the launch commit: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pinned commit: {dg}"
import transformers
assert transformers.__version__ == "5.17.0", transformers.__version__
from experts4bit_qlora import merged_arena, serve_paged, adapter_dir
assert hasattr(merged_arena, "merged_arena_refusals") and hasattr(merged_arena, "refuse_merged_path")
assert "merge" in serve_paged.build_engine.__code__.co_varnames
for mod in (merged_arena, serve_paged, adapter_dir):
    rel = mod.__file__.split("experts4bit_qlora/", 1)[1]
    a = open(mod.__file__, "rb").read(); b = open(f"/root/d0s/e4bsrc/experts4bit_qlora/{rel}", "rb").read()
    assert a == b, f"installed {rel} differs from the checkout's"
import nvme_bake_nf4, nvme_arena  # noqa: F401  (the bake under the base arena and the writer's checkpoint source)
import experts4bit_qlora as e, torch, triton
open("/root/d0s/versions.txt", "a").write(f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\ncc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "gnf4", md.version("grouped-nf4-gemm"), "torch", torch.__version__)
PYT
cat versions.txt | tee -a summary.txt
# ---- the premise, on THIS card: the default build carries no merge record; a merged arena serves as written
(cd $W/e4bsrc && perl -e 'alarm 1200; exec @ARGV' python -m pytest tests/test_merged_arena_gpu.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; LASTL=$(tail -1 logs/premise.log); { echo -n "premise rc=$rc: "; echo "$LASTL"; } | tee -a summary.txt
if [ "$rc" = 0 ] && echo "$LASTL" | grep -q "2 passed" && ! echo "$LASTL" | grep -q skipped; then echo "premise ok" | tee -a summary.txt
else say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25; fi
BOX="python $W/e4bsrc/bench/d0s/d0s_box.py"
if [ "$SHAKE" = 1 ]; then
  say "shake: prepare"
  perl -e "alarm $(step_alarm 1800); exec @ARGV" $BOX prepare --mode shake --out $W/work > logs/prepare.log 2>&1 || { tail -5 logs/prepare.log; say "SHAKE FAILED at prepare"; finish 31; }
else
  say "fetch $MODEL @ $REV"
  perl -e "alarm $(step_alarm 2400); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.txt'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
  SNAP=$(tail -1 logs/fetch.log); [ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
  echo "FETCH $MODEL@$REV" | tee -a summary.txt
  if [ "${D0S_PROBE:-0}" = 1 ]; then   # EXPLORE (never data): the base bake and the writer at both sources for nodq, timed
    say "probe: the base bake and the writer for nodq at both sources (EXPLORE, never data)"
    perl -e "alarm $(step_alarm 9000); exec @ARGV" python $W/e4bsrc/bench/d0s/d0s_writer_probe.py --snapshot "$SNAP" --adapters $W/adapters --out $W/work > logs/probe.log 2>&1 || { tail -5 logs/probe.log; say "PROBE FAILED"; finish 33; }
    tail -3 logs/probe.log | tee -a summary.txt; echo "PROBE_OK -- EXPLORE, never data" | tee -a summary.txt; : > PROBE_OK
    finish 0
  fi
  say "prepare: the base bake, the attention-only adapters, the writer's merges"
  perl -e "alarm $(step_alarm 5400); exec @ARGV" $BOX prepare --mode read --snapshot "$SNAP" --adapters $W/adapters --out $W/work > logs/prepare.log 2>&1 || { tail -5 logs/prepare.log; say "PREPARE FAILED"; finish 31; }
fi
tail -1 logs/prepare.log | tee -a summary.txt
say "the arms, one process each"
perl -e "alarm $(step_alarm 5400); exec @ARGV" $BOX run --out $W/work > logs/arms.log 2>&1; rc=$?
tail -1 logs/arms.log | tee -a summary.txt
if [ "$SHAKE" = 1 ]; then
  python - <<'PYS' || { say "SHAKE FAILED"; finish 32; }
import json
p = json.load(open("/root/d0s/work/prepare.json")); s = json.load(open("/root/d0s/work/arms-status.json"))
cells = {f"{fx}-{c}": v["status"] for fx, r in p["fixtures"].items() for c, v in r["cells"].items()}
assert all(v == "WRITTEN" for v in cells.values()), cells
assert len(s) == 8 and all(v == 0 for v in s.values()), s
print("SHAKE_OK", cells, s)
PYS
  echo "SHAKE_OK -- the whole flow ran on the smoke's tiny model (never data)" | tee -a summary.txt; : > SHAKE_OK
  finish 0
fi
[ "$rc" = 0 ] || { say "AN ARM FAILED (see work/arms-status.json)"; finish 26; }
finish 0
