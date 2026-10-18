#!/bin/bash
# bench/p140/p140_run.sh -- lane P140, BOX side (bench/p140/PREREG-p140.md). Started detached by p140_drive.sh with the
# run's nonce; P140_RUN_NONCE first, then P140_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and P140_SUCCESS.<nonce>
# only when the reducer ran.
#
# Does layer-major accumulation (engines/layer_major.py) cut the step time of expert-offloaded QLoRA training on the
# configuration P137 licensed on an RTX 4090 (experts offloaded, enable_fast_train(dgrad=True)) -- on an RTX 5090?
# A replication of P137 (bench/p137/PREREG-p137.md, Amendments 1-3 folded in). Derived from bench/p137/p137_run.sh by
# named substitutions:
#   pins     e4b E4B_PIN (main with layer-major and its guards), grouped-nf4-gemm 0.46.0 exact, torch 2.8.0+cu128 (the
#            image's), transformers 5.18.0 (TC3's e4b box), bitsandbytes 0.50.2; any other is refused;
#   card     the RTX 5090 class, compute capability 12.0 (the licence would cover that configuration only);
#   arms     A0 = the stock accumulation loop; L1 = layer_major_step; ABBA A0a L1a L1b A0b in ONE process after one
#            load (P137 Amendment 2: p140_box.py --mode abba), each from the post-load adapter snapshot, the same steps,
#            micro-batches and seed;
#   premise  rc 0 with exactly the registered pass count and no skip (tests/test_layer_major.py, test_offload_hold.py,
#            test_offload.py at the pin), listed with -rs;
#   dry      P140_DRY=1: the $0 dry run on a tiny Qwen3-MoE fixture (p140_tiny.py) on any CUDA card, on the BOX'S OWN
#            stack (recorded, never installed: the dry box is shared), the e4b tree on PYTHONPATH. Never a reading.
#   order    refusals; install + tripwire (or the dry stack's record); reducer and box self-tests; premise; the staged
#            data slice's sha256 (P137 Amendment 1);
#            fetch (or the fixture); four arms A0a L1a L1b A0b; reduce
# Refusals: CUDA unusable 18 (the host floor; a torch that will not import stays 10), torch not the pin 28, card class 15,
# disk 13, host RAM 16. Only 13 and 18 name the machine for the launcher's next draw.
# Knobs (recorded; any value off its registered default marks the run a REHEARSAL, NOT a reading):
# P140_GPU_CLASS P140_MIN_DISK_GB P140_MIN_RAM_GB P140_REHEARSAL.
set -uo pipefail
W=${P140_HOME:-/root/p140}; export P140_HOME=$W; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p140: $*"; }
NONCE=${P140_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P140_RUN_NONCE.tmp && mv $W/P140_RUN_NONCE.tmp $W/P140_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P140_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P140_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P140_RUN_ID P140_DEADLINE_EPOCH P140_INSTANCE_ID E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
E4B_PIN=b4366b9af2234730b5df783c5f3b0c7aef3ab41c      # main with #1708 #1710 #1715 (layer-major and its guards); a registered constant
GNF4_SHA=6b7c2e7474d20c8843814dd46dad6cfc3d94b440      # grouped-nf4-gemm 0.46.0, exact; a registered constant
TORCH_WANT=2.8.0+cu128
PREMISE_PASSED=47                                       # test_layer_major 15 + test_offload_hold 13 + test_offload 19, none skipped
[ "$E4B_SHA" = "$E4B_PIN" ] || { say "refusing: E4B_SHA $E4B_SHA is not the registered e4b pin $E4B_PIN"; finish 78; }
DRY=${P140_DRY:-0}
if [ "$DRY" = 1 ]; then
  MODEL=$W/p140-tiny; REV=""                                                                      # p140_tiny.py's fixture
  NEED_FETCH=300; NEED_ABBA=600; DISK_DEF=10; RAM_DEF=16                                           # the fixture's own floors
else
  MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39                            # TC3's pin
  NEED_FETCH=2700; NEED_ABBA=4500; DISK_DEF=120; RAM_DEF=64   # A2: one load + 4 arms
fi
GPU_CLASS=${P140_GPU_CLASS:-5090}; MIN_DISK_GB=${P140_MIN_DISK_GB:-$DISK_DEF}; MIN_RAM_GB=${P140_MIN_RAM_GB:-$RAM_DEF}
REHEARSAL=${P140_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false P140_DRY=$DRY
# every training lever starts unset: TC3's fused offload configuration at its defaults is the subject
unset LAYER_MAJOR E4B_OFFLOAD_STATS E4B_OFFLOAD_ARENA E4B_OFFLOAD_PIN_SLAB E4B_CKPT_OFFLOAD E4B_CHUNKED_LM_LOSS \
      E4B_CHUNKED_EVAL_LOSS E4B_ABSMAX_DQ E4B_MOE_KEEP_LAYERS TRAIN_ARENA OFFLOAD_EXPERTS OFFLOAD_PIN TRITON_INTERPRET
: > summary.txt; echo "$P140_INSTANCE_ID" > INSTANCE_ID
echo "KNOBS run=$P140_RUN_ID e4b=$E4B_SHA gnf4=$GNF4_SHA model=$MODEL@$REV gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB min_ram_gb=$MIN_RAM_GB dry=$DRY" | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$GPU_CLASS" != 5090 ] || [ "$MIN_DISK_GB" != "$DISK_DEF" ] || [ "$MIN_RAM_GB" != "$RAM_DEF" ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
[ "$DRY" = 1 ] && { echo "DRY -- the \$0 dry run on the tiny fixture: NEVER a reading" | tee -a summary.txt; : > DRY; }
# ---- staged pieces, byte-for-byte
for f in p140_box.py p140_reduce.py p140_tiny.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p140/staged.sha256"; finish 9; }
# ---- refusals before anything is installed or fetched: CUDA, the torch pin, the card, the disk, the host RAM
CUDA_PROBE=$(python - <<'PYC' 2>/dev/null | tail -1
import sys
try:
    import torch
except Exception as e:
    print(f"no-torch {type(e).__name__}"); sys.exit(0)
try:
    ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
except Exception:
    ok = False
if not ok:
    print(f"no-cuda torch {torch.__version__}"); sys.exit(0)
cc, p = torch.cuda.get_device_capability(0), torch.cuda.get_device_properties(0)
print(f"ok torch={torch.__version__} cc={cc[0]}.{cc[1]} sms={p.multi_processor_count}")
PYC
)
echo "CUDA_PROBE ${CUDA_PROBE:-none}" | tee -a summary.txt
case "$CUDA_PROBE" in
  ok*) ;;
  no-cuda*) say "REFUSED: torch cannot use the GPU on this host (${CUDA_PROBE}) -- registered host floor"
            echo "refused: cuda unusable (${CUDA_PROBE})" > REFUSAL; echo "BOX_REFUSED cuda=unusable" >> summary.txt; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
if [ "$DRY" = 1 ]; then
  echo "DRY: the box's own torch ($CUDA_PROBE), recorded, not the pin" | tee -a summary.txt
else
  case "$CUDA_PROBE" in *" torch=$TORCH_WANT "*) ;; *) say "REFUSED: the image's torch is not the registered $TORCH_WANT ($CUDA_PROBE)"; echo "refused: torch ($CUDA_PROBE)" > REFUSAL; finish 28;; esac
fi
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid,compute_cap,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^Vendor ID|^CPU\(s\)" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [ "$DRY" = 1 ]; then
  echo "DRY: card '$GPU_NAME' ($CUDA_PROBE) -- the class is not checked" | tee -a summary.txt
else
  case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
  case "$CUDA_PROBE" in *" cc=12.0 "*) ;; *) say "REFUSED: $CUDA_PROBE -- the licence covers sm_120 only"; echo "refused: config $CUDA_PROBE" > REFUSAL; finish 15;; esac
fi
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (a 61 GB checkpoint and the install)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
RAM_GB=$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)
[ "${RAM_GB:-0}" -ge "$MIN_RAM_GB" ] || { say "REFUSED: ${RAM_GB:-?} GiB host RAM < ${MIN_RAM_GB} GiB"; echo "refused: ram ${RAM_GB:-?} GiB" > REFUSAL; finish 16; }
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P140_DEADLINE_EPOCH" ] || { say "STOP-2: $2 needs ${need}s, only $((P140_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( P140_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 600 ] && left=600; echo "$left"; }
export DEBIAN_FRONTEND=noninteractive
if [ "$DRY" = 1 ]; then
  # the dry box is shared: nothing is installed; the e4b tree (staged as e4bsrc/ by the driver) runs from PYTHONPATH
  [ -d $W/e4bsrc/experts4bit_qlora ] || { say "DRY: no e4b tree at $W/e4bsrc"; finish 9; }
  export PYTHONPATH=$W/e4bsrc
  python -c "import experts4bit_qlora.engines.layer_major" || { say "DRY: the e4b tree does not import"; finish 9; }
  E4BSRC=$W/e4bsrc
else
  echo "torch==${TORCH_WANT%%+*}" > $W/constraints.txt
  pipx(){ local log=$1 secs=$2; shift 2
    perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
    say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
    perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
  say "install e4b @$E4B_SHA + gnf4 @$GNF4_SHA (image python; transformers 5.18.0; huggingface_hub < 2; bitsandbytes 0.50.2; torch held at $TORCH_WANT)"
  pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
    "transformers==5.18.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23,<2" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
  pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
  # the e4b tests at the pin, for the premise (they run against the installed package)
  perl -e 'alarm 600; exec @ARGV' git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git $W/e4bsrc > logs/clone.log 2>&1 && git -C $W/e4bsrc checkout -q $E4B_SHA \
    || { say "CLONE FAIL (e4b, for the premise tests)"; finish 9; }
  E4BSRC=$W/e4bsrc
fi
WANT_E4B=$E4B_SHA WANT_GNF4=$GNF4_SHA WANT_TORCH=$TORCH_WANT python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, json, importlib.metadata as md
dry = os.environ.get("P140_DRY") == "1"
import torch, transformers
if not dry:
    d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
    assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the pin: {d}"
    dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
    assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pin: {dg}"
    assert md.version("grouped-nf4-gemm") == "0.46.0", md.version("grouped-nf4-gemm")
    assert md.version("bitsandbytes") == "0.50.2", md.version("bitsandbytes")
    assert transformers.__version__ == "5.18.0", transformers.__version__
    hub = md.version("huggingface_hub")
    assert int(hub.split(".")[0]) < 2, f"huggingface_hub {hub}: P137 Amendment 1 pins < 2 (hub 2.x's httpx2 breaks on this image)"
    assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__
    assert torch.cuda.get_device_capability() == (12, 0), torch.cuda.get_device_capability()
from experts4bit_qlora.engines import layer_major as lm, offload as off
from experts4bit_qlora.engines.fast import enable_fast_train  # noqa: F401
assert lm.SUPPORTED == ("Qwen3MoeForCausalLM",)
assert hasattr(lm, "LayerMajorDriftError") and hasattr(lm, "LayerMajorNotReproducibleError"), "the guards (#1715)"
assert hasattr(off, "hold_forward_residency") and hasattr(off._ExpertOffload, "_held"), "the per-handle hold (#1708)"
assert not any(os.environ.get(k) for k in ("LAYER_MAJOR", "E4B_OFFLOAD_STATS", "E4B_CKPT_OFFLOAD", "E4B_CHUNKED_LM_LOSS"))
import nf4_grouped  # noqa: F401  (the grouped kernels enable_fast_train routes through)


def _has(name):
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


import experts4bit_qlora as e
open(os.path.join(os.environ.get("P140_HOME", "/root/p140"), "versions.txt"), "a").write(
    f"e4b {e.__version__} @{os.environ['WANT_E4B']}{' (dry: the tree)' if dry else ''}\ngnf4 {md.version('grouped-nf4-gemm')}\n"
    f"torch {torch.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n"
    f"huggingface_hub {md.version('huggingface_hub')}\n"
    + "".join(f"{n} {md.version(n)}\n" for n in ("httpx", "httpx2") if _has(n))
    + f"cc {torch.cuda.get_device_capability()}\n")
print("tripwire OK:", e.__version__, "torch", torch.__version__, "transformers", transformers.__version__)
PYT
cat versions.txt | tee -a summary.txt
python $W/p140_reduce.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "REDUCER SELF-TEST FAILED"; finish 21; }
python $W/p140_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { say "BOX SELF-TEST FAILED"; finish 21; }
# ---- the premise, on THIS card, before anything is fetched: layer-major, the hold and offload, exact pass count, no skip
(cd $E4BSRC && perl -e 'alarm 1200; exec @ARGV' python -m pytest tests/test_layer_major.py tests/test_offload_hold.py tests/test_offload.py -q -rs -p no:cacheprovider) > logs/premise.log 2>&1
rc=$?; { echo -n "premise rc=$rc: "; tail -1 logs/premise.log; } | tee -a summary.txt
grep -a "^SKIPPED" logs/premise.log | tee -a summary.txt
if [ "$rc" = 0 ] && tail -1 logs/premise.log | grep -Eq "^=* ?$PREMISE_PASSED passed(, [0-9]+ warnings?)? in "; then
  echo "premise ok ($PREMISE_PASSED passed, none skipped)" | tee -a summary.txt
else
  echo "premise failed" | tee -a summary.txt; say "PREMISE FAILED on this card"; echo "premise failed rc=$rc" > REFUSAL; finish 25
fi
# ---- the training rows (alpaca train[:400], the dataset TC3's recipe trains on): staged by the driver from the controller
# ---- (P137 Amendment 1: no dataset download on the box; p137-4090-1 died in one), checked against its registered sha256
DATA_SHA=68e3d31c194ebd8fee4330808a46642f443aecf4e16b78e5eb39fd21b16079c5
[ -s $W/data.json ] || { say "DATA FAIL: data.json was not staged"; finish 19; }
GOT_SHA=$(sha256sum $W/data.json | cut -d" " -f1)
[ "$GOT_SHA" = "$DATA_SHA" ] || { say "DATA FAIL: data.json is $GOT_SHA, registered $DATA_SHA"; finish 19; }
echo "P140_DATA staged, sha256 $GOT_SHA (registered)" | tee -a summary.txt
# ---- the checkpoint (or the dry fixture)
can_run $NEED_FETCH fetch || finish 40
if [ "$DRY" = 1 ]; then
  say "build the dry fixture"
  perl -e "alarm $(step_alarm 1200); exec @ARGV" python $W/p140_tiny.py --out "$MODEL" > logs/fetch.log 2>&1 || { tail -3 logs/fetch.log; say "FIXTURE FAIL"; finish 11; }
  grep -a "^P140_TINY" logs/fetch.log | tee -a summary.txt; SNAP=$MODEL
else
  say "fetch $MODEL @ $REV"
  perl -e "alarm $(step_alarm 2700); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=8))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
  SNAP=$(tail -1 logs/fetch.log); echo "FETCH $MODEL@$REV $SNAP" | tee -a summary.txt
fi
[ -d "$SNAP" ] || { say "DL FAIL: no snapshot dir"; finish 11; }
# ---- the four arms A0a L1a L1b A0b (ABBA) in ONE process, one load (P137 Amendment 2): each arm starts from the post-load
# ---- adapter snapshot (hash-checked), a new optimizer and the same seed; only the accumulation loop differs
can_run $NEED_ABBA "the four arms" || finish 40
AL=$(step_alarm 7200); say "the four arms, one process (alarm=$AL)"
env E4B_SHA=$E4B_SHA perl -e "alarm $AL; exec @ARGV" python $W/p140_box.py --mode abba --model "$SNAP" \
  --data $W/data.json --outdir $W > logs/abba.log 2>&1
rc=$?
{ echo "abba rc=$rc"; grep -a "^P140_LOAD" logs/abba.log | tail -1; } | tee -a summary.txt
for TAG in A0a L1a L1b A0b; do
  { echo -n "arm $TAG "; grep -a "^P140_ARM {\"tag\": \"$TAG\"" logs/abba.log | tail -1 | cut -c1-600; echo; } | tee -a summary.txt
done
[ "$rc" = 0 ] || { tail -6 logs/abba.log | cut -c1-300 | tee -a summary.txt; say "THE ARMS FAILED (rc=$rc) -- the reducer will VOID"; }
DF=""; [ "$DRY" = 1 ] && DF="--dry"
say "reduce"; python $W/p140_reduce.py --dir $W --out $W/verdict.json --e4b-sha $E4B_SHA $DF 2>&1 | tee -a summary.txt
[ "${PIPESTATUS[0]}" = 0 ] && [ -s $W/verdict.json ] || { say "REDUCER FAILED"; finish 22; }
[ "$DRY" = 1 ] && echo "DRY -- reducer verdict $(python -c "import json; print(json.load(open('$W/verdict.json'))['verdict'])") on the tiny fixture (never a reading)" | tee -a summary.txt
finish 0
