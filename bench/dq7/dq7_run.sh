#!/bin/bash
# BOX side; speaks tc1_drive's nonce contract. No compute creation or approval.
set -uo pipefail
W=${DQ7_W:-/root/tc1}; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
say(){ echo "[$(date -u +%FT%TZ)] dq7: $*" | tee -a summary.txt; }
mkdir -p logs receipts configs hf-cache
: > summary.txt
E4B_SHA=${E4B_SHA:?}
link=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>&1 | head -1)
case "$link" in "NVIDIA GeForce RTX 5090, "3[0-3][0-9][0-9][0-9]) ;; *) say "OUT OF BAND: $link"; finish 19;; esac
python dq3_vram_probe.py > logs/vram_probe.log 2>&1; rc=$?
if [ "$rc" = 3 ]; then echo 'vram floor' > REFUSAL; finish 18; elif [ "$rc" != 0 ]; then finish 9; fi
python dq3_egress_probe.py > logs/egress_probe.log 2>&1; rc=$?
if [ "$rc" = 4 ]; then echo 'egress' > REFUSAL; finish 14; elif [ "$rc" != 0 ]; then finish 9; fi
# Synthetic checkpoint data stays excluded under hf-cache, never fetched into the immutable receipt store.
free_kb=$(df -Pk . | tail -1 | awk '{print $4}')
[ "$free_kb" -ge 120000000 ] || { say "disk below 120 GB for synthetic checkpoints"; finish 9; }
for subject in qwen3_14b llama31_8b qwen3_32b; do cp "$subject.json" configs/ || finish 9; done
LOGGETTA_SHA=$(python -c 'import json; print(json.load(open("runtime.json"))["loggetta_sha"])') || finish 9
export LOGGETTA_SHA
GNF4_SHA=$(python -c 'import json; print(json.load(open("runtime.json"))["gnf4_sha"])') || finish 9
say "install pinned Loggetta $LOGGETTA_SHA and e4b $E4B_SHA"
perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "git+https://github.com/pjordanandrsn/loggetta.git@$LOGGETTA_SHA" \
  "bitsandbytes==0.50.2" "transformers==5.18.0" "peft==0.21.2" accelerate safetensors tokenizers \
  > logs/pip.log 2>&1 || { tail -20 logs/pip.log; finish 9; }
python - <<'PY' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; finish 9; }
import importlib.metadata as md, json, os, torch
from experts4bit_qlora.engines.dense_offload import _bnb_mirror_mismatches
runtime=json.load(open('runtime.json'))
for name, sha in [('loggetta',runtime['loggetta_sha']),('experts4bit-qlora',os.environ['E4B_SHA']),('grouped-nf4-gemm',runtime['gnf4_sha'])]:
    got=json.loads(md.distribution(name).read_text('direct_url.json'))['vcs_info']['commit_id']
    assert got == sha, (name,got,sha)
for name in ('torch','bitsandbytes','transformers','peft'):
    assert md.version(name) == runtime[name], (name,md.version(name),runtime[name])
assert not _bnb_mirror_mismatches()
assert torch.cuda.is_available()
print('DQ7 software tripwire PASS', runtime)
PY
cat logs/tripwire.log | tee -a summary.txt
python dq7_reduce.py --self-test > logs/reducer_selftest.log 2>&1 || finish 9
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,power.limit --format=csv,noheader > forensics.txt
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export TOKENIZERS_PARALLELISM=false
export LOGGETTA_DENSE_EXECUTE=1
say 'synthetic tiny checkpoint and deterministic CUDA proof'
python dq7_subject.py tiny hf-cache/tiny --config-dir configs > logs/build-tiny.log 2>&1 || finish 9
python dq7_arm.py --mode proof --checkpoint hf-cache/tiny --out receipts/proof.json > logs/proof.log 2>&1 \
  || { tail -30 logs/proof.log; say 'FUNCTION_FAIL: proof failed; no reading'; finish 11; }
cp hf-cache/tiny/dq7-subject.json receipts/subject-tiny.json
say 'proof PASS; capacity reading begins'
for subject in qwen3_14b llama31_8b qwen3_32b; do
  [ "$(date +%s)" -lt "$(( ${TC1_DEADLINE_EPOCH:?} - 900 ))" ] || { say "INCOMPLETE: guard has less than 900 s"; finish 11; }
  say "build architecture-only checkpoint $subject"
  python dq7_subject.py "$subject" "hf-cache/$subject" --config-dir configs > "logs/build-$subject.log" 2>&1 || finish 9
  cp "hf-cache/$subject/dq7-subject.json" "receipts/subject-$subject.json"
  seqs='512 2048'; [ "$subject" = qwen3_32b ] && seqs='2048 4096'
  for seq in $seqs; do
    for placement in device stream; do
      tag="read-$subject-$placement-$seq"
      say "$tag"
      perl -e 'alarm 1800; exec @ARGV' python dq7_arm.py --mode read --checkpoint "hf-cache/$subject" \
        --subject "$subject" --placement "$placement" --seq "$seq" --out "receipts/$tag.json" \
        > "logs/$tag.log" 2>&1 || { tail -30 "logs/$tag.log"; say "INCOMPLETE: $tag"; finish 11; }
      say "$tag completed"
    done
  done
done
python dq7_reduce.py receipts > receipts/dq7-read.json 2> logs/reduce.log || finish 12
python -c 'import json; print("DQ7",json.load(open("receipts/dq7-read.json"))["verdict"])' | tee -a summary.txt
finish 0
