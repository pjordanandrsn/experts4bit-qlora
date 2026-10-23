#!/bin/bash
# bench/serve-hybrid/hc_run.sh -- HC's one explore box (DESIGN-HC.md section 5), BOX side. EXPLORE, never a reading.
# Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER (TC1_EXTRA_STAGE carries this file, hc_client.py and bench/p39's
# calib.json), so it speaks tc1_drive's contract: the nonce handshake, summary.txt, and TC1_EXIT_CODE.<nonce> /
# TC1_SUCCESS.<nonce> / TP_DONE.<nonce> at the end, a refusal included. Nothing here creates, destroys or approves compute.
#   install e4b + gnf4 (+ the HTTP server's deps) -> tripwire (commits, CPU kernels, the solver-graphs refusal) -> fetch
#   Qwen3-30B-A3B -> bake the NF4 arena OUTSIDE the fetched tree -> serve_paged per arm, serially, greedy:
#     A  all-vram, eager (E4B_PAGED_GRAPHS=0), the reference
#     B  solver at 8.0 GiB VRAM / 3.0 GiB DRAM, hot_rows 128: VRAM, DRAM and NVMe tiers all non-empty
#     C  solver at 8.0 GiB VRAM / 0 DRAM, hot_rows 128: VRAM and NVMe only (NVMe experts compute on the GPU), the control
#   each arm's texts against A's: identical, or the first diverging character and token. The stop rule (ruled): if A or B
#   does not build, or B's comparison cannot be read, stop -- no retry, no iteration on the box.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] hc: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
ARENA_DIR=/root/hc-arena; ARENA=$ARENA_DIR/qwen3.nf4; PORT=8778; URL=http://127.0.0.1:$PORT
echo "EXPLORE (HC: serve_paged at the solver placement): not a reading" | tee -a summary.txt
for f in hc_client.py calib.json; do [ -s $W/$f ] || { echo "STAGE MISSING: $f" | tee -a summary.txt; finish 9; }; done

free_gb=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
ram_gb=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
[ "${free_gb:-0}" -ge 120 ] || { echo "FLOOR: $free_gb GB free < 120" | tee -a summary.txt; finish 13; }
[ "${mem_mib:-0}" -ge 30000 ] || { echo "FLOOR: $mem_mib MiB card < 30000" | tee -a summary.txt; finish 13; }
[ "${ram_gb:-0}" -ge 60 ] || { echo "FLOOR: $ram_gb GiB RAM < 60" | tee -a summary.txt; finish 13; }

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: gnf4 @$GNF4_SHA + e4b @$E4B_SHA, transformers 5.18.0, the HTTP server's deps"
perl -e 'alarm 2400; exec @ARGV' python -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "transformers==5.18.0" "bitsandbytes==0.50.2" accelerate safetensors "huggingface_hub>=0.23" sentencepiece \
  "fastapi==0.141.1" "uvicorn==0.54.0" > logs/pip.log 2>&1 \
  || { tail -5 logs/pip.log; echo "PIP FAIL" | tee -a summary.txt; finish 9; }
python - <<'PYT' > logs/tripwire.log 2>&1 || { cat logs/tripwire.log; echo "TRIPWIRE FAIL" | tee -a summary.txt; finish 9; }
import importlib.metadata as md, json, os, torch
for dist, want in (("experts4bit-qlora", os.environ["E4B_SHA"]), ("grouped-nf4-gemm", os.environ["GNF4_SHA"])):
    got = json.loads(md.distribution(dist).read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
    assert got == want, f"{dist}: installed {got} != pinned {want}"
assert torch.cuda.is_available()
import cpu_grouped
assert cpu_grouped.cpu_kernels_available(), "the CPU tier's native kernels did not build: the DRAM tier cannot engage"
from nvme_bake_nf4 import bake_nf4  # noqa: F401
import fastapi, uvicorn  # noqa: F401
from experts4bit_qlora.serve_paged import PagedServeConfig
os.environ.update(E4B_PAGED_MODEL="m", E4B_PAGED_PLACEMENT="solver", E4B_PAGED_GRAPHS="1", E4B_PAGED_MAX_SEQS="1")
try:
    PagedServeConfig.from_env()
except ValueError as e:
    assert "solver would capture nothing" in str(e), e
    print("solver-graphs refusal: refused at startup, as #2067 has it")
else:
    raise AssertionError("E4B_PAGED_GRAPHS=1 at solver was accepted: this e4b lacks the startup refusal")
print("hc tripwire OK", md.version("experts4bit-qlora"), md.version("grouped-nf4-gemm"), torch.__version__,
      torch.cuda.get_device_name(), "sm_%d%d" % torch.cuda.get_device_capability())
PYT
cat logs/tripwire.log | tee -a summary.txt
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader | tee forensics.txt
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a forensics.txt; grep -E "MemTotal" /proc/meminfo | tee -a forensics.txt
python hc_client.py --self-test 2>&1 | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || { echo "SELF-TEST FAIL" | tee -a summary.txt; finish 21; }

say "fetch $MODEL@${REV:0:8}"
fetch(){ perl -e 'alarm 3600; exec @ARGV' python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL', revision='$REV', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','*.jinja']))" 2>> logs/fetch.log | tail -1; }
snap=$(fetch); [ -d "$snap" ] || { echo "fetch retry" | tee -a summary.txt; snap=$(fetch); }
[ -d "$snap" ] || { tail -3 logs/fetch.log; echo "FETCH FAILED" | tee -a summary.txt; finish 22; }
echo "fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
say "bake the NF4 arena into $ARENA_DIR (outside the fetched tree)"
mkdir -p $ARENA_DIR
perl -e 'alarm 3600; exec @ARGV' python -c "from nvme_bake_nf4 import bake_nf4; bake_nf4('$snap', '$ARENA')" > logs/bake.log 2>&1
rc=$?; echo "bake rc=$rc $(du -sh "$ARENA" 2>/dev/null | cut -f1) $(tail -1 logs/bake.log | cut -c1-160)" | tee -a summary.txt
[ $rc = 0 ] && [ -s "$ARENA.index.json" ] || finish 23

wait_ready(){ local pid=$1 log=$2 cap=$3 t0 st; t0=$(date +%s)
  while :; do
    st=$(curl -s -m 5 "$URL/health" 2>/dev/null | grep -a -o -E '"status": ?"[a-z]+"' | head -1 | grep -a -o -E '[a-z]+"$' | tr -d '"')
    case "$st" in ready|busy) return 0;; error) return 45;; esac
    kill -0 "$pid" 2>/dev/null || return 45
    [ $(( $(date +%s) - t0 )) -ge "$cap" ] && return 44
    sleep 3
  done; }
stop_server(){ local pid=$1 i; kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done; kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; }
# arm TAG PLACEMENT GRAPHS VRAM_GB DRAM_GB HOT_ROWS -> receipts/run_TAG.json, receipts/health_TAG.json; rc 0 or the build's failure
arm(){ local tag=$1 place=$2 graphs=$3 vram=$4 dram=$5 hot=$6 log=logs/server_$1.log pid rc t0
  say "arm $tag: placement $place, graphs $graphs, tiers $vram/$dram GiB, hot_rows $hot"; t0=$(date +%s)
  setsid env PYTHONPATH= E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$ARENA E4B_PAGED_CALIB=$W/calib.json \
      E4B_PAGED_PLACEMENT=$place E4B_PAGED_GRAPHS=$graphs E4B_PAGED_VRAM_GB=$vram E4B_PAGED_DRAM_GB=$dram E4B_PAGED_HOT_ROWS=$hot \
      E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=1024 E4B_HOST=127.0.0.1 E4B_PORT=$PORT \
      python -m experts4bit_qlora.serve_paged > "$log" 2>&1 < /dev/null &
  pid=$!
  wait_ready $pid "$log" 1200; rc=$?
  curl -s -m 30 "$URL/health" -o receipts/health_$tag.json 2>/dev/null
  if [ $rc != 0 ]; then echo "arm $tag: BUILD FAILED (rc $rc) after $(( $(date +%s) - t0 )) s: $(tail -3 "$log" | tr '\n' ' ' | cut -c1-300)" | tee -a summary.txt
    stop_server $pid; return $rc; fi
  echo "arm $tag: ready in $(( $(date +%s) - t0 )) s; vram_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" | tee -a summary.txt
  perl -e 'alarm 1500; exec @ARGV' python hc_client.py run --url $URL --model $MODEL --arm $tag --max-tokens 32 \
      --out receipts/run_$tag.json > logs/client_$tag.log 2>&1; rc=$?
  tail -1 logs/client_$tag.log | tee -a summary.txt
  stop_server $pid
  [ $rc = 0 ] || { echo "arm $tag: CLIENT FAILED rc $rc" | tee -a summary.txt; return 46; }
  return 0; }
tiers(){ python hc_client.py tiers --arm $1 --config "$snap/config.json" --arena $ARENA --calib $W/calib.json \
           --vram-gb $2 --dram-gb $3 --out receipts/tiers_$1.json 2>&1 | tail -1 | tee -a summary.txt; }
cmp_arm(){ python hc_client.py compare --base receipts/run_A.json --other receipts/run_$1.json --tokenizer "$snap" \
             --out receipts/compare_$1.json 2>&1 | tail -1 | tee -a summary.txt; return "${PIPESTATUS[0]}"; }

arm A all-vram 0 1.2 6.0 64 || { echo "STOP: the reference arm A did not build or answer" | tee -a summary.txt; finish 31; }
tiers B 8.0 3.0
arm B solver auto 8.0 3.0 128 || { echo "STOP: arm B (solver, three tiers) did not build or answer" | tee -a summary.txt; finish 32; }
cmp_arm B || { echo "STOP: B against A could not be compared" | tee -a summary.txt; finish 33; }
tiers C 8.0 0
rc=0
arm C solver auto 8.0 0 128 && cmp_arm C || { echo "arm C (the control) did not complete" | tee -a summary.txt; rc=34; }
echo "HC done: arms $(ls receipts/run_*.json 2>/dev/null | wc -l | tr -d ' '), EXPLORE, not a reading" | tee -a summary.txt
finish $rc
