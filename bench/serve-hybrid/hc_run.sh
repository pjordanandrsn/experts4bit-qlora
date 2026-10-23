#!/bin/bash
# bench/serve-hybrid/hc_run.sh -- HC's one explore box (DESIGN-HC.md section 5), BOX side. EXPLORE, never a reading.
# Started by bench/tc1/tc1_drive.sh as its TC1_RUNNER (TC1_EXTRA_STAGE carries this file, hc_client.py and bench/p39's
# calib.json), so it speaks tc1_drive's contract: the nonce handshake, summary.txt, and TC1_EXIT_CODE.<nonce> /
# TC1_SUCCESS.<nonce> / TP_DONE.<nonce> at the end, a refusal included. Nothing here creates, destroys or approves compute.
#   download probe -> guard fit -> install e4b + gnf4 (+ the HTTP server's deps) -> tripwire (commits, CPU kernels, the
#   solver-graphs refusal) -> fetch Qwen3-30B-A3B -> bake the NF4 arena OUTSIDE the fetched tree -> serve_paged per arm,
#   serially, greedy:
#     A  all-vram, eager (E4B_PAGED_GRAPHS=0), the reference, at C's knobs (8.0 / 0 GiB, hot_rows 128) apart from placement
#     B  solver at 8.0 GiB VRAM / 3.0 GiB DRAM, hot_rows 128: its premise (VRAM, DRAM and NVMe tiers all non-empty, the
#        solver's own split recomputed) is gated before the arm (rc 36)
#     C  solver at 8.0 GiB VRAM / 0 DRAM, hot_rows 128: VRAM and NVMe only (NVMe experts compute on the GPU), the control;
#        its premise (DRAM empty, VRAM and NVMe not) is gated too, and C is skipped, never a stop, when it fails or time is short
#   each arm's texts against A's: identical, or the first diverging character and token. The stop rule (ruled): if A or B
#   does not build, or B's comparison cannot be read, stop -- no retry, no iteration on the box.
# Before any install: the download probe (P231's form; below 40 MB/s is rc 19, HOST_REFUSED, never a strike) and the
# guard-fit check (bench/common/README.md, "Guard fit"; rc 41). Before the fetch, the bake and each arm: can_run on the
# time left (rc 40, TP_DONE written, the evidence so far kept). Each step's alarm is capped by the time left.
set -uo pipefail
W=/root/tc1; cd "$W" || exit 9
NONCE=${TC1_RUN_NONCE:?}; printf '%s\n' "$NONCE" > TC1_RUN_NONCE
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
say(){ echo "[$(date -u +%FT%TZ)] hc: $*"; }
finish(){ local rc=$1; printf '%s\n' "$rc" > "TC1_EXIT_CODE.$NONCE"; [ "$rc" = 0 ] && : > "TC1_SUCCESS.$NONCE"; : > "TP_DONE.$NONCE"; exit "$rc"; }
: > summary.txt; mkdir -p logs receipts
E4B_SHA=${E4B_SHA:?}; GNF4_SHA=${GNF4_SHA:?}; DEADLINE=${TC1_DEADLINE_EPOCH:?}
MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
ARENA_DIR=/root/hc-arena; ARENA=$ARENA_DIR/qwen3.nf4; PORT=8778; URL=http://127.0.0.1:$PORT
INSTALL_S=600           # pip (two git installs, transformers, the HTTP server's deps) and the tripwire
NEED_FETCH=1500         # 57 GB at the probe's 40 MB/s floor is 1,425 s
NEED_BAKE=400           # SV4's bake_nf4 of this checkpoint: 209 s on a 4090
NEED_A=600; NEED_B=1200; NEED_C=900   # a server build, 10 x 32 greedy tokens, the stop; B and C decode eagerly off-GPU
RESERVE=600             # can_run's margin after a step: the fetch-back and the teardown
MIN_DL_MB_S=40
echo "EXPLORE (HC: serve_paged at the solver placement): not a reading" | tee -a summary.txt
for f in hc_client.py calib.json; do [ -s $W/$f ] || { echo "STAGE MISSING: $f" | tee -a summary.txt; finish 9; }; done

free_gb=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
ram_gb=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
[ "${free_gb:-0}" -ge 120 ] || { echo "FLOOR: $free_gb GB free < 120" | tee -a summary.txt; finish 13; }
[ "${mem_mib:-0}" -ge 30000 ] || { echo "FLOOR: $mem_mib MiB card < 30000" | tee -a summary.txt; finish 13; }
[ "${ram_gb:-0}" -ge 60 ] || { echo "FLOOR: $ram_gb GiB RAM < 60" | tee -a summary.txt; finish 13; }

# the download probe (P231's form): one sustained GET of the checkpoint's first shard from the endpoint the fetch uses,
# until 256 MiB AND 20 s, or 60 s. A host below MIN_DL_MB_S, a non-2xx status or an error is refused (19, HOST_REFUSED,
# never a strike). download_probe.json is written on EVERY box, refused or not.
rm -f $W/download_probe.json
DL=$(MIX=$MODEL MIX_REV=$REV FLOOR=$MIN_DL_MB_S OUT=$W/download_probe.json timeout 120 python - <<'PYD' 2>/dev/null | tail -1
import json, os, time, urllib.request
m, rev, tok = os.environ["MIX"], os.environ["MIX_REV"], ""
rec = {"model": m, "revision": rev, "floor_mb_s": int(os.environ["FLOOR"]), "status": None, "bytes": 0, "seconds": None,
       "mb_s": None, "shard": None, "error": None, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    tok = open("/root/.cache/huggingface/token").read().strip()
except OSError:
    pass
def get(url, t):
    req = urllib.request.Request(url)
    if tok:
        req.add_unredirected_header("Authorization", f"Bearer {tok}")
    return urllib.request.urlopen(req, timeout=t)
t0 = None
try:
    meta = json.load(get(f"https://huggingface.co/api/models/{m}/revision/{rev}", 30))
    rec["shard"] = sorted(s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".safetensors"))[0]
    t0 = time.time()
    with get(f"https://huggingface.co/{m}/resolve/{rev}/{rec['shard']}", 60) as r:
        rec["status"] = r.status
        while True:
            dt = time.time() - t0
            if (rec["bytes"] >= 256 << 20 and dt >= 20) or dt >= 60:
                break
            b = r.read(1 << 20)
            if not b:
                break
            rec["bytes"] += len(b)
except Exception as e:  # noqa: BLE001 -- a refusal reason, recorded by name
    rec["status"] = rec["status"] or getattr(e, "code", None)
    rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
finally:
    if t0 is not None:
        rec["seconds"] = round(time.time() - t0, 2)
        if rec["seconds"] and rec["error"] is None:
            rec["mb_s"] = round(rec["bytes"] / rec["seconds"] / 1e6, 1)
    json.dump(rec, open(os.environ["OUT"], "w"), indent=1)
print(f"status={rec['status']} bytes={rec['bytes']} seconds={rec['seconds']} mb_s={rec['mb_s'] if rec['mb_s'] is not None else 'none'}")
PYD
)
[ -s $W/download_probe.json ] || printf '{"status": null, "bytes": null, "mb_s": null, "error": "the probe wrote no record (killed at 120 s or did not run)"}\n' > $W/download_probe.json
echo "DOWNLOAD ${DL:-none}" | tee -a summary.txt
DL_MB_S=${DL##*mb_s=}; DL_MB_S=${DL_MB_S%%.*}
case "$DL" in status=2[0-9][0-9]\ *) ;; *) DL_MB_S=0;; esac
[ "${DL_MB_S:-0}" -ge "$MIN_DL_MB_S" ] 2>/dev/null || { say "REFUSED: download ${DL:-none} -- below ${MIN_DL_MB_S} MB/s"; echo "refused: download ${DL:-none}" > REFUSAL; finish 19; }
# ---- the pre-install fit check (bench/common/README.md, "Guard fit"): before any install. C is not in the fit: it runs
# only if can_run finds its time.
FIT_HEADROOM_PCT=125
FIT=$((INSTALL_S + NEED_FETCH + NEED_BAKE + NEED_A + NEED_B + RESERVE)); FIT_NEED=$((FIT * FIT_HEADROOM_PCT / 100)); LEFT=$((DEADLINE - $(date +%s)))
# rc 41: a refusal before the install, not a start; two consecutive headroom refusals count as one strike
[ "$LEFT" -ge "$FIT_NEED" ] || { say "REFUSED: needs ${FIT_NEED}s (fit ${FIT}s x ${FIT_HEADROOM_PCT}%), ${LEFT}s left"; echo "refused: fit ${FIT_NEED}s, ${LEFT}s left" > REFUSAL; finish 41; }
echo "GUARD_FIT fit=${FIT}s need=${FIT_NEED}s left=${LEFT}s" | tee -a summary.txt
can_run(){ local need=$1 now fit left; now=$(date +%s); fit=$((need + RESERVE)); left=$((DEADLINE - now)); [ "$fit" -le "$left" ] || { say "STOP-2: $2 needs ${fit}s (${need}s + the ${RESERVE} s reserve), only ${left}s left -- skipped (host-limited)"; echo "SKIPPED $2 needs ${fit}s, ${left}s left" >> summary.txt; return 1; }; }
step_alarm(){ local cap=$1 left=$(( DEADLINE - $(date +%s) - RESERVE )); [ "$left" -gt "$cap" ] && left=$cap; [ "$left" -lt 60 ] && left=60; echo "$left"; }

command -v git >/dev/null 2>&1 || perl -e 'alarm 600; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq git' > logs/apt_git.log 2>&1 \
  || { tail -3 logs/apt_git.log; echo "GIT INSTALL FAIL" | tee -a summary.txt; finish 9; }
say "install: gnf4 @$GNF4_SHA + e4b @$E4B_SHA, transformers 5.18.0, the HTTP server's deps"
perl -e "alarm $(step_alarm 2400); exec @ARGV" python -m pip install -q --no-input --prefer-binary \
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

can_run $NEED_FETCH fetch || finish 40
say "fetch $MODEL@${REV:0:8}"
fetch(){ perl -e "alarm $(step_alarm $NEED_FETCH); exec @ARGV" python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL', revision='$REV', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','*.jinja']))" 2>> logs/fetch.log | tail -1; }
snap=$(fetch); [ -d "$snap" ] || { echo "fetch retry" | tee -a summary.txt; snap=$(fetch); }
[ -d "$snap" ] || { tail -3 logs/fetch.log; echo "FETCH FAILED" | tee -a summary.txt; finish 22; }
echo "fetched $(du -shL "$snap" | cut -f1)" | tee -a summary.txt
can_run $NEED_BAKE bake || finish 40
say "bake the NF4 arena into $ARENA_DIR (outside the fetched tree)"
mkdir -p $ARENA_DIR
perl -e "alarm $(step_alarm $NEED_BAKE); exec @ARGV" python -c "from nvme_bake_nf4 import bake_nf4; bake_nf4('$snap', '$ARENA')" > logs/bake.log 2>&1
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
# arm TAG PLACEMENT GRAPHS VRAM_GB DRAM_GB HOT_ROWS NEED_S -> receipts/run_TAG.json, receipts/health_TAG.json; rc 0 or the failure
arm(){ local tag=$1 place=$2 graphs=$3 vram=$4 dram=$5 hot=$6 need=$7 log=logs/server_$1.log pid rc t0
  say "arm $tag: placement $place, graphs $graphs, tiers $vram/$dram GiB, hot_rows $hot"; t0=$(date +%s)
  setsid env PYTHONPATH= E4B_PAGED_MODEL=$MODEL E4B_PAGED_REVISION=$REV E4B_PAGED_ARENA=$ARENA E4B_PAGED_CALIB=$W/calib.json \
      E4B_PAGED_PLACEMENT=$place E4B_PAGED_GRAPHS=$graphs E4B_PAGED_VRAM_GB=$vram E4B_PAGED_DRAM_GB=$dram E4B_PAGED_HOT_ROWS=$hot \
      E4B_PAGED_MAX_SEQS=1 E4B_PAGED_MAX_TOKENS_PER_SEQ=1024 E4B_HOST=127.0.0.1 E4B_PORT=$PORT \
      python -m experts4bit_qlora.serve_paged > "$log" 2>&1 < /dev/null &
  pid=$!
  wait_ready $pid "$log" "$(step_alarm $((need * 2 / 3)))"; rc=$?
  curl -s -m 30 "$URL/health" -o receipts/health_$tag.json 2>/dev/null
  if [ $rc != 0 ]; then echo "arm $tag: BUILD FAILED (rc $rc) after $(( $(date +%s) - t0 )) s: $(tail -3 "$log" | tr '\n' ' ' | cut -c1-300)" | tee -a summary.txt
    stop_server $pid; return $rc; fi
  echo "arm $tag: ready in $(( $(date +%s) - t0 )) s; vram_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" | tee -a summary.txt
  perl -e "alarm $(step_alarm $need); exec @ARGV" python hc_client.py run --url $URL --model $MODEL --arm $tag --max-tokens 32 \
      --out receipts/run_$tag.json > logs/client_$tag.log 2>&1; rc=$?
  tail -1 logs/client_$tag.log | tee -a summary.txt
  stop_server $pid
  [ $rc = 0 ] || { echo "arm $tag: CLIENT FAILED rc $rc" | tee -a summary.txt; return 46; }
  return 0; }
tiers(){ python hc_client.py tiers --arm $1 --config "$snap/config.json" --arena $ARENA --calib $W/calib.json \
           --vram-gb $2 --dram-gb $3 --expect $4 --out receipts/tiers_$1.json 2>&1 | tail -1 | tee -a summary.txt
         return "${PIPESTATUS[0]}"; }
cmp_arm(){ python hc_client.py compare --base receipts/run_A.json --other receipts/run_$1.json --tokenizer "$snap" \
             --out receipts/compare_$1.json 2>&1 | tail -1 | tee -a summary.txt; return "${PIPESTATUS[0]}"; }

can_run $NEED_A "arm A" || finish 40
arm A all-vram 0 8.0 0 128 $NEED_A || { echo "STOP: the reference arm A did not build or answer" | tee -a summary.txt; finish 31; }
tiers B 8.0 3.0 three || { echo "STOP: arm B's premise failed: the solver's split at 8.0/3.0 GiB is not three non-empty tiers" | tee -a summary.txt; finish 36; }
can_run $NEED_B "arm B" || finish 40
arm B solver auto 8.0 3.0 128 $NEED_B || { echo "STOP: arm B (solver, three tiers) did not build or answer" | tee -a summary.txt; finish 32; }
cmp_arm B || { echo "STOP: B against A could not be compared" | tee -a summary.txt; finish 33; }
rc=0
if ! tiers C 8.0 0 nvme-only; then echo "arm C skipped: its premise failed (the split at 8.0/0 GiB is not VRAM + NVMe only)" | tee -a summary.txt; rc=34
elif ! can_run $NEED_C "arm C"; then rc=34
elif ! { arm C solver auto 8.0 0 128 $NEED_C && cmp_arm C; }; then echo "arm C (the control) did not complete" | tee -a summary.txt; rc=34
fi
echo "HC done: arms $(ls receipts/run_*.json 2>/dev/null | wc -l | tr -d ' '), EXPLORE, not a reading" | tee -a summary.txt
finish $rc
