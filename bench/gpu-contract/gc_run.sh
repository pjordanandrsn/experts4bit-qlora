#!/bin/bash
# bench/gpu-contract/gc_run.sh -- the GPU contract, BOX side (bench/gpu-contract/README.md). It runs a named grouped-nf4-gemm
# commit's own GPU kernel tests and experts4bit-qlora's served-route premise tests on ONE card of a declared class, at two
# pinned commits, and records pass counts and every skipped test. It is a correctness contract for a kernel PR's review, not
# a registered reading: nothing here times anything.
#   order: nonce; refusals (CUDA 18, torch import 10, torch not the image's pin 28, card class 15, disk 13); installs at the
#          pins and a tripwire; clean source checkouts at the pins; gnf4's kernel tests; e4b's served tests; summary.
#   exit:  0 both suites passed with no failure or error; 30 a suite failed or errored; 9 an install, checkout or tripwire;
#          78 a refused input.
set -uo pipefail
W=/root/gc; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] gc: $*"; }
NONCE=${GC_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/GC_RUN_NONCE.tmp && mv $W/GC_RUN_NONCE.tmp $W/GC_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > GC_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > GC_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in GC_RUN_ID GC_DEADLINE_EPOCH GC_GNF4_SHA GC_E4B_SHA GC_CARD; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in GC_GNF4_SHA GC_E4B_SHA; do
  val=${!v}; case "$val" in *[!0-9a-f]*|"") say "refusing: $v is not a full lowercase commit sha"; finish 78;; esac
  [ ${#val} -eq 40 ] || { say "refusing: $v is not 40 characters (an unpinned ref)"; finish 78; }
done
TORCH_WANT=2.8.0+cu128
# the two suites: gnf4's kernel contract for the bandwidth GEMV and its dispatch tally (run from the clone's kernel/), and
# experts4bit-qlora's served-route premise (run from its clone's root) -- P133's two premise suites, plus gnf4's dispatch file
GNF4_TESTS="test_nf4_gemv_bw.py test_dispatch_counts.py"
E4B_TESTS="tests/test_decode_graph_buckets.py tests/test_kv_step_select.py tests/test_gemv_bw_served_gpu.py"
MIN_DISK_GB=${GC_MIN_DISK_GB:-30}
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false TRITON_INTERPRET=0
{ echo "GC run $GC_RUN_ID nonce ${NONCE:0:12}"; echo "gnf4 $GC_GNF4_SHA"; echo "e4b $GC_E4B_SHA"; echo "card class $GC_CARD"; echo "required ${GC_REQUIRE:-<none>}"; } | tee summary.txt
CUDA_PROBE=$(python - <<'PYC' 2>/dev/null
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
  no-cuda*) say "REFUSED: torch cannot use the GPU (${CUDA_PROBE})"; echo "refused: cuda unusable" > REFUSAL; finish 18;;
  *) say "DUD BOX (${CUDA_PROBE:-no probe output})"; finish 10;;
esac
case "$CUDA_PROBE" in *" torch=$TORCH_WANT "*) ;; *) say "REFUSED: torch is not the image's $TORCH_WANT"; echo "refused: torch" > REFUSAL; finish 28;; esac
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | tee forensics.txt
python $W/gc_box.py --card "$CUDA_PROBE" --declared "$GC_CARD" | tee -a summary.txt; crc=${PIPESTATUS[0]}
[ "$crc" = 0 ] || { echo "refused: card class" > REFUSAL; finish "$crc"; }
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB"; echo "refused: disk" > REFUSAL; finish 13; }
python $W/gc_box.py --self-test | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || finish 9
echo "torch==${TORCH_WANT%%+*}" > $W/constraints.txt
pipx(){ local log=$1 secs=$2; shift 2
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" > $log 2>&1 && return 0
  say "pip failed ($(tail -1 $log | cut -c1-120)) -- one retry in 20 s"; sleep 20
  perl -e "alarm $secs; exec @ARGV" python -m pip install -q --no-input -c $W/constraints.txt "$@" >> $log 2>&1; }
say "install e4b @$GC_E4B_SHA, then gnf4 @$GC_GNF4_SHA over it (torch held at $TORCH_WANT)"
pipx logs/pip_e4b.log 1800 --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$GC_E4B_SHA" \
  "transformers==5.17.0" "bitsandbytes==0.50.2" datasets accelerate sentencepiece safetensors "huggingface_hub>=0.23" pytest numpy || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
pipx logs/pip_gnf4.log 900 --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GC_GNF4_SHA" || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
# clean checkouts at the pins: a depth-1 fetch of the commit itself, refused unless HEAD is the pin and the tree is clean
checkout(){ local url=$1 sha=$2 dir=$3
  rm -rf "$dir" && mkdir -p "$dir" && git -C "$dir" init -q && perl -e 'alarm 600; exec @ARGV' git -C "$dir" fetch -q --depth 1 "$url" "$sha" \
    && git -C "$dir" checkout -q --detach FETCH_HEAD || return 1
  [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] && [ -z "$(git -C "$dir" status --porcelain)" ]; }
checkout https://github.com/pjordanandrsn/grouped-nf4-gemm.git "$GC_GNF4_SHA" $W/gnf4src || { say "CHECKOUT FAIL (gnf4): not the pin, or dirty"; finish 9; }
checkout https://github.com/pjordanandrsn/experts4bit-qlora.git "$GC_E4B_SHA" $W/e4bsrc || { say "CHECKOUT FAIL (e4b): not the pin, or dirty"; finish 9; }
WANT_E4B=$GC_E4B_SHA WANT_GNF4=$GC_GNF4_SHA WANT_TORCH=$TORCH_WANT GNF4_SRC=$W/gnf4src python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import hashlib, importlib.metadata as md, json, os, pathlib
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
assert d.get("vcs_info", {}).get("commit_id") == os.environ["WANT_E4B"], f"installed e4b is not the pin: {d}"
dg = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}")
assert dg.get("vcs_info", {}).get("commit_id") == os.environ["WANT_GNF4"], f"installed gnf4 is not the pin: {dg}"
import torch, triton, transformers
assert torch.__version__ == os.environ["WANT_TORCH"], torch.__version__
import nf4_grouped
inst = hashlib.sha256(pathlib.Path(nf4_grouped.__file__).read_bytes()).hexdigest()
src = hashlib.sha256((pathlib.Path(os.environ["GNF4_SRC"]) / "kernel" / "nf4_grouped.py").read_bytes()).hexdigest()
assert inst == src, f"the installed nf4_grouped.py ({inst[:12]}) is not the checkout's ({src[:12]})"
import experts4bit_qlora as e
open("/root/gc/versions.txt", "w").write(
    f"e4b {e.__version__} @{os.environ['WANT_E4B']}\ngnf4 {md.version('grouped-nf4-gemm')} @{os.environ['WANT_GNF4']}\n"
    f"nf4_grouped.py sha256 {inst}\ntorch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\n"
    f"bitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK")
PYT
cat versions.txt | tee -a summary.txt
# ---- the time left (P133's can_run): a suite starts only with its need plus a receipt margin before the deadline, and its alarm
# never runs past that margin, so a tight guard still leaves time to fetch the receipts. A skipped suite fails the contract.
MARGIN_S=300
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + MARGIN_S)) -le "$GC_DEADLINE_EPOCH" ] && return 0
  say "STOP: $2 needs ${need}s, only $((GC_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" | tee -a summary.txt; return 1; }
alarm(){ local left=$(( GC_DEADLINE_EPOCH - $(date +%s) - MARGIN_S )); [ "$left" -gt 1800 ] && left=1800; [ "$left" -lt 60 ] && left=60; echo "$left"; }
# ---- the two suites, each with -rs (every skip listed) and a junit report; neither stops the other
if can_run "${GC_NEED_GNF4_S:-900}" "gnf4 kernel tests"; then
  (cd $W/gnf4src/kernel && PYTHONPATH= perl -e "alarm $(alarm); exec @ARGV" python -m pytest $GNF4_TESTS -q -rs -p no:cacheprovider --junitxml=$W/logs/gnf4.xml) > logs/gnf4_tests.log 2>&1
  echo "gnf4 kernel tests rc=$?: $(tail -1 logs/gnf4_tests.log)" | tee -a summary.txt
  grep -E '^SKIPPED' logs/gnf4_tests.log | tee -a summary.txt
fi
if can_run "${GC_NEED_E4B_S:-600}" "e4b served tests"; then
  (cd $W/e4bsrc && PYTHONPATH= perl -e "alarm $(alarm); exec @ARGV" python -m pytest $E4B_TESTS -q -rs -p no:cacheprovider --junitxml=$W/logs/e4b.xml) > logs/e4b_tests.log 2>&1
  echo "e4b served tests rc=$?: $(tail -1 logs/e4b_tests.log)" | tee -a summary.txt
  grep -E '^SKIPPED' logs/e4b_tests.log | tee -a summary.txt
fi
rc=0
for s in gnf4 e4b; do
  [ -s logs/$s.xml ] || { echo "GC_SUITE {\"suite\": \"$s\", \"ok\": false, \"why\": \"no junit report\"}" | tee -a summary.txt; rc=30; continue; }
  python $W/gc_box.py --junit logs/$s.xml --suite $s | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || rc=30
done
# GC_REQUIRE: the tests the review names as the PR's own evidence must PASS, not merely not fail (a skip or an absence fails)
if [ -n "${GC_REQUIRE:-}" ]; then
  python $W/gc_box.py --require "$GC_REQUIRE" --junits logs/gnf4.xml logs/e4b.xml | tee -a summary.txt; [ "${PIPESTATUS[0]}" = 0 ] || rc=30
fi
[ $rc = 0 ] && echo "GC CONTRACT PASSED" | tee -a summary.txt || echo "GC CONTRACT FAILED" | tee -a summary.txt
finish $rc
