#!/bin/bash
# bench/sc2/sc1g-r/sc1g_r_run.sh -- lane SC1g amendment A4 (#846), box R, BOX side. Started detached by sc1g_r_drive.sh with
# the run's nonce; SC1G_R_RUN_NONCE first, then SC1G_R_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# SC1G_R_SUCCESS.<nonce> only when R's own verdict is R_OK.
#
# One >= 80 GB card (H100 NVL, declared $3.50/h, maintainer-approved 2026-10-05 per tc1c's precedent). K0 controls on THIS
# host first (kl_fidelity --controls), then the pinned gpt-oss-20b fetch, then sc1g_ref.py: the bf16-dequant reference
# scored decode-shaped over the five registered windows (artifacts ref_<src>.npz), the two calibration pairs (decode vs
# prefill; reference vs its NF4 fake-quant), and R's verdict (r_verdict.json, OK / UNREAD / VOID per check). Patterned on
# bench/p44/p44b_run.sh (P44-b ran the same reference on an H100 NVL). Nothing here creates, destroys or approves compute.
set -uo pipefail
W=/root/sc1g_r; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] sc1g_r: $*"; }
NONCE=${SC1G_R_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/SC1G_R_RUN_NONCE.tmp && mv $W/SC1G_R_RUN_NONCE.tmp $W/SC1G_R_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > SC1G_R_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > SC1G_R_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in SC1G_R_RUN_ID SC1G_R_DEADLINE_EPOCH SC1G_R_INSTANCE_ID GNF4_SHA MODEL_REV; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in GNF4_SHA MODEL_REV; do case "${!v}" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac; done
[ ${#GNF4_SHA} -eq 40 ] && [ ${#MODEL_REV} -eq 40 ] || { say "refusing: a pin is not a 40-char sha"; finish 78; }
# the controller fetches the receipts with rsync; the pytorch devel image ships none (tc1c-h100-19 lost its receipts that way)
command -v rsync > /dev/null || perl -e 'alarm 300; exec @ARGV' sh -c 'apt-get update -qq && apt-get install -y -qq rsync' > $W/logs/apt_rsync.log 2>&1
echo "rsync $(command -v rsync > /dev/null && echo present || echo MISSING -- the driver falls back to scp)" > $W/rsync.txt
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
MIN_MBPS=${SC1G_R_MIN_MBPS:-20}; MIN_VRAM_GB=${SC1G_R_MIN_VRAM_GB:-80}; MIN_DISK_GB=${SC1G_R_MIN_DISK_GB:-80}
: > summary.txt; echo "$SC1G_R_INSTANCE_ID" > INSTANCE_ID
for f in sc1g_r_run.sh sc1g_ref.py sc1g_kl.py kl_fidelity.py window_shas.json staged-r.sha256 \
         windows/k8_window_conv1.json windows/k8_window_conv2.json windows/k8_window_conv3.json windows/k8_window_conv4.json windows/k8_window_wikitext.json; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged-r.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/sc2/sc1g-r/staged-r.sha256"; finish 9; }
python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h /root | tail -1 | tee -a forensics.txt
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
FREE_GB=$(df -BG --output=avail /root 2>/dev/null | tail -1 | tr -dc 0-9)
if [ "${FREE_GB:-0}" -lt "$MIN_DISK_GB" ]; then say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (host-limited, not a result)"; echo "refused: disk" > REFUSAL; finish 13; fi
if [ "${VRAM_MB:-0}" -lt $(( MIN_VRAM_GB * 1000 )) ]; then say "REFUSED: card holds ${VRAM_MB} MiB < ${MIN_VRAM_GB} GB -- not the registered class"; echo "refused: vram class" > REFUSAL; finish 15; fi
say "egress pre-flight: HF CDN, 50 MB range, 20 s cap (floor ${MIN_MBPS} MB/s)"
# Python, not curl: sc1g-r-6's curl probe read 0.0 MB/s on a RunPod H100 NVL with its stderr discarded, so whether curl was
# missing or the host could not reach the CDN is unknown. Same URL, 50 MB range, 20 s cap and floor; status, final host and
# any error go to logs/egress.log, and curl's presence to forensics.txt. Two outcomes, never conflated: a probe that RAISES
# before reading any byte (HTTP 403/429, TLS, DNS, an import) prints ERROR and exits rc 9 (harness/host fault, reason logged);
# rc 14 means only "measured slow": bytes were read, under the floor (a timeout mid-read included). The SC1G_R_EGRESS_* overrides
# exist for the tests; the box never sets them.
echo "curl $(command -v curl || echo MISSING)" >> forensics.txt
MBPS=$(SC1G_R_EGRESS_URL=${SC1G_R_EGRESS_URL:-https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors} \
       SC1G_R_EGRESS_TIMEOUT=${SC1G_R_EGRESS_TIMEOUT:-20} SC1G_R_EGRESS_CAP_S=${SC1G_R_EGRESS_CAP_S:-20} python3 - 2>> logs/egress.log <<'PYE'
import os, sys, time, urllib.request
url, cap = os.environ["SC1G_R_EGRESS_URL"], float(os.environ["SC1G_R_EGRESS_CAP_S"])
req = urllib.request.Request(url, headers={"Range": "bytes=0-52428800", "User-Agent": "sc1g-r-egress"})
t0, n, err = time.time(), 0, None
try:
    with urllib.request.urlopen(req, timeout=float(os.environ["SC1G_R_EGRESS_TIMEOUT"])) as r:
        print(f"egress status {r.status} final {r.geturl()[:100]}", file=sys.stderr)
        while time.time() - t0 < cap:
            b = r.read(1 << 20)
            if not b:
                break
            n += len(b)
except Exception as e:
    err = e
    print(f"egress probe error after {n} bytes: {e!r}"[:400], file=sys.stderr)
print("ERROR" if err is not None and n == 0 else round(n / max(time.time() - t0, 1e-3) / 1e6, 1))
PYE
)
MBPS=${MBPS:-ERROR}; WHY=$(tail -1 logs/egress.log 2>/dev/null | cut -c1-200)
if [ "$MBPS" = ERROR ]; then
  say "REFUSED: the egress probe failed before reading any byte (harness/host fault, not a slow host): $WHY"
  echo "hf_cdn_mbps=ERROR" >> forensics.txt; echo "refused: egress probe error: $WHY" > REFUSAL; finish 9; fi
say "HF CDN ${MBPS} MB/s ($WHY)"; echo "hf_cdn_mbps=$MBPS" >> forensics.txt
if python3 -c "import sys; sys.exit(0 if float('$MBPS') < float('$MIN_MBPS') else 1)"; then
  say "REFUSED: egress ${MBPS} MB/s < ${MIN_MBPS} (host-limited): $WHY"
  echo "refused: egress $MBPS MB/s" > REFUSAL; finish 14; fi
say "install transformers 5.16.1 (+ the reference's deps) and gnf4 @$GNF4_SHA (the NF4 cross-check only)"
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input --prefer-binary "transformers==5.16.1" accelerate safetensors \
  "huggingface_hub>=0.23" numpy > logs/pip.log 2>&1 || { tail -4 logs/pip.log; say "PIP FAIL"; finish 9; }
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import os, importlib.metadata as md
import torch, transformers
from transformers import Mxfp4Config  # noqa: F401  (the gpt-oss dequant reference)
from nf4_grouped import dequant_ref  # noqa: F401
from nf4_pack_ref import quantize_pack_nf4  # noqa: F401
assert transformers.__version__ == "5.16.1", transformers.__version__
open("/root/sc1g_r/versions.txt", "w").write(f"gnf4 {md.version('grouped-nf4-gemm')} @{os.environ['GNF4_SHA']}\ntorch {torch.__version__} cuda {torch.version.cuda}\n"
                                             f"transformers {transformers.__version__}\ngpt-oss-20b @{os.environ['MODEL_REV']}\n")
print("tripwire OK: torch", torch.__version__, "transformers", transformers.__version__, "gnf4", md.version("grouped-nf4-gemm"))
PYT
cat versions.txt | tee -a summary.txt
# ---- K0 controls on THIS host: the instrument's own gate (P44's rule: no KL row without them)
perl -e 'alarm 600; exec @ARGV' python $W/kl_fidelity.py --controls --out $W/k0.json > logs/k0.log 2>&1 || { tail -5 logs/k0.log; say "K0 CONTROLS FAILED"; echo "K0 FAILED" >> summary.txt; finish 16; }
python -c "import json; r=json.load(open('$W/k0.json')); assert r['all_passed']; print('K0 all_passed')" | tee -a summary.txt || { say "K0 receipt not passing"; finish 16; }
arm_alarm(){ local l=$(( SC1G_R_DEADLINE_EPOCH - $(date +%s) - 300 )); [ "$l" -lt 600 ] && l=600; echo "$l"; }
say "fetch openai/gpt-oss-20b @ $MODEL_REV"
perl -e "alarm $(arm_alarm); exec @ARGV" python - "openai/gpt-oss-20b" "$MODEL_REV" <<'PYF' > logs/fetch.log 2>&1 || { tail -3 logs/fetch.log; say "FETCH FAIL"; finish 11; }
import os, sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.jinja"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}"); print("STAGED", os.path.basename(os.path.realpath(p)))
PYF
GOT=$(grep -a "^STAGED " logs/fetch.log | tail -1 | awk '{print $2}'); [ "$GOT" = "$MODEL_REV" ] || { say "PIN MISMATCH staged=$GOT"; finish 12; }
say "PIN OK $GOT"; echo "FETCH gpt-oss-20b @$GOT" >> summary.txt
say "sc1g_ref.py: reference (dequant-to-bf16), five windows decode-shaped, the two calibration pairs, R's verdict"
perl -e "alarm $(arm_alarm); exec @ARGV" python -u $W/sc1g_ref.py --model openai/gpt-oss-20b --rev "$MODEL_REV" --windows $W/windows \
    --shas $W/window_shas.json --k0 $W/k0.json --out $W/ref > logs/sc1g_ref.log 2>&1; rc=$?
grep -aE "^(SC1G_R|R )" logs/sc1g_ref.log | tee -a summary.txt
[ -s $W/ref/r_verdict.json ] || { tail -5 logs/sc1g_ref.log; say "NO VERDICT (rc=$rc)"; finish 30; }
(cd $W/ref && sha256sum ref_*.npz) | tee $W/ref/SHA256SUMS | sed "s/^/ARTIFACT /" | tee -a summary.txt
say "----- summary -----"; cat summary.txt
[ $rc = 0 ] && grep -q '"verdict": "R_OK"' $W/ref/r_verdict.json && finish 0
finish 31
