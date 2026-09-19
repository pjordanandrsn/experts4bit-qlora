#!/usr/bin/env bash
# bench/p43/g4/p43_g4_run.sh -- BOX SIDE of P43 T2 (bench/p43/P43-PREREG.md): the e4b#558 LAYER-1 adjudication
# with the bf16 oracle RESIDENT on an 80 GB-class card. Lineage: the mini's untracked probe_run4.sh (tp4-g4layer1-*),
# committed here so a manifest can pin it. Same venv recipe as the tp4 e4b arm; pinned checkpoint; then the probe.
#
# Two pre-flights the earlier attempts lacked, each a REFUSAL ROW rather than a stall: (1) HF CDN egress >= P43_MIN_MBPS
# (two attempts died 90 min inside a download or a model load on hosts the bandwidth gate could not see); (2) the card
# holds >= P43_MIN_VRAM_GB (the oracle is ~52 GB bf16; a 32 GB card would CPU-offload it and stream the model per chunk).
set -uo pipefail
W=/root/probe; cd "$W" || exit 20
mkdir -p logs
say(){ echo "[$(date -u +%FT%TZ)] p43-g4: $*" | tee -a "$W/outer.log"; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
: "${E4B_SHA:?}"; : "${GNF4_SHA:?}"; : "${MID:?}"; : "${REV:?}"
TF_VER=${P43_TF_VER:-5.17.0}; BNB_VER=${P43_BNB_VER:-0.50.2}; PEFT_VER=${P43_PEFT_VER:-0.20.0}
MIN_MBPS=${P43_MIN_MBPS:-20}; MIN_VRAM_GB=${P43_MIN_VRAM_GB:-80}; ROWS=${P43_ROWS:-128}; CHUNK=${P43_CHUNK:-8}
PY=$W/venv-e4b/bin/python
mark(){ echo "$1" > "$W/PROBE_EXIT_CODE"; [ "$1" = 0 ] && touch "$W/PROBE_SUCCESS"; touch "$W/PROBE_DONE"; exit "$1"; }

say "host: $(grep MemTotal /proc/meminfo) | $(df -h /root | tail -1 | awk '{print $4" free"}')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$W/outer.log" > forensics.txt
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${VRAM_MB:-0}" -lt $(( MIN_VRAM_GB * 1000 )) ]; then say "REFUSED: card holds ${VRAM_MB} MiB < ${MIN_VRAM_GB} GB -- not the registered class"; echo "refused: vram class" > REFUSAL; mark 15; fi

say "egress pre-flight: HF CDN, 50 MB range, 20 s cap (floor ${MIN_MBPS} MB/s)"
BPS=$(curl -sSL --max-time 20 -r 0-52428800 -o /dev/null -w '%{speed_download}' https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors 2>/dev/null || echo 0)
MBPS=$(python3 -c "print(round(float('${BPS:-0}')/1e6,1))"); say "HF CDN ${MBPS} MB/s"; echo "hf_cdn_mbps=$MBPS" >> forensics.txt
if python3 -c "import sys; sys.exit(0 if float('${MBPS:-0}') < float('$MIN_MBPS') else 1)"; then say "REFUSED: egress ${MBPS} MB/s < ${MIN_MBPS} -- host-limited, not a probe result"; echo "refused: egress $MBPS MB/s" > REFUSAL; mark 14; fi

say "venv-e4b: e4b @$E4B_SHA + gnf4 @$GNF4_SHA + transformers==$TF_VER"
python -m venv --system-site-packages "$W/venv-e4b" || { say "VENV FAIL"; mark 9; }
perl -e 'alarm 2400; exec @ARGV' "$PY" -m pip install -q --no-input --prefer-binary \
  "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" \
  "transformers==$TF_VER" "bitsandbytes==$BNB_VER" "peft==$PEFT_VER" accelerate safetensors \
  "huggingface_hub>=0.23" sentencepiece tiktoken > logs/pip.log 2>&1 || { say "PIP FAIL"; tail -20 logs/pip.log; mark 10; }
"$PY" -c "import torch,transformers,accelerate,experts4bit_qlora as e; assert torch.cuda.is_available(); print('e4b',e.__version__,'tf',transformers.__version__,'accel',accelerate.__version__)" | tee -a "$W/outer.log" versions.txt
"$PY" -c "import experts4bit_qlora.loader as l; assert hasattr(l, 'UnplaceableTensorError'), 'e4b tree predates 0.36.1'" || { say "TRIPWIRE FAIL (e4b)"; mark 10; }

say "tokens pre-flight"
"$PY" - "$W/tokens_gemma4.json" <<'PYF' | tee -a "$W/outer.log" || { say "TOKENS PREFLIGHT FAIL"; mark 13; }
import hashlib, json, sys
b = open(sys.argv[1], "rb").read(); t = json.loads(b)
assert isinstance(t.get("train"), list) and len(t["train"]) >= 128, "tokens_gemma4.json has < 128 train rows"
print("tokens preflight OK: %d train rows, pad_id %s, file sha256 %s, inner sha %s" % (len(t["train"]), t.get("pad_id"), hashlib.sha256(b).hexdigest(), t.get("sha256")))
PYF

say "fetch $MID (pin $REV)"
perl -e 'alarm 5400; exec @ARGV' "$PY" - "$MID" <<'PYF' > logs/fetch.log 2>&1
import os, sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt", "merges.txt", "vocab.json", "*.tiktoken", "*.jinja"], max_workers=4)
print(f"staged in {(time.time()-t0)/60:.1f} min at {p}")
print("STAGED", os.path.basename(os.path.realpath(p)))
PYF
rc=$?; tail -2 logs/fetch.log | head -1 | tee -a "$W/outer.log"
[ $rc -ne 0 ] && { say "FETCH FAIL rc=$rc"; tail -5 logs/fetch.log; mark 11; }
GOT=$(grep -a "^STAGED " logs/fetch.log | tail -1 | awk '{print $2}')
RDIR=/root/.cache/huggingface/hub/models--${MID//\//--}
[ "$GOT" = "$REV" ] || { say "PIN MISMATCH staged=$GOT != pin=$REV -- ABORTING, not coerced"; mark 12; }
mkdir -p "$RDIR/refs" && printf '%s' "$REV" > "$RDIR/refs/main"
say "PIN OK staged=$GOT == pin"

PROBE=${P43_G4_PROBE:-layer1}            # layer1 (T2) | layer_sweep (T2b, amendment 1)
case "$PROBE" in
  layer1)      SCRIPT=gemma4_layer1_probe.py; OUT=gemma4_layer1_probe.json; EXTRA="";;
  layer_sweep) SCRIPT=gemma4_layer_sweep.py; OUT=gemma4_layer_sweep.json; EXTRA="--noise-floor ${P43_NOISE_FLOOR:-1e-4}";;
  *) say "REFUSED: unknown probe '$PROBE'"; echo "refused: probe $PROBE" > REFUSAL; mark 16;;
esac
[ -s "$W/$SCRIPT" ] || { say "REFUSED: $SCRIPT was not staged"; echo "refused: probe script missing" > REFUSAL; mark 16; }
say "running probe $PROBE ($ROWS rows, chunk $CHUNK, real positions only; oracle resident)"
( while :; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits)"; sleep 2; done ) > "$W/vram.txt" 2>/dev/null & SP=$!
HF_HUB_OFFLINE=1 perl -e 'alarm 3600; exec @ARGV' "$PY" -u "$W/$SCRIPT" \
  --model "$MID" --revision "$REV" --tokens "$W/tokens_gemma4.json" --rows "$ROWS" --chunk "$CHUNK" $EXTRA \
  --out "$W/$OUT" 2>&1 | tee "$W/logs/probe.log"
prc=${PIPESTATUS[0]}; kill "$SP" 2>/dev/null
say "probe exit rc=$prc"
mark "$prc"
