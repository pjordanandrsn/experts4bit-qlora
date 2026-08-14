#!/bin/bash
# The PREREG's 284B run. Detached on the mini so teardown survives the session.
#
# Differences from the parity autopilot, all forced by this run holding something
# expensive:
#   * 3 h cap, matching the prior V4 experiment's hardcap.
#   * the payload runs DETACHED on the pod and this polls, so receipts are pulled
#     CONTINUOUSLY -- a deadline kill then loses minutes, not the run.
#   * HOLD is honoured: a failure after the 160 GB download / 147 GB bake keeps
#     the pod for inspection and lets the wall-clock backstop be the bill cap.
#     Cheap-to-recreate tears down; expensive-to-recreate holds.
set -u
. "$HOME/.runpod/secrets.env"
REST="https://rest.runpod.io/v1"
GQL="https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
PUB=$(cat "$HOME/.runpod/id_ed25519.pub")
KEY="$HOME/.runpod/id_ed25519"
SRC="$HOME/v4run"
OUTD="$HOME/v4-out"
CAP_MIN=180
DEADLINE=$(( $(date +%s) + CAP_MIN * 60 ))
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

say() { echo "[$(date -u +%FT%TZ)] $*"; }
PODID=""; STATUS="NOT_STARTED"; NOTE=""; COST="?"; HELD=0
rm -rf "$OUTD"; mkdir -p "$OUTD"

create() {
  local body resp pid
  body=$(jq -nc --arg pk "$PUB" --arg gpu "$1" '{
    name:"v4-mxfp4-arena-train",
    imageName:"runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204",
    cloudType:"SECURE", gpuTypeIds:[$gpu], gpuCount:1, interruptible:false,
    supportPublicIp:true, ports:["22/tcp"],
    containerDiskInGb:420,
    env:{PUBLIC_KEY:$pk}}')
  resp=$(curl -s -m 90 -X POST "$REST/pods" -H "Authorization: Bearer $RUNPOD_API_KEY" \
         -H "Content-Type: application/json" -d "$body")
  pid=$(printf '%s' "$resp" | jq -r '.id // empty')
  if [ -n "$pid" ]; then PODID="$pid"; say "secured $1 -> $pid"; return 0; fi
  say "  $1 unavailable: $(printf '%s' "$resp" | jq -rc '.error // .' | head -c 160)"
  return 1
}

teardown() {
  [ -n "$PODID" ] || return 0
  say "teardown $PODID"
  curl -s -m 40 -X DELETE -o /dev/null -w '  delete http %{http_code}\n' \
    -H "Authorization: Bearer $RUNPOD_API_KEY" "$REST/pods/$PODID"
  sleep 8
  local code
  code=$(curl -s -m 25 -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $RUNPOD_API_KEY" "$REST/pods/$PODID")
  say "  verify GET /pods/$PODID -> $code (404 = gone)"
  [ "$code" = "404" ] || NOTE="$NOTE [MANUAL-ACTION: pod $PODID still visible ($code)]"
}

pull() {   # pull receipts; tolerate misses, never abort the loop
  scp $SSHO -i "$KEY" -P "$PORT" -r "root@$IP:/root/v4out/." "$OUTD/" > /dev/null 2>&1
  scp $SSHO -i "$KEY" -P "$PORT" "root@$IP:/root/v4/run.log" "$OUTD/run.log" > /dev/null 2>&1
}

# 24 GB shapes FIRST -- the claim is "one 24 GB card". The 48 GB fallbacks are
# still a fair test because v4_run.py caps the process to 24 GiB on a bigger card.
SHAPES=("NVIDIA RTX A5000" "NVIDIA GeForce RTX 3090" "NVIDIA A40" "NVIDIA L40S")
NSHAPES=${#SHAPES[@]}; SI=0; IP=""; PORT=""

while [ -z "$IP" ] && [ "$SI" -lt "$NSHAPES" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
  PODID=""
  while [ -z "$PODID" ] && [ "$SI" -lt "$NSHAPES" ]; do
    create "${SHAPES[$SI]}" || SI=$((SI + 1))
  done
  [ -n "$PODID" ] || { STATUS="NO_STOCK"; NOTE="every laddered secure shape out of stock"; break; }

  nohup bash "$HOME/v4-backstop.sh" "$PODID" "$DEADLINE" \
    >> "$HOME/v4-backstop.log" 2>&1 </dev/null & disown
  say "backstop armed on $PODID until $(date -u -r $DEADLINE +%FT%TZ 2>/dev/null || echo $DEADLINE)"
  echo "$PODID" > "$HOME/v4-pod.id"

  COST=$(curl -s -m 25 -H "Authorization: Bearer $RUNPOD_API_KEY" "$REST/pods/$PODID" \
         | jq -r '.costPerHr // "?"')
  say "billed rate (read back, not the listing): \$$COST/hr"

  for i in $(seq 1 40); do
    sleep 15
    R=$(curl -s -m 30 "$GQL" -H "Content-Type: application/json" \
        -d '{"query":"query{pod(input:{podId:\"'"$PODID"'\"}){runtime{uptimeInSeconds ports{ip publicPort privatePort isIpPublic}}}}"}')
    UP=$(printf '%s' "$R" | jq -r '.data.pod.runtime.uptimeInSeconds // 0')
    IP=$(printf '%s' "$R" | jq -r '[.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true)][0].ip // empty')
    PORT=$(printf '%s' "$R" | jq -r '[.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true)][0].publicPort // empty')
    [ -n "$IP" ] && [ -n "$PORT" ] && { say "endpoint $IP:$PORT after $((i*15))s (uptime ${UP}s)"; break; }
    # WEDGE THRESHOLD, measured 2026-08-14 rather than assumed. An A/B/C probe on
    # A40 had two healthy pods first report a running container at ~167-178 s
    # (uptime 2 s and 13 s on the 180 s poll). The old threshold was 120 s, which
    # sat in the middle of that distribution and converted slow starters into
    # "wedges" this launcher then tore down -- so the shape-correlated wedge rate
    # recorded earlier is substantially an artifact of THIS check, not a provider
    # property. Real wedges exist (the probe's B arm sat at 0 for the full 180 s),
    # so the check stays; it just has to be well past the observed start latency.
    if [ "$i" -ge 20 ] && [ "$UP" = "0" ]; then
      say "WEDGE: uptime 0 at $((i*15))s on ${SHAPES[$SI]}"; STATUS="WEDGED"; break
    fi
  done

  if [ -z "$IP" ] || [ -z "$PORT" ]; then
    [ "$STATUS" = "WEDGED" ] || STATUS="NO_ENDPOINT"
    teardown; PODID=""; SI=$((SI + 1))
    say "advancing to next shape ($((NSHAPES - SI)) left)"
  fi
done

if [ -n "$PODID" ] && [ -n "$IP" ]; then
  for i in $(seq 1 24); do
    ssh $SSHO -i "$KEY" -p "$PORT" root@"$IP" true 2>/dev/null && break
    sleep 10
  done
  say "=== ship payload ==="
  ssh -n $SSHO -i "$KEY" -p "$PORT" root@"$IP" "mkdir -p /root/v4 /root/v4out" > /dev/null 2>&1
  scp $SSHO -i "$KEY" -P "$PORT" "$SRC/v4-payload.sh" "$SRC/v4_run.py" \
      "$SRC/e4b_overlay.tgz" root@"$IP":/root/v4/ > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    STATUS="SCP_FAILED"; NOTE="could not ship payload"
    teardown
  else
    # DETACHED, so receipts can be pulled while it runs (a deadline kill then
    # loses minutes, not the whole run).
    say "=== launch payload (detached) ==="
    ssh -n $SSHO -i "$KEY" -p "$PORT" root@"$IP" \
      "cd /root/v4 && nohup bash v4-payload.sh > run.log 2>&1 </dev/null & echo launched" \
      </dev/null
    STATUS="RUNNING"

    while [ "$(date +%s)" -lt "$DEADLINE" ]; do
      sleep 60
      pull
      TAIL=$(ssh -n $SSHO -i "$KEY" -p "$PORT" root@"$IP" "tail -2 /root/v4/run.log" 2>/dev/null)
      [ -n "$TAIL" ] && say "  $TAIL"
      if ssh -n $SSHO -i "$KEY" -p "$PORT" root@"$IP" "test -f /root/v4out/DONE" 2>/dev/null; then
        say "DONE seen"; break
      fi
      if ssh -n $SSHO -i "$KEY" -p "$PORT" root@"$IP" "test -f /root/v4out/HOLD" 2>/dev/null; then
        [ $HELD -eq 0 ] && say "HOLD seen — pod now holds expensive state"
        HELD=1
      fi
    done

    pull
    if [ -s "$OUTD/result.json" ]; then
      STATUS=$(jq -r '.status // "UNKNOWN"' "$OUTD/result.json")
      NOTE=$(jq -r '.note // ""' "$OUTD/result.json")
      say "evidence verified: result.json present, status=$STATUS"
    else
      STATUS="NO_RESULT"; NOTE="no result.json before the deadline"
    fi

    # HOLD without DONE means the run failed AFTER the expensive stages. Keep it.
    if ssh -n $SSHO -i "$KEY" -p "$PORT" root@"$IP" "test -f /root/v4out/DONE" 2>/dev/null; then
      teardown
    else
      HELD=1
      NOTE="$NOTE [HELD: pod $PODID kept (holds the 160 GB ckpt / 147 GB arena). ssh root@$IP -p $PORT. The backstop terminates it at the cap.]"
      say "$NOTE"
    fi
  fi
fi

say "STATUS=$STATUS"
python3 - "$STATUS" "$NOTE" "$PODID" "$COST" "$OUTD" <<'PYEOF' || true
import json, os, smtplib, sys
from email.message import EmailMessage
from email.utils import formatdate
status, note, podid, cost, outd = sys.argv[1:6]
L = [f"DeepSeek-V4-Flash 284B QLoRA from a native MXFP4 arena (PREREG amendment 2)",
     f"status : {status}", f"pod    : {podid} (${cost}/hr secure on-demand)",
     f"note   : {note}", ""]

def grab(fn, keys):
    p = os.path.join(outd, fn)
    if not os.path.exists(p):
        L.append(f"{fn}: (absent)"); return
    try:
        d = json.load(open(p))
    except Exception as e:
        L.append(f"{fn}: unparsable {e}"); return
    L.append(f"--- {fn} ---")
    for k in keys:
        if k in d:
            L.append(f"  {k} = {d[k]}")

grab("env.json", ["gpu", "vram_gib", "torch", "triton", "experts4bit-qlora", "grouped-nf4-gemm"])
grab("bake.json", ["seconds", "n_layers", "n_experts_per_layer", "row_stride", "segments"])
grab("arm_arena.json", ["outcome", "G1_modules_patched", "G4", "G4_enable_fast_train_returned",
                        "load_seconds", "P2_peak_gib", "P2_in_band_4_to_8", "P2_under_24",
                        "P3_frozen_bytes_unchanged", "P4_loss_finite_and_moves",
                        "losses", "step_seconds", "hot_rows", "vram"])
grab("arm_stock.json", ["outcome", "error_type", "failed_on_memory", "error"])
for f in ("run.log",):
    p = os.path.join(outd, f)
    if os.path.exists(p):
        L += ["", f"--- {f} (tail) ---"] + open(p, errors="replace").read().splitlines()[-25:]

cred = {}
for line in open(os.path.expanduser("~/.config/mailcow-notify/cred")):
    k, _, v = line.strip().partition("="); cred[k] = v
m = EmailMessage()
m["Subject"] = f"[v4-arena-train] {status}"
m["From"] = m["To"] = cred["MAIL_USER"]; m["Date"] = formatdate(localtime=True)
m.set_content("\n".join(L))
with smtplib.SMTP_SSL("mx.cerinamroth.com", 465, timeout=45) as s:
    s.login(cred["MAIL_USER"], cred["MAIL_PASS"]); s.send_message(m)
print("mail sent")
PYEOF
say "autopilot complete"
exit 0
