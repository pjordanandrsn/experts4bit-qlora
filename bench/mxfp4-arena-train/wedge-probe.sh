#!/bin/bash
# Is the uptime-0 wedge OURS or the provider's?
#
# A5000 has wedged 0/4 today under MY body. The body that came up reliably in
# earlier work (`up2xa5000.sh`) differs in exactly two ways: it attaches
# `volumeInGb:150` at `/workspace`, and it asks for `containerDiskInGb:80`
# instead of 420. This runs the same shape under three bodies, one variable at a
# time, and reports which reach a live container.
#
# Every pod is torn down as soon as it is classified, and each id is pinned into
# its own teardown. Worst case ~3 pods x ~3 min x $0.27/hr = ~$0.04.
set -u
. "$HOME/.runpod/secrets.env"
REST="https://rest.runpod.io/v1"
GQL="https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
PUB=$(cat "$HOME/.runpod/id_ed25519.pub")
SHAPE="${SHAPE:-NVIDIA RTX A5000}"

say() { echo "[$(date -u +%FT%TZ)] $*"; }

kill_pod() {
  curl -s -m 30 -X DELETE -o /dev/null "$REST/pods/$1" \
    -H "Authorization: Bearer $RUNPOD_API_KEY"
  sleep 5
  local c
  c=$(curl -s -m 20 -o /dev/null -w '%{http_code}' \
      -H "Authorization: Bearer $RUNPOD_API_KEY" "$REST/pods/$1")
  say "    torn down $1 -> GET $c"
}

probe() {  # probe <label> <extra-json>
  local label="$1" extra="$2" body resp pid up ip
  body=$(jq -nc --arg pk "$PUB" --arg gpu "$SHAPE" --argjson x "$extra" '{
    name:"wedge-probe", imageName:"runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204",
    cloudType:"SECURE", gpuTypeIds:[$gpu], gpuCount:1, interruptible:false,
    supportPublicIp:true, ports:["22/tcp"], env:{PUBLIC_KEY:$pk}} + $x')
  resp=$(curl -s -m 90 -X POST "$REST/pods" -H "Authorization: Bearer $RUNPOD_API_KEY" \
         -H "Content-Type: application/json" -d "$body")
  pid=$(printf '%s' "$resp" | jq -r '.id // empty')
  if [ -z "$pid" ]; then
    say "  $label: NO STOCK / rejected: $(printf '%s' "$resp" | jq -rc '.error // .' | head -c 120)"
    return
  fi
  say "  $label: created $pid ($extra)"
  for i in $(seq 1 12); do          # 12 x 15 s = 180 s, well past the 120 s mark
    sleep 15
    R=$(curl -s -m 25 "$GQL" -H "Content-Type: application/json" \
        -d '{"query":"query{pod(input:{podId:\"'"$pid"'\"}){runtime{uptimeInSeconds ports{ip privatePort isIpPublic}}}}"}')
    up=$(printf '%s' "$R" | jq -r '.data.pod.runtime.uptimeInSeconds // 0')
    ip=$(printf '%s' "$R" | jq -r '[.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true)][0].ip // empty')
    if [ -n "$ip" ]; then say "  $label: LIVE at ${i}x15s (uptime ${up}s) -> $ip"; kill_pod "$pid"; return; fi
    if [ "$up" != "0" ]; then say "  $label: container STARTED (uptime ${up}s), ports pending"; fi
  done
  say "  $label: WEDGED — uptime still $up after 180 s"
  kill_pod "$pid"
}

say "=== shape under test: $SHAPE ==="
probe "A my-body   (disk 420, NO volume)" '{"containerDiskInGb":420}'
probe "B proven    (disk 80, volume 150)" '{"containerDiskInGb":80,"volumeInGb":150,"volumeMountPath":"/workspace"}'
probe "C small-disk (disk 80, NO volume)" '{"containerDiskInGb":80}'
say "=== done ==="
