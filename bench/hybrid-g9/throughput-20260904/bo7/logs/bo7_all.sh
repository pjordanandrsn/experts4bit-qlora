#!/bin/bash
set -uo pipefail
cd /tmp/bo7
ID=$(python3 - <<'PY'
import json,os,urllib.request,urllib.parse
k=os.environ["VAST_API_KEY"]
def api(u,m="GET",b=None):
    r=urllib.request.Request(u,data=(json.dumps(b).encode() if b else None),
      headers={"Authorization":"Bearer "+k,"Content-Type":"application/json"},method=m)
    return json.load(urllib.request.urlopen(r,timeout=90))
q={"gpu_name":{"eq":"RTX 5090"},"num_gpus":{"eq":1},"disk_space":{"gte":320},"cpu_ram":{"gte":98304},
   "rentable":{"eq":True},"verified":{"eq":True},"reliability2":{"gte":0.98},
   "inet_down":{"gte":800},"type":"ask","order":[["dlperf_per_dphtotal","desc"]],"limit":60}
offers=api("https://console.vast.ai/api/v0/bundles/?q="+urllib.parse.quote(json.dumps(q))).get("offers",[])
for o in offers[:40]:
    if o.get("dph_total",9)>1.20: continue
    if o.get("cuda_max_good",0) and o["cuda_max_good"]<12.4: continue
    cc = int(o.get("compute_cap") or 0)
    if cc < 800: continue   # Triton kernels need sm_80+; Turing fails every arm
    try: bad = set(int(x) for x in open("/tmp/bo7/bad_machines").read().split())
    except Exception: bad = set()
    if o.get("machine_id") in bad: continue
    try:
        r=api("https://console.vast.ai/api/v0/asks/%d/"%o["id"],"PUT",
          {"client_id":"me","image":"pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel",
           "disk":320,"label":"bo7","onstart":"touch /root/OK","runtype":"ssh_direct"})
        if not r.get("success"):
            r=api("https://console.vast.ai/api/v0/asks/%d/"%o["id"],"PUT",
              {"client_id":"me","image":"pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel",
               "disk":320,"label":"bo7","onstart":"touch /root/OK","runtype":"ssh"})
        if r.get("success"):
            open("/tmp/bo7/last_machine", "w").write(str(o.get("machine_id") or 0))
            print(r["new_contract"]); break
    except Exception: pass
PY
)
[ -z "$ID" ] && { echo "RENT FAILED"; exit 1; }
echo "rented $ID"
KEY="$HOME/.vast/id_ed25519"; H=""; P=""
SSHOPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=3"
destroy(){
  python3 - <<PY2
import json,os,time,urllib.request
k=os.environ["VAST_API_KEY"]
def api(u,m="GET"):
    return json.load(urllib.request.urlopen(urllib.request.Request(u,
      headers={"Authorization":"Bearer "+k},method=m),timeout=60))
for attempt in range(4):
    d=api("https://console.vast.ai/api/v1/instances/"); assert "instances" in d
    left=[i["id"] for i in d["instances"]]
    if $ID not in left:
        print("TEARDOWN PROVEN: absent from list (attempt",attempt,") remaining=",left); break
    for i in d["instances"]:
        if str(i.get("id"))=="$ID" and i.get("label")=="bo7":
            try: print("destroy attempt",attempt,"->",api("https://console.vast.ai/api/v0/instances/$ID/","DELETE"))
            except Exception as e: print("destroy err:",str(e)[:100])
    time.sleep(10)
else:
    print("TEARDOWN FAILED: still listed after 4 attempts; remaining=",left)
PY2
}
for t in $(seq 1 110); do
  read H P ST < <(python3 -c "
import json,os,urllib.request
k=os.environ['VAST_API_KEY']
r=urllib.request.Request('https://console.vast.ai/api/v1/instances',headers={'Authorization':'Bearer '+k})
d=json.load(urllib.request.urlopen(r,timeout=30))
assert 'instances' in d
for i in d['instances']:
    if str(i.get('id'))=='$ID':
        msg=(i.get('status_msg') or '')
        # the API says outright when a container failed to start (host CDI
        # breakage, image pull failure, OCI errors). Waiting out the ssh
        # window against a dead container is pure waste -- read it.
        dead = ('Error response from daemon' in msg
                or 'failed to start containers' in msg
                or (i.get('cur_state')=='stopped' and i.get('intended_status')=='stopped'))
        # a layer pull that reports 'Retrying' is failing, not progressing:
        # one draw burned its whole 46-min window on 'Retrying in 1 second'
        stalled = 'Retrying in' in msg
        host=i.get('ssh_host') or 'NONE'; port=i.get('ssh_port') or 0
        ports=i.get('ports') or {}
        m=ports.get('22/tcp') if isinstance(ports,dict) else None
        if m and i.get('public_ipaddr') and os.environ.get('PREFER_PROXY','0')!='1':
            try: host=i['public_ipaddr']; port=int(m[0]['HostPort'])
            except Exception: pass
        st = 'DEAD' if dead else ('PULLSTALL' if stalled else (i.get('actual_status') or 'unknown'))
        print(host, port, st)
        break
else: print('NONE 0 GONE')")
  if [ "${ST:-}" = "PULLSTALL" ]; then
    PULLBAD=$((${PULLBAD:-0}+1))
    if [ "${PULLBAD:-0}" -ge 6 ]; then
      echo "IMAGE PULL RETRYING x6 -- host cannot fetch the image; blacklisting and re-rolling"
      [ -f /tmp/bo7/last_machine ] && cat /tmp/bo7/last_machine >> /tmp/bo7/bad_machines && echo "" >> /tmp/bo7/bad_machines
      destroy; exit 1
    fi
  else
    PULLBAD=0
  fi
  if [ "${ST:-}" = "DEAD" ]; then
    echo "CONTAINER FAILED TO START (api status_msg) -- not waiting out the ssh window"
    python3 -c "
import json,os,urllib.request
k=os.environ['VAST_API_KEY']
r=urllib.request.Request('https://console.vast.ai/api/v1/instances',headers={'Authorization':'Bearer '+k})
for i in json.load(urllib.request.urlopen(r,timeout=30))['instances']:
    if str(i.get('id'))=='$ID': print('  reason:', (i.get('status_msg') or '')[:200])
" 2>/dev/null
    [ -f /tmp/bo7/last_machine ] && cat /tmp/bo7/last_machine >> /tmp/bo7/bad_machines && echo "" >> /tmp/bo7/bad_machines
    destroy; exit 1
  fi
  ERR=""
  if [ "$H" != "NONE" ] && [ "$P" != "0" ]; then
    OUT=$(ssh -i "$KEY" -p "$P" $SSHOPTS -o IdentitiesOnly=yes "root@$H" 'echo AUTH_OK' 2>&1)
    echo "$OUT" | grep -q AUTH_OK && { echo "auth ok $H:$P (try $t)"; break; }
    ERR=$(echo "$OUT" | grep -oE "Permission denied|Connection refused|Connection timed out|Operation timed out|Network is unreachable|Connection closed|No route to host" | head -1)
    case "$ERR" in *"timed out"|*unreachable|"") [ "${ST:-}" = "running" ] && { ERR="${ERR:-timeout}"; TMO=$((${TMO:-0}+1)); } ;; *) TMO=0 ;; esac
    if [ "${TMO:-0}" -ge 8 ] && [ "${PREFER_PROXY:-0}" != "1" ]; then
      echo "DIRECT PORT DEAD x8 ($H:$P never answers) -- falling back to the vast proxy route"
      export PREFER_PROXY=1; TMO=0
    fi
    # key rejected on a box the API calls RUNNING is bimodal in practice:
    # the key lands within ~5 min or never. Do not wait out 1000 s on never.
    if [ "$ERR" = "Permission denied" ] && [ "${ST:-}" = "running" ]; then
      DENIED=$((${DENIED:-0}+1))
    else
      DENIED=0
    fi
    if [ "${DENIED:-0}" -ge 10 ]; then
      echo "KEY NEVER INJECTED: 'Permission denied (publickey)' x10 on a running box -- blacklisting and re-rolling"
      [ -f /tmp/bo7/last_machine ] && cat /tmp/bo7/last_machine >> /tmp/bo7/bad_machines && echo "" >> /tmp/bo7/bad_machines
      destroy; exit 1
    fi
  fi
  [ $((t % 5)) -eq 0 ] && echo "auth try $t: host=$H port=$P state=${ST:-?} reason=${ERR:-none}"
  if [ "$t" = "110" ]; then
    echo "AUTH TIMEOUT -- destroying $ID"
    [ -f /tmp/bo7/last_machine ] && cat /tmp/bo7/last_machine >> /tmp/bo7/bad_machines && echo "" >> /tmp/bo7/bad_machines
    destroy; exit 1
  fi
  sleep 25
done
# --- hard-kill guard: survives this script; verified alive below ---
nohup bash -c "sleep 36000; cd /tmp/bo7 && set -a && . \$HOME/.vast/secrets.env && set +a && python3 -c \"
import json,os,urllib.request
k=os.environ['VAST_API_KEY']
r=urllib.request.Request('https://console.vast.ai/api/v0/instances/$ID/',headers={'Authorization':'Bearer '+k},method='DELETE')
print('HARDKILL-$ID ->', json.load(urllib.request.urlopen(r,timeout=60)))\"" \
  >> /tmp/bo7/guard.log 2>&1 & disown
sleep 1
gpid=""
for _p in $(pgrep -f "sleep 36000"); do
  ps -o command= -p "$_p" | grep -q "instances/$ID/" && gpid="$_p"
done
# Match the guard for THIS instance, never just any sleeping guard: a stale
# guard from a previous box satisfies a bare pgrep and would report ALIVE
# while this box runs unguarded (the failure mode that orphaned a pod).
if [ -n "$gpid" ]; then
  echo "GUARD ALIVE (10h hard-kill armed for $ID, pid $gpid)"
# ---- network pre-flight: this campaign pulls ~80 GB of checkpoints plus a
# multi-GB install, and a box whose link crawls burns the whole alarm window
# before failing. Measure it in 15 s and re-roll instead (observed: 0.1 MB/s).
PF=$(perl -e 'alarm 90; exec @ARGV' ssh -i "$KEY" -p "$P" $SSHOPTS "root@$H" \
  'for u in "https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/main/model-00001-of-00016.safetensors" "https://cdn-lfs.hf.co/" ; do
     out=$(curl -s -L -o /dev/null -m 25 -r 0-104857600 -w "%{speed_download} %{http_code}" "$u" 2>/dev/null); rc=$?
     spd=$(echo "$out" | cut -d" " -f1); code=$(echo "$out" | cut -d" " -f2)
     echo "PF rc=$rc http=$code speed=$spd url=$u"
     awk -v s="${spd:-0}" "BEGIN{exit !(s+0 > 15728640)}" && break
   done')
echo "$PF" | sed "s/^/  /"
BWMB=$(echo "$PF" | awk "{for(i=1;i<=NF;i++) if(\$i ~ /^speed=/){split(\$i,a,\"=\"); if(a[2]+0>m) m=a[2]+0}} END{printf \"%d\", m/1048576}")
echo "net pre-flight: ${BWMB:-0} MB/s"
# a zero now comes WITH curl's exit code and http status, so "slow" and
# "curl could not connect at all" are distinguishable in the log
if [ "${BWMB:-0}" -lt 15 ]; then
  echo "SLOW BOX (${BWMB:-0} MB/s < 15) -- blacklisting and re-rolling"
  [ -f /tmp/bo7/last_machine ] && cat /tmp/bo7/last_machine >> /tmp/bo7/bad_machines && echo "" >> /tmp/bo7/bad_machines
  kill $gpid 2>/dev/null; destroy; exit 1
fi
else
  echo "GUARD NOT VERIFIED -- destroying and aborting"; destroy; exit 1
fi
SCP="scp -q -i $KEY -P $P $SSHOPTS"
SSH="ssh -i $KEY -p $P $SSHOPTS root@$H"
$SSH 'mkdir -p /root/bo7'
$SSH 'mkdir -p /root/bo7/hook'
$SCP bo7_run.sh k8_bake.py step_decomp.py calib.json models.txt "root@$H:/root/bo7/"
$SCP usercustomize.py "root@$H:/root/bo7/hook/"
$SSH 'mkdir -p /root/.cache/huggingface'
$SCP "$HOME/.config/hf/token" "root@$H:/root/.cache/huggingface/token"
$SSH 'chmod 600 /root/.cache/huggingface/token' && echo "hf token installed"
perl -e 'alarm 40; exec @ARGV' ssh -i "$KEY" -p "$P" $SSHOPTS "root@$H" 'chmod +x /root/bo7/bo7_run.sh && setsid nohup /root/bo7/bo7_run.sh > /root/bo7/outer.log 2>&1 < /dev/null & sleep 2; echo LAUNCHED' ; echo "launch issued (alarm-capped)"
cat > /tmp/bo7/conn.env <<CONN
IID=$ID
H=$H
P=$P
CONN
echo "conn.env written ($ID @ $H:$P)"
exit 0
