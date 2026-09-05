#!/bin/bash
# bo7_queue.sh (mini): start the throughput census once bo6 and its repeat controls are done (TP2_DONE on the bo6 box),
# or once bo6 has been torn down (nothing left to wait for). Deadline 8 h.
cd /tmp/bo7 || exit 1
K=~/.vast/id_ed25519; deadline=$(( $(date +%s) + 28800 ))
while :; do
  [ -f /tmp/bo6/TEARDOWN_PROVEN ] && { echo "bo6 torn down -> launching bo7 $(date -u +%FT%TZ)"; break; }
  if [ -f /tmp/bo6/conn.env ]; then . /tmp/bo6/conn.env
    ssh -q -i $K -p $P -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes root@$H 'test -f /root/bo6/TP2_DONE' 2>/dev/null && { echo "bo6b done -> launching bo7 $(date -u +%FT%TZ)"; break; }
  fi
  [ $(date +%s) -gt $deadline ] && { echo "bo7 queue: 8 h deadline, not launched"; exit 1; }
  sleep 120
done
set -a; . ~/.vast/secrets.env; set +a
bash /tmp/bo7/bo7_all.sh > /tmp/bo7/bo7_all.log 2>&1
echo "bo7_all rc=$?"
