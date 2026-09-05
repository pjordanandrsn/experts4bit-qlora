#!/bin/bash
# box-side: wait for bo7's TP_DONE, then run bo7b.sh (P35 amendment 2). Started with setsid nohup so it survives the ssh session.
cd /root/bo7 || exit 1
while [ ! -f /root/bo7/TP_DONE ]; do sleep 60; done
sleep 30
bash /root/bo7/bo7b.sh >> /root/bo7/outer.log 2>&1
