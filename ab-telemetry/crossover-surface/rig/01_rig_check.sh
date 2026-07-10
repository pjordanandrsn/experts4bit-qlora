#!/usr/bin/env bash
# 01_rig_check.sh — Session 4 rig verification. Run FIRST on the root bare-metal host.
# Verifies the substrate BEFORE any number is trusted: raw NVMe present, RAID-0 buildable,
# GDS/cuFile loadable (or CPU-bounce, recorded), PCIe topology, root caps (drop_caches).
# Emits receipts/rig.json. NON-DESTRUCTIVE except the explicit RAID-0 build (guarded).
set -u
OUT="${OUT:-receipts}"; mkdir -p "$OUT"
echo "=== root + caps ==="
[ "$(id -u)" = 0 ] && echo "root: yes" || { echo "root: NO — Session 4 needs root, abort"; exit 1; }
echo 1 > /proc/sys/vm/drop_caches 2>/dev/null && echo "drop_caches: WRITABLE (real cache drop between arms)" || echo "drop_caches: DENIED — not a true root host (a pod?), abort"
echo "=== raw NVMe data drives (exclude the OS disk) ==="
lsblk -d -o NAME,SIZE,TYPE,MODEL | grep -iE 'nvme' || echo "(no nvme?)"
NVME=$(lsblk -dn -o NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print "/dev/"$1}')
echo "candidate data drives: $NVME"
echo "=== PCIe topology (drives + GPU same root complex? P2P?) ==="
lspci | grep -iE 'nvidia|non-volatile|nvme' | head -8
nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader 2>/dev/null
echo "=== GDS / nvidia-fs availability ==="
if lsmod | grep -q nvidia_fs; then echo "nvidia-fs: loaded (GDS path available)"
elif modprobe nvidia-fs 2>/dev/null && lsmod | grep -q nvidia_fs; then echo "nvidia-fs: loaded now (GDS available)"
else echo "nvidia-fs: NOT available -> datapath = cpu_bounce (record it, hold constant)"; fi
command -v gdscheck >/dev/null 2>&1 && gdscheck -p 2>/dev/null | grep -iE 'cuFile|GDS|supported' | head -5
echo "=== RAID-0 build (guarded: set BUILD_RAID=1 and STRIPE_DRIVES='/dev/nvmeXn1 /dev/nvmeYn1') ==="
if [ "${BUILD_RAID:-0}" = 1 ] && [ -n "${STRIPE_DRIVES:-}" ]; then
  N=$(echo $STRIPE_DRIVES | wc -w)
  echo "building /dev/md0 RAID-0 over $N drives: $STRIPE_DRIVES"
  mdadm --create /dev/md0 --level=0 --raid-devices=$N $STRIPE_DRIVES --run
  mkfs.xfs -f /dev/md0 && mkdir -p /mnt/stripe && mount /dev/md0 /mnt/stripe
  echo "stripe mounted at /mnt/stripe"
else
  echo "(skipped — set BUILD_RAID=1 + STRIPE_DRIVES to build; single-drive arm can use a raw drive's xfs)"
fi
python3 - "$OUT" <<'PY'
import json,sys,subprocess,os
def sh(c):
    try: return subprocess.run(c,shell=True,capture_output=True,text=True).stdout.strip()
    except: return None
rec={
 "root": os.getuid()==0,
 "drop_caches_writable": os.access("/proc/sys/vm/drop_caches", os.W_OK),
 "nvme_drives": sh("lsblk -dn -o NAME,SIZE,TYPE | awk '$3==\"disk\" && $1 ~ /nvme/'"),
 "gpu": sh("nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader"),
 "nvidia_fs_loaded": "nvidia_fs" in (sh("lsmod")or""),
 "datapath": "GDS" if "nvidia_fs" in (sh("lsmod")or"") else "cpu_bounce",
}
json.dump(rec,open(os.path.join(sys.argv[1],"rig.json"),"w"),indent=2)
print("wrote", os.path.join(sys.argv[1],"rig.json")); print(json.dumps(rec,indent=2))
PY
