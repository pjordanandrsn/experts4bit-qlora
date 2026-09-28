# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
# Runs where the checkpoints live (stdlib only): pull every router tensor out of the
# safetensors shards into one small raw file + a JSON index.
import json
import struct
import sys
import os
import glob
src, dst, pat = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
index, off = {}, 0
with open(dst + ".bin", "wb") as out:
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        if os.path.basename(f).startswith("._"):
            continue
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            for name, meta in hdr.items():
                if name == "__metadata__" or not any(p in name for p in pat):
                    continue
                a, b = meta["data_offsets"]
                fh.seek(8 + n + a)
                buf = fh.read(b - a)
                out.write(buf)
                index[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "off": off, "len": b - a}
                off += b - a
json.dump(index, open(dst + ".json", "w"), indent=0)
print(len(index), "tensors", off, "bytes")
