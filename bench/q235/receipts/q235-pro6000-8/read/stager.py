"""A rolling SSD stage of a Hugging Face snapshot for a layer-streamed reader (layer_stream_ref.py), so its memory-mapped reads come
off the SSD pool instead of the HDD pool without holding a full copy (the 235B snapshot is 439 GiB; the SSD pool has 416 GiB free).

FARM mirrors SNAPSHOT: the small files are copied; every *.safetensors entry is at all times EITHER a symlink to the snapshot's own
content-addressed blob OR a complete copy whose sha256, taken from the copy on disk, equals that blob's name. A copy is written to a
hidden temporary, fsynced, hashed, and only then renamed over the symlink; an evicted copy is replaced by the symlink again (rename).
So the reader sees the same bytes whichever pool serves a shard, and a failed or slow stage only costs time.

Progress is the reader's own output: PROGRESS/layerNNN.pt per finished layer, so c = their count is the layer being read. Wanted:
shards holding any layer in [c - behind, c + ahead]; evicted: staged shards whose last layer is below c - behind. Exits on --stop,
restoring every symlink. One JSON line per event in --log.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, shutil, time

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("farm"); ap.add_argument("progress")
ap.add_argument("--ahead", type=int, default=8); ap.add_argument("--behind", type=int, default=1)
ap.add_argument("--ready", type=int, required=True, help="write --ready-file once every shard of layers [0, READY] is staged")
ap.add_argument("--ready-file", required=True); ap.add_argument("--stop", required=True); ap.add_argument("--log", required=True)
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
LOG = open(a.log, "a")


def log(event, **kw):
    LOG.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **kw}) + "\n"); LOG.flush()


os.makedirs(a.farm)
blob = {}
for name in sorted(os.listdir(a.snapshot)):
    src = os.path.join(a.snapshot, name)
    if name.endswith(".safetensors"):
        blob[name] = os.path.realpath(src)
        assert re.fullmatch(r"[0-9a-f]{64}", os.path.basename(blob[name])), (name, blob[name])
        os.symlink(blob[name], os.path.join(a.farm, name))
    else:
        shutil.copyfile(src, os.path.join(a.farm, name))
wmap = json.load(open(os.path.join(a.snapshot, "model.safetensors.index.json")))["weight_map"]
n = json.load(open(os.path.join(a.snapshot, "config.json")))["num_hidden_layers"]
span = {}
for key, f in wmap.items():
    m = re.match(r"model\.layers\.(\d+)\.", key)
    i = int(m.group(1)) if m else (0 if "embed_tokens" in key else n - 1)    # embeddings before layer 0; norm and head after the last
    lo, hi = span.get(f, (i, i)); span[f] = (min(lo, i), max(hi, i))
assert set(span) == set(blob), "index and snapshot disagree on the shards"
order = sorted(span, key=lambda f: span[f])
log("farm", snapshot=a.snapshot, farm=a.farm, shards=len(order), layers=n, ahead=a.ahead, behind=a.behind)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage(f):
    tmp, t0 = os.path.join(a.farm, f".{f}.part"), time.time()
    with open(blob[f], "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 16 << 20); fo.flush(); os.fsync(fo.fileno())
    t1 = time.time(); got = sha256(tmp)
    if got != os.path.basename(blob[f]):
        os.unlink(tmp); return f, False, dict(sha256=got, copy_s=round(t1 - t0, 1))
    os.rename(tmp, os.path.join(a.farm, f))
    return f, True, dict(sha256=got, bytes=os.path.getsize(os.path.join(a.farm, f)), copy_s=round(t1 - t0, 1), hash_s=round(time.time() - t1, 1))


def evict(f):
    tmp = os.path.join(a.farm, f".{f}.link")
    os.symlink(blob[f], tmp); os.rename(tmp, os.path.join(a.farm, f))


def done_layers():
    try:
        return sum(1 for x in os.listdir(a.progress) if re.fullmatch(r"layer\d{3}\.pt", x))
    except FileNotFoundError:
        return 0


staged, failed, busy, ready = set(), set(), {}, False
with cf.ThreadPoolExecutor(a.workers) as pool:
    while True:
        for fut in [x for x in busy if x.done()]:
            f, ok, kw = fut.result(); del busy[fut]
            (staged if ok else failed).add(f); log("staged" if ok else "MISMATCH", shard=f, layers=list(span[f]), **kw)
        c = done_layers()
        for f in sorted(staged):
            if span[f][1] < c - a.behind:
                evict(f); staged.discard(f); log("evicted", shard=f, at_layer=c)
        if not ready and all(f in staged for f in order if span[f][0] <= a.ready):
            open(a.ready_file, "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime())); ready = True; log("ready", at_layer=c)
        if os.path.exists(a.stop):
            break
        inflight = set(busy.values())
        for f in order:
            if len(busy) >= a.workers:
                break
            lo, hi = span[f]
            if hi >= c - a.behind and lo <= c + a.ahead and f not in staged | failed | inflight:
                busy[pool.submit(stage, f)] = f; inflight.add(f)
        time.sleep(1)
    for fut in list(busy):
        f, ok, kw = fut.result()
        if ok:
            staged.add(f)
        else:
            os.path.exists(os.path.join(a.farm, f".{f}.part")) and os.unlink(os.path.join(a.farm, f".{f}.part"))
for f in sorted(staged):
    evict(f)
log("stopped", at_layer=done_layers(), evicted=len(staged), mismatches=sorted(failed))
