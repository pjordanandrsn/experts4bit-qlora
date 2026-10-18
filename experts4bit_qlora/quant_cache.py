"""On-disk cache of the quantized experts (``E4B_QUANT_CACHE=DIR``; bench/quant-cache/DESIGN.md).

Every offloaded start of :func:`experts4bit_qlora.loader.load_moe_4bit_streaming` rebuilds the same 4-bit experts from the
checkpoint: per-expert reads, a host-side stack, a device quantize, a device-to-host copy per layer. With this variable set,
the first load writes each MoE layer's packed experts (``gate_up_proj``, ``down_proj`` and their fp32 absmax) to
``DIR/<key>/`` and every later load with the same key reads them back instead.

The key is the sha256 of everything the bytes depend on (:func:`cache_key`): the checkpoint identity, the per-layer scheme,
the load dtype, the layout, the bitsandbytes and torch versions and :data:`QUANT_FORMAT_VERSION` -- not the package version,
so a release that does not touch the quantizer keeps every cache. The GPU is recorded, not keyed: a hit on another GPU
re-quantizes one layer and compares it before it is trusted (the loader does that; a difference makes the load a miss under
a key with the device folded in).

Safety: only safetensors files and a JSON manifest are read, never a pickle, so a cache copied from elsewhere can supply
bytes but not code. Every file is checked against the manifest's sha256 before a hit is used; a mismatch refuses
(:class:`QuantCacheCorrupt`) unless ``E4B_QUANT_CACHE_REBUILD=1``. A miss writes into a temporary directory and renames it
into place; a partial write is never read, and a writer that loses the rename to a concurrent one uses the winner's.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
import warnings

import torch

ENV = "E4B_QUANT_CACHE"
ENV_REBUILD = "E4B_QUANT_CACHE_REBUILD"
#: Bumped whenever the bytes a given checkpoint quantizes to may change: ``fuse_experts`` / ``stack_experts`` (the layout),
#: ``ExpertsNbit._quantize_stack`` (the quantizer call) or the extra-state schema. tests/test_quant_cache.py pins the
#: source of those functions against this number, so changing one without bumping it fails CI.
QUANT_FORMAT_VERSION = 1
#: The on-disk layout of this module (manifest fields, file names).
CACHE_FORMAT = 1
NAMES = ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")
_FREE_MARGIN = 1.05
_STALE_S = 2 * 3600          # a temporary directory untouched this long is a crashed writer's


class QuantCacheCorrupt(RuntimeError):
    """A cache hit whose files do not match its manifest: refused rather than trained on."""


def _newest_mtime(path):
    try:
        return max([os.path.getmtime(path)] + [os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path)])
    except OSError:
        return 0.0


def cache_dir_requested():
    v = os.environ.get(ENV, "").strip()
    return v or None


def _sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def checkpoint_identity(model_id, snapshot_dir, commit_hash):
    """The checkpoint half of the key: the resolved commit for a Hub id, otherwise the sha256 of every safetensors shard and
    the index in the local directory (a local directory has no revision to trust)."""
    if commit_hash:
        return {"model_id": str(model_id), "commit": str(commit_hash)}
    files = sorted(f for f in os.listdir(snapshot_dir)
                   if f.endswith(".safetensors") or f == "model.safetensors.index.json")
    return {"model_id": os.path.basename(os.path.normpath(str(model_id))),
            "files": {f: _sha256_file(os.path.join(snapshot_dir, f)) for f in files}}


def cache_key(fields: dict) -> str:
    """sha256 of the canonical JSON of ``fields`` (sorted keys, no whitespace)."""
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def key_fields(*, checkpoint, model_type, layer_specs, dtype, has_gate, shape):
    """Everything the cached bytes depend on. ``layer_specs`` maps layer index -> ``[quant_type, blocksize]`` or ``None``."""
    import bitsandbytes
    return {"checkpoint": checkpoint, "model_type": str(model_type),
            "layer_specs": {str(k): (list(v) if v is not None else None) for k, v in sorted(layer_specs.items())},
            "dtype": str(dtype).replace("torch.", ""), "has_gate": bool(has_gate), "shape": dict(shape),
            "compress_statistics": False, "bitsandbytes": bitsandbytes.__version__,
            "torch": torch.__version__.split("+")[0], "quant_format": QUANT_FORMAT_VERSION, "cache_format": CACHE_FORMAT}


def _device_record():
    if not torch.cuda.is_available():
        return {"name": "cpu", "capability": None}
    return {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}


class QuantCache:
    """One load's view of the cache: ``mode`` is ``"hit"`` (read every MoE layer from disk) or ``"miss"`` (write each layer
    after it is built, publish at :meth:`finish`)."""

    def __init__(self, root, fields, log=print):
        self.root, self.log = root, log
        self.fields = dict(fields)
        self.key = cache_key(self.fields)
        self.dir = os.path.join(root, self.key)
        self.device = _device_record()
        self.needs_device_check = False
        self._tmp = None
        self._layers = {}
        self._write_ok = True
        self.manifest = None
        self.mode = "miss"
        man = os.path.join(self.dir, "manifest.json")
        if os.path.isfile(man):
            self._open_hit(man)
        if self.mode == "miss":
            os.makedirs(root, exist_ok=True)
            self._sweep_stale_tmp()

    # ---- hit
    def _open_hit(self, man):
        with open(man, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("cache_format") != CACHE_FORMAT or m.get("key") != self.key or m.get("fields") != self.fields:
            raise QuantCacheCorrupt(f"{man}: the manifest does not describe key {self.key[:12]} (a foreign or edited cache)")
        bad = []
        for i, ent in m["layers"].items():
            path = os.path.join(self.dir, ent["file"])
            if (not os.path.isfile(path) or os.path.getsize(path) != ent["bytes"] or _sha256_file(path) != ent["sha256"]):
                bad.append(ent["file"])
        if bad:
            if os.environ.get(ENV_REBUILD) == "1":
                self.log(f"[quant-cache] {len(bad)} file(s) do not match the manifest under {self.dir}; "
                         f"{ENV_REBUILD}=1: rebuilding")
                shutil.rmtree(self.dir, ignore_errors=True)
                return
            raise QuantCacheCorrupt(f"quant cache {self.dir}: {len(bad)} file(s) do not match the manifest's sha256 "
                                     f"({', '.join(bad[:3])}{' ...' if len(bad) > 3 else ''}); refusing to train on it. "
                                     f"Remove the directory or set {ENV_REBUILD}=1.")
        self.manifest, self.mode = m, "hit"
        self.needs_device_check = m.get("device", {}).get("name") != self.device["name"]

    def has_layer(self, i):
        return self.mode == "hit" and str(i) in self.manifest["layers"]

    def load_layer(self, i):
        """The four packed tensors of MoE layer ``i`` as CPU tensors, and the extra state recorded with them."""
        from safetensors.torch import load_file
        ent = self.manifest["layers"][str(i)]
        t = load_file(os.path.join(self.dir, ent["file"]), device="cpu")
        return {n: t[n] for n in NAMES}, ent["extra_state"]

    def device_mismatch_to_miss(self):
        """A hit on another GPU whose re-quantized layer differs: fold the device into the key and become a miss."""
        warnings.warn(f"[quant-cache] layer bytes quantized on {self.device['name']} differ from the cache's "
                      f"({self.manifest['device']['name']}): rebuilding under a device-specific key", RuntimeWarning,
                      stacklevel=2)
        self.fields["device"] = self.device
        self.key = cache_key(self.fields)
        self.dir = os.path.join(self.root, self.key)
        self.manifest, self.mode, self.needs_device_check = None, "miss", False
        man = os.path.join(self.dir, "manifest.json")
        if os.path.isfile(man):
            self._open_hit(man)
        if self.mode == "miss":
            self._sweep_stale_tmp()

    # ---- miss
    def _sweep_stale_tmp(self):
        """Remove this key's temporary directories left by a crashed writer: only ones untouched for _STALE_S, so a
        concurrent writer's directory, still being filled, is never removed under it."""
        now = time.time()
        for d in os.listdir(self.root):
            path = os.path.join(self.root, d)
            if d.startswith(f".tmp-{self.key[:16]}-") and now - _newest_mtime(path) > _STALE_S:
                shutil.rmtree(path, ignore_errors=True)

    def write_layer(self, i, tensors: dict, extra_state: dict, n_layers_expected: int):
        """Write MoE layer ``i`` from its (pinned, CPU) homes. Never raises: a failure warns and disables the write."""
        if self.mode != "miss" or not self._write_ok:
            return
        try:
            from safetensors.torch import save_file
            if self._tmp is None:
                need = sum(t.numel() * t.element_size() for t in tensors.values()) * n_layers_expected * _FREE_MARGIN
                free = shutil.disk_usage(self.root).free
                if free < need:
                    self._write_ok = False
                    self.log(f"[quant-cache] not written: {free / 1e9:.1f} GB free under {self.root} < "
                             f"{need / 1e9:.1f} GB needed")
                    return
                self._tmp = tempfile.mkdtemp(prefix=f".tmp-{self.key[:16]}-{uuid.uuid4().hex[:8]}-", dir=self.root)
            fname = f"layer_{i:03d}.safetensors"
            path = os.path.join(self._tmp, fname)
            save_file({n: tensors[n].detach().to("cpu").contiguous() for n in NAMES}, path)
            self._layers[str(i)] = {"file": fname, "bytes": os.path.getsize(path), "sha256": _sha256_file(path),
                                    "extra_state": dict(extra_state)}
        except Exception as e:  # a cache write must never fail the load
            self._write_ok = False
            self.log(f"[quant-cache] not written: {type(e).__name__}: {e}")

    def finish(self):
        """Publish a miss's directory atomically. Returns the mode actually in effect (``hit``, ``wrote``, ``not-written``)."""
        if self.mode == "hit":
            return "hit"
        if not self._write_ok or self._tmp is None or not self._layers:
            if self._tmp:
                shutil.rmtree(self._tmp, ignore_errors=True)
            return "not-written"
        man = {"cache_format": CACHE_FORMAT, "key": self.key, "fields": self.fields, "device": self.device,
               "layers": self._layers}
        with open(os.path.join(self._tmp, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.rename(self._tmp, self.dir)
        except OSError:
            # A concurrent writer published this key first: use its directory if it verifies, drop ours either way.
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
            try:
                QuantCache(self.root, self.fields, log=self.log)
                return "wrote (a concurrent writer's copy verified)"
            except QuantCacheCorrupt:
                return "not-written"
        self._tmp = None
        return "wrote"
