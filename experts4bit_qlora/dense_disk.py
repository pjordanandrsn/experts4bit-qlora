"""Serve a model's DENSE (non-expert) weights from disk, byte-for-byte.

``dense_offload`` keeps each decoder layer's dense weights in pinned host RAM and
streams one layer at a time to the GPU. That is bit-identical and fast, but the host
RAM it needs is the whole dense side of the model at once -- 114.4 GB for Kimi-K3,
which is more than a cheap single-GPU pod has. This module moves that floor to disk:
a layer's weights are read from the checkpoint on demand, into a pinned staging
buffer, and copied to the device exactly as before.

**Nothing is transformed.** The bytes handed to the GPU are the bytes in the
checkpoint, at their own dtype. That is the whole point -- the alternative way to fit
a 114 GB dense side on a small card is to quantize it, which changes the model. This
does not.

Why safetensors directly instead of a baked arena: a safetensors file already IS a
flat file with an exact byte range per tensor, no compression and no framing between
them. Baking a second copy into a custom arena would cost another 114 GB on disk and
add a format to keep honest, for no gain. The one thing the arena format buys --
4096-aligned rows, so reads can use ``O_DIRECT`` -- is recovered here by reading the
aligned window that CONTAINS the tensor and slicing the middle out.

``O_DIRECT`` matters because dense weights are read once per token and never reused:
page cache can only cost memory, never save a read.

Usage::

    src = DenseDiskSource("/path/to/dense-snapshot")   # dir of *.safetensors
    hs = enable_dense_offload(model, "cuda", source=src)
"""
from __future__ import annotations

import json
import os
import struct
from typing import Iterable

import torch

from .util import log

# safetensors dtype tags -> torch, with the itemsize we need for offset math.
_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "U16": torch.uint16 if hasattr(torch, "uint16") else torch.int16,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "I32": torch.int32, "U32": torch.uint32 if hasattr(torch, "uint32") else torch.int32,
    "F32": torch.float32, "I64": torch.int64, "F64": torch.float64,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8),
    "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8),
}

_ALIGN = 4096          # O_DIRECT wants offset, length and buffer all aligned to this

# Linux caps a single read at 0x7FFFF000 (~2.1 GB) no matter how much you ask for, so
# a tensor bigger than that comes back SHORT rather than erroring. K3 hits this on
# embed_tokens and lm_head at 2.35 GB each. Read in aligned chunks and loop; the chunk
# size is a module constant so a test can shrink it and exercise the loop on a tensor
# small enough to live in a repo.
_MAX_READ = 1 << 30    # 1 GiB, a multiple of _ALIGN


class _Located:
    """Where one tensor's bytes live, and what shape to read them back as."""

    __slots__ = ("path", "offset", "nbytes", "shape", "dtype")

    def __init__(self, path, offset, nbytes, shape, dtype):
        self.path, self.offset, self.nbytes = path, offset, nbytes
        self.shape, self.dtype = tuple(shape), dtype


class DenseDiskSource:
    """Byte-exact random access to the tensors of a safetensors directory or file.

    Holds one file descriptor per shard and one shared pinned staging buffer. The
    buffer is shared on purpose: reads here are followed by a SYNCHRONOUS device
    copy, so it is free again the moment ``fetch`` returns. See
    :meth:`_DenseOffload._copy_home_to_device` for why the copy stays synchronous on
    this path.
    """

    def __init__(self, path: str, *, direct: bool = True, staging_bytes: int = 0):
        self.path = path
        self.direct = direct
        self._fds: dict[str, int] = {}
        self._staging: torch.Tensor | None = None
        self._staging_ok = True        # False once we learn the buffer is unaligned
        self.reads = 0
        self.read_bytes = 0
        self.slack_bytes = 0           # extra bytes pulled in for alignment

        files = []
        if os.path.isdir(path):
            files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
            files = [os.path.join(path, f) for f in files]
        elif path.endswith(".safetensors"):
            files = [path]
        if not files:
            raise FileNotFoundError(f"no .safetensors under {path!r}")

        self.tensors: dict[str, _Located] = {}
        for f in files:
            for name, loc in _read_header(f).items():
                if name in self.tensors:
                    raise ValueError(f"{name!r} appears in two shards; index is ambiguous")
                self.tensors[name] = loc
        self.bytes = sum(t.nbytes for t in self.tensors.values())
        if staging_bytes:
            self._alloc_staging(staging_bytes)

    # ------------------------------------------------------------------ read --
    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            flags = os.O_RDONLY
            if self.direct and hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT
            try:
                fd = os.open(path, flags)
            except OSError:
                if not (self.direct and hasattr(os, "O_DIRECT")):
                    raise
                # some filesystems (tmpfs, certain network mounts) reject O_DIRECT
                self.direct = False
                fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def _alloc_staging(self, nbytes: int) -> None:
        n = max(nbytes, _ALIGN) + _ALIGN        # room to slide to an aligned start
        buf = torch.empty(n, dtype=torch.uint8)
        try:
            buf = buf.pin_memory()
        except (RuntimeError, AssertionError):
            pass                                # pageable works, just slower
        self._staging = buf
        # cudaHostAlloc returns page-aligned memory, but do not take that on faith:
        # an unaligned destination makes O_DIRECT fail with EINVAL, which would look
        # like a corrupt checkpoint rather than a buffer problem.
        self._staging_ok = (buf.data_ptr() % _ALIGN) == 0
        if self.direct and not self._staging_ok:
            log("  dense_disk: staging buffer not 4096-aligned; using buffered reads")
            self.direct = False

    def staging_for(self, nbytes: int) -> torch.Tensor:
        if self._staging is None or self._staging.numel() < nbytes + _ALIGN:
            self._alloc_staging(nbytes)
        return self._staging

    def fetch(self, name: str) -> torch.Tensor:
        """Read ``name``'s bytes into the shared staging buffer and view them.

        The returned tensor aliases the staging buffer: valid until the next
        ``fetch``. Callers copy out of it immediately.
        """
        loc = self.tensors.get(name)
        if loc is None:
            raise KeyError(f"{name!r} not in {self.path}")

        # Settle `direct` BEFORE computing the window. `staging_for` can flip it to
        # False — an unaligned staging buffer cannot serve an O_DIRECT read — and a
        # window computed under the old value would then be read into a buffer that no
        # longer satisfies the contract, failing with EINVAL that looks like a corrupt
        # checkpoint. The request covers the largest window any branch below can want,
        # so the second call is a no-op.
        buf = self.staging_for(loc.nbytes + 2 * _ALIGN)
        if self.direct:
            lo = (loc.offset // _ALIGN) * _ALIGN
            hi = -(-(loc.offset + loc.nbytes) // _ALIGN) * _ALIGN
            pad = loc.offset - lo
        else:
            lo, hi, pad = loc.offset, loc.offset + loc.nbytes, 0
        span = hi - lo

        buf = self.staging_for(span)
        # slide the read to an aligned start INSIDE the buffer, so both ends of the
        # O_DIRECT contract hold (offset aligned, length aligned, address aligned)
        skew = (-buf.data_ptr()) % _ALIGN if self.direct else 0
        window = buf[skew:skew + span]
        fd = self._fd(loc.path)
        mv = memoryview(window.numpy())
        # Only the bytes this tensor needs are guaranteed to exist: for the LAST
        # tensor in a file the aligned window runs past EOF, so a read of the full
        # span legitimately comes back short. Demanding the whole span turned that
        # normal case into an error.
        need = pad + loc.nbytes
        got = 0
        while got < need:
            # every partial read stays aligned: `got` advances by a multiple of
            # _ALIGN, so offset, length and buffer address all remain aligned, which
            # O_DIRECT requires of each individual call and not merely of the whole
            n = os.preadv(fd, [mv[got:got + min(_MAX_READ, span - got)]], lo + got)
            if n <= 0:
                raise IOError(f"short read for {name}: {got} of {span} bytes at {lo} "
                              f"(read returned {n})")
            got += n
        self.reads += 1
        self.read_bytes += span
        self.slack_bytes += span - loc.nbytes

        raw = window[pad:pad + loc.nbytes]
        return raw.view(loc.dtype).view(loc.shape)

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._staging = None

    def stats(self) -> dict:
        return {"tensors": len(self.tensors), "bytes": self.bytes,
                "reads": self.reads, "read_bytes": self.read_bytes,
                "slack_bytes": self.slack_bytes, "direct": self.direct}

    def __repr__(self) -> str:
        return (f"DenseDiskSource({self.path!r}, {len(self.tensors)} tensors, "
                f"{self.bytes/1e9:.1f} GB, direct={self.direct})")


class DiskHome:
    """A ``_DenseOffload`` home whose bytes live on disk rather than in host RAM.

    Quacks like the pinned CPU tensor it replaces for the three things
    ``_DenseOffload`` asks of a home: ``.dtype`` and ``.shape`` to size the device
    buffer, a materialized CPU tensor to copy from, and a real tensor for
    ``state_dict()`` while the layer is evicted.
    """

    __slots__ = ("source", "name", "shape", "dtype", "nbytes")

    def __init__(self, source: DenseDiskSource, name: str):
        loc = source.tensors[name]
        self.source, self.name = source, name
        self.shape, self.dtype, self.nbytes = loc.shape, loc.dtype, loc.nbytes

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self) -> int:
        return self.nbytes // max(self.numel(), 1)

    def view_now(self) -> torch.Tensor:
        """A CPU view of this tensor, valid only until the next fetch on the source."""
        return self.source.fetch(self.name)

    def materialize(self) -> torch.Tensor:
        """An OWNED copy — for ``state_dict()``, which must outlive the buffer."""
        return self.view_now().clone()

    def __repr__(self) -> str:
        return f"DiskHome({self.name!r}, {tuple(self.shape)}, {self.dtype})"


def _read_header(path: str) -> dict[str, _Located]:
    """Parse a safetensors header into byte ranges, without reading any tensor."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(hlen))
    data0 = 8 + hlen
    out: dict[str, _Located] = {}
    for name, meta in head.items():
        if name == "__metadata__":
            continue
        dt = _DTYPES.get(meta["dtype"])
        if dt is None:
            raise ValueError(f"unhandled safetensors dtype {meta['dtype']!r} for {name}")
        lo, hi = meta["data_offsets"]
        out[name] = _Located(path, data0 + lo, hi - lo, meta["shape"], dt)
    return out


def disk_homes_for(source: DenseDiskSource, prefix: str,
                   names: Iterable[str]) -> dict[str, DiskHome]:
    """Build ``{attr_path: DiskHome}`` for the tensors of one layer.

    ``prefix`` is the layer's key prefix in the checkpoint (e.g.
    ``"language_model.model.layers.7."``); ``names`` are layer-relative keys such as
    ``"self_attn.q_proj.weight"``.
    """
    homes = {}
    for n in names:
        full = prefix + n
        if full in source.tensors:
            homes[n] = DiskHome(source, full)
    return homes
