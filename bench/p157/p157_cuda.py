# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P157's driver and NVML probes (bench/p157/PREREG-p157.md), through ctypes so the box needs no extra install.

- :func:`occupancy`: the driver's own count of a kernel's resident CTAs per SM at a block size and a dynamic
  shared-memory size (``cuOccupancyMaxActiveBlocksPerMultiprocessor``), the check behind every occupancy pin;
- :func:`func_attrs` / :func:`device_attrs`: registers, local memory, shared memory and the preferred carveout of a
  loaded kernel; the SM's shared memory, the L2 and the SM count of the device;
- :func:`capture_kernel_nodes`: the kernel nodes of the CUDA graph being captured on a stream, read back while the
  capture is open (grid, block, ``sharedMemBytes``), so a pinned arm's pad is checked where it is timed;
- :class:`ClockSampler`: NVML's SM clock, memory clock, power (the averaged draw, the instantaneous draw and the
  ENFORCED limit, so a drifting cap is told apart from boost) and clock-event reasons, sampled while a graph replays and
  tagged with the arm replaying.
"""
from __future__ import annotations

import ctypes
import statistics
import threading
import time

CU_FUNC = {"max_threads": 0, "static_shared": 1, "local_bytes": 3, "num_regs": 4, "max_dynamic_shared": 8,
           "carveout": 9}
CU_DEV = {"sm_count": 16, "l2_bytes": 38, "max_shared_per_sm": 81, "max_regs_per_sm": 82,
          "max_shared_per_block_optin": 97, "max_blocks_per_sm": 106, "reserved_shared_per_block": 111}
_KERNEL_NODE = 0                                # CU_GRAPH_NODE_TYPE_KERNEL
_FI_POWER_INSTANT = 186                         # NVML_FI_DEV_POWER_INSTANT (mW)
_lib = None


def _cuda():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libcuda.so.1")
    return _lib


def _ok(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: CUresult {rc}")


def occupancy(function: int, block: int, dynamic_shared: int) -> int:
    n = ctypes.c_int(0)
    _ok(_cuda().cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(n), ctypes.c_void_p(function),
                                                            ctypes.c_int(block), ctypes.c_size_t(dynamic_shared)),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor")
    return n.value


def func_attrs(function: int) -> dict:
    out = {}
    for name, code in CU_FUNC.items():
        v = ctypes.c_int(0)
        rc = _cuda().cuFuncGetAttribute(ctypes.byref(v), ctypes.c_int(code), ctypes.c_void_p(function))
        out[name] = v.value if rc == 0 else f"CUresult {rc}"
    return out


def device_attrs(device: int = 0) -> dict:
    out = {}
    for name, code in CU_DEV.items():
        v = ctypes.c_int(0)
        rc = _cuda().cuDeviceGetAttribute(ctypes.byref(v), ctypes.c_int(code), ctypes.c_int(device))
        out[name] = v.value if rc == 0 else f"CUresult {rc}"
    return out


class _KernelNodeParams(ctypes.Structure):     # CUDA_KERNEL_NODE_PARAMS_v2
    _fields_ = [("func", ctypes.c_void_p), ("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint),
                ("gridDimZ", ctypes.c_uint), ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint),
                ("blockDimZ", ctypes.c_uint), ("sharedMemBytes", ctypes.c_uint), ("kernelParams", ctypes.c_void_p),
                ("extra", ctypes.c_void_p), ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]


def capture_kernel_nodes(stream: int) -> list:
    """The kernel nodes of the graph being captured on ``stream`` (call it inside the capture): [(func, (gx, gy, gz),
    (bx, by, bz), sharedMemBytes)]. Querying a graph under capture is permitted (cuStreamGetCaptureInfo)."""
    cu = _cuda()
    status, cid, graph = ctypes.c_int(0), ctypes.c_uint64(0), ctypes.c_void_p(0)
    deps, ndeps = ctypes.c_void_p(0), ctypes.c_size_t(0)
    _ok(cu.cuStreamGetCaptureInfo_v2(ctypes.c_void_p(stream), ctypes.byref(status), ctypes.byref(cid),
                                     ctypes.byref(graph), ctypes.byref(deps), ctypes.byref(ndeps)),
        "cuStreamGetCaptureInfo_v2")
    if status.value != 1 or not graph.value:      # CU_STREAM_CAPTURE_STATUS_ACTIVE
        raise RuntimeError(f"stream is not capturing (status {status.value})")
    n = ctypes.c_size_t(0)
    _ok(cu.cuGraphGetNodes(graph, None, ctypes.byref(n)), "cuGraphGetNodes")
    nodes = (ctypes.c_void_p * n.value)()
    _ok(cu.cuGraphGetNodes(graph, nodes, ctypes.byref(n)), "cuGraphGetNodes")
    get = getattr(cu, "cuGraphKernelNodeGetParams_v2", None) or cu.cuGraphKernelNodeGetParams
    out = []
    for i in range(n.value):
        t = ctypes.c_int(-1)
        _ok(cu.cuGraphNodeGetType(ctypes.c_void_p(nodes[i]), ctypes.byref(t)), "cuGraphNodeGetType")
        if t.value != _KERNEL_NODE:
            continue
        p = _KernelNodeParams()
        _ok(get(ctypes.c_void_p(nodes[i]), ctypes.byref(p)), "cuGraphKernelNodeGetParams")
        out.append((p.func, (p.gridDimX, p.gridDimY, p.gridDimZ), (p.blockDimX, p.blockDimY, p.blockDimZ),
                    p.sharedMemBytes))
    return out


class _FieldValue(ctypes.Structure):            # nvmlFieldValue_t
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int),
                ("value", ctypes.c_ulonglong)]


class ClockSampler(threading.Thread):
    """NVML samples every ``period`` seconds while :attr:`tag` names the arm replaying (None between replays).
    :meth:`summary` gives each arm's median SM clock, memory clock and power, and its clock-event reasons."""

    def __init__(self, index: int = 0, period: float = 0.002):
        super().__init__(daemon=True)
        self.tag, self.period, self.samples, self.error = None, period, [], None
        self._stop_evt = threading.Event()
        try:
            self.nv = ctypes.CDLL("libnvidia-ml.so.1")
            _ok(self.nv.nvmlInit_v2(), "nvmlInit_v2")
            self.h = ctypes.c_void_p(0)
            _ok(self.nv.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self.h)),
                "nvmlDeviceGetHandleByIndex_v2")
            self.reasons = (getattr(self.nv, "nvmlDeviceGetCurrentClocksEventReasons", None)
                            or self.nv.nvmlDeviceGetCurrentClocksThrottleReasons)
        except Exception as e:                    # noqa: BLE001  (recorded: the reducer labels every arm "clock unread")
            self.error = f"{type(e).__name__}: {e}"

    def _read(self):
        sm, mem, mw, rs, lim = (ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_uint(0), ctypes.c_ulonglong(0),
                                ctypes.c_uint(0))
        self.nv.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(sm))
        self.nv.nvmlDeviceGetClockInfo(self.h, 2, ctypes.byref(mem))
        self.nv.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(mw))
        self.reasons(self.h, ctypes.byref(rs))
        self.nv.nvmlDeviceGetEnforcedPowerLimit(self.h, ctypes.byref(lim))
        fv = _FieldValue(fieldId=_FI_POWER_INSTANT)
        inst = None
        if self.nv.nvmlDeviceGetFieldValues(self.h, 1, ctypes.byref(fv)) == 0 and fv.nvmlReturn == 0:
            inst = (fv.value & 0xFFFFFFFF) / 1000.0          # an unsigned int in the union's low word
        return sm.value, mem.value, mw.value / 1000.0, rs.value, lim.value / 1000.0, inst

    def run(self):
        if self.error:
            return
        while not self._stop_evt.is_set():
            tag = self.tag
            if tag is not None:
                try:
                    self.samples.append((tag, *self._read()))
                except Exception as e:            # noqa: BLE001
                    self.error = f"{type(e).__name__}: {e}"
                    return
            time.sleep(self.period)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)

    def summary(self) -> dict:
        out = {"error": self.error, "n": len(self.samples), "period_s": self.period}
        arms = sorted({s[0] for s in self.samples})
        for arm in arms:
            xs = [s for s in self.samples if s[0] == arm]
            reasons = {}
            for s in xs:
                reasons[hex(s[4])] = reasons.get(hex(s[4]), 0) + 1
            inst = [s[6] for s in xs if s[6] is not None]
            out[arm] = {"n": len(xs), "sm_mhz_median": statistics.median(s[1] for s in xs),
                        "sm_mhz_min": min(s[1] for s in xs), "sm_mhz_max": max(s[1] for s in xs),
                        "mem_mhz_median": statistics.median(s[2] for s in xs),
                        "power_w_median": statistics.median(s[3] for s in xs),
                        "power_instant_w_median": statistics.median(inst) if inst else None,
                        "power_instant_w_max": max(inst) if inst else None,
                        "enforced_limit_w": sorted({s[5] for s in xs}), "reasons": reasons}
        return out
