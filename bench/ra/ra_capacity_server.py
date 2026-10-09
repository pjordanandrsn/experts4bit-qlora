#!/usr/bin/env python3
"""Owned RA capacity server: release defaults, frozen counters before capture."""
from __future__ import annotations

import argparse
import copy
import ctypes
import os
import signal
import socket
import sys
from pathlib import Path

import ra_env
import ra_serving
import ra_stage
import ra_trace


def parent_death_guard(parent_pid):
    if sys.platform != "linux" or ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("Linux parent-death guard unavailable")
    if os.getppid() != parent_pid:
        raise RuntimeError("capacity supervisor already exited")


def instrumented_app(server, instrument, cfg, *, listener=None):
    original = server.build_engine
    observed = {}
    attempted = False

    def build(config):
        nonlocal attempted
        if attempted:
            raise RuntimeError("capacity engine may build only once")
        attempted = True
        parts, kernels, forwards = ra_serving.build_instrumented(server, instrument, config, builder=original, listener=listener)
        observed.update(parts=parts, kernels=kernels, forwards=forwards)
        return parts

    server.build_engine = build
    app = server.create_app(cfg)

    @app.post("/_ra/close-trace")
    def close_trace(timeout: float):
        return ra_trace.close_engine(app.state.engine, timeout)

    @app.get("/_ra/evidence")
    def evidence():
        if not observed:
            return {"ready": False}
        parts = observed["parts"]
        return {"ready": True, "scope": "build-capture+warm+burst+point",
                "config": {k: v for k, v in vars(cfg).items() if k != "token"},
                "fusions": {k: copy.deepcopy(parts.info[k]) for k in instrument.CENSUS_KEYS},
                "fusion_modes": copy.deepcopy(parts.info["fusion_modes"]),
                "resolved_defaults": copy.deepcopy(observed["forwards"].ra_defaults),
                "kernels": observed["kernels"].snapshot(), "forward_counts": observed["forwards"].snapshot()}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--socket-fd", required=True, type=int)
    ap.add_argument("--parent-pid", required=True, type=int)
    ap.add_argument("--instruments", required=True, type=Path)
    args = ap.parse_args()
    parent_death_guard(args.parent_pid)
    if not args.instruments.is_absolute():
        raise ValueError("absolute instrument path required")
    ra_env.check_current("capacity", os.environ)
    ra_stage.verify(args.instruments)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.instruments))
    import torch
    import uvicorn
    import p115_quality
    from experts4bit_qlora import serve_paged
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one usable CUDA GPU required")
    prop = torch.cuda.get_device_properties(0)
    if "RTX 5090" not in prop.name or (prop.major, prop.minor) != (12, 0):
        raise ValueError("registered RTX 5090/sm_120 required")
    listener = socket.socket(fileno=args.socket_fd)
    if listener.family != socket.AF_INET or listener.getsockname()[0] != "127.0.0.1" or \
            not listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN):
        raise ValueError("owned loopback listener required")
    cfg = serve_paged.PagedServeConfig.from_env()
    cfg.host, cfg.port = listener.getsockname()
    app = instrumented_app(serve_paged, p115_quality, cfg, listener=listener)
    uvicorn.Server(uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="info")).run(sockets=[listener])


if __name__ == "__main__":
    main()
