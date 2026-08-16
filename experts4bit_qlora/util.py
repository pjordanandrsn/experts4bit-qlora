import os
import time


def log(msg: str) -> None:
    """Timestamped, flushed stdout line (so progress shows up promptly under ``python -u``)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


#: cgroup layouts, newest first: (limit, usage, stat, page-cache key in stat).
_CGROUP_LAYOUTS = (
    ("v2", "memory.max", "memory.current", "memory.stat", "file"),
    ("v1", "memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes",
     "memory/memory.stat", "cache"),
)

#: cgroup v1 spells "unlimited" as a near-2**63 sentinel rather than a word.
_V1_UNLIMITED = 2 ** 62


def container_free_bytes(root: str = "/sys/fs/cgroup") -> tuple[int | None, dict]:
    """Host memory actually available to THIS container, with the page cache counted free.

    Returns ``(bytes_or_None, detail)``. ``None`` means "could not measure" — an
    unlimited cgroup, no memory controller, or an unreadable file — and the caller
    must fall back explicitly rather than treat it as zero.

    Two traps, both of which have cost a rented run:

    * **``free``/``psutil`` report the HOST.** A pod showed 256 cores and 1 TB
      where the real limits were 27.2 CPUs and 125 GB. Sizing ``hot_rows`` off the
      host figure pins more DRAM than exists.
    * **The page cache is counted as USED.** Both cgroup versions include it in
      current usage, so straight after writing a 138 GiB arena
      ``limit - current`` read **18.3 MB** — and a caller that believed it fell
      back to an unvalidated floor. The cache is reclaimable, so it is added back.

    Both v2 and v1 are handled because both turn up in practice: RunPod's
    `runpod/pytorch` pods and QNAP's Container Station are **v1**, while most
    modern hosts are v2. A v2-only reader silently measures nothing on either.

    ``root`` is a parameter so this is testable without a container.
    """
    for ver, lim_p, cur_p, stat_p, cache_key in _CGROUP_LAYOUTS:
        lim_f = os.path.join(root, lim_p)
        if not os.path.exists(lim_f):
            continue
        try:
            raw = open(lim_f).read().strip()
            if raw == "max":
                return None, {"cgroup": ver, "limit": "max (unlimited)"}
            limit = int(raw)
            if limit >= _V1_UNLIMITED:
                return None, {"cgroup": ver, "limit": limit, "note": "unlimited sentinel"}
            current = int(open(os.path.join(root, cur_p)).read().strip())
            cache = 0
            for line in open(os.path.join(root, stat_p)):
                key, _, val = line.partition(" ")
                if key == cache_key:
                    cache = int(val)
                    break
            in_use = max(0, current - cache)
            return max(0, limit - in_use), {
                "cgroup": ver, "limit": limit, "current": current,
                "page_cache": cache, "in_use": in_use,
            }
        except Exception as exc:                       # unreadable / malformed
            return None, {"cgroup": ver, "error": f"{type(exc).__name__}: {exc}"}
    return None, {"error": f"no cgroup v2 or v1 memory controller under {root}"}
