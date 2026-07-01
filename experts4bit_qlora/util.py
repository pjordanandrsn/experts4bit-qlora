import time


def log(msg: str) -> None:
    """Timestamped, flushed stdout line (so progress shows up promptly under ``python -u``)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
