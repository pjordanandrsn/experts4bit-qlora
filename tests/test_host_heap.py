"""engines.host_heap: the int4 levers hand their freed host heap back. Where the C library has no ``malloc_trim`` the
call is a no-op that says so."""
import ctypes

import pytest

from experts4bit_qlora.engines import host_heap


def test_release_is_a_bool_and_safe_to_repeat():
    assert host_heap.release_freed_host_heap() in (True, False)
    assert host_heap.release_freed_host_heap() in (True, False)


def test_without_malloc_trim_it_does_nothing(monkeypatch):
    monkeypatch.setattr(host_heap, "_TRIM", [None])
    assert host_heap.release_freed_host_heap() is False


def test_on_glibc_the_trim_is_found(monkeypatch):
    try:
        ctypes.CDLL("libc.so.6").malloc_trim
    except (OSError, AttributeError):
        pytest.skip("no glibc malloc_trim here")
    monkeypatch.setattr(host_heap, "_TRIM", [])
    host_heap.release_freed_host_heap()
    assert host_heap._TRIM[0] is not None


def test_releasing_cached_pinned_memory_is_safe_anywhere(monkeypatch):
    """Without CUDA (or a torch without the private hook) it does nothing and says so; with it, it runs the hook."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert host_heap.release_cached_pinned_memory() is False
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: calls.append(1), raising=False)
    assert host_heap.release_cached_pinned_memory() is True and calls == [1]


def test_the_server_build_hands_its_freed_heap_back():
    """build_engine trims once it is built and reports it (``host_heap_trimmed`` in its info): measured, the build
    leaves 0.34 GB (OLMoE-1B-7B, RTX A2000 host) of freed heap resident otherwise."""
    import inspect

    pytest.importorskip("torch")
    from experts4bit_qlora import serve_paged

    src = inspect.getsource(serve_paged.build_engine)
    trim, ready = src.find("release_freed_host_heap()"), src.find('log(f"ready:')
    assert 0 < trim < ready and '"host_heap_trimmed"' in src
    pinned = src.find("release_cached_pinned_memory()")
    assert 0 < pinned < ready and '"pinned_cache_released"' in src
