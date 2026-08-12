# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""CUDA-graph capture of the decode step, and a probe that says whether a model supports it.

Why: #108 measured kernel LAUNCHES as the dominant recurring per-token host cost
(288.8 ms of 562.4 ms host, 45,501 launches on OLMoE/3090). Graph replay collapses
that to one launch per step. `torch.compile(mode="reduce-overhead")` cannot be
borrowed for it -- inductor dies on the engine's layout with
``aot_autograd() does not yet handle input mutations on views with different
dtypes``, which is exactly the one-uint8-store-viewed-as-int64-and-float32
row-block. Raw ``torch.cuda.CUDAGraph`` capture of the engine step already works
and is tested, so this builds on that instead.

The blocker was dynamic shapes: a growing KV cache gave 51 distinct sizes, one
graph each. ``StaticCache`` preallocates to ``max_length`` so every decode step
has identical shapes and ONE graph serves the whole generation. The cost is real
and worth stating: the cache is allocated for the full length up front.

**Support is not assumed.** Capture fails loudly on a host sync or a
data-dependent launch inside the region, and a graph that captures a stale
pointer replays happily while returning garbage -- so :func:`probe_capture`
verifies replay against eager and REFUSES rather than degrading. Architectures
differ in ways that matter here (sliding-window and SSM state are fixed-size and
should be favourable; hybrids need attention cache and recurrent state handled
together), so the support matrix is measured per model, never predicted.
"""
from __future__ import annotations

import torch


def _static_cache(model, max_length: int, batch: int = 1):
    from transformers import StaticCache
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return StaticCache(config=cfg, max_batch_size=batch, max_cache_len=max_length,
                       device=model.device, dtype=next(model.parameters()).dtype)


class CapturedDecoder:
    """One captured decode step, replayed per token.

    Buffers are persistent and the graph reads them by address, so a step is
    ``copy_`` the new token in, bump the position, ``replay()``, read the logits.
    """

    def __init__(self, model, cache, cur_token, cache_position, graph, logits):
        self.model, self.cache = model, cache
        self.cur_token, self.cache_position = cur_token, cache_position
        self.graph, self.logits = graph, logits

    def step(self, token: torch.Tensor) -> torch.Tensor:
        # Replay FIRST, then advance. `cache_position` already points at the slot
        # this token belongs in; incrementing before the replay would write it one
        # slot too far and leave a hole. (That off-by-one produced a stream that
        # diverged on the very first replayed token.)
        self.cur_token.copy_(token.view(1, 1))
        self.graph.replay()
        self.cache_position += 1
        return self.logits[:, -1:].argmax(-1)


def capture_decode(model, input_ids, max_length: int, warmup: int = 3):
    """Prefill eagerly, then capture ONE decode step against static buffers.

    Returns ``(CapturedDecoder, first_token)``. Raises if capture is not possible
    on this model -- a host sync or data-dependent launch inside the region makes
    ``torch.cuda.graph`` throw, which is the honest signal that the model is not
    supported rather than something to work around.
    """
    cache = _static_cache(model, max_length)
    n = input_ids.shape[-1]
    pos = torch.arange(n, device=model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, cache_position=pos, use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)

    cur_token = nxt.clone()                                   # static input buffer
    cache_position = torch.tensor([n], device=model.device)   # static, bumped in place

    # Warm up on a side stream: allocator and any lazy buffers must be settled
    # before capture, or the graph bakes an address that is about to change.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(warmup):
            model(input_ids=cur_token, past_key_values=cache,
                  cache_position=cache_position, use_cache=True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        captured = model(input_ids=cur_token, past_key_values=cache,
                         cache_position=cache_position, use_cache=True)

    # Warmup and capture each ran a real forward, and each WROTE into the cache at
    # `cache_position`. That state is garbage for generation, so re-prefill to put
    # the cache back to exactly the post-prompt state before any token is stepped.
    # Skipping this leaves the first generated token reading polluted slots.
    cache.reset()
    with torch.no_grad():
        out2 = model(input_ids=input_ids, past_key_values=cache,
                     cache_position=pos, use_cache=True)
    nxt = out2.logits[:, -1:].argmax(-1)
    cur_token.copy_(nxt)
    cache_position.fill_(n)
    return CapturedDecoder(model, cache, cur_token, cache_position, g, captured.logits), nxt


def probe_capture(model, input_ids, max_new_tokens: int = 16) -> dict:
    """Does capture work on THIS model, and does replay match eager?

    A captured graph that reads a stale pointer replays without error and returns
    plausible-looking tokens, so 'it ran' is not evidence. The gate is token-stream
    equality against an eager generate on the same inputs.
    """
    report = {"captured": False, "matches_eager": None, "n_new": max_new_tokens,
              "error": None, "eager": None, "replayed": None}
    n = input_ids.shape[-1]
    try:
        with torch.no_grad():                      # eager reference, same greedy path
            ref = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, use_cache=True)
        report["eager"] = ref[0, n:].tolist()
    except Exception as e:                          # a model that cannot even generate
        report["error"] = f"eager generate failed: {type(e).__name__}: {e}"
        return report

    try:
        dec, first = capture_decode(model, input_ids, max_length=n + max_new_tokens + 2)
        report["captured"] = True
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return report

    got = [int(first)]
    tok = first
    for _ in range(max_new_tokens - 1):
        tok = dec.step(tok)
        got.append(int(tok))
    torch.cuda.synchronize()
    report["replayed"] = got
    report["matches_eager"] = (got == report["eager"])
    return report
