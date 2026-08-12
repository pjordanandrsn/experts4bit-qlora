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

    def reset(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Re-prefill and rewind to the post-prompt state; returns the first token.

        Required between generations: `step()` advances `cache_position` and never
        rewinds, so a second run continues past the end of a `max_length`-sized
        cache and trips `index_copy_(): index out of bounds`. The graph itself is
        reusable — only the cache and the position need rewinding.
        """
        self.cache.reset()
        n = input_ids.shape[-1]
        pos = torch.arange(n, device=input_ids.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=self.cache,
                             cache_position=pos, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
        self.cur_token.copy_(nxt)
        self.cache_position.fill_(n)
        return nxt

    def logits_for(self, token: torch.Tensor) -> torch.Tensor:
        """Replay on `token` and return the raw logits instead of the argmax.

        Same graph, same advance -- only the projection to a token is skipped, so
        callers can compare distributions rather than argmax winners.
        """
        self.cur_token.copy_(token.view(1, 1))
        self.graph.replay()
        self.cache_position += 1
        return self.logits[:, -1].clone()

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


def _eager_greedy(model, input_ids, n_new: int, forced=None):
    """Incremental eager greedy decode against a StaticCache.

    `generate()` cannot serve as the reference here. It applies stopping criteria
    and generation-config processors, while the captured path emits exactly
    `n_new` raw argmax steps -- so an early EOS shows up as a token-stream
    mismatch that has nothing to do with capture, and a bit-identical
    teacher-forced delta would then relabel that length difference a "tie".
    Decoding here by the same rule removes the confound instead of excusing it.

    It must also be INCREMENTAL. Full-sequence attention and single-token
    attention use different kernels and reduction orders and differ by a few ulp
    in pure eager; comparing against a one-shot forward charges that to capture.

    With `forced`, teacher-forces those tokens instead of its own argmax and
    returns the logits produced at each step.
    """
    n = input_ids.shape[-1]
    cache = _static_cache(model, n + n_new + 2)
    toks, logits = [], []
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache,
                    cache_position=torch.arange(n, device=input_ids.device), use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
        for j in range(n_new):
            tok = nxt if forced is None else torch.tensor(
                [[forced[j]]], dtype=input_ids.dtype, device=input_ids.device)
            toks.append(int(tok))
            if j == n_new - 1:
                break
            out = model(input_ids=tok, past_key_values=cache,
                        cache_position=torch.tensor([n + j], device=input_ids.device),
                        use_cache=True)
            logits.append(out.logits[0, -1].float())
            nxt = out.logits[:, -1:].argmax(-1)
    return toks, logits


def probe_capture(model, input_ids, max_new_tokens: int = 16) -> dict:
    """Does capture work on THIS model, and does replay match eager?

    A captured graph that reads a stale pointer replays without error and returns
    plausible-looking tokens, so 'it ran' is not evidence. The gate is token-stream
    equality against eager decode driven by the SAME rule -- raw argmax, fixed
    length, no stopping criteria -- so a mismatch can only mean capture.
    """
    report = {"captured": False, "matches_eager": None, "n_new": max_new_tokens,
              "error": None, "eager": None, "replayed": None}
    n = input_ids.shape[-1]
    try:
        report["eager"], _ = _eager_greedy(model, input_ids, max_new_tokens)
    except Exception as e:                          # a model that cannot even decode
        report["error"] = f"eager decode failed: {type(e).__name__}: {e}"
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
    if not report["matches_eager"]:
        # Teacher-forcing FIRST: it is the authoritative test. The logit-gap
        # heuristic below cannot tell a tie from a defect on its own -- on a
        # random-weight fixture it called a 1.7-ulp gap "real" while replay was
        # in fact bit-identical to eager. Gap only adds colour once this decides.
        tf = _teacher_forced_delta(model, dec, input_ids, report["eager"])
        report["teacher_forced"] = tf
        div = _classify_divergence(model, input_ids, report["eager"], got)
        if tf.get("max_abs_delta") == 0.0 and tf.get("steps"):
            div["verdict"] = "tie (replay bit-identical to eager under teacher forcing)"
        report["divergence"] = div
    return report


def _teacher_forced_delta(model, dec, input_ids, eager_tokens) -> dict:
    """Max |logits_replay - logits_eager| when BOTH are fed the same tokens.

    Free-running argmax equality conflates two things: whether replay computes
    the same function, and whether the model has ties to trip over. Random-weight
    fixtures are almost all ties, so that test says nothing about capture there.
    Forcing the same token stream down both paths removes the compounding and
    isolates the only question capture can answer wrongly -- does the graph
    compute what eager computes.
    """
    out = {"max_abs_delta": None, "steps": 0, "note": None}
    try:
        _, ref = _eager_greedy(model, input_ids, len(eager_tokens), forced=eager_tokens)
        dec.reset(input_ids)
        worst, steps = 0.0, 0
        for j, t in enumerate(eager_tokens[:-1]):
            got = dec.logits_for(torch.tensor([[t]], dtype=input_ids.dtype,
                                              device=input_ids.device))
            if j < len(ref):
                worst = max(worst, float((got.float() - ref[j]).abs().max()))
                steps += 1
        out["max_abs_delta"], out["steps"] = worst, steps
    except Exception as e:
        out["note"] = f"{type(e).__name__}: {e}"[:140]
    return out


def _classify_divergence(model, input_ids, eager, got) -> dict:
    """Locate the first mismatched token and describe how close that argmax was.

    Deliberately does NOT decide tie-vs-defect. It used to, off a one-ulp gap
    threshold, and it was wrong in the direction that matters: it labelled a
    bit-identical replay "real". The gap and the ulp are useful colour -- they say
    whether the two tokens were in a photo finish -- but the verdict comes from
    `_teacher_forced_delta` alone, and stays `undetermined` when that cannot run.
    A confident wrong answer is worse here than no answer.
    """
    i = next((j for j in range(min(len(eager), len(got))) if eager[j] != got[j]), None)
    out = {"index": i, "eager_token": None, "replay_token": None,
           "gap": None, "verdict": "unknown"}
    if i is None:
        out["verdict"] = "length mismatch only"
        return out
    out["eager_token"], out["replay_token"] = eager[i], got[i]
    try:
        prefix = torch.cat(
            [input_ids, torch.tensor([eager[:i]], dtype=input_ids.dtype,
                                     device=input_ids.device)], dim=-1)
        with torch.no_grad():
            logits = model(input_ids=prefix, use_cache=False).logits[0, -1].float()
        gap = float(abs(logits[eager[i]] - logits[got[i]]))
        out["gap"] = gap
        # Reported as CONTEXT, never as a verdict. A one-ulp threshold cannot
        # separate a tie from a defect: measured here, a 0.0039 gap against a
        # 0.0023 ulp -- 1.7x, comfortably "real" by any such rule -- sat on a replay
        # that was bit-identical to eager. The gap says how close the race was, not
        # who was right. Only `_teacher_forced_delta` can answer that, and when it
        # cannot run there is no answer to give.
        out["ulp"] = float(max(abs(logits[eager[i]]), abs(logits[got[i]]))) * 2 ** -8
        out["verdict"] = "undetermined (teacher-forced comparison is the only verdict)"
    except Exception as e:
        out["verdict"] = f"unclassifiable: {type(e).__name__}: {e}"[:120]
    return out
