# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Chunked causal-LM loss for TRAINING (``E4B_CHUNKED_LM_LOSS``, ``auto`` by default): the full-vocabulary logits are never materialised.

Hugging Face's causal-LM loss (``ForCausalLMLoss``) runs the LM head over every token, upcasts the whole ``[tokens, vocab]``
logits to fp32 and takes the cross-entropy over them; autograd keeps the fp32 log-probabilities for backward and builds fp32 and
bf16 gradients of the same size. At Qwen3's vocabulary (151,936) one fp32 copy of a 4,096-token row is 2.32 GiB -- the exact
allocation a rented RTX 5090 failed on at step 1 of TC1 amendment 39's packed 4,096-token regime (box ``tc1-5090-86``, 29.5 of
31.36 GiB in use). This module computes the same loss over chunks of tokens: per chunk, ``lm_head`` -> the architecture's own
logits transform -> fp32 -> cross-entropy (sum), each chunk under :func:`torch.utils.checkpoint.checkpoint`, so nothing of a chunk
is kept for backward and backward recomputes one chunk at a time. The loss's peak is one chunk's logits, not the row's.

Semantics are Hugging Face's, term by term: labels shifted by one inside each row (or ``shift_labels`` as given), ``ignore_index``
(-100 unless the caller passes another), the mean over the non-ignored tokens -- or the sum over ``num_items_in_batch`` when the
caller passes it -- and the router auxiliary loss added the way the architecture's forward adds it. Rows whose label is ignored
are not run through the head at all (their stock contribution to the loss and to every gradient is exactly zero); finding them
costs one host sync per training forward, after the decoder. What differs from the stock path:

- the summation order of the loss (per-chunk sums, then their sum) and of a trainable head's weight gradient: fp32 rounding,
  test-pinned in ``tests/test_chunked_lm_loss.py``;
- on CUDA, the head's backward matmul runs at a chunk's row count rather than the row's, and cuBLAS picks its bf16 split-K
  reduction by shape (``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction``, on by default). On an RTX A2000
  the hidden-state gradient differed from stock in about a third of its elements by bf16 rounding (relative L2 3-5e-3), and
  was ``torch.equal`` to stock with that flag off;
- time: one extra LM-head matmul and cross-entropy forward per chunk in backward (the recompute). Its size on a target card
  is a registered lane's to read; the RTX A2000 this was built on is a correctness testbed.

Which forwards take it: a forward that passes ``labels`` with gradients enabled, ``logits_to_keep`` 0 and the dict return.
Everything else -- generation, a ``torch.no_grad`` evaluation, ``return_dict=False`` -- runs the stock forward untouched, so a
held-out loss is the stock path's bit for bit. A forward that takes it returns the model's own output class with ``.loss`` set
and ``.logits`` None: nothing on e4b's training path or the TC1 harness reads the logits of a training forward.

Held-out evaluation, on by default since TC1 amendment 60 (``E4B_CHUNKED_EVAL_LOSS`` unset or ``1``; ``0``: the stock forward, as above).
A ``torch.no_grad`` forward
that passes ``labels`` -- an evaluation loop's -- builds the full logits AND Hugging Face's loss upcasts them to fp32 and takes the
log-softmax of the copy: on one packed 4,096-token row at Qwen3's vocabulary that is 1.16 GiB of bf16 logits plus 2 x 2.32 GiB of
fp32. On TC1's packed rows that evaluation is e4b's run peak: 26.88 GB against a training-phase peak of 26.59 (25.85 with
``E4B_CKPT_OFFLOAD=1``), TC1 amendment 58's read. With the switch on, such a forward whose stock fp32 logits would reach
:data:`EVAL_MIN_LOGITS_BYTES` runs WITHOUT ``labels`` -- so its logits are the stock forward's, bit for bit, and are returned as
usual -- and the loss is then computed from those logits in fp32 chunks of the chunk size (:func:`chunked_lm_loss_from_logits`),
with the router auxiliary loss added as the forward adds it. Nothing but the loss's fp32 summation order differs, and only above the
gate. Under the gate, without labels, with ``logits_to_keep``, a tuple return, or a model the table refused, the forward is the
stock one. The eval path rides on the training patch: ``E4B_CHUNKED_LM_LOSS=0`` patches nothing, so evaluation is stock whatever
this variable says. TC1 amendment 60 read it on Qwen3-30B-A3B's packed rows on one RTX 5090 in torch 2.12
(``e4b.train.chunked-eval-loss.packed-4k.5090.2026-10-07``): the evaluation-phase peak fell 26.88 -> 22.50 GB, below the 25.85 GB
training phase, with step-0 held-out identical and held-out at N within 0.00005. A held-out loss above the gate therefore differs
from the stock one by fp32 summation order only; one compared across this default (before / after, or against ``0``) is not
byte-identical there. Under the gate -- TC1's field-recipe evaluation rows peak at 411 tokens, 0.23 GiB -- nothing changes.

Which models: the Hugging Face causal-LM classes in :data:`SUPPORTED`. The post-``lm_head`` code of each was read in
transformers' source, and each is test-pinned against the stock loss and gradients on a tiny config. Anything else is REFUSED (a
``RuntimeWarning`` with the reason; the model keeps its stock loss): a class not in the table (Gemma-4's final-logit softcap
among them), a forward or ``loss_function`` replaced by another library, an ``lm_head`` already patched or hooked, a head whose
width is not the vocabulary. Every training forward also re-checks the transform at run time: ``lm_head`` hands the forward a
one-element probe, and the logits the forward returns must be exactly the table's transform of it (the probe object itself for
an identity head). Any other change made after ``lm_head`` disables the chunked loss for that model with a warning, and the call
re-runs stock.

``E4B_CHUNKED_LM_LOSS`` = ``auto`` (the DEFAULT, also when unset): chunks of :data:`DEFAULT_CHUNK` tokens, but only for a training
forward whose stock fp32 logits (positions x vocabulary x 4 bytes) would reach :data:`AUTO_MIN_LOGITS_BYTES`; a smaller one runs the
stock forward untouched (``small_calls``). ``1`` or a chunk size in tokens: every training forward chunks. ``0``: the stock loss
everywhere, nothing patched. Unset, a model the table refuses keeps its stock loss without a warning (the switch was not asked for);
set, the refusal warns. TC1 amendment 44 made ``auto`` the default (with amendment 43's packed rows fitting under it). At the
field recipe, where a micro-batch's logits are 0.3-0.6 GiB, chunking is a cost: 1.049 of the shipped arm's step on a host-bound
RTX 5090 (TC1 amendment 41). On packed 4,096-token rows (2.32 GiB a row at Qwen3's vocabulary) it is what lets e4b train at all
(amendments 39 and 40). The gate separates the two by size alone.
``enable_fast_train`` applies it and ``disable_fast_train`` unwinds it, like the other training-path switches; the CLI trainer
(``python -m experts4bit_qlora.train``) applies it itself; :func:`enable_chunked_lm_loss` is the direct call.
"""
from __future__ import annotations

import functools
import inspect
import os
import warnings

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "AUTO_MIN_LOGITS_BYTES",
    "CHUNKED_LM_LOSS_STATS",
    "DEFAULT_CHUNK",
    "SUPPORTED",
    "EVAL_MIN_LOGITS_BYTES",
    "chunked_causal_lm_loss",
    "chunked_eval_loss_requested",
    "chunked_lm_loss_from_logits",
    "chunked_lm_loss_by_default",
    "chunked_lm_loss_min_bytes",
    "chunked_lm_loss_refusal",
    "chunked_lm_loss_requested",
    "disable_chunked_lm_loss",
    "enable_chunked_lm_loss",
]

#: Tokens per chunk when ``E4B_CHUNKED_LM_LOSS=1``. On an RTX A2000 at Qwen3's vocabulary (151,936) and hidden 2048 the loss's
#: own peak is 0.90 GiB at 512-token chunks, 1.77 at 1,024 and 8.11 stock, at 4,096 tokens (allocator bytes;
#: bench/chunked-lm-loss/receipts/lm_head_loss_a2000.json).
DEFAULT_CHUNK = 512

#: ``E4B_CHUNKED_LM_LOSS=auto``'s gate: a training forward chunks when its stock fp32 logits would take at least this many bytes.
#: 1 GiB sits between TC1's field recipe (Qwen3's Alpaca micro-batches: 0.29 GiB median, 0.64 GiB largest over a 60-step run,
#: 0.86 GiB for the two longest rows of the 1,200) and one packed 4,096-token row (2.32 GiB). Mixtral's 32,000-token vocabulary
#: stays under it at 4,096 tokens (0.49 GiB): its logits are not the problem there.
AUTO_MIN_LOGITS_BYTES = 1 << 30

#: ``E4B_CHUNKED_EVAL_LOSS``'s gate: a no-grad forward with labels computes its loss from the logits in chunks when its stock
#: fp32 logits would take at least this many bytes (the training gate's value: TC1's field-recipe evaluation rows stay stock).
EVAL_MIN_LOGITS_BYTES = AUTO_MIN_LOGITS_BYTES

#: ``patched`` models, training forwards that took the chunked loss / ran stock (no labels, ``logits_to_keep``, a tuple return) /
#: ran stock under ``auto``'s size gate (``small_calls``), run-time refusals, and enable-time refusals by class name -> reason, so a
#: training census can say the switch served the step. With the eval switch on (the default): no-grad forwards with labels whose loss
#: was computed from the logits in chunks (``eval_chunked_calls``) / that ran stock (``eval_stock_calls``: under the gate,
#: ``logits_to_keep``, a tuple return).
CHUNKED_LM_LOSS_STATS = {"patched": 0, "chunked_calls": 0, "stock_calls": 0, "small_calls": 0, "runtime_refusals": 0,
                         "refused": {}, "eval_chunked_calls": 0, "eval_stock_calls": 0}

_ENV = "E4B_CHUNKED_LM_LOSS"
_EVAL_ENV = "E4B_CHUNKED_EVAL_LOSS"


def _identity(logits, config):
    return logits


def _logits_scaling(logits, config):
    return logits / config.logits_scaling                 # Granite: `logits = logits / self.config.logits_scaling`


def _to_fp32(logits, config):
    return logits.float()                                 # Nemotron-H: `self.lm_head(...).float()`


#: Hugging Face causal-LM class -> (transformers model package, the forward's post-``lm_head`` transform, the attribute its
#: forward scales the router auxiliary loss by, or None when it adds none). Read in transformers 5.x's modeling source; every
#: row is pinned by tests/test_chunked_lm_loss.py (loss and every gradient against the stock forward, aux loss on).
SUPPORTED = {
    "Qwen3MoeForCausalLM": ("qwen3_moe", _identity, "router_aux_loss_coef"),
    "Qwen3ForCausalLM": ("qwen3", _identity, None),          # dense; transformers 5.18: logits = lm_head(h), no aux (DQ4)
    "Qwen3_5MoeForCausalLM": ("qwen3_5_moe", _identity, "router_aux_loss_coef"),
    "MixtralForCausalLM": ("mixtral", _identity, "router_aux_loss_coef"),
    "OlmoeForCausalLM": ("olmoe", _identity, "router_aux_loss_coef"),
    "GptOssForCausalLM": ("gpt_oss", _identity, "router_aux_loss_coef"),
    "Ernie4_5_MoeForCausalLM": ("ernie4_5_moe", _identity, "router_aux_loss_coef"),
    "GraniteMoeForCausalLM": ("granitemoe", _logits_scaling, "router_aux_loss_coef"),
    "GraniteMoeSharedForCausalLM": ("granitemoeshared", _logits_scaling, "router_aux_loss_coef"),
    "GraniteMoeHybridForCausalLM": ("granitemoehybrid", _logits_scaling, "router_aux_loss_coef"),
    "Lfm2MoeForCausalLM": ("lfm2_moe", _identity, None),
    "NemotronHForCausalLM": ("nemotron_h", _to_fp32, None),
}


def chunked_lm_loss_requested():
    """``None`` (off) or the chunk size in tokens, from ``E4B_CHUNKED_LM_LOSS`` (unset / ``auto`` / ``1`` / ``on`` =
    :data:`DEFAULT_CHUNK`; ``0`` / ``off`` = None)."""
    v = os.environ.get(_ENV, "").strip().lower()
    if v in ("0", "off", "false", "no", "none"):
        return None
    if v in ("", "1", "on", "true", "yes", "auto"):
        return DEFAULT_CHUNK
    try:
        n = int(v)
    except ValueError:
        raise ValueError(f"{_ENV} must be 0 / 1 / auto or a chunk size in tokens, got {v!r}") from None
    if n < 2:
        raise ValueError(f"{_ENV} must be 0 / 1 or a chunk size in tokens (>= 2), got {v!r}")
    return n


def chunked_lm_loss_min_bytes():
    """``auto``'s size gate (:data:`AUTO_MIN_LOGITS_BYTES`) when ``E4B_CHUNKED_LM_LOSS`` is ``auto`` or unset (the default), else
    ``None`` (every training forward chunks)."""
    return AUTO_MIN_LOGITS_BYTES if os.environ.get(_ENV, "").strip().lower() in ("", "auto") else None


def chunked_lm_loss_by_default():
    """True when ``E4B_CHUNKED_LM_LOSS`` is unset or empty: the switch runs as e4b's default, not because it was asked for, so a
    refusal is recorded without a warning."""
    return os.environ.get(_ENV, "").strip() == ""


def chunked_eval_loss_requested():
    """``E4B_CHUNKED_EVAL_LOSS``: unset / ``1`` / ``on`` True (the default since TC1 amendment 60), ``0`` / ``off`` False (evaluation
    runs stock)."""
    v = os.environ.get(_EVAL_ENV, "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return False
    if v in ("", "1", "on", "true", "yes"):
        return True
    raise ValueError(f"{_EVAL_ENV} must be 0 or 1, got {v!r}")


def _chunk_ce_sum(h, y, *, head, transform, config, ignore_index):
    """One chunk of Hugging Face's loss: ``lm_head``, the architecture's transform, the fp32 upcast, cross-entropy summed."""
    return F.cross_entropy(transform(head(h), config).float(), y, ignore_index=ignore_index, reduction="sum")


def chunked_causal_lm_loss(hidden, labels, head, *, chunk=DEFAULT_CHUNK, transform=_identity, config=None,
                           num_items_in_batch=None, ignore_index=-100, shift_labels=None):
    """``ForCausalLMLoss(transform(head(hidden)), labels, ...)`` without materialising the logits.

    ``hidden`` is the decoder's final hidden states ``[batch, seq, hidden]`` (what the forward hands ``lm_head``), ``labels`` the
    forward's ``[batch, seq]`` labels, shifted here exactly as Hugging Face shifts them (padded with ``ignore_index`` at the end
    of each row, then ``[..., 1:]``) unless ``shift_labels`` is given. Rows with an ignored label never reach the head. The
    supervised rows go through ``head`` -> ``transform`` -> fp32 -> cross-entropy (sum) ``chunk`` at a time, each chunk under
    non-reentrant :func:`~torch.utils.checkpoint.checkpoint` when a gradient is wanted, so a chunk's logits are dropped after its
    forward and recomputed for its backward (the autocast and RNG state restored as the checkpoint does). The sum is then
    divided as Hugging Face divides it: by the number of supervised tokens (``mean``; nan when there are none, as there), or by
    ``num_items_in_batch`` when given."""
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    y = shift_labels.reshape(-1).to(hidden.device)
    h = hidden.reshape(-1, hidden.shape[-1])
    if y.numel() != h.shape[0]:
        raise ValueError(f"{y.numel()} labels for {h.shape[0]} hidden rows")
    keep = y != ignore_index
    count = keep.sum()
    idx = keep.nonzero().squeeze(1)                        # the one host sync: how many rows the head runs on
    if idx.numel() < y.numel():
        h, y = h.index_select(0, idx), y.index_select(0, idx)
    fn = functools.partial(_chunk_ce_sum, head=head, transform=transform, config=config, ignore_index=ignore_index)
    grad = torch.is_grad_enabled() and (h.requires_grad or any(p.requires_grad for p in head.parameters()))
    total = None
    for s in range(0, max(h.shape[0], 1), chunk):          # nothing supervised: one empty chunk -> 0, stock's nan, zero grads
        part = checkpoint(fn, h[s:s + chunk], y[s:s + chunk], use_reentrant=False) if grad else fn(h[s:s + chunk], y[s:s + chunk])
        total = part if total is None else total + part
    if num_items_in_batch is None:
        return total / count                               # nll_loss's mean: the supervised sum over the supervised count
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(total.device)
    return total / num_items_in_batch


def chunked_lm_loss_from_logits(logits, labels, *, chunk=DEFAULT_CHUNK, num_items_in_batch=None, ignore_index=-100,
                                shift_labels=None):
    """``ForCausalLMLoss(logits, labels, ...)`` with the fp32 upcast and the cross-entropy taken ``chunk`` rows at a time.

    Hugging Face's loss upcasts the whole ``[batch, seq, vocab]`` logits to fp32 and takes the log-softmax of that copy; here each
    chunk of ``chunk`` token rows is upcast and summed on its own, so the loss's transient is two chunks' fp32 rows instead of
    two copies of the row. Labels are shifted as there (padded with ``ignore_index``, then ``[..., 1:]``) unless ``shift_labels``
    is given; the sum is divided by the supervised count (``mean``; nan when there is none, as there) or by
    ``num_items_in_batch``. Only the order of the fp32 summation differs from the stock loss. No gradient flows through the
    chunk loop's bookkeeping, but the result is differentiable in ``logits`` like the stock loss."""
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    y = shift_labels.reshape(-1).to(logits.device)
    z = logits.reshape(-1, logits.shape[-1])
    if y.numel() != z.shape[0]:
        raise ValueError(f"{y.numel()} labels for {z.shape[0]} logit rows")
    total = None
    for s in range(0, max(z.shape[0], 1), chunk):          # no rows at all: one empty chunk -> 0, then stock's nan
        part = F.cross_entropy(z[s:s + chunk].float(), y[s:s + chunk], ignore_index=ignore_index, reduction="sum")
        total = part if total is None else total + part
    if num_items_in_batch is None:
        return total / (y != ignore_index).sum()
    if torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(total.device)
    return total / num_items_in_batch


def _target(model):
    """The causal-LM module whose forward computes the loss: ``model`` itself when its class is in :data:`SUPPORTED`, else the
    outermost supported causal LM below it (a PEFT wrapper's base model), else ``model`` (to be refused under its own name)."""
    for m in model.modules():
        if type(m).__name__ in SUPPORTED:
            return m
    return model


def chunked_lm_loss_refusal(model):
    """Why the chunked loss cannot take ``model``'s training forward, or None when it can."""
    target = _target(model)
    cls = type(target)
    name = cls.__name__
    if name not in SUPPORTED:
        return f"{name} is not in the chunked-loss table (its post-lm_head logits path is not verified)"
    pkg, _transform, aux_attr = SUPPORTED[name]
    want_mod = f"transformers.models.{pkg}.modeling_{pkg}"
    if cls.__module__ != want_mod:
        return f"{name} comes from {cls.__module__}, not {want_mod}"
    fwd = getattr(cls, "forward", None)
    if getattr(fwd, "__module__", None) != want_mod or getattr(fwd, "__qualname__", None) != f"{name}.forward":
        return (f"{name}.forward has been replaced (by {getattr(fwd, '__module__', '?')}."
                f"{getattr(fwd, '__qualname__', '?')}); the table describes transformers' own forward")
    if "forward" in vars(target):
        return f"{name}'s forward is already replaced on this instance"
    if "labels" not in inspect.signature(fwd).parameters:
        return f"{name}.forward takes no `labels`"
    try:
        from transformers.loss.loss_utils import ForCausalLMLoss
    except ImportError:
        return "transformers.loss.loss_utils.ForCausalLMLoss is not importable"
    try:
        loss_fn = target.loss_function
    except Exception as e:                                 # an older transformers without the property, or a broken override
        return f"{name}.loss_function cannot be read ({type(e).__name__}: {e})"
    if loss_fn is not ForCausalLMLoss:
        return f"{name}.loss_function is {getattr(loss_fn, '__qualname__', loss_fn)!r}, not ForCausalLMLoss"
    head = getattr(target, "lm_head", None)
    if not isinstance(head, torch.nn.Module):
        return f"{name} has no lm_head module"
    if target.get_output_embeddings() is not head:
        return f"{name}.get_output_embeddings() is not its lm_head"
    if "forward" in vars(head):
        return f"{name}.lm_head's forward is already replaced on the instance"
    if head._forward_hooks or head._forward_pre_hooks:
        return f"{name}.lm_head carries forward hooks, which would see the probe instead of the logits"
    vocab = getattr(target, "vocab_size", None) or getattr(target.config, "vocab_size", None)
    width = getattr(head, "out_features", None)
    if width is None or width != vocab:
        return f"{name}.lm_head is {width} wide, the vocabulary {vocab}"
    if getattr(target.config, "final_logit_softcapping", None) is not None:
        return f"{name}'s config sets final_logit_softcapping"
    if aux_attr is not None and not hasattr(target, aux_attr):
        return f"{name} has no {aux_attr} to scale the router auxiliary loss by"
    return None


class _ChunkedState:
    """What a patched model carries: the chunk size, its head and transform, where ``labels`` / ``logits_to_keep`` sit among
    the forward's positional arguments, and the per-call capture of ``lm_head``'s input."""

    def __init__(self, head, transform, aux_attr, chunk, labels_pos, ltk_pos, min_bytes=None, eval_loss=False):
        self.head, self.transform, self.aux_attr, self.chunk = head, transform, aux_attr, int(chunk)
        self.eval_loss = bool(eval_loss)
        self.labels_pos, self.ltk_pos = labels_pos, ltk_pos
        self.min_bytes, self.vocab = min_bytes, int(head.out_features)
        self.capture = self.probe = self.refused = None


def _arg(args, kwargs, pos, name, default=None):
    if pos is not None and len(args) > pos:
        return args[pos]
    return kwargs.get(name, default)


def _chunked_forward(self, cls_forward, args, kwargs):
    st = self.__dict__.get("_e4b_chunked_lm_loss")
    if st is None or st.refused is not None:
        return cls_forward(self, *args, **kwargs)
    if not torch.is_grad_enabled():
        return _eval_forward(self, st, cls_forward, args, kwargs)
    labels = _arg(args, kwargs, st.labels_pos, "labels")
    ltk = _arg(args, kwargs, st.ltk_pos, "logits_to_keep", 0)
    return_dict = kwargs.get("return_dict")
    if return_dict is None:
        return_dict = getattr(self.config, "return_dict", True)
    if labels is None or not (isinstance(ltk, int) and ltk == 0) or not return_dict:
        CHUNKED_LM_LOSS_STATS["stock_calls"] += 1
        return cls_forward(self, *args, **kwargs)
    if st.min_bytes is not None and labels.numel() * st.vocab * 4 < st.min_bytes:   # `auto`: the stock fp32 logits are small
        CHUNKED_LM_LOSS_STATS["small_calls"] += 1
        return cls_forward(self, *args, **kwargs)
    if st.labels_pos is not None and len(args) > st.labels_pos:
        a2, k2 = args[:st.labels_pos] + (None,) + args[st.labels_pos + 1:], kwargs
    else:
        a2, k2 = args, dict(kwargs, labels=None)
    w = getattr(st.head, "weight", None)
    probe = torch.full((1, 1, 1), 2.0, dtype=w.dtype if w is not None and w.is_floating_point() else torch.float32,
                       device=w.device if w is not None else None)
    st.capture, st.probe = [], probe
    try:
        out = cls_forward(self, *a2, **k2)                 # the stock decoder; lm_head records its input and returns the probe
    finally:
        cap, st.capture, st.probe = st.capture, None, None
    why = None
    got = out.get("logits") if hasattr(out, "get") else None
    if len(cap) != 1:
        why = f"lm_head ran {len(cap)} times in one forward"
    else:
        want = st.transform(probe, self.config)
        if want is probe:
            ok = got is probe
        else:
            ok = (torch.is_tensor(got) and got.shape == want.shape and got.dtype == want.dtype and torch.equal(got, want))
        if not ok:
            why = "the forward changed lm_head's output in a way the chunked-loss table does not reproduce"
    if why is not None:
        st.refused = why
        CHUNKED_LM_LOSS_STATS["runtime_refusals"] += 1
        warnings.warn(f"[e4b.chunked_lm_loss] disabled on {type(self).__name__}: {why}; this call and every later one run "
                      "the stock loss", RuntimeWarning, stacklevel=3)
        del out, cap
        return cls_forward(self, *args, **kwargs)
    loss = chunked_causal_lm_loss(cap[0], labels, st.head, chunk=st.chunk, transform=st.transform, config=self.config,
                                  num_items_in_batch=kwargs.get("num_items_in_batch"),
                                  ignore_index=kwargs.get("ignore_index", -100), shift_labels=kwargs.get("shift_labels"))
    aux = out.get("aux_loss")
    if aux is not None and st.aux_attr is not None:       # `loss += self.router_aux_loss_coef * aux_loss.to(loss.device)`
        loss = loss + getattr(self, st.aux_attr) * aux.to(loss.device)
    fields = {k: v for k, v in out.items() if k not in ("loss", "logits")}
    CHUNKED_LM_LOSS_STATS["chunked_calls"] += 1
    return type(out)(loss=loss, **fields)


def _eval_forward(self, st, cls_forward, args, kwargs):
    """A no-grad forward: the stock one, unless the eval switch was on (the default) when the model was patched and the forward
    passes ``labels`` whose stock fp32 logits reach :data:`EVAL_MIN_LOGITS_BYTES`. Then it runs without ``labels`` (the logits
    are the stock forward's) and the loss comes from :func:`chunked_lm_loss_from_logits`, plus the router auxiliary loss."""
    labels = _arg(args, kwargs, st.labels_pos, "labels")
    if not st.eval_loss or labels is None:
        return cls_forward(self, *args, **kwargs)
    ltk = _arg(args, kwargs, st.ltk_pos, "logits_to_keep", 0)
    return_dict = kwargs.get("return_dict")
    if return_dict is None:
        return_dict = getattr(self.config, "return_dict", True)
    if not (isinstance(ltk, int) and ltk == 0) or not return_dict or labels.numel() * st.vocab * 4 < EVAL_MIN_LOGITS_BYTES:
        CHUNKED_LM_LOSS_STATS["eval_stock_calls"] += 1
        return cls_forward(self, *args, **kwargs)
    if st.labels_pos is not None and len(args) > st.labels_pos:
        a2, k2 = args[:st.labels_pos] + (None,) + args[st.labels_pos + 1:], kwargs
    else:
        a2, k2 = args, dict(kwargs, labels=None)
    out = cls_forward(self, *a2, **k2)                    # the stock forward without labels: its logits, no loss
    loss = chunked_lm_loss_from_logits(out.get("logits"), labels, chunk=st.chunk, num_items_in_batch=kwargs.get("num_items_in_batch"),
                                       ignore_index=kwargs.get("ignore_index", -100), shift_labels=kwargs.get("shift_labels"))
    aux = out.get("aux_loss")
    if aux is not None and st.aux_attr is not None:       # `loss += self.router_aux_loss_coef * aux_loss.to(loss.device)`
        loss = loss + getattr(self, st.aux_attr) * aux.to(loss.device)
    CHUNKED_LM_LOSS_STATS["eval_chunked_calls"] += 1
    return type(out)(loss=loss, **{k: v for k, v in out.items() if k != "loss"})


def _make_forward(cls_forward):
    @functools.wraps(cls_forward)                          # inspect.signature (generation's kwarg checks) sees the real one
    def forward(self, *args, **kwargs):
        return _chunked_forward(self, cls_forward, args, kwargs)
    return forward


def enable_chunked_lm_loss(model, chunk=None, verbose: bool = False, min_logits_bytes=None, quiet_refusal: bool = False,
                           eval_loss=None) -> int:
    """Route ``model``'s training forwards (``labels`` given, gradients on) through :func:`chunked_causal_lm_loss` in chunks
    of ``chunk`` tokens (default :data:`DEFAULT_CHUNK`). With ``min_logits_bytes`` (``auto``'s gate) a forward whose stock fp32
    logits would take fewer bytes runs the stock forward. Returns 1 when patched, 0 when refused (a ``RuntimeWarning`` names
    the reason and the model keeps its stock loss) or already patched (then only the chunk size, the gate and the eval switch
    change). With ``quiet_refusal`` (the default-on path) a refusal is recorded in :data:`CHUNKED_LM_LOSS_STATS` without the
    warning. ``eval_loss`` (default: ``E4B_CHUNKED_EVAL_LOSS``) also routes large no-grad forwards with labels through
    :func:`chunked_lm_loss_from_logits` (see the module docstring)."""
    chunk = DEFAULT_CHUNK if chunk is None else int(chunk)
    eval_loss = chunked_eval_loss_requested() if eval_loss is None else bool(eval_loss)
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    target = _target(model)
    st = target.__dict__.get("_e4b_chunked_lm_loss")
    if st is not None:
        st.chunk, st.min_bytes, st.eval_loss = chunk, min_logits_bytes, eval_loss
        return 0
    why = chunked_lm_loss_refusal(target)
    if why is not None:
        CHUNKED_LM_LOSS_STATS["refused"][type(target).__name__] = why
        if quiet_refusal:
            return 0
        warnings.warn(f"[e4b.chunked_lm_loss] refused: {why}; the stock loss runs", RuntimeWarning, stacklevel=2)
        if verbose:
            print(f"[e4b.chunked_lm_loss] refused: {why}; the stock loss runs")
        return 0
    cls = type(target)
    _pkg, transform, aux_attr = SUPPORTED[cls.__name__]
    params = [p for p in inspect.signature(cls.forward).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)][1:]      # drop `self`
    names = [p.name for p in params]
    st = _ChunkedState(target.lm_head, transform, aux_attr, chunk, names.index("labels"),
                       names.index("logits_to_keep") if "logits_to_keep" in names else None, min_logits_bytes, eval_loss)
    head = st.head
    head_forward = type(head).forward

    def _head_forward(*args, **kwargs):
        if st.capture is not None:
            st.capture.append(args[0] if args else next(iter(kwargs.values())))
            return st.probe
        return head_forward(head, *args, **kwargs)

    head.forward = _head_forward
    target._e4b_chunked_lm_loss = st
    target.forward = _make_forward(cls.forward).__get__(target, cls)
    CHUNKED_LM_LOSS_STATS["patched"] += 1
    if verbose:
        gate = ("" if min_logits_bytes is None else
                f"; only where the stock fp32 logits would reach {min_logits_bytes / 2**30:.2f} GiB, smaller ones run stock")
        print(f"[e4b.chunked_lm_loss] chunked LM loss on {cls.__name__} ({chunk}-token chunks): training forwards with labels "
              f"never materialise the [tokens, {head.out_features}] logits{gate}")
        if eval_loss:
            print(f"[e4b.chunked_lm_loss] held-out loss (E4B_CHUNKED_EVAL_LOSS, default on): no-grad forwards with labels whose fp32 logits reach "
                  f"{EVAL_MIN_LOGITS_BYTES / 2**30:.2f} GiB take their loss from the logits in {chunk}-token chunks")
    return 1


def disable_chunked_lm_loss(model) -> int:
    """Undo :func:`enable_chunked_lm_loss` on every patched causal LM in ``model``; returns how many were restored."""
    n = 0
    for m in list(model.modules()):
        st = m.__dict__.get("_e4b_chunked_lm_loss")
        if st is None:
            continue
        m.__dict__.pop("forward", None)
        st.head.__dict__.pop("forward", None)
        del m._e4b_chunked_lm_loss
        CHUNKED_LM_LOSS_STATS["patched"] = max(0, CHUNKED_LM_LOSS_STATS["patched"] - 1)
        n += 1
    return n
