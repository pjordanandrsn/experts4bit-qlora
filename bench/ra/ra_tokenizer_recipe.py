"""Pure fixed RA tokenizer recipes; caller tokenizer provenance is not proved.

No imports of datasets, transformers, torch or releases; no file/network/process
access. A future reviewed guarded caller must bind sources, tokenizer assets,
runtime, complete input bytes and actual consumer evidence independently.
"""
from __future__ import annotations

import hashlib
import json

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
)
MODEL = {"proof": ("granite", "ibm-granite/granite-3.1-3b-a800m-instruct"),
         "reading": ("qwen3", "Qwen/Qwen3-30B-A3B")}
SEQ, EVAL_N, OFFSET, PROMPT = 2048, 8, 4096, 512
MAX_TEXT_BYTES, MAX_IDS = 16 * 1024 * 1024, 2_000_000


def require(ok, message):
    if not ok:
        raise ValueError("tokenizer recipe: " + message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def token_ids(value, *, maximum):
    require(type(value) is list and len(value) <= maximum and
            all(type(t) is int and t >= 0 for t in value), "complete bounded integer IDs")
    return list(value)


def training(data, tokenizer, *, battery, model_revision, dataset_name, dataset_sha256):
    """Reproduce TC1's unpacked Alpaca preparation, including all eval calls.

    Identity arguments label the result; this pure function cannot authenticate
    them or the supplied tokenizer. It does not load or bind dataset bytes.
    """
    require(type(battery) is str and battery in MODEL and type(model_revision) is str and len(model_revision) == 40 and
            all(c in '0123456789abcdef' for c in model_revision), "battery/full model revision")
    require(type(dataset_name) is str and dataset_name and '/' not in dataset_name and
            '\\' not in dataset_name and dataset_name not in ('.', '..') and
            type(dataset_sha256) is str and len(dataset_sha256) == 64 and
            all(c in '0123456789abcdef' for c in dataset_sha256), "dataset labels")
    require(type(data) is dict and type(data.get('train')) is list and type(data.get('eval')) is list and
            len(data['train']) == 1200 and len(data['eval']) == 48, "registered data row counts")
    eos = getattr(tokenizer, 'eos_token', None) or ''
    require(type(eos) is str, "tokenizer EOS text")
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        pad_id = getattr(tokenizer, 'eos_token_id', None)
    require(pad_id is None or type(pad_id) is int and pad_id >= 0, "tokenizer pad/EOS ID")
    for row in data['train'] + data['eval']:
        require(type(row) is dict and type(row.get('instruction')) is str and type(row.get('output')) is str and
                (row.get('input') is None or type(row.get('input')) is str), "Alpaca row text")
        require(len(ALPACA_PROMPT.format(row['instruction'], row.get('input', '') or '', row['output']).encode()) +
                len(eos.encode()) <= MAX_TEXT_BYTES, "bounded Alpaca text")

    def encode(rows):
        result = []
        for row in rows:
            text = ALPACA_PROMPT.format(row['instruction'], row.get('input', '') or '', row['output']) + eos
            encoded = tokenizer(text, truncation=True, max_length=SEQ)
            ids = token_ids(getattr(encoded, 'input_ids', None), maximum=SEQ)
            if len(ids) >= 8:
                result.append(ids)
        return result

    train, ev = encode(data['train']), encode(data['eval'])[:EVAL_N]
    require(train and len(ev) == EVAL_N, "prepared training/eval minimum")
    body = json.dumps({'train': train, 'eval': ev}, separators=(',', ':')).encode()
    fam, model = MODEL[battery]
    return {'fam': fam, 'dataset': dataset_name, 'dataset_sha256': dataset_sha256,
            'tokenizer': model, 'revision': model_revision, 'tokenizer_class': type(tokenizer).__name__,
            'format': ALPACA_PROMPT, 'template': 'alpaca', 'eos': eos, 'pad_id': pad_id, 'seq': SEQ,
            'n_train': len(train), 'n_eval': len(ev), 'train_tokens': sum(map(len, train)),
            'sha256': digest(body), 'train': train, 'eval': ev}


def wikitext(rows, tokenizer, *, battery):
    """Tokenize complete joined text, then project fixed decode/quality windows.

    The full ID stream is returned for comparison. Corpus authority, tensor
    provenance and runtime consumption are outside this pure function.
    """
    require(type(battery) is str and battery in MODEL and type(rows) is list and 0 < len(rows) <= 10_000 and
            all(type(t) is str for t in rows), "bounded WikiText rows/battery")
    require(sum(len(t.encode()) for t in rows) + 2 * len(rows) <= MAX_TEXT_BYTES, "bounded WikiText bytes")
    text = '\n\n'.join(t for t in rows if t.strip())
    batch = getattr(tokenizer(text, return_tensors='pt'), 'input_ids', None)
    shape = getattr(batch, 'shape', ())
    require(getattr(batch, 'ndim', None) == 2 and isinstance(shape, (tuple, list)) and
            len(shape) == 2 and all(type(n) is int for n in shape) and shape[0] == 1,
            "one full tokenizer tensor row")
    tensor = batch[0]
    require(callable(getattr(tensor, 'numel', None)) and callable(getattr(tensor, 'tolist', None)),
            "full tokenizer tensor methods")
    count = tensor.numel()
    require(type(count) is int and MAX_IDS >= count >= 15 * OFFSET + PROMPT and
            shape[1] == count, "complete full-text token count")
    ids = token_ids(tensor.tolist(), maximum=MAX_IDS)
    require(len(ids) == count, "full tensor/list count")
    cont = 32 if battery == 'proof' else 128
    decode = [ids[k * OFFSET:k * OFFSET + PROMPT] for k in range(16)]
    quality = [ids[k * OFFSET:k * OFFSET + PROMPT + cont] for k in range(12)]
    require(all(len(w) == PROMPT for w in decode) and
            all(len(w) == PROMPT + cont for w in quality), "complete fixed windows")
    return {'joined_text_sha256': digest(text.encode()), 'joined_text_bytes': len(text.encode()),
            'input_ids': ids, 'decode_prompts': decode, 'quality_windows': {'wikitext': quality}}


def compare(expected, observed):
    """Compare complete JSON projections; booleans/floats cannot equal integers."""
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(canonical(expected) == canonical(observed), "complete projection differs")
