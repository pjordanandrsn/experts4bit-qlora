"""Indexed tokenizer assets and declared configuration, without tokenizer execution.

Selected Linux Python -I -S -B and inherited SIGKILL guard are required by
verify/CLI. audit permits independent retained-record evaluation. No native
asset-read, backend class, token IDs, consumer, GPU or launch proof.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

LIMIT = 32 * 1024 * 1024
PEERS = ('ra_tokenizer_assets', 'ra_checkpoint', 'ra_inputs', 'ra_stage')
KEYS = {'schema', 'input_spec', 'stage', 'input_lock', 'input_lock_sha256', 'expected_parent'}
COMMON = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt'}
POLICIES = {
    'Qwen/Qwen3-30B-A3B': ('qwen3_moe', 'Qwen2Tokenizer', COMMON),
    'ibm-granite/granite-3.1-3b-a800m-instruct':
        ('granitemoe', 'GPT2Tokenizer', COMMON | {'added_tokens.json', 'special_tokens_map.json'}),
}


def require(ok, message):
    if not ok:
        raise ValueError('tokenizer assets: ' + message)


def peers():
    base = Path(__file__).absolute().parent
    name = 'ra_checkpoint'
    require(name not in sys.modules or Path(sys.modules[name].__file__).absolute() == base / (name + '.py'),
            'foreign helper preloaded')
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, base / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    cp = sys.modules[name]
    return cp.load_inputs(), cp


def manifest_fields(manifest):
    require(isinstance(manifest, dict) and set(manifest) == KEYS and
            type(manifest['schema']) is int and manifest['schema'] == 1 and
            type(manifest['expected_parent']) is int and manifest['expected_parent'] > 0, 'fixed manifest')
    return {k: v for k, v in manifest.items() if k != 'expected_parent'}


def parent_guard(manifest):
    manifest_fields(manifest)
    require(sys.platform == 'linux' and sys.flags.isolated and sys.flags.no_site and
            sys.dont_write_bytecode, 'selected Linux python -I -S -B required')
    parent = os.getppid()
    require(parent == manifest['expected_parent'], 'expected live parent')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    observed = ctypes.c_int()
    require(libc.prctl(2, ctypes.cast(ctypes.byref(observed), ctypes.c_void_p).value, 0, 0, 0) == 0 and
            observed.value == signal.SIGKILL and os.getppid() == parent, 'inherited SIGKILL parent guard')
    return {'pid': os.getpid(), 'expected_parent': parent, 'observed_parent': os.getppid(),
            'signal': observed.value, 'platform': 'linux'}


def parse(data):
    inputs, _ = peers()
    require(type(data) is bytes and 0 < len(data) <= LIMIT, 'bounded JSON bytes')
    return json.loads(data, object_pairs_hook=inputs.unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite tokenizer JSON')))


def configuration(model, payloads):
    """Parse declared configuration only; never infer the actual runtime class."""
    require(model in POLICIES and set(payloads) == POLICIES[model][2], 'fixed registered asset set')
    config, tokenizer = parse(payloads['config.json']), parse(payloads['tokenizer_config.json'])
    require(isinstance(config, dict) and isinstance(tokenizer, dict), 'configuration objects')
    model_type, declared_class, _ = POLICIES[model]
    require(config.get('model_type') == model_type and tokenizer.get('tokenizer_class') == declared_class,
            'registered configuration family/class declaration')
    # Fixed local constructor derives paths from the reviewed root only. These
    # fields would introduce remote code or caller-controlled asset resolution.
    overrides = {'auto_map', 'tokenizer_file', 'vocab_file', 'merges_file', 'added_tokens_file',
                 'special_tokens_map_file', 'tokenizer_config_file', 'fast_tokenizer_files',
                 'tokenizer_class'}
    require(not (set(config) & overrides) and not (set(tokenizer) & (overrides - {'tokenizer_class'})),
            'remote code/asset/class override')
    decoder = tokenizer.get('added_tokens_decoder')
    require(isinstance(decoder, dict) and 0 < len(decoder) <= 10000, 'complete configured added tokens')
    tokens = {}
    fields = {'content', 'lstrip', 'normalized', 'rstrip', 'single_word', 'special'}
    for key, row in decoder.items():
        require(isinstance(key, str) and len(key) <= 10 and key.isascii() and key.isdecimal() and str(int(key)) == key and
                0 <= int(key) < 2 ** 31 and isinstance(row, dict) and set(row) == fields and
                isinstance(row['content'], str) and bool(row['content']) and
                all(type(row[k]) is bool for k in fields - {'content'}), 'typed complete added-token declaration')
        require(row['content'] not in tokens, 'ambiguous added-token content')
        tokens[row['content']] = int(key)
    special = {}
    for name in ('eos', 'pad'):
        value = tokenizer.get(name + '_token')
        require(isinstance(value, str) and value in tokens and decoder[str(tokens[value])]['special'] is True,
                'declared EOS/pad token identity')
        token_id = tokens[value]
        if name + '_token_id' in config:
            require(type(config[name + '_token_id']) is int and config[name + '_token_id'] == token_id,
                    'model/config special ID differs')
        if name + '_token_id' in tokenizer:
            require(type(tokenizer[name + '_token_id']) is int and tokenizer[name + '_token_id'] == token_id,
                    'tokenizer/config special ID differs')
        special[name] = {'text': value, 'declared_added_token_id': token_id}
    for name in ('add_bos_token', 'add_eos_token', 'split_special_tokens', 'clean_up_tokenization_spaces'):
        require(name not in tokenizer or type(tokenizer[name]) is bool, 'typed tokenizer mode')
    backend = parse(payloads['tokenizer.json'])
    require(isinstance(backend, dict) and isinstance(backend.get('model'), dict) and
            backend['model'].get('type') == 'BPE', 'declared BPE asset')
    added = backend.get('added_tokens')
    require(isinstance(added, list) and len(added) == len(decoder), 'complete asset/config added-token set')
    observed = {}
    for row in added:
        require(isinstance(row, dict) and set(row) == fields | {'id'} and type(row['id']) is int and
                str(row['id']) in decoder and str(row['id']) not in observed, 'typed unique asset added token')
        require(all(type(row[k]) is type(decoder[str(row['id'])][k]) and
                    row[k] == decoder[str(row['id'])][k] for k in fields), 'asset/config added-token body differs')
        observed[str(row['id'])] = {k: row[k] for k in fields}
    require(observed == decoder, 'complete asset/config added-token equality')
    return {'model_type': model_type, 'declared_tokenizer_class': declared_class,
            'special_tokens': special, 'added_token_count': len(decoder),
            'backend_asset_model_type': 'BPE', 'runtime_class_observed': False,
            'native_asset_read_observed': False}


def bind(root, authority, expected):
    inputs, cp = peers()
    model = authority['model']
    require(model in POLICIES, 'fixed registered model')
    names = POLICIES[model][2]
    require(names <= set(expected) and names <= set(authority['files']), 'required registered tokenizer assets')
    before = inputs.inventory(root)
    require(before == expected, 'complete reviewed checkpoint inventory')
    payloads, records = {}, {}
    for name in sorted(names):
        path = inputs.absolute(root) / name
        row, identity = expected[name], authority['files'][name]
        require(type(identity['size']) is int and 0 < identity['size'] <= LIMIT and
                row['size'] == identity['size'], 'bounded indexed asset size')
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as src:
            require(0 < os.fstat(src.fileno()).st_size <= LIMIT, 'bounded local asset size')
            data = src.read(LIMIT + 1)
        require(len(data) == row['size'] and hashlib.sha256(data).hexdigest() == row['sha256'] and
                inputs.file_record(path) == row, 'asset read drift')
        if identity['lfs_sha256'] is None:
            require(cp.git_blob(data) == identity['git_blob_oid'], 'indexed ordinary Git asset bytes')
        else:
            require(row['sha256'] == identity['lfs_sha256'], 'indexed materialized LFS asset bytes')
        payloads[name], records[name] = data, row
    declared = configuration(model, payloads)
    require(inputs.inventory(root) == before, 'checkpoint drift during asset/config binding')
    return {'files': records, 'file_count': len(records), 'configuration': declared}


def audit(manifest, fetch):
    inputs, cp = peers()
    m = manifest_fields(manifest)
    base = Path(__file__).absolute().parent
    paths = [base / (n + '.py') for n in PEERS] + [base / 'source-pins.json']
    paths += [inputs.absolute(m[k]) for k in ('input_spec', 'input_lock')]
    before = {str(p): inputs.file_record(p) for p in paths}
    spec, lock, common = cp.check(m)
    authority = cp.observe(lock['model'], lock['revision'], fetch)
    bound = bind(inputs.absolute(spec['trees']['checkpoint']), authority, lock['trees']['checkpoint'])
    require(cp.observe(lock['model'], lock['revision'], fetch) == authority and
            bind(inputs.absolute(spec['trees']['checkpoint']), authority, lock['trees']['checkpoint']) == bound and
            cp.check(m) == (spec, lock, common) and {str(p): inputs.file_record(p) for p in paths} == before,
            'authority/input/helper drift')
    return {'schema': 1, 'authority': authority, 'assets': bound, 'common_inputs': common,
            'execution_inputs': before, 'proves_indexed_tokenizer_asset_and_config_equality': True,
            'proves_tokenizer_execution': False, 'proves_native_asset_consumption': False,
            'proves_tokenizer_regeneration': False, 'proves_runtime_consumption': False,
            'proves_calibration_authority': False, 'proves_publisher_signature': False,
            'proves_gpu_engagement': False, 'launch_authority': False}


def verify(manifest, fetch):
    guard = parent_guard(manifest)
    result = audit(manifest, fetch)
    require(parent_guard(manifest) == guard, 'parent guard drift')
    return {**result, 'parent_guard': guard}


def retained(manifest, output, process, manifest_sha256):
    """Parent re-audit after its OWN verified process record; records are not authenticated here."""
    inputs, cp = peers()
    manifest_fields(manifest)
    require(cp.sha(manifest_sha256, 64) and isinstance(process, dict), 'manifest digest/process record')
    output = inputs.absolute(output)
    guard = process.get('parent_death_guard', {})
    require(process.get('status') == 'OK' and type(process.get('returncode')) is int and
            process['returncode'] == 0 and process.get('cleanup_complete') is True and
            type(process.get('pid')) is int and process['pid'] > 0 and
            guard.get('required') is True and guard.get('verified') is True and
            type(guard.get('expected_parent')) is int and
            guard.get('expected_parent') == manifest['expected_parent'], 'verified process/guard/cleanup required')
    expected_guard = {'pid': process['pid'], 'expected_parent': manifest['expected_parent'],
                      'observed_parent': manifest['expected_parent'], 'signal': signal.SIGKILL, 'platform': 'linux'}
    result = inputs.read_json(output / 'result.json')
    rows = result.get('observations')
    require(isinstance(rows, list) and len(rows) == 4, 'four retained responses required')
    _, lock, _ = cp.check(manifest_fields(manifest))
    names, values, last = {'result.json'}, [], None
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) ==
                {'url', 'file', 'sha256', 'bytes', 'started_at', 'finished_at'} and
                row['url'] == (cp.urls(lock['model'], lock['revision']) * 2)[index] and
                row['file'] == f'response-{index:02d}.json' and type(row['bytes']) is int and
                0 < row['bytes'] <= cp.LIMIT and cp.sha(row['sha256'], 64), 'fixed bounded raw response')
        path = output / row['file']
        require(inputs.file_record(path) == {'sha256': row['sha256'], 'size': row['bytes']}, 'retained raw bytes drift')
        start, end = [datetime.datetime.fromisoformat(row[k]) for k in ('started_at', 'finished_at')]
        require(start.utcoffset() == end.utcoffset() == datetime.timedelta(0) and start <= end and
                (last is None or last <= start), 'sequential UTC clock reads')
        last = end
        names.add(row['file'])
        values.append(inputs.read_json(path))
    require({p.name for p in output.iterdir()} == names and
            all(p.is_file() and not p.is_symlink() for p in output.iterdir()), 'exact five-file receipt inventory')
    before = inputs.inventory(output)
    iterator = iter(values)
    expected = audit(manifest, lambda url: next(iterator))
    complete = {**expected, 'parent_guard': expected_guard, 'status': 'PASS', 'pid': process['pid'],
                'manifest_sha256': manifest_sha256, 'observations': rows}
    require(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) ==
            json.dumps(complete, sort_keys=True, separators=(',', ':'), allow_nan=False) and
            inputs.inventory(output) == before, 'parent complete asset/config/result differs')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    inputs, cp = peers()
    manifest = inputs.read_json(args.manifest)
    m = manifest_fields(manifest)
    spec = inputs.read_json(m['input_spec'])
    require(isinstance(spec, dict) and set(spec) == inputs.SPEC and isinstance(spec['trees'], dict) and
            set(spec['trees']) == inputs.TREES and isinstance(spec['files'], dict) and
            set(spec['files']) == inputs.FILES, 'common input roles')
    out = inputs.absolute(args.out)
    protected = [inputs.absolute(v) for v in spec['trees'].values()] + [inputs.absolute(v) for v in spec['files'].values()]
    protected += [inputs.absolute(m[k]) for k in ('input_spec', 'input_lock', 'stage')]
    protected += [inputs.absolute(args.manifest), Path(__file__).absolute().parent]
    require(not out.exists() and all(not (out == p or out in p.parents or p in out.parents) for p in protected),
            'fresh protected receipt directory')
    initial = inputs.file_record(args.manifest)
    out.mkdir()
    fetch = None
    result = {'schema': 1, 'status': 'FAILED', 'pid': os.getpid(), 'manifest_sha256': initial['sha256']}
    try:
        guard = parent_guard(manifest)
        checked = cp.check(m)
        fetch = cp.Collector(out, cp.urls(checked[1]['model'], checked[1]['revision']))
        result = verify(manifest, fetch)
        require(len(fetch.records) == 4 and {p.name for p in out.iterdir()} == {r['file'] for r in fetch.records} and
                inputs.file_record(args.manifest) == initial and cp.check(m) == checked and
                parent_guard(manifest) == guard and
                all(inputs.file_record(out / row['file']) == {'sha256': row['sha256'], 'size': row['bytes']}
                    for row in fetch.records) and
                {name: inputs.file_record(name) for name in result['execution_inputs']} == result['execution_inputs'],
                'final manifest/input/helper/guard/receipt drift')
        result.update(status='PASS', pid=os.getpid(), manifest_sha256=initial['sha256'])
    except Exception as error:
        result = {'schema': 1, 'status': 'FAILED', 'error_type': type(error).__name__, 'error': str(error),
                  'pid': os.getpid(), 'manifest_sha256': initial['sha256']}
        raise
    finally:
        result['observations'] = fetch.records if fetch is not None else []
        with (out / 'result.json').open('x') as dest:
            dest.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
