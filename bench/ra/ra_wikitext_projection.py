"""Proposed raw WikiText source, complete rows and registered text projection gate.

Selected Linux Python -I -S -B with the inherited SIGKILL parent guard only.
No release imports, tokenizer execution, native consumption or launch authority.
The pure audit supports independent retained-record evaluation by the parent.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import datetime
import hashlib
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

PEERS = ('ra_wikitext_projection', 'ra_wikitext_text', 'ra_wikitext',
         'ra_checkpoint', 'ra_inputs', 'ra_stage')
KEYS = {'schema', 'input_spec', 'stage', 'input_lock', 'input_lock_sha256', 'expected_parent'}


def require(ok, message):
    if not ok:
        raise ValueError('WikiText projection: ' + message)


def peers():
    base = Path(__file__).absolute().parent
    for name in ('ra_wikitext', 'ra_wikitext_text'):
        require(name not in sys.modules or Path(sys.modules[name].__file__).absolute() == base / (name + '.py'),
                'foreign helper preloaded')
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, base / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    raw, text = sys.modules['ra_wikitext'], sys.modules['ra_wikitext_text']
    inputs, cp = raw.peers()
    return raw, text, inputs, cp


def raw_manifest(manifest):
    require(isinstance(manifest, dict) and set(manifest) == KEYS and
            type(manifest['schema']) is int and manifest['schema'] == 1 and
            type(manifest['expected_parent']) is int and manifest['expected_parent'] > 0, 'fixed manifest')
    return {k: v for k, v in manifest.items() if k != 'expected_parent'}


def parent_guard(manifest):
    raw_manifest(manifest)
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
            'signal': observed.value, 'platform': sys.platform}


def recipe(stage):
    raw, _, inputs, _ = peers()
    base = raw.recipe(stage)
    expected = [ast.parse(s).body[0] for s in (
        "text = '\\n\\n'.join(t for t in ds['text'] if t.strip())",
        "ids = tok(text, return_tensors='pt').input_ids[0]",
    )]
    for filename, name in (('p109_box.py', 'wikitext_rows'), ('p97_box.py', 'wikitext_windows')):
        tree = ast.parse((inputs.absolute(stage) / filename).read_bytes())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(functions) == 1, 'registered complete text function')
        for target, wanted in zip(('text', 'ids'), expected):
            statements = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Assign) and
                          any(isinstance(t, ast.Name) and t.id == target for t in n.targets)]
            require(len(statements) == 1 and ast.dump(statements[0], include_attributes=False) ==
                    ast.dump(wanted, include_attributes=False), 'registered full text join/tokenizer call')
    return {**base, 'row_filter': 't.strip()', 'join_separator': '\n\n',
            'ordered_rows': True, 'tokenizer_argument_is_complete_text': True}


def audit(manifest, fetch):
    """Regenerate complete rows/text from reviewed raw bytes, without invoking a tokenizer."""
    m = raw_manifest(manifest)
    raw, text, inputs, _ = peers()
    base = Path(__file__).absolute().parent
    paths = [base / (n + '.py') for n in PEERS] + [base / 'source-pins.json']
    paths += [inputs.absolute(m[k]) for k in ('input_spec', 'input_lock')]
    before = {str(p): inputs.file_record(p) for p in paths}
    spec, lock, common, authority = raw.check(m)
    source_recipe = recipe(m['stage'])
    indexed = raw.observe(authority['revision'], fetch)
    path = inputs.absolute(spec['trees']['datasets']) / raw.FILENAME
    archive = inputs.file_record(path)
    identity = indexed['files'][raw.FILENAME]
    require(archive == {'sha256': identity['lfs_sha256'], 'size': identity['size']} and
            12 <= archive['size'] <= text.MAX_BYTES, 'indexed bounded raw archive')
    data = path.read_bytes()
    require(len(data) == archive['size'] and hashlib.sha256(data).hexdigest() == archive['sha256'] and
            inputs.file_record(path) == archive, 'raw archive read drift')
    rows = text.decode(data)
    ordered, joined = text.project(rows)
    require(raw.observe(authority['revision'], fetch) == indexed and
            raw.check(m) == (spec, lock, common, authority) and inputs.file_record(path) == archive and
            recipe(m['stage']) == source_recipe and {str(p): inputs.file_record(p) for p in paths} == before,
            'authority/input/helper/recipe drift')
    result = {'schema': 1, 'authority': indexed, 'archive': archive, 'recipe': source_recipe,
              'execution_inputs': before,
              'common_inputs': common, 'row_count': len(rows), 'nonblank_rows': sum(bool(t.strip()) for t in rows),
              'ordered_rows': {'sha256': hashlib.sha256(ordered).hexdigest(), 'size': len(ordered)},
              'joined_text': {'sha256': hashlib.sha256(joined).hexdigest(), 'size': len(joined)},
              'proves_raw_wikitext_source_and_text_projection': True,
              'proves_tokenizer_execution': False, 'proves_runtime_consumption': False,
              'proves_native_loader_revision_binding': False, 'proves_calibration_authority': False,
              'proves_publisher_signature': False, 'proves_gpu_engagement': False, 'launch_authority': False}
    return result, ordered, joined


def verify(manifest, fetch):
    before = parent_guard(manifest)
    result, ordered, joined = audit(manifest, fetch)
    require(parent_guard(manifest) == before, 'parent guard drift')
    result['parent_guard'] = before
    return result, ordered, joined


def retained(manifest, output, process, manifest_sha256):
    """Parent re-audit of complete retained bytes after a verified process return.

    The caller supplies its own ra_process.run record, not a child assertion.
    This component does not acquire a process, establish launch clearance or
    authenticate caller-supplied process records.
    """
    raw, _, inputs, cp = peers()
    output = inputs.absolute(output)
    guard = process.get('parent_death_guard', {})
    require(process.get('status') == 'OK' and type(process.get('returncode')) is int and
            process['returncode'] == 0 and process.get('cleanup_complete') is True and
            type(process.get('pid')) is int and process['pid'] > 0 and
            guard.get('required') is True and guard.get('verified') is True and
            guard.get('expected_parent') == manifest['expected_parent'], 'verified process/guard/cleanup required')
    expected_guard = {'pid': process['pid'], 'expected_parent': manifest['expected_parent'],
                      'observed_parent': manifest['expected_parent'], 'signal': signal.SIGKILL, 'platform': 'linux'}
    result = inputs.read_json(output / 'result.json')
    rows = result.get('observations')
    require(isinstance(rows, list) and len(rows) == 4, 'four retained responses required')
    _, _, _, authority = raw.check(raw_manifest(manifest))
    values, names = [], {'result.json', 'ordered-rows.json', 'joined-text.txt'}
    last = None
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) ==
                {'url', 'file', 'sha256', 'bytes', 'started_at', 'finished_at'} and
                row['url'] == (raw.urls(authority['revision']) * 2)[index] and
                row['file'] == f'response-{index:02d}.json' and type(row['bytes']) is int and
                0 < row['bytes'] <= cp.LIMIT and cp.sha(row['sha256'], 64), 'fixed bounded raw response')
        path = output / row['file']
        require(inputs.file_record(path) == {'sha256': row['sha256'], 'size': row['bytes']}, 'retained raw bytes drift')
        start, end = [datetime.datetime.fromisoformat(row[k]) for k in ('started_at', 'finished_at')]
        require(start.utcoffset() == end.utcoffset() == datetime.timedelta(0) and start <= end and
                (last is None or last <= start), 'sequential UTC clock reads')
        last = end
        values.append(inputs.read_json(path))
        names.add(row['file'])
    require({p.name for p in output.iterdir()} == names and
            all(p.is_file() and not p.is_symlink() for p in output.iterdir()), 'exact seven-file receipt inventory')
    before = inputs.inventory(output)
    iterator = iter(values)
    expected, ordered, joined = audit(manifest, lambda url: next(iterator))
    require((output / 'ordered-rows.json').read_bytes() == ordered and
            (output / 'joined-text.txt').read_bytes() == joined and
            result == {**expected, 'parent_guard': expected_guard, 'status': 'PASS', 'pid': process['pid'],
                       'manifest_sha256': manifest_sha256, 'observations': rows} and
            inputs.inventory(output) == before, 'parent complete rows/text/result differs')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    raw, _, inputs, cp = peers()
    manifest = inputs.read_json(args.manifest)
    m = raw_manifest(manifest)
    spec = inputs.read_json(m['input_spec'])
    require(isinstance(spec, dict) and set(spec) == inputs.SPEC and
            isinstance(spec['trees'], dict) and set(spec['trees']) == inputs.TREES and
            isinstance(spec['files'], dict) and set(spec['files']) == inputs.FILES, 'common input roles')
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
        checked = raw.check(m)
        authority = checked[3]
        base = Path(__file__).absolute().parent
        paths = [base / (n + '.py') for n in PEERS] + [base / 'source-pins.json']
        paths += [inputs.absolute(m[k]) for k in ('input_spec', 'input_lock')]
        pins = {str(p): inputs.file_record(p) for p in paths}
        fetch = cp.Collector(out, raw.urls(authority['revision']))
        result, ordered, joined = verify(manifest, fetch)
        require(inputs.file_record(args.manifest) == initial, 'manifest drift')
        require(len(fetch.records) == 4 and {p.name for p in out.iterdir()} ==
                {r['file'] for r in fetch.records}, 'exact raw receipt inventory')
        for name, payload in (('ordered-rows.json', ordered), ('joined-text.txt', joined)):
            with (out / name).open('xb') as dest:
                dest.write(payload)
        require(inputs.file_record(out / 'ordered-rows.json') == result['ordered_rows'] and
                inputs.file_record(out / 'joined-text.txt') == result['joined_text'], 'written projection bytes')
        require(inputs.file_record(args.manifest) == initial and raw.check(m) == checked and
                parent_guard(manifest) == guard and
                {str(p): inputs.file_record(p) for p in paths} == pins and
                {p.name for p in out.iterdir()} == {r['file'] for r in fetch.records} |
                {'ordered-rows.json', 'joined-text.txt'}, 'final input/helper/guard/receipt drift')
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
