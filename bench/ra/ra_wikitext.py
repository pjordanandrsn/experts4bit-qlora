"""Bind the reviewed WikiText test archive to repeated Hub index records.

No Parquet decoding, tokenizer execution, model consumption or GPU evidence.
The frozen loader recipe stays unchanged. Selected Python -I -S -B only;
there are no release imports or payload downloads in this gate.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path, PurePosixPath

REPO = 'Salesforce/wikitext'
CONFIG = 'wikitext-2-raw-v1'
FILENAME = CONFIG + '/test-00000-of-00001.parquet'
AUTHORITY = 'wikitext-authority.json'
PEERS = ('ra_wikitext', 'ra_checkpoint', 'ra_inputs', 'ra_stage')


def require(ok, message):
    if not ok:
        raise ValueError('WikiText binding: ' + message)


def peers():
    base = Path(__file__).absolute().parent
    for name in ('ra_stage', 'ra_inputs', 'ra_checkpoint'):
        require(name not in sys.modules or Path(sys.modules[name].__file__).absolute() == base / (name + '.py'),
                'foreign helper preloaded')
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, base / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return sys.modules['ra_inputs'], sys.modules['ra_checkpoint']


def urls(revision):
    _, cp = peers()
    require(cp.sha(revision, 40), 'reviewed full dataset revision')
    root = 'https://huggingface.co/api/datasets/' + REPO
    return (root + '/revision/' + revision + '?blobs=true',
            root + '/tree/' + revision + '?recursive=true&expand=false')


def recipe(stage):
    inputs, _ = peers()
    stage = inputs.absolute(stage)
    inputs.ra_stage.verify(stage)
    manifest = inputs.read_json(Path(stage) / 'stage-manifest.json')
    for filename, name, source in (
        ('p109_box.py', 'wikitext_rows', 'bench/p109/p109_box.py'),
        ('p97_box.py', 'wikitext_windows', 'bench/p97/p97_box.py'),
    ):
        require(manifest['files'].get(filename, {}).get('source_path') == source and
                manifest['files'][filename].get('registered') is True, 'registered corpus recipe')
        tree = ast.parse((Path(stage) / filename).read_bytes())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(functions) == 1, 'unique corpus function')
        calls = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Name) and n.func.id == 'load_dataset']
        require(len(calls) == 1, 'unique dataset loader')
        call = calls[0]
        require([ast.literal_eval(v) for v in call.args] == [REPO, CONFIG] and
                len(call.keywords) == 1 and call.keywords[0].arg == 'split' and
                ast.literal_eval(call.keywords[0].value) == 'test', 'fixed registered dataset/config/split')
    return {'repo': REPO, 'config': CONFIG, 'split': 'test', 'filename': FILENAME,
            'native_loaders_have_revision_argument': False}


def observe(revision, fetch):
    _, cp = peers()
    info, tree = [fetch(url) for url in urls(revision)]
    require(isinstance(info, dict) and info.get('id') == REPO and info.get('sha') == revision and
            info.get('private') is False and info.get('gated') is False and info.get('disabled') is False,
            'public dataset revision identity')
    siblings = info.get('siblings')
    require(isinstance(siblings, list) and 0 < len(siblings) <= 1000 and
            isinstance(tree, list) and 0 < len(tree) <= 2000, 'bounded complete indexes')
    files, dirs = {}, set()
    for row in tree:
        require(isinstance(row, dict), 'tree row')
        name = cp.safe_name(row.get('path'))
        require(name not in files and name not in dirs and cp.sha(row.get('oid'), 40), 'unique Git tree address')
        if row.get('type') == 'directory':
            dirs.add(name)
            continue
        require(row.get('type') == 'file' and type(row.get('size')) is int and row['size'] >= 0, 'tree file')
        binding = {'size': row['size'], 'git_blob_oid': row['oid'], 'lfs_sha256': None}
        lfs = row.get('lfs')
        if lfs is not None:
            require(isinstance(lfs, dict) and set(lfs) == {'oid', 'size', 'pointerSize'} and
                    cp.sha(lfs['oid'], 64) and type(lfs['size']) is int and lfs['size'] == row['size'], 'tree LFS')
            pointer = ('version https://git-lfs.github.com/spec/v1\noid sha256:' + lfs['oid'] +
                       '\nsize ' + str(lfs['size']) + '\n').encode()
            require(type(lfs['pointerSize']) is int and lfs['pointerSize'] == len(pointer) and
                    cp.git_blob(pointer) == row['oid'], 'canonical LFS pointer Git address')
            binding['lfs_sha256'] = lfs['oid']
        files[name] = binding
    require(dirs == {str(p) for n in files for p in PurePosixPath(n).parents if str(p) != '.'},
            'complete directory projection')
    seen = set()
    for row in siblings:
        require(isinstance(row, dict), 'sibling row')
        name = cp.safe_name(row.get('rfilename'))
        require(name in files and name not in seen and row.get('blobId') == files[name]['git_blob_oid'] and
                type(row.get('size')) is int and row['size'] == files[name]['size'], 'info/tree identities')
        seen.add(name)
        lfs = row.get('lfs')
        if files[name]['lfs_sha256'] is None:
            require(lfs is None, 'surplus sibling LFS')
        else:
            pointer = ('version https://git-lfs.github.com/spec/v1\noid sha256:' + files[name]['lfs_sha256'] +
                       '\nsize ' + str(files[name]['size']) + '\n').encode()
            require(isinstance(lfs, dict) and set(lfs) == {'sha256', 'size', 'pointerSize'} and
                    lfs['sha256'] == files[name]['lfs_sha256'] and type(lfs['size']) is int and
                    lfs['size'] == files[name]['size'] and type(lfs['pointerSize']) is int and
                    lfs['pointerSize'] == len(pointer), 'sibling LFS identity')
    require(seen == set(files) and FILENAME in files and files[FILENAME]['lfs_sha256'] is not None,
            'complete indexed test archive')
    require({n for n in files if n.startswith(CONFIG + '/test-')} == {FILENAME}, 'single registered test shard')
    return {'repo': REPO, 'revision': revision, 'files': dict(sorted(files.items()))}


def check(manifest):
    inputs, cp = peers()
    require(set(manifest) == {'schema', 'input_spec', 'stage', 'input_lock', 'input_lock_sha256'} and
            type(manifest['schema']) is int and manifest['schema'] == 1 and
            cp.sha(manifest['input_lock_sha256'], 64), 'reviewed manifest')
    spec = inputs.read_json(manifest['input_spec'])
    common = inputs.verify(spec, manifest['stage'], manifest['input_lock'], manifest['input_lock_sha256'])
    lock = inputs.read_json(manifest['input_lock'])
    root = inputs.absolute(spec['trees']['datasets'])
    authority = inputs.read_json(root / AUTHORITY)
    require(set(authority) == {'schema', 'repo', 'config', 'split', 'revision', 'filename'} and
            type(authority['schema']) is int and authority['schema'] == 1 and cp.sha(authority['revision'], 40) and
            {k: authority[k] for k in ('repo', 'config', 'split', 'filename')} ==
            {k: recipe(manifest['stage'])[k] for k in ('repo', 'config', 'split', 'filename')}, 'reviewed authority recipe')
    require(lock['trees']['datasets'].get(AUTHORITY) == inputs.file_record(root / AUTHORITY) and
            lock['trees']['datasets'].get(FILENAME) == inputs.file_record(root / FILENAME), 'reviewed archive/authority inventory')
    return spec, lock, common, authority


def verify(manifest, fetch):
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, 'python -I -S -B required')
    inputs, _ = peers()
    base = Path(__file__).absolute().parent
    paths = [base / (n + '.py') for n in PEERS] + [base / 'source-pins.json']
    paths += [inputs.absolute(manifest[k]) for k in ('input_spec', 'input_lock')]
    before = {str(p): inputs.file_record(p) for p in paths}
    spec, lock, common, authority = check(manifest)
    indexed = observe(authority['revision'], fetch)
    path = inputs.absolute(spec['trees']['datasets']) / FILENAME
    row = inputs.file_record(path)
    identity = indexed['files'][FILENAME]
    require(row == {'sha256': identity['lfs_sha256'], 'size': identity['size']}, 'materialized test archive bytes')
    # Container envelope only, deliberately no parser or content claim.
    require(row['size'] >= 12, 'Parquet envelope size')
    with path.open('rb') as stream:
        require(stream.read(4) == b'PAR1', 'Parquet initial magic')
        stream.seek(-8, 2)
        footer = stream.read(8)
    require(footer[4:] == b'PAR1' and 0 < int.from_bytes(footer[:4], 'little') <= row['size'] - 12,
            'Parquet footer envelope')
    require(inputs.file_record(path) == row and observe(authority['revision'], fetch) == indexed and
            check(manifest) == (spec, lock, common, authority) and
            {str(p): inputs.file_record(p) for p in paths} == before, 'authority/input/helper drift')
    return {'schema': 1, 'authority': indexed, 'archive': row, 'recipe': recipe(manifest['stage']),
            'common_inputs': common, 'proves_hub_index_raw_wikitext_equality': True,
            'proves_parquet_decoding': False, 'proves_joined_text_equality': False,
            'proves_tokenizer_execution': False, 'proves_runtime_consumption': False,
            'proves_native_loader_revision_binding': False, 'proves_publisher_signature': False,
            'proves_gpu_engagement': False, 'launch_authority': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    inputs, cp = peers()
    manifest = inputs.read_json(args.manifest)
    spec, _, _, authority = check(manifest)
    out = inputs.absolute(args.out)
    protected = [inputs.absolute(v) for v in spec['trees'].values()] + [inputs.absolute(v) for v in spec['files'].values()]
    protected += [inputs.absolute(manifest[k]) for k in ('input_spec', 'input_lock', 'stage')]
    protected += [inputs.absolute(args.manifest), Path(__file__).absolute().parent]
    require(not out.exists() and all(not (out == p or out in p.parents or p in out.parents) for p in protected),
            'fresh protected receipt directory')
    out.mkdir()
    initial = inputs.file_record(args.manifest)
    fetch = cp.Collector(out, urls(authority['revision']))
    try:
        result = verify(manifest, fetch)
        require(inputs.file_record(args.manifest) == initial, 'manifest drift')
        result.update(status='PASS', pid=os.getpid(), manifest_sha256=initial['sha256'])
    except Exception as error:
        result = {'schema': 1, 'status': 'FAILED', 'error_type': type(error).__name__,
                  'pid': os.getpid(), 'manifest_sha256': initial['sha256']}
        raise
    finally:
        result['observations'] = fetch.records
        with (out / 'result.json').open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
