"""Bind TC1 prepared data to its registered Alpaca source and shuffle recipe.

This proves source-byte and prepared-projection equality only. No WikiText,
calibration, tokenizer execution, runtime consumption, publisher signature,
GPU or launch authority. Run selected Python -I -S -B; no release imports.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path

PEERS = ('ra_alpaca', 'ra_inputs', 'ra_stage')
SOURCE_NAME = 'alpaca_data_cleaned.json'
RECIPE_NAMES = {'REPO_ID', 'REVISION', 'FILE', 'FILE_SHA256', 'SEED', 'N_TRAIN', 'N_EVAL'}


def require(ok, message):
    if not ok:
        raise ValueError('Alpaca binding: ' + message)


def load_inputs():
    base = Path(__file__).absolute().parent
    for name in ('ra_stage', 'ra_inputs'):
        require(name not in sys.modules or Path(sys.modules[name].__file__).absolute() == base / (name + '.py'),
                'foreign helper preloaded')
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, base / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return sys.modules['ra_inputs']


def recipe(stage):
    """Read literal constants from verified registered bytes; never execute them."""
    inputs = load_inputs()
    stage = inputs.absolute(stage)
    inputs.ra_stage.verify(stage)
    path = stage / 'tp4_alpaca.py'
    manifest = inputs.read_json(inputs.absolute(stage) / 'stage-manifest.json')
    require(manifest['files'].get(path.name, {}).get('source_path') == 'bench/tp4/tp4_alpaca.py' and
            manifest['files'][path.name].get('registered') is True, 'registered source builder required')
    tree = ast.parse(path.read_bytes())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
                elif isinstance(target, ast.Tuple) and all(isinstance(n, ast.Name) for n in target.elts):
                    names += [n.id for n in target.elts]
            if set(names) & RECIPE_NAMES:
                require(set(names) <= RECIPE_NAMES and not set(names) & set(values), 'unique literal recipe constants')
                value = ast.literal_eval(node.value)
                if len(names) == 1:
                    values[names[0]] = value
                else:
                    require(isinstance(value, tuple) and len(value) == len(names), 'literal recipe tuple')
                    values.update(zip(names, value))
    require(set(values) == RECIPE_NAMES and values['REPO_ID'] == 'unsloth/alpaca-cleaned' and
            values['FILE'] == SOURCE_NAME and isinstance(values['REVISION'], str) and
            re.fullmatch(r'[0-9a-f]{40}', values['REVISION']) and isinstance(values['FILE_SHA256'], str) and
            re.fullmatch(r'[0-9a-f]{64}', values['FILE_SHA256']), 'fixed source recipe identity')
    require(all(type(values[k]) is int for k in ('SEED', 'N_TRAIN', 'N_EVAL')) and
            (values['SEED'], values['N_TRAIN'], values['N_EVAL']) == (3407, 1200, 48),
            'registered split/seed')
    return values


def project(source, values):
    """Independent implementation of the frozen tp4_alpaca.build projection."""
    inputs = load_inputs()
    require(inputs.file_record(source)['sha256'] == values['FILE_SHA256'], 'registered raw source bytes differ')
    rows = inputs.read_json(source)
    require(isinstance(rows, list) and len(rows) == 51760, 'registered source row count')
    indices = list(range(len(rows)))
    random.Random(values['SEED']).shuffle(indices)
    selected = [rows[i] for i in indices[:values['N_TRAIN'] + values['N_EVAL']]]
    clean = []
    for row in selected:
        require(isinstance(row, dict) and all(isinstance(row.get(k), str) for k in ('instruction', 'output')) and
                isinstance(row.get('input', ''), str), 'selected source text fields')
        clean.append({'instruction': row['instruction'], 'input': row.get('input', ''), 'output': row['output']})
    result = {'source': {'repo_id': values['REPO_ID'], 'revision': values['REVISION'], 'file': values['FILE'],
                         'file_sha256': values['FILE_SHA256']},
              'seed': values['SEED'], 'n_train': values['N_TRAIN'], 'n_eval': values['N_EVAL'],
              'train': clean[:values['N_TRAIN']], 'eval': clean[values['N_TRAIN']:]}
    return json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def check(manifest):
    inputs = load_inputs()
    require(set(manifest) == {'schema', 'input_spec', 'stage', 'input_lock', 'input_lock_sha256'} and
            type(manifest['schema']) is int and manifest['schema'] == 1, 'manifest schema')
    require(isinstance(manifest['input_lock_sha256'], str) and
            re.fullmatch(r'[0-9a-f]{64}', manifest['input_lock_sha256']), 'reviewed lock digest')
    spec = inputs.read_json(manifest['input_spec'])
    common = inputs.verify(spec, manifest['stage'], manifest['input_lock'], manifest['input_lock_sha256'])
    lock = inputs.read_json(manifest['input_lock'])
    return spec, lock, common


def verify(manifest):
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, 'python -I -S -B required')
    inputs = load_inputs()
    base = Path(__file__).absolute().parent
    paths = [base / (n + '.py') for n in PEERS] + [base / 'source-pins.json']
    paths += [inputs.absolute(manifest[k]) for k in ('input_spec', 'input_lock')]
    before = {str(p): inputs.file_record(p) for p in paths}
    spec, lock, common = check(manifest)
    values = recipe(manifest['stage'])
    source = inputs.absolute(spec['trees']['datasets']) / SOURCE_NAME
    source_before = inputs.file_record(source)
    require(lock['trees']['datasets'].get(SOURCE_NAME) == source_before, 'raw source outside reviewed dataset inventory')
    projected = project(source, values)
    data = inputs.absolute(spec['files']['train_data'])
    data_before = inputs.file_record(data)
    require(data_before == {'sha256': hashlib.sha256(projected).hexdigest(), 'size': len(projected)} and
            data.read_bytes() == projected, 'prepared Alpaca projection bytes differ')
    require(check(manifest) == (spec, lock, common) and source_before == inputs.file_record(source) and
            data_before == inputs.file_record(data) and recipe(manifest['stage']) == values and
            {str(p): inputs.file_record(p) for p in paths} == before, 'input/helper/source drift')
    return {'schema': 1, 'recipe': values, 'source': source_before, 'prepared_data': data_before,
            'source_rows': 51760, 'train_rows': 1200, 'eval_rows': 48, 'common_inputs': common,
            'proves_registered_alpaca_source_and_projection': True, 'proves_wikitext_authority': False,
            'proves_calibration_authority': False, 'proves_tokenizer_execution': False,
            'proves_runtime_consumption': False, 'proves_publisher_signature': False,
            'proves_gpu_engagement': False, 'launch_authority': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    inputs = load_inputs()
    manifest = inputs.read_json(args.manifest)
    spec, _, _ = check(manifest)
    out = inputs.absolute(args.out)
    protected = [inputs.absolute(v) for v in spec['trees'].values()] + [inputs.absolute(v) for v in spec['files'].values()]
    protected += [inputs.absolute(manifest[k]) for k in ('input_spec', 'input_lock', 'stage')]
    protected += [inputs.absolute(args.manifest), Path(__file__).absolute().parent]
    require(not out.exists() and all(not (out == p or out in p.parents or p in out.parents) for p in protected),
            'fresh separate receipt directory')
    out.mkdir()
    initial = inputs.file_record(args.manifest)
    try:
        result = verify(manifest)
        require(inputs.file_record(args.manifest) == initial, 'manifest changed')
        result.update(status='PASS', pid=os.getpid(), manifest_sha256=initial['sha256'])
    except Exception as error:
        result = {'schema': 1, 'status': 'FAILED', 'error_type': type(error).__name__,
                  'pid': os.getpid(), 'manifest_sha256': initial['sha256']}
        raise
    finally:
        with (out / 'result.json').open('x') as dest:
            dest.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
