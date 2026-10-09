"""Fixed worker continuation after same-process handoff; no launch authority."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import runpy
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).absolute().parent
PYTOOLS = set('''ra_abba ra_capacity ra_capacity_server ra_child_guard ra_closure ra_env ra_fallback
ra_handoff ra_inputs ra_normalize ra_prestartup ra_process ra_provenance ra_publication ra_quality
ra_reduce ra_routes ra_serving ra_source_binding ra_source_metadata ra_stage ra_trace ra_training
ra_wheel_lock ra_verified_worker'''.split())
TOOLS = {n + '.py' for n in PYTOOLS} | {'fallback-adapters.json', 'route-adapters.json', 'source-pins.json',
    'startup-audit-adapters.json', 'startup-execution-proposal.json', 'proof-image.json',
    'proof-wheel-lock.json', 'PREREG-ra.md'}
TARGETS = {'training': 'ra_training.py', 'training_profile': 'ra_training.py',
           'capacity': 'ra_capacity.py', 'decode': 'ra_serving.py', 'quality': 'ra_serving.py'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and
            not any(p.is_symlink() for p in (path, *path.parents)), 'worker regular absolute file')
    return path


def digest(path):
    return hashlib.sha256(regular(path).read_bytes()).hexdigest()


def check_tools(pins):
    require(set(pins) == TOOLS, 'worker complete tool pin set')
    require({p.stem for p in HERE.glob('ra_*.py')} == PYTOOLS, 'worker unknown/missing RA Python tool')
    for name, expected in pins.items():
        require(digest(HERE / name) == expected, 'worker tool bytes: ' + name)
    for name, module in sys.modules.items():
        if name.split('.')[0] in PYTOOLS:
            require(Path(getattr(module, '__file__', '')).absolute() == HERE / (name.split('.')[0] + '.py'),
                    'worker preloaded RA module owner')


def argv(spec):
    phase = spec['phase']
    target = HERE / TARGETS[phase]
    args = [str(target), '--spec', spec['native_spec']['path'], '--instruments', spec['stage']['path'],
            '--out', spec['native_out']]
    if phase == 'training_profile':
        args.append('--profile')
    elif phase in ('decode', 'quality'):
        args.extend(['--mode', phase])
    return args


def execute(spec, spec_path, spec_hash, handoff, handoff_path, manifest):
    require(set(spec) == {'schema', 'phase', 'tools', 'native_spec', 'stage', 'native_out', 'receipt'} and
            type(spec['schema']) is int and spec['schema'] == 1 and spec['phase'] in TARGETS,
            'worker spec fields/phase')
    require(handoff['status'] == 'PASSED' and handoff['phase'] == 'COMPLETE' and
            handoff['payload']['proves_installed_payload'] and handoff['proves_requested_release_imports'],
            'completed same-process handoff required')
    receipt = Path(spec['receipt'])
    output = Path(spec['native_out'])
    require(receipt.is_absolute() and not receipt.exists() and output.is_absolute() and not output.exists() and
            not any(p.is_symlink() for p in (receipt, *receipt.parents, output, *output.parents)),
            'fresh worker output/receipt')
    require(receipt != output and output not in receipt.parents and receipt not in output.parents,
            'worker receipt/output overlap')
    record = {'schema': 1, 'status': 'FAILED', 'phase': 'PRECONDITIONS', 'worker_phase': spec['phase'],
              'spec_sha256': spec_hash, 'handoff_sha256': digest(handoff_path), 'pid': os.getpid(),
              'handoff_evidence': json.loads(json.dumps(handoff)),
              'proves_gpu_engagement': False, 'release_cleared': False,
              'nested_workers_verified': False}
    try:
        check_tools(spec['tools'])
        require(digest(spec_path) == spec_hash, 'worker spec changed')
        for pin in (spec['native_spec'], spec['stage']):
            require(set(pin) == {'path', 'sha256'}, 'worker input pin fields')
        native = regular(spec['native_spec']['path'])
        stage = Path(spec['stage']['path'])
        require(stage.is_absolute() and stage.is_dir() and
                not any(p.is_symlink() for p in (stage, *stage.parents)), 'worker regular stage')
        protected = [HERE, stage, Path(handoff['payload']['venv']), native, Path(spec_path), Path(handoff_path)]
        require(not any(a == b or a in b.parents or b in a.parents for a in (receipt, output) for b in protected),
                'worker output overlaps protected input')
        require(digest(native) == spec['native_spec']['sha256'] and
                digest(stage / 'stage-manifest.json') == spec['stage']['sha256'], 'worker native/stage pin')
        # RA modules become importable only after complete tool hashes pass.
        paths, environment, old_argv = list(sys.path), dict(os.environ), list(sys.argv)
        require(str(HERE) not in sys.path and str(stage) not in sys.path, 'worker preloaded tool/stage path')
        sys.path.insert(0, str(HERE))
        ra_stage = importlib.import_module('ra_stage')
        ra_env = importlib.import_module('ra_env')
        staged = ra_stage.verify(stage)
        require(digest(stage / 'source-pins.json') == spec['tools']['source-pins.json'], 'worker registered source pins')
        ra_env.check_current(spec['phase'], os.environ)
        pre = importlib.import_module('ra_handoff').load_prestartup()
        helper = pre.load_provenance()
        sites = {Path(sysconfig.get_path(k, vars={'base': handoff['payload']['venv'],
                                                'platbase': handoff['payload']['venv']}))
                 for k in ('purelib', 'platlib')}
        config = Path(handoff['payload']['venv']) / 'pyvenv.cfg'
        executable = Path(sys.executable).resolve()
        executable_hash = handoff['payload']['executable_sha256']
        base = Path(sys._base_executable).resolve()
        prefix = sys.prefix
        meta = list(sys.meta_path)
        archive_hashes = {regular(p['path']): p['sha256'] for p in manifest['payload']['wheels']}
        image = regular(manifest['payload']['image'])
        installed = {p: digest(p) for site in sites for p in site.rglob('*') if p.is_file()}
        require(hashlib.sha256(json.dumps({str(p.relative_to(Path(prefix))): h for p, h in installed.items()},
                sort_keys=True, separators=(',', ':')).encode()).hexdigest() == handoff['installed_inventory_sha256'],
                'worker installed bytes changed since handoff')
        require(all(digest(p) == expected for p, expected in archive_hashes.items()) and
                digest(config) == handoff['payload']['pyvenv_cfg_sha256'] and
                digest(image) == manifest['payload']['image_sha256'], 'worker bootstrap inputs changed before target')
        # Only wheel payload, fixed verified tools, frozen stage and stdlib may
        # be loaded by the continuation. This is not per-call/GPU engagement.
        allowed = {HERE / n for n in TOOLS} | {stage / n for n in staged['files']}
        stdlib = Path(sysconfig.get_path('stdlib')).resolve()
        owners = {p: 'verified-installation' for p in installed}
        record.update(phase='WORKER', tools_sha256=hashlib.sha256(
            json.dumps(spec['tools'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            nested_workers_verified=False)
        sys.argv = argv(spec)
        try:
            try:
                runpy.run_path(sys.argv[0], run_name='__main__')
            except SystemExit as error:
                require(error.code is None or type(error.code) is int and error.code == 0, 'worker nonzero SystemExit')
        finally:
            sys.argv = old_argv
        record['phase'] = 'POST_WORKER'
        check_tools(spec['tools'])
        require(digest(spec_path) == spec_hash and digest(native) == spec['native_spec']['sha256'],
                'worker spec/native bytes changed')
        ra_stage.verify(stage)
        require(output.exists(), 'worker native output missing')
        require(digest(handoff_path) == record['handoff_sha256'] and
                digest(image) == manifest['payload']['image_sha256'], 'worker bootstrap evidence changed')
        require(all(digest(p) == expected for p, expected in archive_hashes.items()), 'worker archive changed')
        require(all(not p.is_symlink() for site in sites for p in site.rglob('*')), 'worker site symlink')
        require({p: digest(p) for site in sites for p in site.rglob('*') if p.is_file()} == installed,
                'worker installed bytes changed')
        require(Path(sys.executable).resolve() == executable and digest(executable) == executable_hash and
                Path(sys._base_executable).resolve() == base and
                digest(base) == handoff['payload']['base_executable_sha256'] and sys.prefix == prefix and
                digest(config) == handoff['payload']['pyvenv_cfg_sha256'], 'worker interpreter/config changed')
        require(sys.path in ( [str(HERE), *paths], [str(stage), str(HERE), *paths]) and
                dict(os.environ) == environment, 'worker search path/environment changed')
        require(len(sys.meta_path) == len(meta) and all(a is b for a, b in zip(sys.meta_path, meta)),
                'worker finder order changed')
        importlib.import_module('ra_handoff').origins(helper, installed, owners, stdlib, allowed)
        record.update(status='WORKER_RETURNED_PENDING_GATES', phase='COMPLETE')
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with receipt.open('x') as stream:
            stream.write(json.dumps(record, sort_keys=True, indent=2) + '\n')
    return record
