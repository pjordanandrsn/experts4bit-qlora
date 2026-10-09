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
NATIVE = {'tc1_training': 'training', 'tc1_training_profile': 'training_profile',
          'sc2_warm': 'capacity', 'sc2_burst': 'capacity', 'sc2_end': 'capacity', 'capacity_server': 'capacity'}
CURRENT = None  # Only installed by this module during a verified wrapper continuation.
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
    native_phase = spec.get('phase') in NATIVE
    sc2_phase = spec.get('phase', '').startswith('sc2_')
    server_phase = spec.get('phase') == 'capacity_server'
    fields = {'schema', 'phase', 'tools', 'native_spec', 'stage', 'native_out', 'receipt'}
    require(set(spec) == fields | ({'handoff_manifest'} if native_phase else set()) | ({'sc2'} if sc2_phase else {'server'} if server_phase else set()) and
            type(spec['schema']) is int and spec['schema'] == (4 if server_phase else 3 if sc2_phase else 2 if native_phase else 1) and
            spec['phase'] in (NATIVE if native_phase else TARGETS), 'worker spec fields/phase')
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
        ra_env.check_current(NATIVE.get(spec['phase'], spec['phase']), os.environ)
        if native_phase:
            pin = spec['handoff_manifest']
            require(set(pin) == {'path', 'sha256'} and digest(pin['path']) == pin['sha256'] and
                    json.loads(regular(pin['path']).read_bytes()) == manifest, 'nested handoff manifest binding')
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
        global CURRENT
        require(CURRENT is None, 'recursive in-process worker context')
        if server_phase:
            capacity = importlib.import_module('ra_capacity')
            native_spec = json.loads(native.read_bytes())
            server_pin, parent_out = server_arguments(spec)
            capacity.prepared(capacity.owned_traces(native_spec, parent_out), stage)
            require(Path(native_spec['venv']) / 'bin/python' == Path(sys.executable), 'server selected interpreter')
            output.mkdir()
            generated = [str(HERE / 'ra_capacity_server.py'), '--socket-fd', str(server_pin['listener']['fd']),
                '--parent-pid', str(server_pin['parent_pid']), '--instruments', str(stage),
                '--receipt', str(output / 'lifecycle.json')]
            sys.argv = generated
        elif sc2_phase:
            capacity = importlib.import_module('ra_capacity')
            native_spec = json.loads(native.read_bytes())
            point, base_url, parent_out = sc2_arguments(spec)
            effective = capacity.owned_traces(native_spec, parent_out)
            model, prompts, _, _, driver = capacity.prepared(effective, stage, driver_only=True)
            generated = capacity.driver_command(Path(native_spec['venv']) / 'bin/python', stage, base_url,
                model, native_spec['prompts']['path'], point, output / sc2_filename(point))
            require(Path(generated[0]).resolve() == executable, 'nested selected interpreter')
            require('sc2_driver.py' in staged['files'] and staged['files']['sc2_driver.py']['registered'],
                    'nested SC2 frozen instrument required')
            output.mkdir()
            sys.path.insert(0, str(stage))
            sys.argv = generated[2:]
        elif native_phase:
            training = importlib.import_module('ra_training')
            native_spec = json.loads(native.read_bytes())
            generated, _, _ = training.command(native_spec, stage, output,
                                                profile=spec['phase'] == 'tc1_training_profile')
            require(Path(generated[0]).resolve() == executable, 'nested selected interpreter')
            require('tc1_arm.py' in staged['files'] and staged['files']['tc1_arm.py']['registered'],
                    'nested TC1 frozen instrument required')
            output.mkdir()
            (output / 'native').mkdir()
            (output / 'adapters').mkdir()
            sys.path.insert(0, str(stage))
            sys.argv = generated[2:]
        else:
            sys.argv = argv(spec)
            CURRENT = {'spec': spec, 'spec_path': str(spec_path), 'spec_hash': spec_hash,
                       'manifest': manifest, 'handoff': handoff, 'handoff_path': str(handoff_path)}
        try:
            try:
                runpy.run_path(sys.argv[0], run_name='__main__')
            except SystemExit as error:
                require(error.code is None or type(error.code) is int and error.code == 0, 'worker nonzero SystemExit')
        finally:
            CURRENT = None
            sys.argv = old_argv
        if server_phase:
            capacity.ra_process.check_input(native_spec['prompts'])
            result = json.loads(regular(output / 'lifecycle.json').read_bytes())
            require(result['status'] == 'SERVER_RETURNED_PENDING_GATES' and result['pid'] == os.getpid() and
                    result['parent_pid'] == server_pin['parent_pid'] and result['listener'] == server_pin['listener'] and
                    result['closed']['closed'] is True and result['closed']['thread_alive'] is False and
                    result['engine_thread_alive'] is False, 'server lifecycle binding')
            require(result['trace_sha256'] == {n: digest(parent_out / n) for n in
                    ('request-trace.jsonl', 'step-trace.jsonl')}, 'server trace bytes changed')
            record.update(native_receipt_sha256=digest(output / 'lifecycle.json'), fixed_argv_sha256=hashlib.sha256(
                json.dumps(generated, separators=(',', ':')).encode()).hexdigest())
        elif sc2_phase:
            capacity.ra_process.check_input(native_spec['prompts'])
            native_result = output / sc2_filename(point)
            capacity.check_native(json.loads(native_result.read_bytes()), point, driver, base_url, model, prompts)
            record.update(native_receipt_sha256=digest(native_result), fixed_argv_sha256=hashlib.sha256(
                json.dumps(generated, separators=(',', ':')).encode()).hexdigest())
        elif native_phase:
            for key in ('data', 'tokens', 'prereg'):
                training.ra_process.check_input(native_spec[key])
            fam, _ = training.MODEL[native_spec['battery']]
            component = NATIVE[spec['phase']]
            native_result = output / 'native' / (fam + '_e4b_' + component + '.json')
            training.check_native(json.loads(native_result.read_bytes()), native_spec,
                json.loads((stage / 'source-pins.json').read_bytes()), profile=component == 'training_profile')
            record.update(native_receipt_sha256=digest(native_result), fixed_argv_sha256=hashlib.sha256(
                json.dumps(generated, separators=(',', ':')).encode()).hexdigest())
        record['phase'] = 'POST_WORKER'
        check_tools(spec['tools'])
        require(digest(spec_path) == spec_hash and digest(native) == spec['native_spec']['sha256'],
                'worker spec/native bytes changed')
        ra_stage.verify(stage)
        if native_phase:
            require(digest(spec['handoff_manifest']['path']) == spec['handoff_manifest']['sha256'],
                    'nested handoff manifest changed')
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


def tc1_child(spec, stage, out, *, profile):
    """Derive a fresh frozen child from the live verified wrapper, never a command."""
    require(CURRENT is not None, 'verified TC1 parent context required')
    parent = CURRENT['spec']
    component = 'training_profile' if profile else 'training'
    require(parent['phase'] == component and parent['native_out'] == str(out) and
            parent['stage']['path'] == str(stage) and
            json.loads(regular(parent['native_spec']['path']).read_bytes()) == spec and
            digest(CURRENT['spec_path']) == CURRENT['spec_hash'], 'nested TC1 parent binding')
    check_tools(parent['tools'])
    mp = out / 'child-handoff-spec.json'
    wp = out / 'child-worker-spec.json'
    hp = out / 'child-handoff.json'
    rp = out / 'child-worker.json'
    target = out / 'frozen'
    with mp.open('x') as stream:
        stream.write(json.dumps(CURRENT['manifest'], sort_keys=True, indent=2) + '\n')
    child = {'schema': 2, 'phase': 'tc1_' + component, 'tools': dict(parent['tools']),
             'native_spec': dict(parent['native_spec']), 'stage': dict(parent['stage']),
             'native_out': str(target), 'receipt': str(rp),
             'handoff_manifest': {'path': str(mp), 'sha256': digest(mp)}}
    with wp.open('x') as stream:
        stream.write(json.dumps(child, sort_keys=True, indent=2) + '\n')
    command = [str(Path(sys.executable)), '-I', '-S', '-B', str(HERE / 'ra_handoff.py'),
               '--manifest', str(mp), '--out', str(hp), '--worker-spec', str(wp), '--worker-sha256', digest(wp)]
    return command, target, {'manifest': child['handoff_manifest'],
        'worker_spec': {'path': str(wp), 'sha256': digest(wp)}, 'handoff_receipt': str(hp), 'worker_receipt': str(rp)}


def check_tc1_child(binding, process, *, profile):
    """Do not promote exit zero without its independently joined child receipts."""
    for name in ('manifest', 'worker_spec'):
        require(digest(binding[name]['path']) == binding[name]['sha256'], 'nested TC1 input changed')
    handoff = json.loads(regular(binding['handoff_receipt']).read_bytes())
    record = json.loads(regular(binding['worker_receipt']).read_bytes())
    require(process['status'] == 'OK' and process.get('cleanup_complete') and
            record['status'] == 'WORKER_RETURNED_PENDING_GATES' and record['phase'] == 'COMPLETE' and
            record['worker_phase'] == ('tc1_training_profile' if profile else 'tc1_training') and
            record['pid'] == process['pid'] and record['spec_sha256'] == binding['worker_spec']['sha256'] and
            record['handoff_sha256'] == digest(binding['handoff_receipt']) and
            handoff['status'] == 'PASSED' and handoff['manifest_sha256'] == binding['manifest']['sha256'] and
            record['handoff_evidence'] == handoff, 'nested TC1 startup receipt join')
    return {'status': 'VERIFIED_TC1_CHILD_STARTUP_PENDING_ENGAGEMENT', 'worker_sha256': digest(binding['worker_receipt']),
            'handoff_sha256': digest(binding['handoff_receipt']), 'pid': process['pid'],
            'proves_gpu_engagement': False, 'release_cleared': False}


def sc2_filename(point):
    return ('capacity' if point[0] == 'end' else 'capacity_' + point[0]) + '.json'


def sc2_arguments(spec):
    from urllib.parse import urlsplit
    import ra_capacity
    pin = spec['sc2']
    require(set(pin) == {'base', 'parent_out'}, 'SC2 fixed invocation fields')
    url = urlsplit(pin['base'])
    require(url.scheme == 'http' and url.hostname == '127.0.0.1' and url.username is None and
            url.password is None and not url.path and not url.query and not url.fragment and
            type(url.port) is int and 0 < url.port < 65536 and
            pin['base'] == 'http://127.0.0.1:' + str(url.port), 'SC2 owned numeric loopback base')
    label = spec['phase'].removeprefix('sc2_')
    point = next(p for p in ra_capacity.POINTS if p[0] == label)
    parent_out = Path(pin['parent_out'])
    require(parent_out.is_absolute() and parent_out.is_dir() and
            not any(p.is_symlink() for p in (parent_out, *parent_out.parents)) and
            spec['native_out'] == str(parent_out / (label + '-driver') / 'frozen'), 'SC2 fixed child output')
    return point, pin['base'], parent_out


def sc2_child(spec, stage, out, *, base, point):
    """Derive one of three fixed driver phases from the live verified capacity wrapper."""
    require(CURRENT is not None, 'verified SC2 parent context required')
    parent = CURRENT['spec']
    require(parent['phase'] == 'capacity' and parent['native_out'] == str(out) and
            parent['stage']['path'] == str(stage) and
            json.loads(regular(parent['native_spec']['path']).read_bytes()) == spec and
            digest(CURRENT['spec_path']) == CURRENT['spec_hash'], 'nested SC2 parent binding')
    import ra_capacity
    require(point in ra_capacity.POINTS, 'SC2 registered point required')
    check_tools(parent['tools'])
    own = out / (point[0] + '-driver')
    own.mkdir(exist_ok=False)
    mp, wp, hp, rp = [own / n for n in ('handoff-spec.json', 'worker-spec.json', 'handoff.json', 'worker.json')]
    with mp.open('x') as stream:
        stream.write(json.dumps(CURRENT['manifest'], sort_keys=True, indent=2) + '\n')
    child = {'schema': 3, 'phase': 'sc2_' + point[0], 'tools': dict(parent['tools']),
             'native_spec': dict(parent['native_spec']), 'stage': dict(parent['stage']),
             'native_out': str(own / 'frozen'), 'receipt': str(rp),
             'sc2': {'base': base, 'parent_out': str(out)},
             'handoff_manifest': {'path': str(mp), 'sha256': digest(mp)}}
    sc2_arguments(child)
    with wp.open('x') as stream:
        stream.write(json.dumps(child, sort_keys=True, indent=2) + '\n')
    command = [str(Path(sys.executable)), '-I', '-S', '-B', str(HERE / 'ra_handoff.py'),
               '--manifest', str(mp), '--out', str(hp), '--worker-spec', str(wp), '--worker-sha256', digest(wp)]
    return command, Path(child['native_out']) / sc2_filename(point), {
        'manifest': child['handoff_manifest'], 'worker_spec': {'path': str(wp), 'sha256': digest(wp)},
        'handoff_receipt': str(hp), 'worker_receipt': str(rp)}


def check_sc2_child(binding, process, *, point, native):
    for name in ('manifest', 'worker_spec'):
        require(digest(binding[name]['path']) == binding[name]['sha256'], 'nested SC2 input changed')
    handoff = json.loads(regular(binding['handoff_receipt']).read_bytes())
    record = json.loads(regular(binding['worker_receipt']).read_bytes())
    require(process['status'] == 'OK' and process.get('cleanup_complete') and
            record['status'] == 'WORKER_RETURNED_PENDING_GATES' and record['phase'] == 'COMPLETE' and
            record['worker_phase'] == 'sc2_' + point[0] and record['pid'] == process['pid'] and
            record['spec_sha256'] == binding['worker_spec']['sha256'] and
            record['handoff_sha256'] == digest(binding['handoff_receipt']) and
            handoff['status'] == 'PASSED' and handoff['manifest_sha256'] == binding['manifest']['sha256'] and
            record['handoff_evidence'] == handoff and record['native_receipt_sha256'] == digest(native),
            'nested SC2 startup receipt join')
    return {'status': 'VERIFIED_SC2_DRIVER_STARTUP_PENDING_GATES', 'worker_sha256': digest(binding['worker_receipt']),
            'handoff_sha256': digest(binding['handoff_receipt']), 'pid': process['pid'],
            'native_receipt_sha256': record['native_receipt_sha256'],
            'proves_gpu_engagement': False, 'release_cleared': False, 'server_startup_verified': False}


def server_arguments(spec):
    import ra_capacity_server
    pin = spec['server']
    require(set(pin) == {'listener', 'parent_pid', 'parent_out'} and
            type(pin['parent_pid']) is int and pin['parent_pid'] > 1 and
            os.getppid() == pin['parent_pid'], 'server live parent binding')
    parent_out = Path(pin['parent_out'])
    require(parent_out.is_absolute() and parent_out.is_dir() and
            not any(p.is_symlink() for p in (parent_out, *parent_out.parents)) and
            spec['native_out'] == str(parent_out / 'server-child' / 'native'), 'server fixed output')
    require(set(pin['listener']) == {'fd', 'device', 'inode', 'address'} and
            ra_capacity_server.listener_identity(pin['listener']['fd']) == pin['listener'],
            'server inherited socket identity')
    return pin, parent_out


def server_child(spec, stage, out, listener):
    require(CURRENT is not None, 'verified server parent context required')
    parent = CURRENT['spec']
    require(parent['phase'] == 'capacity' and parent['native_out'] == str(out) and
            parent['stage']['path'] == str(stage) and
            json.loads(regular(parent['native_spec']['path']).read_bytes()) == spec and
            digest(CURRENT['spec_path']) == CURRENT['spec_hash'], 'nested server parent binding')
    import ra_capacity_server
    check_tools(parent['tools'])
    own = out / 'server-child'
    own.mkdir(exist_ok=False)
    mp, wp, hp, rp = [own / n for n in ('handoff-spec.json', 'worker-spec.json', 'handoff.json', 'worker.json')]
    with mp.open('x') as stream:
        stream.write(json.dumps(CURRENT['manifest'], sort_keys=True, indent=2) + '\n')
    child = {'schema': 4, 'phase': 'capacity_server', 'tools': dict(parent['tools']),
             'native_spec': dict(parent['native_spec']), 'stage': dict(parent['stage']),
             'native_out': str(own / 'native'), 'receipt': str(rp),
             'server': {'listener': ra_capacity_server.listener_identity(listener.fileno()),
                        'parent_pid': os.getpid(), 'parent_out': str(out)},
             'handoff_manifest': {'path': str(mp), 'sha256': digest(mp)}}
    with wp.open('x') as stream:
        stream.write(json.dumps(child, sort_keys=True, indent=2) + '\n')
    command = [str(Path(sys.executable)), '-I', '-S', '-B', str(HERE / 'ra_handoff.py'),
               '--manifest', str(mp), '--out', str(hp), '--worker-spec', str(wp), '--worker-sha256', digest(wp)]
    return command, {'manifest': child['handoff_manifest'],
        'worker_spec': {'path': str(wp), 'sha256': digest(wp)}, 'handoff_receipt': str(hp),
        'worker_receipt': str(rp), 'lifecycle': str(own / 'native' / 'lifecycle.json'), 'server': child['server']}


def check_server_ready(binding, process, evidence):
    for name in ('manifest', 'worker_spec'):
        require(digest(binding[name]['path']) == binding[name]['sha256'], 'server input changed')
    handoff = json.loads(regular(binding['handoff_receipt']).read_bytes())
    require(handoff['status'] == 'PASSED' and handoff['phase'] == 'COMPLETE' and
            handoff['manifest_sha256'] == binding['manifest']['sha256'] and
            handoff['payload']['proves_installed_payload'] and handoff['proves_requested_release_imports'] and
            evidence.get('ready') is True and type(evidence.get('pid')) is int and evidence['pid'] == process['pid'] and
            process['parent_death_guard'].get('verified') is True, 'server ready/startup/guard join')
    return digest(binding['handoff_receipt'])


def check_server_child(binding, process, closed, traces):
    check_server_ready(binding, process, {'ready': True, 'pid': process['pid']})
    handoff = json.loads(regular(binding['handoff_receipt']).read_bytes())
    record = json.loads(regular(binding['worker_receipt']).read_bytes())
    native = json.loads(regular(binding['lifecycle']).read_bytes())
    require(process['status'] == 'OK' and process['returncode'] == 0 and process.get('cleanup_complete') is True and
            record['status'] == 'WORKER_RETURNED_PENDING_GATES' and record['phase'] == 'COMPLETE' and
            record['worker_phase'] == 'capacity_server' and record['pid'] == process['pid'] and
            record['spec_sha256'] == binding['worker_spec']['sha256'] and
            record['handoff_sha256'] == digest(binding['handoff_receipt']) and record['handoff_evidence'] == handoff and
            record['native_receipt_sha256'] == digest(binding['lifecycle']) and
            native['status'] == 'SERVER_RETURNED_PENDING_GATES' and native['pid'] == process['pid'] and
            native['parent_pid'] == binding['server']['parent_pid'] and native['listener'] == binding['server']['listener'] and
            native['closed'] == closed and closed.get('closed') is True and closed.get('thread_alive') is False and
            native['engine_thread_alive'] is False and native['trace_sha256'] == traces,
            'server normal shutdown receipt join')
    return {'status': 'VERIFIED_SERVER_STARTUP_RETURN_PENDING_GATES', 'pid': process['pid'],
            'worker_sha256': digest(binding['worker_receipt']), 'handoff_sha256': digest(binding['handoff_receipt']),
            'lifecycle_sha256': digest(binding['lifecycle']), 'proves_gpu_engagement': False,
            'proves_native_context_absence': False, 'release_cleared': False}
