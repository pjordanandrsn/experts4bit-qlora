"""Synthetic SC2 startup controls, never HTTP/5090 engagement evidence."""
import hashlib
import importlib.util
import json
import os
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location('sc2_worker_fixture', ROOT / 'tests/test_ra_verified_worker.py')
f = importlib.util.module_from_spec(sp)
sp.loader.exec_module(f)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def setup(tmp_path, *, label='warm', nested=False, effect=''):
    manifest, job, python, tool, _ = f.setup(tmp_path, phase='capacity')
    stage = Path(job['stage']['path'])
    model = 'ibm-granite/granite-3.1-3b-a800m-instruct'
    pins = json.loads((stage / 'source-pins.json').read_text())
    pins['models'] = {model: 'synthetic-fixture-revision'}
    rows = [[i] + [0] * 511 for i in range(16)]
    pf = {'rows': rows, 'prompts_sha256': hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    pp = tmp_path / 'prompts.json'
    pp.write_text(json.dumps(pf))
    arena, calib = tmp_path / 'arena', tmp_path / 'calib'
    arena.write_text('synthetic arena')
    calib.write_text('{}')
    inputs = {'battery': 'proof', 'venv': str(python.parent.parent), 'cache': str(tmp_path / 'cache'),
        'threads': 1, 'allocator': 'expandable_segments:True', 'prompts': {'path': str(pp), 'sha256': sha(pp)},
        'deadline_epoch_s': time.time() + 1200, 'startup_s': 10, 'driver_s': 30,
        'fixture': {'E4B_PAGED_MODEL': model, 'E4B_PAGED_REVISION': pins['models'][model],
                    'E4B_PAGED_ARENA': str(arena), 'E4B_PAGED_CALIB': str(calib),
                    'E4B_PAGED_MAX_TOKENS_PER_SEQ': '2048', 'E4B_PAGED_CHUNK_TOKENS': '512'}}
    native = Path(job['native_spec']['path'])
    native.write_text(json.dumps(inputs))
    # Real native plan function, substituted CPU main; only OWN synthetic pins change.
    body = (ROOT / 'bench/sc2/sc2_driver.py').read_text().split('if __name__ == "__main__":')[0]
    stub = """
if __name__ == '__main__':
    from pathlib import Path
    import os
    assert 'ra_probe_e4b' in sys.modules and 'ra_probe_gnf4' in sys.modules
    a = dict(zip(sys.argv[2::2],sys.argv[3::2]))
    mode,rate,n,seed=a['--mode'],float(a['--rate']),int(a['--n']),int(a['--seed'])
    pf=json.loads(Path(a['--prompts']).read_text())
    pl=[list(p) for p in plan(mode,rate,n,seed,16,64,256)]
    row={'base':a['--base'],'model':a['--model'],'profile':a['--profile'],'mode':mode,
         'rate':rate,'n':n,'seed':seed,'max_tokens_range':[64,256],
         'prompts_sha256':pf['prompts_sha256'],'plan':pl,'summary':{'valid':n,'invalid':0},
         'requests':[{'i':i,'prompt_index':p,'planned_s':t,'max_tokens':m,'status':200,'valid':True,
                      'prompt_tokens':512,'prompt_len':512,'completion_tokens':m,'finish_reason':'length',
                      'ttft_s':.1,'tpot_s':.01} for i,(t,p,m) in enumerate(pl)]}
    path=Path(a['--out']);path.write_text(json.dumps(row))
    path.parent.joinpath('target-pid.json').write_text(json.dumps({'pid':os.getpid(),'argv':sys.argv}))
"""
    (stage / 'sc2_driver.py').write_text(body + stub + effect + '\n')
    pins['instruments']['bench/sc2/sc2_driver.py'] = sha(stage / 'sc2_driver.py')
    (stage / 'source-pins.json').write_text(json.dumps(pins))
    (tool / 'source-pins.json').write_bytes((stage / 'source-pins.json').read_bytes())
    staged = json.loads((stage / 'stage-manifest.json').read_text())
    staged['source_pins_sha256'] = sha(stage / 'source-pins.json')
    staged['files']['sc2_driver.py'] = {'source_path': 'bench/sc2/sc2_driver.py', 'registered': True,
                                     'sha256': sha(stage / 'sc2_driver.py')}
    (stage / 'stage-manifest.json').write_text(json.dumps(staged))
    (stage / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + p.name + '\n'
                                    for p in sorted(stage.iterdir()) if p.name != 'SHA256SUMS'))
    (tool / 'ra_capacity.py').write_bytes((ROOT / 'bench/ra/ra_capacity.py').read_bytes())
    parent_out = tmp_path / 'parent-out'
    parent_out.mkdir()
    job.update(tools={n: sha(tool / n) for n in f.worker.TOOLS},
               native_spec={'path': str(native), 'sha256': sha(native)},
               stage={'path': str(stage), 'sha256': sha(stage / 'stage-manifest.json')},
               native_out=str(parent_out / (label + '-driver') / 'frozen'))
    if nested:
        wrapper = """import argparse,json,os,sys
from pathlib import Path
import ra_capacity as c,ra_verified_worker as w
p=argparse.ArgumentParser();p.add_argument('--spec');p.add_argument('--instruments');p.add_argument('--out');a=p.parse_args()
spec=json.loads(Path(a.spec).read_text());out=Path(a.out);out.mkdir()
point=next(p for p in c.POINTS if p[0]==LABEL)
cmd,native,binding=w.sc2_child(spec,Path(a.instruments),out,base='http://127.0.0.1:12345',point=point)
env,_=c.ra_env.clean(os.environ,component='capacity',fixture=c.owned_traces(spec,out)['fixture'],
    venv=Path(spec['venv']),cache=Path(spec['cache']),threads=spec['threads'],allocator=spec['allocator'])
process=c.ra_process.run(cmd,env=env,cwd=out,log=out/'driver.log',receipt=out/'process.json',
    deadline=spec['deadline_epoch_s'],timeout=spec['driver_s'])
evidence=w.check_sc2_child(binding,process,point=point,native=native)
(out/'component.json').write_text(json.dumps(evidence))
""".replace('LABEL', repr(label))
        (tool / 'ra_capacity.py').write_bytes((ROOT / 'bench/ra/ra_capacity.py').read_bytes() +
            b'\n# Synthetic fixture wrapper replaces main only.\n')
        # Keep module API available to the child, substitute only __main__ execution.
        body = (tool / 'ra_capacity.py').read_text().split('if __name__ == "__main__":')[0]
        (tool / 'ra_capacity.py').write_text(body + "if __name__ == '__main__':\n" +
            '\n'.join('    ' + line for line in wrapper.splitlines()) + '\n')
        job.update(native_out=str(tmp_path / 'nested-out'), tools={n: sha(tool / n) for n in f.worker.TOOLS})
    else:
        Path(job['native_out']).parent.mkdir()
        job.update(schema=3, phase='sc2_' + label, sc2={'base': 'http://127.0.0.1:12345', 'parent_out': str(parent_out)},
                   handoff_manifest={'path': str(tmp_path / 'handoff-spec.json'),
                                     'sha256': hashlib.sha256(json.dumps(manifest).encode()).hexdigest()})
    return manifest, job, python, tool


def run(tmp_path, manifest, job, python, tool):
    env, _ = f.worker_environment.clean(os.environ, component='capacity',
        fixture={'E4B_PAGED_MAX_TOKENS_PER_SEQ': '2048', 'E4B_PAGED_CHUNK_TOKENS': '512'},
        venv=Path(manifest['payload']['venv']), cache=tmp_path / 'worker-cache',
        threads=1, allocator='expandable_segments:True')
    return f.run(tmp_path, manifest, job, python, tool, env=env)


@pytest.mark.parametrize('label', ['warm', 'burst', 'end'])
@pytest.mark.parametrize('nested', [False, True])
def test_fixed_sc2_after_startup_same_pid_and_nested_join(tmp_path, label, nested):
    m, j, py, tool = setup(tmp_path, label=label, nested=nested)
    result = run(tmp_path, m, j, py, tool)
    assert result.returncode == 0, result.stderr
    out = Path(j['native_out'])
    target = out / (label + '-driver') / 'frozen' if nested else out
    rec = json.loads((out / (label + '-driver') / 'worker.json' if nested else tmp_path / 'worker.json').read_text())
    assert json.loads((target / 'target-pid.json').read_text())['pid'] == rec['pid']
    filename = 'capacity.json' if label == 'end' else 'capacity_' + label + '.json'
    assert rec['native_receipt_sha256'] == sha(target / filename)
    assert rec['proves_gpu_engagement'] is rec['release_cleared'] is False
    if nested:
        assert json.loads((tmp_path / 'worker.json').read_text())['pid'] != rec['pid']
        ev = json.loads((out / 'component.json').read_text())
        assert ev['pid'] == rec['pid'] and ev['server_startup_verified'] is False


@pytest.mark.parametrize('mutation', ['manifest_pin', 'native_spec', 'unregistered', 'wrong_phase', 'extra_flag',
    'foreign_base', 'base_path', 'base_auth', 'base_query', 'base_ipv6', 'output', 'schema', 'point_override'])
def test_driver_entry_mutants_refuse_before_marker(tmp_path, mutation):
    m, j, py, tool = setup(tmp_path)
    if mutation == 'manifest_pin':
        j['handoff_manifest']['sha256'] = '0' * 64
    elif mutation == 'native_spec':
        j['native_spec']['sha256'] = '0' * 64
    elif mutation == 'unregistered':
        p = Path(j['stage']['path']) / 'stage-manifest.json'
        data = json.loads(p.read_text())
        data['files']['sc2_driver.py']['registered'] = False
        p.write_text(json.dumps(data))
        j['stage']['sha256'] = sha(p)
    elif mutation == 'wrong_phase':
        j['phase'] = 'sc2_external'
    elif mutation == 'extra_flag':
        p = Path(j['native_spec']['path'])
        data = json.loads(p.read_text())
        data['extra_flags'] = ['--n', '1']
        p.write_text(json.dumps(data))
        j['native_spec']['sha256'] = sha(p)
    elif mutation == 'output':
        j['native_out'] = str(tmp_path / 'foreign-out')
    elif mutation == 'schema':
        j['schema'] = 2
    elif mutation == 'point_override':
        j['sc2']['seed'] = 0
    else:
        j['sc2']['base'] = {'foreign_base': 'http://example.invalid:12345', 'base_path': 'http://127.0.0.1:1/',
            'base_auth': 'http://user@127.0.0.1:1', 'base_query': 'http://127.0.0.1:1?q=1',
            'base_ipv6': 'http://[::1]:1'}[mutation]
    result = run(tmp_path, m, j, py, tool)
    assert result.returncode != 0
    assert not (Path(j['native_out']) / 'target-pid.json').exists()


@pytest.mark.parametrize('effect', ["    sys.path.append('/foreign')", "    sys.meta_path.reverse()",
    "    os.environ['E4B_FUSED_RMSNORM']='1'", "    path.write_text('{}')", "    raise SystemExit(5)",
    "    Path(__file__).write_text('mutated')"])
def test_nested_mutants_retain_failed_child_and_parent(tmp_path, effect):
    m, j, py, tool = setup(tmp_path, nested=True, effect=effect)
    result = run(tmp_path, m, j, py, tool)
    assert result.returncode != 0
    out = Path(j['native_out'])
    assert json.loads((out / 'warm-driver/handoff.json').read_text())['status'] == 'PASSED'
    assert json.loads((out / 'warm-driver/worker.json').read_text())['status'] == 'FAILED'
    assert json.loads((tmp_path / 'worker.json').read_text())['status'] == 'FAILED'
    assert json.loads((out / 'process.json').read_text())['status'] == 'PROCESS_FAILED'


def test_capacity_wrapper_wires_all_three_fixed_drivers(tmp_path, monkeypatch):
    sp = importlib.util.spec_from_file_location('capacity_composition', ROOT / 'tests/test_ra_capacity.py')
    c = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(c)
    calls = []
    def child(spec, stage, out, *, base, point):
        target = out / (point[0] + '-driver')
        target.mkdir()
        path = target / 'native.json'
        cmd = c.capacity.driver_command(Path(spec['venv']) / 'bin/python', stage, base,
            c.capacity.ra_training.MODEL[spec['battery']][1], spec['prompts']['path'], point, path)
        return cmd, path, {'synthetic': point[0]}
    def join(binding, process, *, point, native):
        assert binding['synthetic'] == point[0] and native.is_file()
        calls.append(point[0])
        return {'synthetic': point[0], 'proves_gpu_engagement': False, 'native_receipt_sha256': sha(native)}
    monkeypatch.setitem(c.sys.modules, 'ra_verified_worker', types.SimpleNamespace(CURRENT={},
                       sc2_child=child, check_sc2_child=join))
    c.test_owned_socket_sequence_retention_and_failure_cleanup(tmp_path, monkeypatch, None)
    record = json.loads((tmp_path / 'out/server.json').read_text())
    assert calls == ['warm', 'burst', 'end']
    assert len(record['verified_driver_startups']) == 3
    assert record['server_startup_verified'] is record['nested_workers_verified'] is False


@pytest.mark.parametrize('mutation', [None, 'process_status', 'cleanup', 'pid', 'worker_status', 'worker_phase',
    'phase', 'spec_sha', 'handoff_sha', 'handoff_status', 'evidence', 'native_hash', 'native_bytes',
    'manifest_bytes', 'spec_bytes', 'missing_receipt'])
def test_child_join_requires_receipts_and_native_bytes(tmp_path, mutation):
    # Supplied observations exercise only the join; they are never startup/GPU proof.
    mp, wp, hp, rp, np = [tmp_path / n for n in ('manifest', 'spec', 'handoff', 'worker', 'native')]
    mp.write_text('{}')
    wp.write_text('{}')
    np.write_text('{}')
    handoff = {'status': 'PASSED', 'manifest_sha256': sha(mp)}
    hp.write_text(json.dumps(handoff))
    record = {'status': 'WORKER_RETURNED_PENDING_GATES', 'phase': 'COMPLETE', 'worker_phase': 'sc2_warm',
              'pid': 123, 'spec_sha256': sha(wp), 'handoff_sha256': sha(hp), 'handoff_evidence': handoff,
              'native_receipt_sha256': sha(np)}
    process = {'status': 'OK', 'cleanup_complete': True, 'pid': 123}
    binding = {'manifest': {'path': str(mp), 'sha256': sha(mp)},
               'worker_spec': {'path': str(wp), 'sha256': sha(wp)},
               'handoff_receipt': str(hp), 'worker_receipt': str(rp)}
    changes = {'pid': ('pid', 124), 'worker_status': ('status', 'FAILED'),
               'worker_phase': ('worker_phase', 'sc2_burst'), 'phase': ('phase', 'WORKER'),
               'spec_sha': ('spec_sha256', '0' * 64), 'handoff_sha': ('handoff_sha256', '0' * 64),
               'evidence': ('handoff_evidence', {}), 'native_hash': ('native_receipt_sha256', '0' * 64)}
    if mutation in changes:
        key, value = changes[mutation]
        record[key] = value
    elif mutation == 'process_status':
        process['status'] = 'TIMEOUT'
    elif mutation == 'cleanup':
        process['cleanup_complete'] = False
    elif mutation == 'handoff_status':
        handoff['status'] = 'FAILED'
        hp.write_text(json.dumps(handoff))
        record['handoff_sha256'] = sha(hp)
    elif mutation in ('native_bytes', 'manifest_bytes', 'spec_bytes'):
        {'native_bytes': np, 'manifest_bytes': mp, 'spec_bytes': wp}[mutation].write_text('changed')
    rp.write_text(json.dumps(record))
    if mutation == 'missing_receipt':
        binding['worker_receipt'] = str(tmp_path / 'absent')
    point = ('warm', 'serial', 0, 4, 999)
    if mutation:
        with pytest.raises(ValueError):
            f.worker.check_sc2_child(binding, process, point=point, native=np)
    else:
        result = f.worker.check_sc2_child(binding, process, point=point, native=np)
        assert result['proves_gpu_engagement'] is result['release_cleared'] is result['server_startup_verified'] is False


def test_single_driver_reserves_its_phase_without_repeating_parent_sequence(tmp_path):
    sp = importlib.util.spec_from_file_location('capacity_deadline_fixture', ROOT / 'tests/test_ra_capacity.py')
    c = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(c)
    spec, stage, _ = c.prepared(tmp_path)
    spec['deadline_epoch_s'] = time.time() + 660
    with pytest.raises(ValueError, match='ten minutes'):
        c.capacity.prepared(spec, stage)
    assert c.capacity.prepared(spec, stage, driver_only=True)[0] == spec['fixture']['E4B_PAGED_MODEL']


@pytest.mark.parametrize('mutation', [None, 'source_bytes', 'existing_destination'])
def test_native_retention_checks_joined_bytes_and_fresh_projection(tmp_path, mutation):
    sp = importlib.util.spec_from_file_location('capacity_retention_fixture', ROOT / 'tests/test_ra_capacity.py')
    c = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(c)
    source, dest = tmp_path / 'native', tmp_path / 'projection'
    source.write_bytes(b'{"native":"exact bytes"}\n')
    expected = sha(source)
    if mutation == 'source_bytes':
        source.write_bytes(b'changed after child join')
    elif mutation == 'existing_destination':
        dest.write_bytes(b'previous observation')
    if mutation:
        with pytest.raises((ValueError, FileExistsError)):
            c.capacity.retain_driver(source, dest, expected)
        assert not dest.exists() or dest.read_bytes() == b'previous observation'
    else:
        c.capacity.retain_driver(source, dest, expected)
        assert dest.read_bytes() == source.read_bytes()
