"""Synthetic frozen TC1 and nested startup controls, never GPU evidence."""
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tc1_worker_fixture', ROOT / 'tests/test_ra_verified_worker.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def setup(tmp_path, *, profile=False, nested=False, effect=''):
    phase = 'training_profile' if profile else 'training'
    manifest, job, python, tool, marker = f.setup(tmp_path, phase=phase)
    stage = Path(job['stage']['path'])
    pins = json.loads((stage / 'source-pins.json').read_text())
    pins['models'] = {'ibm-granite/granite-3.1-3b-a800m-instruct': 'synthetic-fixture-revision'}
    inputs = {'battery': 'proof', 'venv': str(python.parent.parent), 'cache': str(tmp_path / 'cache'),
              'threads': 1, 'allocator': 'expandable_segments:True', 'expect_trainable': 100,
              'deadline_epoch_s': time.time() + 1200, 'timeout_s': 45}
    for n in ('data', 'prereg'):
        path = tmp_path / n
        path.write_text('synthetic fixture\n')
        inputs[n] = {'path': str(path), 'sha256': sha(path)}
    body = {'train': [list(range(8))], 'eval': [list(range(8)) for _ in range(8)]}
    tk = {**body, 'fam': 'granite', 'tokenizer': list(pins['models'])[0],
          'revision': list(pins['models'].values())[0], 'dataset_sha256': inputs['data']['sha256'],
          'template': 'alpaca', 'seq': 2048, 'sha256': hashlib.sha256(
              json.dumps(body, separators=(',', ':')).encode()).hexdigest()}
    token = tmp_path / 'tokens'
    token.write_text(json.dumps(tk))
    inputs['tokens'] = {'path': str(token), 'sha256': sha(token)}
    native = Path(job['native_spec']['path'])
    native.write_text(json.dumps(inputs))
    row = {'status': 'CPU_SYNTHETIC_ONLY', 'framework': 'e4b', 'arm': 'fused', 'fam': 'granite',
           'model': tk['tokenizer'], 'revision': tk['revision'], 'tag': phase, 'steps': 20, 'seq': 2048,
           'micro_batch': 2, 'accum': 4, 'autocast': False, 'r': 16, 'alpha': 16, 'lr': 2e-4,
           'seed': 3407, 'offload': False, 'attn_4bit': True, 'adapter_dtype': 'fp32',
           'lora_init': 'matched:3407', 'dgrad': True, 'template': 'alpaca', 'expect_trainable': 100,
           'profile_steps': 10 if profile else 0, 'profile_warm': 10, 'prereg': inputs['prereg']['path'],
           'tokens': {'sha256': tk['sha256'], 'pack': False, 'eval_rows_used': 8},
           'optimizer': 'adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5',
           'eval_curve': [{'step': 0}, {'step': 20}],
           'lr_per_step': [round(2e-4 * min(i / 5, (20 - i) / 15), 8) for i in range(20)],
           'profile': {'profiled_steps': 10} if profile else None}
    stub = ('import sys,os,json\nfrom pathlib import Path\n'
            "assert 'ra_probe_e4b' in sys.modules and 'ra_probe_gnf4' in sys.modules\n"
            'options=dict(zip(sys.argv[1::2],sys.argv[2::2]))\n'
            "assert options['--steps']=='20' and options['--autocast']=='0'\n"
            "assert options['--profile-steps']==" + repr('10' if profile else '0') + '\n'
            "path=Path(options['--out'])/" + repr('granite_e4b_' + phase + '.json') + '\n'
            'path.write_text(json.dumps(' + repr(row) + '))\n'
            "path.parent.parent.joinpath('target-pid.json').write_text(json.dumps({'pid':os.getpid(),'argv':sys.argv}))\n"
            + effect + '\n')
    (stage / 'tc1_arm.py').write_text(stub)
    pins['instruments']['bench/tc1/tc1_arm.py'] = sha(stage / 'tc1_arm.py')
    (stage / 'source-pins.json').write_text(json.dumps(pins))
    (tool / 'source-pins.json').write_bytes((stage / 'source-pins.json').read_bytes())
    staged = json.loads((stage / 'stage-manifest.json').read_text())
    staged['source_pins_sha256'] = sha(stage / 'source-pins.json')
    staged['files']['tc1_arm.py'] = {'source_path': 'bench/tc1/tc1_arm.py', 'registered': True,
                                   'sha256': sha(stage / 'tc1_arm.py')}
    (stage / 'stage-manifest.json').write_text(json.dumps(staged))
    (stage / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + p.name + '\n'
                                    for p in sorted(stage.iterdir()) if p.name != 'SHA256SUMS'))
    (tool / 'ra_training.py').write_bytes((ROOT / 'bench/ra/ra_training.py').read_bytes())
    job.update(tools={n: sha(tool / n) for n in f.worker.TOOLS},
               native_spec={'path': str(native), 'sha256': sha(native)},
               stage={'path': str(stage), 'sha256': sha(stage / 'stage-manifest.json')},
               native_out=str(tmp_path / 'native-result'))
    if not nested:
        job.update(schema=2, phase='tc1_' + phase,
                   handoff_manifest={'path': str(tmp_path / 'handoff-spec.json'),
                                     'sha256': hashlib.sha256(json.dumps(manifest).encode()).hexdigest()})
    return manifest, job, python, tool, marker


def run(tmp_path, manifest, job, python, tool):
    # TC1's fixed entry phases are worker-spec values. Both training fixtures
    # use the production empty feature environment, even when the job mutates.
    env, _ = f.worker_environment.clean(os.environ, component='training', fixture={},
        venv=Path(manifest['payload']['venv']), cache=tmp_path / 'worker-cache',
        threads=1, allocator='expandable_segments:True')
    return f.run(tmp_path, manifest, job, python, tool, env=env)


@pytest.mark.parametrize('profile', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_fixed_tc1_after_verified_startup_same_pid_and_nested_receipt(tmp_path, profile, nested):
    manifest, job, python, tool, _ = setup(tmp_path, profile=profile, nested=nested)
    p = run(tmp_path, manifest, job, python, tool)
    assert p.returncode == 0, p.stderr
    out = Path(job['native_out'])
    target = out / 'frozen' if nested else out
    row = json.loads((target / 'target-pid.json').read_text())
    receipt = json.loads((out / 'child-worker.json' if nested else tmp_path / 'worker.json').read_text())
    assert row['pid'] == receipt['pid']
    assert row['argv'][0] == str(Path(job['stage']['path']) / 'tc1_arm.py')
    assert receipt['native_receipt_sha256'] == sha(target / 'native' /
        ('granite_e4b_training_profile.json' if profile else 'granite_e4b_training.json'))
    assert receipt['proves_gpu_engagement'] is receipt['release_cleared'] is False
    if nested:
        parent = json.loads((tmp_path / 'worker.json').read_text())
        component = json.loads((out / 'component.json').read_text())
        assert parent['pid'] != row['pid']
        assert component['verified_child_startup']['pid'] == row['pid']
        assert component['verified_child_startup']['proves_gpu_engagement'] is False


@pytest.mark.parametrize('mutation', ['manifest_pin', 'native_spec', 'stage_body', 'unregistered', 'wrong_phase',
                                      'extra_flag', 'interpreter'])
def test_native_entry_mutants_refuse(tmp_path, mutation):
    manifest, job, python, tool, _ = setup(tmp_path)
    if mutation == 'manifest_pin':
        job['handoff_manifest']['sha256'] = '0' * 64
    elif mutation == 'native_spec':
        job['native_spec']['sha256'] = '0' * 64
    elif mutation == 'stage_body':
        (Path(job['stage']['path']) / 'tc1_arm.py').write_text('raise RuntimeError("mutant")')
    elif mutation == 'unregistered':
        stage = Path(job['stage']['path'])
        staged = json.loads((stage / 'stage-manifest.json').read_text())
        staged['files']['tc1_arm.py']['registered'] = False
        (stage / 'stage-manifest.json').write_text(json.dumps(staged))
        job['stage']['sha256'] = sha(stage / 'stage-manifest.json')
    elif mutation == 'wrong_phase':
        job['phase'] = 'tc1_external'
    else:
        path = Path(job['native_spec']['path'])
        data = json.loads(path.read_text())
        data['extra_flags' if mutation == 'extra_flag' else 'venv'] = ['--steps', '1'] if mutation == 'extra_flag' else '/foreign'
        path.write_text(json.dumps(data))
        job['native_spec']['sha256'] = sha(path)
    p = run(tmp_path, manifest, job, python, tool)
    assert p.returncode != 0
    assert not (Path(job['native_out']) / 'target-pid.json').exists()


@pytest.mark.parametrize('effect', ["sys.path.append('/foreign')", "sys.meta_path.reverse()",
    "os.environ['E4B_FUSED_RMSNORM']='1'", "path.write_text('{}')", "raise SystemExit(5)",
    "Path(__file__).write_text('mutated')"])
def test_nested_failures_preserve_handoff_and_refuse_parent_success(tmp_path, effect):
    manifest, job, python, tool, _ = setup(tmp_path, nested=True, effect=effect)
    p = run(tmp_path, manifest, job, python, tool)
    assert p.returncode != 0
    out = Path(job['native_out'])
    assert json.loads((out / 'child-handoff.json').read_text())['status'] == 'PASSED'
    assert json.loads((out / 'child-worker.json').read_text())['status'] == 'FAILED'
    assert json.loads((tmp_path / 'worker.json').read_text())['status'] == 'FAILED'
    assert json.loads((out / 'process.json').read_text())['status'] == 'PROCESS_FAILED'


@pytest.mark.parametrize('mutation', [None, 'process_status', 'cleanup', 'pid', 'worker_status', 'worker_phase',
                                      'phase', 'spec_sha', 'handoff_sha', 'handoff_status', 'evidence',
                                      'manifest_bytes', 'spec_bytes', 'missing_receipt'])
def test_child_receipt_join_cannot_be_replaced_by_exit_zero(tmp_path, mutation):
    # Caller-provided synthetic observations exercise the join only, never
    # establish real startup, process identity, GPU engagement or clearance.
    mp, wp, hp, rp = [tmp_path / n for n in ('manifest', 'spec', 'handoff', 'worker')]
    mp.write_text('{}')
    wp.write_text('{}')
    handoff = {'status': 'PASSED', 'manifest_sha256': sha(mp)}
    hp.write_text(json.dumps(handoff))
    record = {'status': 'WORKER_RETURNED_PENDING_GATES', 'phase': 'COMPLETE', 'worker_phase': 'tc1_training',
              'pid': 123, 'spec_sha256': sha(wp), 'handoff_sha256': sha(hp), 'handoff_evidence': handoff}
    process = {'status': 'OK', 'cleanup_complete': True, 'pid': 123}
    binding = {'manifest': {'path': str(mp), 'sha256': sha(mp)},
               'worker_spec': {'path': str(wp), 'sha256': sha(wp)},
               'handoff_receipt': str(hp), 'worker_receipt': str(rp)}
    if mutation == 'process_status':
        process['status'] = 'TIMEOUT'
    elif mutation == 'cleanup':
        process['cleanup_complete'] = False
    elif mutation == 'pid':
        record['pid'] += 1
    elif mutation == 'worker_status':
        record['status'] = 'FAILED'
    elif mutation == 'worker_phase':
        record['worker_phase'] = 'tc1_training_profile'
    elif mutation == 'phase':
        record['phase'] = 'WORKER'
    elif mutation == 'spec_sha':
        record['spec_sha256'] = '0' * 64
    elif mutation == 'handoff_sha':
        record['handoff_sha256'] = '0' * 64
    elif mutation == 'handoff_status':
        handoff['status'] = 'FAILED'
        hp.write_text(json.dumps(handoff))
        record['handoff_sha256'] = sha(hp)
    elif mutation == 'evidence':
        record['handoff_evidence'] = {}
    elif mutation == 'manifest_bytes':
        mp.write_text('changed')
    elif mutation == 'spec_bytes':
        wp.write_text('changed')
    rp.write_text(json.dumps(record))
    if mutation == 'missing_receipt':
        binding['worker_receipt'] = str(tmp_path / 'absent')
    if mutation:
        with pytest.raises(ValueError):
            f.worker.check_tc1_child(binding, process, profile=False)
    else:
        result = f.worker.check_tc1_child(binding, process, profile=False)
        assert result['proves_gpu_engagement'] is result['release_cleared'] is False
