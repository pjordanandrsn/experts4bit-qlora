"""Synthetic same-process worker controls; no real wrappers/GPU engagement."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ra_worker_fixtures', ROOT / 'tests/test_ra_handoff.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
spec = importlib.util.spec_from_file_location('worker_controls', ROOT / 'bench/ra/ra_verified_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(tmp_path, *, phase='decode', effect='', hook=True, copies=False):
    manifest, python, site, tool, _, marker = fixtures.setup(tmp_path, hook=hook, copies=copies)
    # Each control owns its narrowed synthetic pins and substituted target.
    # Production source pins and real wrappers are never modified.
    for name in worker.TOOLS:
        if name != 'source-pins.json' and not (tool / name).exists():
            shutil.copyfile(ROOT / 'bench/ra' / name, tool / name)
    stage = tmp_path / 'stage'
    stage.mkdir()
    body = b'"""Synthetic instrument, never executed."""\n'
    (stage / 'fixture.py').write_bytes(body)
    pins = {'source_commit': hashlib.sha1(b'synthetic-worker-fixture').hexdigest(),
            'instruments': {'bench/synthetic/fixture.py': hashlib.sha256(body).hexdigest()}}
    (stage / 'source-pins.json').write_text(json.dumps(pins))
    shutil.copyfile(stage / 'source-pins.json', tool / 'source-pins.json')
    staged = {'schema': 1, 'source_commit': pins['source_commit'], 'source_pins_sha256': sha(stage / 'source-pins.json'),
              'files': {'fixture.py': {'source_path': 'bench/synthetic/fixture.py', 'registered': True,
                                     'sha256': hashlib.sha256(body).hexdigest()}}}
    (stage / 'stage-manifest.json').write_text(json.dumps(staged))
    (stage / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + p.name + '\n' for p in sorted(stage.iterdir())))
    target = tool / worker.TARGETS[phase]
    target.write_text("import argparse,os,sys,json\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--spec');p.add_argument('--instruments');"
        "p.add_argument('--out');p.add_argument('--mode');p.add_argument('--profile',action='store_true');a=p.parse_args()\n"
        "assert 'ra_probe_e4b' in sys.modules and 'ra_probe_gnf4' in sys.modules\n"
        "Path(a.out).write_text(json.dumps({'pid':os.getpid(),'mode':a.mode,'profile':a.profile}))\n" + effect + '\n')
    native = tmp_path / 'native-spec.json'
    native.write_text('{}\n')
    job = {'schema': 1, 'phase': phase, 'tools': {n: sha(tool / n) for n in worker.TOOLS},
           'native_spec': {'path': str(native), 'sha256': sha(native)},
           'stage': {'path': str(stage), 'sha256': sha(stage / 'stage-manifest.json')},
           'native_out': str(tmp_path / 'native-result.json'), 'receipt': str(tmp_path / 'worker.json')}
    return manifest, job, python, tool, marker


def run(tmp_path, manifest, job, python, tool, *, env=None, expected=None, after_load=None):
    if env is None:
        env = dict(os.environ)
        if job['phase'] == 'quality':
            env['E4B_PAGED_GRAPHS'] = '0'
        elif job['phase'] == 'capacity':
            env.update(E4B_PAGED_MAX_TOKENS_PER_SEQ='2048', E4B_PAGED_CHUNK_TOKENS='512')
    m = tmp_path / 'handoff-spec.json'
    m.write_text(json.dumps(manifest))
    s = tmp_path / 'worker-spec.json'
    s.write_text(json.dumps(job))
    args = [str(python), '-I', '-S', '-B', str(tool / 'ra_handoff.py'), '--manifest', str(m),
        '--out', str(tmp_path / 'handoff.json'), '--worker-spec', str(s), '--worker-sha256', expected or sha(s)]
    if after_load:
        code = ("import sys,runpy,importlib.machinery\noriginal=importlib.machinery.SourceFileLoader.exec_module\n"
                "def injected(loader,module):\n    original(loader,module)\n"
                "    if module.__name__=='ra_verified_worker':\n        " + after_load + "\n"
                "importlib.machinery.SourceFileLoader.exec_module=injected\nsys.argv=" + repr(args[4:]) +
                "\nrunpy.run_path(" + repr(str(tool / 'ra_handoff.py')) + ",run_name='__main__')\n")
        args = [str(python), '-I', '-S', '-B', '-c', code]
    return subprocess.run(args, text=True, capture_output=True, env=env, timeout=45)


@pytest.mark.parametrize('phase', list(worker.TARGETS))
def test_fixed_worker_begins_after_handoff_in_same_process(tmp_path, phase):
    manifest, job, python, tool, marker = setup(tmp_path, phase=phase)
    p = run(tmp_path, manifest, job, python, tool)
    assert p.returncode == 0, p.stderr
    receipt = json.loads((tmp_path / 'worker.json').read_bytes())
    native = json.loads((tmp_path / 'native-result.json').read_bytes())
    handoff = json.loads((tmp_path / 'handoff.json').read_bytes())
    assert handoff['status'] == 'PASSED' and marker.exists()
    assert native['pid'] == receipt['pid']
    assert native['mode'] == (phase if phase in ('decode', 'quality') else None)
    assert native['profile'] is (phase == 'training_profile')
    assert receipt['status'] == 'WORKER_RETURNED_PENDING_GATES'
    assert receipt['nested_workers_verified'] is receipt['proves_gpu_engagement'] is receipt['release_cleared'] is False


@pytest.mark.parametrize('copies', [False, True])
def test_worker_without_startup_hook(tmp_path, copies):
    manifest, job, python, tool, marker = setup(tmp_path, hook=False, copies=copies)
    p = run(tmp_path, manifest, job, python, tool)
    assert p.returncode == 0, p.stderr


def test_installed_mutation_between_handoff_and_worker_refuses_before_target(tmp_path):
    manifest, job, python, tool, marker = setup(tmp_path)
    # Derive from the retained installed layout, without assuming minor width.
    site_file = next((Path(manifest['payload']['venv']) / 'lib').glob('python*/site-packages/ra_probe_e4b.py'))
    effect = "__import__('pathlib').Path(" + repr(str(site_file)) + ").write_text('changed')"
    p = run(tmp_path, manifest, job, python, tool, after_load=effect)
    assert p.returncode != 0
    record = json.loads((tmp_path / 'worker.json').read_text())
    assert record['phase'] == 'PRECONDITIONS' and 'since handoff' in record['error']
    assert json.loads((tmp_path / 'handoff.json').read_text())['status'] == 'PASSED'
    assert not Path(job['native_out']).exists()


@pytest.mark.parametrize('mutation', ['helper_pin', 'missing_tool', 'extra_pin', 'tool_bytes', 'unknown_tool',
    'native_pin', 'stage_pin', 'source_pins', 'stage_bytes', 'tool_symlink', 'stage_symlink',
    'phase', 'native_fields', 'stage_fields', 'native_relative', 'output_overlap', 'existing_output',
    'existing_receipt', 'env_override', 'spec_pin'])
def test_refusals_before_target(tmp_path, mutation):
    manifest, job, python, tool, marker = setup(tmp_path)
    expected, env = None, None
    if mutation == 'helper_pin':
        job['tools']['ra_verified_worker.py'] = '0' * 64
    elif mutation == 'missing_tool':
        del job['tools']['ra_training.py']
    elif mutation == 'extra_pin':
        job['tools']['external.py'] = '0' * 64
    elif mutation == 'tool_bytes':
        (tool / 'ra_env.py').write_text('raise RuntimeError("must not import")\n')
    elif mutation == 'unknown_tool':
        (tool / 'ra_unknown.py').write_text('raise RuntimeError("must not import")\n')
    elif mutation == 'native_pin':
        job['native_spec']['sha256'] = '0' * 64
    elif mutation == 'stage_pin':
        job['stage']['sha256'] = '0' * 64
    elif mutation == 'source_pins':
        (tool / 'source-pins.json').write_text('{}\n')
        job['tools']['source-pins.json'] = sha(tool / 'source-pins.json')
    elif mutation == 'stage_bytes':
        (Path(job['stage']['path']) / 'fixture.py').write_text('changed=1\n')
    elif mutation == 'tool_symlink':
        path = tool / 'ra_env.py'
        owned = tmp_path / 'owned-ra-env.py'
        path.rename(owned)
        path.symlink_to(owned)
    elif mutation == 'stage_symlink':
        stage = Path(job['stage']['path'])
        owned = tmp_path / 'owned-stage'
        stage.rename(owned)
        stage.symlink_to(owned, target_is_directory=True)
    elif mutation == 'phase':
        job['phase'] = 'external-command'
    elif mutation in ('native_fields', 'stage_fields'):
        job['native_spec' if mutation == 'native_fields' else 'stage']['extra'] = True
    elif mutation == 'native_relative':
        job['native_spec']['path'] = 'native-spec.json'
    elif mutation == 'output_overlap':
        job['native_out'] = str(tool / 'new-native.json')
    elif mutation == 'existing_output':
        Path(job['native_out']).write_text('already exists')
    elif mutation == 'existing_receipt':
        Path(job['receipt']).write_text('already exists')
    elif mutation == 'env_override':
        env = dict(os.environ, E4B_PAGED_FUSED_QKV='1')
    else:
        expected = '0' * 64
    p = run(tmp_path, manifest, job, python, tool, env=env, expected=expected)
    assert p.returncode != 0
    assert not Path(job['native_out']).exists() or mutation == 'existing_output'
    if mutation == 'existing_receipt':
        assert Path(job['receipt']).read_text() == 'already exists'
    if mutation == 'spec_pin':
        assert not marker.exists()


@pytest.mark.parametrize('effect', ["sys.path.append('/foreign')", "sys.meta_path.reverse()",
    "os.environ['RA_WORKER_MUTANT']='1'", "Path(a.spec).write_text('changed')",
    "Path(a.instruments).joinpath('fixture.py').write_text('changed')",
    "Path(__file__).with_name('ra_env.py').write_text('changed')",
    "import ra_probe_e4b;Path(ra_probe_e4b.__file__).write_text('changed')",
    "Path(a.out).parent.joinpath('handoff.json').write_text('changed')",
    "Path(a.out).parent.joinpath('synthetic-cpu-image.json').write_text('changed')",
    "Path(sys.prefix).joinpath('pyvenv.cfg').write_text('changed')",
    "Path(a.out).unlink()",
    "import types;m=types.ModuleType('foreign');m.__file__='/foreign/module.py';sys.modules['foreign']=m",
    "sys._base_executable='/foreign/python'", "raise RuntimeError('target failed')", "raise SystemExit(3)"])
def test_post_target_mutations_retain_failed_worker_and_passed_handoff(tmp_path, effect):
    manifest, job, python, tool, marker = setup(tmp_path, effect=effect)
    p = run(tmp_path, manifest, job, python, tool)
    assert p.returncode != 0
    receipt = json.loads((tmp_path / 'worker.json').read_bytes())
    assert receipt['status'] == 'FAILED' and receipt['phase'] in ('WORKER', 'POST_WORKER')
    assert receipt['handoff_evidence']['status'] == 'PASSED'
    if "joinpath('handoff.json')" not in effect:
        assert json.loads((tmp_path / 'handoff.json').read_bytes())['status'] == 'PASSED'
    assert Path(job['native_out']).exists() or effect == "Path(a.out).unlink()"
