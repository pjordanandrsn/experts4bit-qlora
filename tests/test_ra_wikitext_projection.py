"""Standalone projection CPU controls with named synthetic guard/transport only."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_ra_wikitext as raw_fixture
import test_ra_wikitext_text as parser_fixture

ROOT = Path(__file__).resolve().parents[1]
template = raw_fixture.template


@pytest.fixture
def owned(template, tmp_path):
    root = raw_fixture.owned.__wrapped__(template, tmp_path)
    for name in ('ra_wikitext_text.py', 'ra_wikitext_projection.py'):
        (root / 'tools' / name).write_bytes((ROOT / 'bench/ra' / name).read_bytes())
    data, _ = parser_fixture.archive()
    (root / 'inputs/datasets/wikitext-2-raw-v1/test-00000-of-00001.parquet').write_bytes(data)
    raw_fixture.write(root / 'transport.json', raw_fixture.records(data))
    manifest = json.loads((root / 'manifest.json').read_bytes())
    manifest['expected_parent'] = os.getpid()
    raw_fixture.write(root / 'manifest.json', manifest)
    raw_fixture.reseal(root)
    return root


def invoke(root, before='', during='', after='', guard=True, parent=False, out=None, process_change=''):
    """Explicit script substitutes only transport and (when named) host guard."""
    code = 'import sys,json,os,signal\nfrom pathlib import Path\n'
    code += 'sys.path.insert(0,' + repr(str(root / 'tools')) + ')\nimport ra_wikitext_projection as g\n'
    code += 'raw,text,i,cp=g.peers();m=i.read_json(' + repr(str(root / 'manifest.json')) + ')\n'
    code += 'data=i.read_json(' + repr(str(root / 'transport.json')) + ');count=0\n'
    code += 'class Replay(cp.Collector):\n def __call__(self,url):\n  global count\n  if url not in self.allowed:raise ValueError("unsafe control URL")\n  v=data[count];payload=json.dumps(v).encode();name="response-%02d.json"%count;count+=1;path=self.out/name;path.write_bytes(payload);self.records.append({"url":url,"file":name,"sha256":i.file_record(path)["sha256"],"bytes":len(payload),"started_at":cp.clock(),"finished_at":cp.clock()})\n'
    code += ''.join('  ' + line + '\n' for line in during.splitlines())
    code += '  return v\ncp.Collector=Replay\n'
    if guard:
        code += 'g.parent_guard=lambda m:{"pid":os.getpid(),"expected_parent":m["expected_parent"],"observed_parent":m["expected_parent"],"signal":signal.SIGKILL,"platform":"linux"}\n'
    code += before + '\n'
    output = out or root / 'out'
    code += 'sys.argv=["gate","--manifest",' + repr(str(root / 'manifest.json')) + ',"--out",' + repr(str(output)) + ']\ng.main()\n'
    code += after + '\n'
    if parent:
        code += 'p={"status":"OK","returncode":0,"cleanup_complete":True,"pid":os.getpid(),"parent_death_guard":{"required":True,"verified":True,"expected_parent":m["expected_parent"]}}\n'
        code += process_change + '\n'
        code += 'g.retained(m,Path(' + repr(str(output)) + '),p,i.file_record(' + repr(str(root / 'manifest.json')) + ')["sha256"])\n'
    path = root / 'explicit-host-synthetic-guard-and-replay.py'
    path.write_text(code)
    return subprocess.run([sys.executable, '-I', '-S', '-B', '-O', str(path)], capture_output=True, text=True, timeout=40)


def test_complete_projection_and_parent_regeneration(owned):
    q = invoke(owned, parent=True)
    assert q.returncode == 0, q.stderr
    result = json.loads((owned / 'out/result.json').read_bytes())
    assert result['row_count'] == 4 and result['nonblank_rows'] == 2
    assert (owned / 'out/joined-text.txt').read_bytes() == 'alpha\n\né\n'.encode()
    assert json.loads((owned / 'out/ordered-rows.json').read_bytes()) == ['', 'alpha', 'é\n', ' \t']
    assert result['proves_raw_wikitext_source_and_text_projection'] is True
    assert all(result[k] is False for k in result if k.startswith('proves_') and k != 'proves_raw_wikitext_source_and_text_projection')
    assert len(list((owned / 'out').iterdir())) == 7


@pytest.mark.parametrize('change', ['extra', 'schema', 'parent_bool', 'parent_zero', 'lock_hash'])
def test_fixed_manifest(owned, change):
    path = owned / 'manifest.json'
    m = json.loads(path.read_bytes())
    if change == 'extra':
        m['command'] = 'caller code'
    elif change == 'schema':
        m['schema'] = True
    elif change == 'parent_bool':
        m['expected_parent'] = True
    elif change == 'parent_zero':
        m['expected_parent'] = 0
    else:
        m['input_lock_sha256'] = '0' * 64
    raw_fixture.write(path, m)
    q = invoke(owned)
    assert q.returncode != 0


def test_host_guard_refuses_with_failed_receipt(owned):
    q = invoke(owned, guard=False)
    assert q.returncode != 0
    result = json.loads((owned / 'out/result.json').read_bytes())
    assert result['status'] == 'FAILED' and result['observations'] == []
    assert 'selected Linux' in result['error']


@pytest.mark.parametrize('change', ['rows', 'join', 'summary', 'extra_summary', 'raw', 'resealed_raw', 'url', 'clock', 'sequence', 'record_extra', 'extra_file', 'symlink'])
def test_parent_refuses_resealed_or_incomplete_receipts(owned, change):
    after = 'out=Path(' + repr(str(owned / 'out')) + ');r=i.read_json(out/"result.json")\n'
    if change in ('rows', 'join'):
        name, key = ('ordered-rows.json', 'ordered_rows') if change == 'rows' else ('joined-text.txt', 'joined_text')
        after += f'(out/{name!r}).write_bytes(b"resealed mutant");r[{key!r}]=i.file_record(out/{name!r})\n'
    elif change == 'summary':
        after += 'r["row_count"]=999\n'
    elif change == 'extra_summary':
        after += 'r["cleared"]=True\n'
    elif change in ('raw', 'resealed_raw'):
        after += 'v=i.read_json(out/"response-00.json");v["sha"]="b"*40;(out/"response-00.json").write_text(json.dumps(v))\n'
        if change == 'resealed_raw':
            after += 'r["observations"][0].update(sha256=i.file_record(out/"response-00.json")["sha256"],bytes=(out/"response-00.json").stat().st_size)\n'
    elif change == 'url':
        after += 'r["observations"][0]["url"]="http://unsafe.test/"\n'
    elif change == 'clock':
        after += 'r["observations"][0]["started_at"]="2026-01-01T00:00:00+01:00"\n'
    elif change == 'sequence':
        after += 'r["observations"][3]["started_at"]="2000-01-01T00:00:00+00:00"\n'
    elif change == 'record_extra':
        after += 'r["observations"][0]["authority"]=True\n'
    elif change == 'extra_file':
        after += '(out/"extra").write_bytes(b"extra")\n'
    else:
        after += '(out/"joined-text.txt").rename(out/"moved");(out/"joined-text.txt").symlink_to(out/"moved")\n'
    after += '(out/"result.json").write_text(json.dumps(r))\n'
    q = invoke(owned, after=after, parent=True)
    assert q.returncode != 0, q.stdout


@pytest.mark.parametrize('change', ['archive', 'helper', 'input', 'manifest', 'stage', 'extra_receipt', 'index'])
def test_mid_observation_drift_retains_failure_and_prefix(owned, change):
    paths = {'archive': owned / 'inputs/datasets/wikitext-2-raw-v1/test-00000-of-00001.parquet',
             'helper': owned / 'tools/ra_wikitext_text.py', 'input': owned / 'input-spec.json',
             'manifest': owned / 'manifest.json', 'stage': owned / 'stage/p109_box.py'}
    if change in paths:
        during = 'if count==3:Path(' + repr(str(paths[change])) + ').write_bytes(Path(' + repr(str(paths[change])) + ').read_bytes()+b" ")'
    elif change == 'extra_receipt':
        during = 'if count==3:(self.out/"extra").write_bytes(b"unexpected")'
    else:
        during = 'if count==3:v["sha"]="b"*40'
    q = invoke(owned, during=during)
    assert q.returncode != 0
    result = json.loads((owned / 'out/result.json').read_bytes())
    assert result['status'] == 'FAILED' and len(result['observations']) >= 3
    assert result['error_type'] in ('ValueError', 'SyntaxError', 'Refused')


@pytest.mark.parametrize('role', ['datasets', 'stage', 'helper', 'ancestor', 'existing'])
def test_output_protection(owned, role):
    out = {'datasets': owned / 'inputs/datasets/new', 'stage': owned / 'stage/new',
           'helper': owned / 'tools/new', 'ancestor': owned, 'existing': owned / 'transport.json'}[role]
    q = invoke(owned, out=out)
    assert q.returncode != 0 and 'fresh protected' in q.stderr


@pytest.mark.parametrize('change', ['separator', 'filter', 'column', 'tokenizer', 'slice'])
def test_registered_recipe_not_caller_text(owned, change):
    replacements = {'separator': (b'"\\n\\n".join', b'" ".join'), 'filter': (b'if t.strip()', b'if t'),
                    'column': (b'ds["text"]', b'ds["other"]'), 'tokenizer': (b'return_tensors="pt"', b'return_tensors="np"'),
                    'slice': (b'tok(text,', b'tok(text[:5],')}
    p = owned / 'stage/p109_box.py'
    original = p.read_bytes()
    a, b = replacements[change]
    assert a in original
    p.write_bytes(original.replace(a,b))
    # Owned synthetic pinset is resealed to reach the independent AST recipe check.
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    for pins_path in (owned / 'stage/source-pins.json', owned / 'tools/source-pins.json'):
        pins = json.loads(pins_path.read_bytes())
        pins['instruments']['bench/p109/p109_box.py'] = digest
        raw_fixture.write(pins_path,pins)
    stage = owned / 'stage'
    m = json.loads((stage/'stage-manifest.json').read_bytes())
    m['source_pins_sha256'] = raw_fixture.sha(stage/'source-pins.json')
    m['files']['p109_box.py']['sha256']=digest
    raw_fixture.write(stage/'stage-manifest.json',m)
    (stage/'SHA256SUMS').write_text(''.join(raw_fixture.sha(q)+'  '+q.name+'\n' for q in sorted(stage.iterdir()) if q.name!='SHA256SUMS'))
    q = invoke(owned)
    assert q.returncode != 0
    assert 'registered full text join/tokenizer call' in q.stderr


def test_foreign_preloaded_parser_refuses(owned):
    q = invoke(owned, before='import types\nsys.modules["ra_wikitext_text"]=types.SimpleNamespace(__file__="/foreign/ra_wikitext_text.py")')
    assert q.returncode != 0 and 'foreign helper' in q.stderr


@pytest.mark.parametrize('change', ['status', 'returncode', 'null_returncode', 'bool_returncode', 'cleanup', 'pid', 'guard', 'parent'])
def test_parent_requires_verified_process_and_cleanup(owned, change):
    mutations = {'status': 'p["status"]="TIMEOUT"', 'returncode': 'p["returncode"]=1',
                 'null_returncode': 'p["returncode"]=None', 'bool_returncode': 'p["returncode"]=False',
                 'cleanup': 'p["cleanup_complete"]=False', 'pid': 'p["pid"]=True',
                 'guard': 'p["parent_death_guard"]["verified"]=False',
                 'parent': 'p["parent_death_guard"]["expected_parent"]+=1'}
    q = invoke(owned, parent=True, process_change=mutations[change])
    assert q.returncode != 0 and 'verified process/guard/cleanup required' in q.stderr


@pytest.mark.parametrize('target', ['manifest', 'helper', 'input'])
def test_post_output_drift_refuses(owned, target):
    path = {'manifest': owned / 'manifest.json', 'helper': owned / 'tools/ra_wikitext_text.py',
            'input': owned / 'input-spec.json'}[target]
    before = 'original_record=i.file_record\ndef changing_record(path):\n record=original_record(path)\n if Path(path).name=="joined-text.txt":\n  p=Path(' + repr(str(path)) + ');p.write_bytes(p.read_bytes()+b" ")\n return record\ni.file_record=changing_record'
    q = invoke(owned, before=before)
    assert q.returncode != 0
    result = json.loads((owned / 'out/result.json').read_bytes())
    assert result['status'] == 'FAILED' and len(result['observations']) == 4
    assert 'final input/helper/guard/receipt drift' in result['error']


@pytest.mark.parametrize('target', ['spec', 'helper', 'source_pins'])
def test_parent_binds_child_input_and_helper_bytes(owned, target):
    path = {'spec': owned / 'input-spec.json', 'helper': owned / 'tools/ra_wikitext_text.py',
            'source_pins': owned / 'tools/source-pins.json'}[target]
    after = 'p=Path(' + repr(str(path)) + ');p.write_bytes(p.read_bytes()+b" ")'
    q = invoke(owned, after=after, parent=True)
    assert q.returncode != 0
    assert 'parent complete rows/text/result differs' in q.stderr or 'stage pins differ from registered RA pins' in q.stderr
