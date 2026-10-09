"""Synthetic WikiText index/archive controls, not decoding or GPU proof."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import test_ra_inputs as inputs_fixture

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def records(raw):
    size, digest = len(raw), hashlib.sha256(raw).hexdigest()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {size}\n".encode()
    name = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
    info = {
        "id": "Salesforce/wikitext",
        "sha": REVISION,
        "private": False,
        "gated": False,
        "disabled": False,
        "siblings": [
            {
                "rfilename": name,
                "blobId": git_blob(pointer),
                "size": size,
                "lfs": {"sha256": digest, "size": size, "pointerSize": len(pointer)},
            }
        ],
    }
    tree = [
        {"type": "directory", "path": "wikitext-2-raw-v1", "oid": "d" * 40},
        {
            "type": "file",
            "path": name,
            "oid": git_blob(pointer),
            "size": size,
            "lfs": {"oid": digest, "size": size, "pointerSize": len(pointer)},
        },
    ]
    return [info, tree, copy.deepcopy(info), copy.deepcopy(tree)]


@pytest.fixture(scope="session")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("wikitext-raw-fixture").resolve()
    tools = root / "tools"
    tools.mkdir()
    for name in ("ra_wikitext.py", "ra_checkpoint.py", "ra_inputs.py", "ra_stage.py"):
        (tools / name).write_bytes((ROOT / "bench/ra" / name).read_bytes())
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    used = {"bench/p109/p109_box.py", "bench/p97/p97_box.py", "bench/p115/p115_quality.py", "bench/sc2/sc2_driver.py"}
    pins["instruments"] = {k: v for k, v in pins["instruments"].items() if k in used}
    write(tools / "source-pins.json", pins)
    stage = root / "stage"
    stage.mkdir()
    entries = {}
    for name, digest in pins["instruments"].items():
        raw = (ROOT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        (stage / Path(name).name).write_bytes(raw)
        entries[Path(name).name] = {"source_path": name, "registered": True, "sha256": digest}
    write(stage / "source-pins.json", pins)
    write(
        stage / "stage-manifest.json",
        {
            "schema": 1,
            "source_commit": pins["source_commit"],
            "source_pins_sha256": sha(stage / "source-pins.json"),
            "files": entries,
        },
    )
    (stage / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in sorted(stage.iterdir())))
    spec = inputs_fixture.fixture(root / "inputs")
    datasets = Path(spec["trees"]["datasets"])
    filename = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
    raw = b"PAR1synthetic envelope only; not decoded Parquet" + (10).to_bytes(4, "little") + b"PAR1"
    (datasets / "wikitext-2-raw-v1").mkdir()
    (datasets / filename).write_bytes(raw)
    write(
        datasets / "wikitext-authority.json",
        {
            "schema": 1,
            "repo": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "revision": REVISION,
            "filename": filename,
        },
    )
    write(root / "input-spec.json", spec)
    write(root / "transport.json", records(raw))
    return root


@pytest.fixture
def owned(template, tmp_path):
    root = tmp_path.resolve() / "own"
    shutil.copytree(template, root)
    spec = json.loads((root / "input-spec.json").read_bytes())
    for group in ("trees", "files"):
        spec[group] = {k: str(root / Path(v).relative_to(template)) for k, v in spec[group].items()}
    write(root / "input-spec.json", spec)
    manifest = {
        "schema": 1,
        "input_spec": str(root / "input-spec.json"),
        "stage": str(root / "stage"),
        "input_lock": str(root / "lock.json"),
        "input_lock_sha256": "0" * 64,
    }
    write(root / "manifest.json", manifest)
    reseal(root)
    return root


def reseal(root):
    code = "import sys,json\nfrom pathlib import Path\nsys.path.insert(0," + repr(str(root / "tools")) + ")\n"
    code += "import ra_wikitext as g\ni,_=g.peers();m=i.read_json(" + repr(str(root / "manifest.json")) + ")\n"
    code += (
        "p=Path(m['input_lock']);p.write_text(json.dumps(i.inspect(i.read_json(m['input_spec']),m['stage'])));m['input_lock_sha256']=i.file_record(p)['sha256'];Path("
        + repr(str(root / "manifest.json"))
        + ").write_text(json.dumps(m))\n"
    )
    q = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, timeout=30)
    assert q.returncode == 0, q.stderr


def invoke(root, mutate="", flags=True, out=None):
    out = out or root / "out"
    code = (
        "import sys,json\nfrom pathlib import Path\nsys.path.insert(0,"
        + repr(str(root / "tools"))
        + ")\nimport ra_wikitext as g\ni,cp=g.peers()\n"
    )
    code += "data=i.read_json(" + repr(str(root / "transport.json")) + ");it=iter(data);count=0\n"
    code += "class Fake(cp.Collector):\n def __call__(self,url):\n  global count\n  assert url in self.allowed\n  v=next(it);raw=json.dumps(v).encode();name='response-%02d.json'%count;count+=1;path=self.out/name;path.write_bytes(raw);self.records.append({'url':url,'file':name,'sha256':i.file_record(path)['sha256'],'bytes':len(raw),'started_at':cp.clock(),'finished_at':cp.clock()});return v\n"
    code += (
        mutate
        + "\ncp.Collector=Fake\nsys.argv=['gate','--manifest',"
        + repr(str(root / "manifest.json"))
        + ", '--out',"
        + repr(str(out))
        + "]\ng.main()\n"
    )
    (root / "control.py").write_text(code)
    return subprocess.run(
        [sys.executable, *(["-I", "-S", "-B"] if flags else []), str(root / "control.py")],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "revision",
        "repo",
        "config",
        "split",
        "filename",
        "extra",
        "bool_schema",
        "payload",
        "magic",
        "footer",
        "footer_size",
        "recipe",
        "symlink",
    ],
)
def test_reviewed_raw_archive_and_fixed_recipe(owned, mutation):
    root = owned
    spec = json.loads((root / "input-spec.json").read_bytes())
    datasets = Path(spec["trees"]["datasets"])
    authority = datasets / "wikitext-authority.json"
    archive = datasets / "wikitext-2-raw-v1/test-00000-of-00001.parquet"
    if mutation in ("revision", "repo", "config", "split", "filename", "extra", "bool_schema"):
        v = json.loads(authority.read_bytes())
        if mutation == "revision":
            v["revision"] = "b" * 40
        elif mutation == "extra":
            v["transport"] = "caller URL"
        elif mutation == "bool_schema":
            v["schema"] = True
        else:
            v[mutation] = "mutant"
        write(authority, v)
        reseal(root)
    elif mutation in ("payload", "magic", "footer", "footer_size"):
        raw = archive.read_bytes()
        if mutation == "payload":
            raw = raw[:5] + b"mutant" + raw[5:]
        elif mutation == "magic":
            raw = b"bad!" + raw[4:]
        elif mutation == "footer":
            raw = raw[:-4] + b"bad!"
        else:
            raw = raw[:-8] + (len(raw) + 1).to_bytes(4, "little") + raw[-4:]
        archive.write_bytes(raw)
        # Reconcile the envelope mutants with their OWN resealed authority.
        if mutation != "payload":
            write(root / "transport.json", records(raw))
        reseal(root)
    elif mutation == "recipe":
        p = root / "stage/p109_box.py"
        p.write_bytes(p.read_bytes().replace(b"Salesforce/wikitext", b"mutant/wikitext"))
    elif mutation == "symlink":
        archive.rename(archive.with_suffix(".moved"))
        archive.symlink_to(archive.with_suffix(".moved"))
    change = ""
    q = invoke(root, change)
    if mutation:
        assert q.returncode != 0, q.stdout
    else:
        assert q.returncode == 0, q.stderr
        result = json.loads((root / "out/result.json").read_bytes())
        assert result["proves_hub_index_raw_wikitext_equality"] is True
        assert all(
            result[k] is False
            for k in (
                "proves_parquet_decoding",
                "proves_joined_text_equality",
                "proves_tokenizer_execution",
                "proves_runtime_consumption",
                "proves_native_loader_revision_binding",
                "proves_gpu_engagement",
                "launch_authority",
            )
        )
        assert len(result["observations"]) == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "id",
        "revision",
        "private",
        "gated",
        "disabled",
        "duplicate",
        "missing",
        "extra_shard",
        "blob",
        "lfs_hash",
        "pointer",
        "size_bool",
        "dir",
        "unsafe",
        "changed",
        "ordinary",
    ],
)
def test_complete_hub_index_reconciliation(owned, mutation):
    path = owned / "transport.json"
    data = json.loads(path.read_bytes())
    info, tree = data[:2]
    if mutation in ("id", "revision", "private", "gated", "disabled"):
        info[{"revision": "sha"}.get(mutation, mutation)] = (
            True if mutation in ("private", "gated", "disabled") else "bad"
        )
    elif mutation == "duplicate":
        info["siblings"].append(copy.deepcopy(info["siblings"][0]))
    elif mutation == "missing":
        info["siblings"] = []
    elif mutation == "extra_shard":
        sibling = copy.deepcopy(info["siblings"][0])
        row = copy.deepcopy(tree[1])
        sibling["rfilename"] = sibling["rfilename"].replace("00000", "00001")
        row["path"] = sibling["rfilename"]
        info["siblings"].append(sibling)
        tree.append(row)
    elif mutation == "blob":
        info["siblings"][0]["blobId"] = "0" * 40
    elif mutation == "lfs_hash":
        info["siblings"][0]["lfs"]["sha256"] = "0" * 64
    elif mutation == "pointer":
        tree[1]["lfs"]["pointerSize"] += 1
    elif mutation == "size_bool":
        tree[1]["size"] = True
    elif mutation == "dir":
        tree.pop(0)
    elif mutation == "unsafe":
        tree[1]["path"] = "../escaped"
    elif mutation == "changed":
        data[2]["sha"] = "b" * 40
    elif mutation == "ordinary":
        del tree[1]["lfs"]
        del info["siblings"][0]["lfs"]
    write(path, data)
    q = invoke(owned)
    assert q.returncode != 0, q.stdout
    result = json.loads((owned / "out/result.json").read_bytes())
    assert result["status"] == "FAILED" and result["observations"]


def test_no_site_flags_and_protected_output(owned):
    q = invoke(owned, flags=False)
    assert q.returncode != 0 and "-I -S -B required" in q.stderr
    q = invoke(owned, out=Path(json.loads((owned / "input-spec.json").read_bytes())["trees"]["datasets"]) / "out")
    assert q.returncode != 0 and "protected receipt" in q.stderr
