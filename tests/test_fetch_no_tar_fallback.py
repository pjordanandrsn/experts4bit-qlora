"""bench/common rule 8 in the G1/G3 smoke drivers: the fetch is rsync --no-links with NO tar fallback (a box is root and
can make its own rsync fail on purpose; a tar stream would import links), and a link in the fetched tree is refused
before anything reads it. The drivers' own fetch blocks run here against stubbed rsync and tar."""
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
KITS = {"g1": ("bench/kv-pool-g/g1-smoke/g1_smoke_drive.sh", "g1s"), "g3": ("bench/kv-pool-g/g3-smoke/g3_smoke_drive.sh", "g3s")}


def _fetch_block(kit):
    path, sub = KITS[kit]
    src = (REPO / path).read_text()
    start = src.index(f'rm -rf "$RUN_DIR/{sub}"')
    end = src.index('say "fetched $(ls')
    return src[start:end], sub


def _run(tmp_path, kit, rsync_body):
    block, sub = _fetch_block(kit)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "rsync").write_text("#!/bin/bash\n" + rsync_body + "\n")
    (bin_ / "tar").write_text('#!/bin/bash\necho TAR_CALLED >> "$TAR_LOG"\nexit 0\n')
    for f in ("rsync", "tar"):
        os.chmod(bin_ / f, 0o755)
    script = ('say(){ echo "$*"; }\nset -uo pipefail\nRUN_DIR="$1"; HOST=h; PORT=1; W=/root/w; E4B_RENT_SSH_OPTS=""\n'
              'SSH="false"\n' + block + 'echo FETCH_OK\n')
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    env = {"PATH": f"{bin_}:/usr/bin:/bin", "TAR_LOG": str(tmp_path / "tar.log"), "DEST_SUB": sub}
    out = subprocess.run(["bash", "-c", script, "drive", str(run_dir)], capture_output=True, text=True, env=env)
    return out, (tmp_path / "tar.log").exists()


@pytest.mark.parametrize("kit", sorted(KITS))
def test_the_fetch_carries_no_links_and_no_tar_fallback(kit):
    block, _ = _fetch_block(kit)
    assert "rsync -az --no-links " in block and "tar -C" not in block and "| tar" not in block


@pytest.mark.parametrize("kit", sorted(KITS))
def test_a_failed_rsync_is_a_fetch_failure_with_no_second_path(tmp_path, kit):
    out, tar_called = _run(tmp_path, kit, "exit 23")
    assert out.returncode == 22 and "FETCH_OK" not in out.stdout and not tar_called, out.stdout + out.stderr


@pytest.mark.parametrize("kit", sorted(KITS))
def test_a_link_in_the_fetched_tree_is_refused_before_any_read(tmp_path, kit):
    body = 'dest="${@: -1}"; mkdir -p "$dest"; echo n > "$dest/G1S_RUN_NONCE"; ln -s /etc/hosts "$dest/summary.txt"; exit 0'
    out, _ = _run(tmp_path, kit, body)
    assert out.returncode == 22 and "symlink" in out.stdout and "FETCH_OK" not in out.stdout, out.stdout + out.stderr


@pytest.mark.parametrize("kit", sorted(KITS))
def test_a_clean_fetch_passes(tmp_path, kit):
    body = 'dest="${@: -1}"; mkdir -p "$dest"; echo n > "$dest/summary.txt"; exit 0'
    out, tar_called = _run(tmp_path, kit, body)
    assert out.returncode == 0 and "FETCH_OK" in out.stdout and not tar_called, out.stdout + out.stderr
