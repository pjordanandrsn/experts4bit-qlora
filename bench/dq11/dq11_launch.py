"""Mandatory CPU wheel gate BEFORE the guarded rental controller can quote."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from dq11_bootstrap import preflight_urls
from dq11_common import file_sha
from dq11_science_stage import stage


def launch(manifest_path, approval, gate_output, *, repository=None, gate_only=False):
    if any(key.startswith("DQ11_REHEARSAL") for key in os.environ) or "CUBLAS_WORKSPACE_CONFIG" in os.environ:
        raise ValueError("science launch refuses inherited rehearsal/proof overrides BEFORE gate/quote/rental")
    repository = Path(repository or Path(__file__).resolve().parents[2]).resolve()
    manifest_path, gate_output = Path(manifest_path).resolve(), Path(gate_output).resolve()
    if gate_output.is_relative_to(repository):
        raise ValueError("gate output must be outside the source checkout")
    manifest = json.loads(manifest_path.read_text())
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip():
        raise ValueError("DQ11 refuses dirty launch source")
    if subprocess.call(["git", "-C", str(repository), "merge-base", "--is-ancestor", head, "origin/main"]):
        raise ValueError("DQ11 refuses unmerged launch source")
    if manifest["heads"]["e4b"] != head:
        raise ValueError("manifest does not bind the launch source")
    directory = repository / "bench/dq11"
    stage(directory, Path(os.environ["DQ11_ASSET_DIR"]))
    report = preflight_urls(directory)
    report.update(e4b_sha=head, utc=datetime.now(timezone.utc).isoformat(),
                  manifest_sha256=file_sha(manifest_path), science_sha256=file_sha(directory / "science.sha256"))
    gate_output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError("DQ11 wheel fetch gate refused BEFORE quote/rental; see " + str(gate_output))
    if gate_only:
        return 0
    guard = Path(os.environ.get("ADERTHA_REPO", str(Path.home() / "code/adertha-main"))) / "tools/pod-launch.sh"
    bash = os.environ.get("POD_LAUNCH_BASH", "/opt/homebrew/bin/bash" if Path("/opt/homebrew/bin/bash").exists() else "bash")
    env = dict(os.environ, E4B_REPO=str(repository))
    return subprocess.run([bash, str(guard), str(manifest_path), approval], env=env, check=False).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("approval")
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--gate-only", action="store_true", help="CPU gate before read-only quote inspection; no rental")
    args = parser.parse_args()
    try:
        code = launch(args.manifest, args.approval, args.gate_output, gate_only=args.gate_only)
    except (ValueError, OSError) as error:
        parser.exit(78, str(error) + "\n")
    raise SystemExit(code)
