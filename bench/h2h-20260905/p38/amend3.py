# P38 amendment 3: the Unsloth branch resolves the pinned snapshot directory directly (same bytes); huggingface_hub 1.30.0's
# offline snapshot_download(revision=<sha>) raises LocalEntryNotFoundError on this cache even though the snapshot is present.
import re, pathlib, shutil
p = pathlib.Path("/root/p38/p38_arm.py"); bak = pathlib.Path("/root/p38/p38_arm.py.pre_amend3")
if bak.exists(): shutil.copy(bak, p)          # start from the unpatched harness
s = p.read_text()
m = re.search(r"^(\s*)local = snapshot_download\(a\.model, revision=a\.revision\)(.*)$", s, re.M)
assert m, "pattern not found"
ind = m.group(1)
new = (ind + "_cand = os.path.join(os.path.expanduser(os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/hub')), 'models--' + a.model.replace('/', '--'), 'snapshots', a.revision)  # amendment 3: the pinned snapshot dir, same bytes\n"
       + ind + "local = _cand if os.path.isdir(_cand) else snapshot_download(a.model, revision=a.revision)" + m.group(2))
s = s[:m.start()] + new + s[m.end():]
assert re.search(r"^import .*\bos\b|^import os", s, re.M), "os not imported"
p.write_text(s); print("patched")
