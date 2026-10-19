"""support_probe.py's scrub: a recorded error keeps its module-relative tail and loses every host root.

Rows under bench/support/rows/ are committed to a public repository. The model path was already scrubbed; exception text
also quotes the interpreter's own directories (a transformers ImportError names its site-packages file), so
``scrub_host_paths`` replaces sys.prefix, sys.base_prefix, the site-packages roots and the home directory with ``...``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "support"))

from support_probe import _host_roots, scrub_host_paths  # noqa: E402

TAIL = "transformers/models/auto/modeling_auto.py"


def _import_error(root: str) -> str:
    return f"ImportError: cannot import name 'X' from 'transformers' ({root}/{TAIL})"


def test_every_host_root_of_this_interpreter_is_scrubbed_and_the_tail_survives():
    roots = _host_roots()
    assert roots, "no host roots found: the scrub would be a no-op"
    for r in roots:
        out = scrub_host_paths(_import_error(r))
        assert TAIL in out, (r, out)
        assert not any(x in out for x in roots), (r, out)


def test_each_prefix_form_is_a_candidate():
    import site
    roots = set(_host_roots())
    forms = {sys.prefix, sys.base_prefix, str(Path.home())}
    forms.update(getattr(site, "getsitepackages", lambda: [])())
    forms.add(getattr(site, "getusersitepackages", lambda: "")())
    for f in forms:
        if f and len(f) > 1 and Path(f).anchor != f:
            assert f in roots or str(Path(f).resolve()) in roots, f


def test_longest_first_keeps_the_module_relative_tail_whatever_the_order_given():
    home = "/home/u"
    prefix = home + "/.venvs/e4b"
    sp = prefix + "/lib/python3.14/site-packages"
    user_sp = home + "/.local/lib/python3.14/site-packages"
    for order in ([home, prefix, sp, user_sp], [sp, user_sp, prefix, home], [prefix, home, user_sp, sp]):
        assert scrub_host_paths(_import_error(sp), order) == _import_error("...")
        assert scrub_host_paths(_import_error(user_sp), order) == _import_error("...")
        assert scrub_host_paths(f"{prefix}/bin/python -m x", order) == ".../bin/python -m x"
        assert scrub_host_paths(f"cache at {home}/.cache/huggingface", order) == "cache at .../.cache/huggingface"


def test_a_root_is_replaced_only_at_a_path_boundary():
    roots = ["/root"]
    assert scrub_host_paths("/root/x.py", roots) == ".../x.py"
    assert scrub_host_paths("in '/root'", roots) == "in '...'"
    assert scrub_host_paths("/root", roots) == "..."
    assert scrub_host_paths("/rootfs/x.py", roots) == "/rootfs/x.py"


def test_the_filesystem_root_and_one_character_roots_are_never_candidates():
    assert all(len(r) > 1 and Path(r).anchor != r for r in _host_roots())


def test_the_probe_scrubs_host_roots_after_the_model_path():
    src = (ROOT / "bench" / "support" / "support_probe.py").read_text(encoding="utf-8")
    body = src[src.index("    def scrub(text: str) -> str:"):src.index("    def fail(stage")]
    assert body.index(".replace(args.model, Path(args.model).name)") < body.index("return scrub_host_paths(text)")
