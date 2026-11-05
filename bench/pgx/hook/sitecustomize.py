"""Loads the PGX routing hook (``usercustomize`` in this directory) in a venv, where Python never runs usercustomize.

``site`` imports ``usercustomize`` only when ``site.ENABLE_USER_SITE`` is true, and a venv created without
``--system-site-packages`` sets it false. Each engine on the box runs in its own such venv, so the hook would load in
no SGLang process. ``sitecustomize`` is imported in every interpreter, venv or not, from the first ``sys.path`` entry
that holds it, so this directory on PYTHONPATH loads the hook either way (an interpreter that also runs usercustomize
finds it already in ``sys.modules``). Found on the $0 path by ``tests/test_pgx_kit.py``, before any box ran it.
"""
import usercustomize  # noqa: F401
