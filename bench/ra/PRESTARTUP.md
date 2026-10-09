# Installed payload before startup

Run `ra_prestartup.py --manifest <absolute path> --out <new absolute path>`
with the selected venv's `python -I -S -B`. The schema-1 manifest contains
`venv`, the retained `wheels` path/hash pins, `image` and `image_sha256`, and
`startup_registry_sha256`. Its image pin binds the running interpreter binary,
version, ABI and platform. `pyvenv.cfg` must exclude system site packages and
identify CPython's `_base_executable` directory, including when `bin/python`
is a copied executable. Both selected and base binaries must match the image
hash and are rechecked after installer-template imports; an optional config
`executable` must identify that same base. Receipts distinguish copied from
symlinked executables. This avoids assuming a copied venv binary lives in the
base directory.

The check derives the venv site path without enabling site, including on Python
3.11 where `-S` leaves `sys.prefix` at the base. Installed payload, complete
distribution inventory, RECORD ownership and metadata are checked before
installer imports. It then adds only verified site paths to derive generated
scripts from verified pip bytes, checks loaded module ownership and rehashes
installed files. Unlisted files, bytecode, symlinks, editable origins, unknown
installer metadata and modified or missing scripts refuse.

The proposed setuptools registry permits inspection of exact startup bytes;
no `.pth` runs, no release module is imported, and the existing import probe
still refuses all hooks. This verifies a supplied venv's installed payload.
It grants no full-image installation, release/source binding, GPU engagement,
startup execution or launch authority. Synthetic CPU venvs are controls; the
full Linux proof installation remains part of the reviewed proof run.

Native selected-Linux controls materialized two tiny synthetic wheels (eight
payload files) in copied and symlinked venvs; both passed without startup or
release imports. A copied-binary mutation executed CPython, then refused at
the image-hash gate. The native host libc differs from the proof image.
